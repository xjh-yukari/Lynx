"""Lynx 适配器：进程内调用 AgentRunner，并采集内部诊断事件。

本文件是 bench 里唯一允许导入 lynx_agent 的模块。诊断数据（压缩摘要、
工具结果回收、记忆召回）只用于展示，不参与计分。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from time import monotonic
from typing import Any

from agent_bench.adapters.base import AdapterError, BaseAdapter, BenchEnv
from agent_bench.env_mcp.server import TOOL_NAMES
from agent_bench.models import TurnResult

READ_ONLY_TOOLS = (
    "notes_get",
    "notes_search",
    "orders_lookup",
    "documents_read",
    "web_page",
    "sandbox_read_file",
    "sandbox_list_files",
)

SAFETY_TOKENS = 1024


class LynxAdapter(BaseAdapter):
    kind = "lynx"

    def __init__(self, name: str, settings, context) -> None:
        super().__init__(name, settings, context)
        self._agent = None
        self._provider = None
        self._config = None
        self._store = None
        self._events: list[dict[str, Any]] = []
        self.close_error: str | None = None

    # ---- 生命周期 ----
    async def setup(self, workspace: Path, env: BenchEnv) -> None:
        await super().setup(workspace, env)
        if self.workspace is None:
            raise AdapterError("Lynx 适配器缺少工作区")
        if self.settings.kind != "lynx":
            raise AdapterError("LynxAdapter 只接受 lynx 类型的配置")
        self._config = self._build_config()
        if self._config.memory.enabled and Path(self._config.memory.executable).is_file():
            self.capabilities.add("cross_session_memory")
            os.environ.setdefault("LYNX_BENCH_MODE", "1")
        else:
            self.capabilities.discard("cross_session_memory")
        if self._config.mcp.servers:
            self.capabilities.add("mcp")
        await self.new_session()

    async def new_session(self) -> str:
        await self._close_agent()
        if self._config is None:
            raise AdapterError("Lynx 适配器尚未完成 setup")
        from lynx_agent.agent import AgentRunner
        from lynx_agent.providers import create_provider
        from lynx_agent.session import SessionStore

        self._store = SessionStore(self.sessions_dir)
        session = self._store.create()
        self._provider = create_provider(self._config)
        tools = self._tools()
        self._agent = AgentRunner(self._config, self._provider, self._store, session,
                                  tools=tools, on_event=self._on_event)
        try:
            await self._agent.__aenter__()
        except Exception as error:
            raise AdapterError(f"Lynx 启动失败：{type(error).__name__}") from error
        return self._agent.session_id or ""

    async def reset_memory(self) -> None:
        await self._close_agent()
        if self.memory_dir.exists():
            shutil.rmtree(self.memory_dir, ignore_errors=True)
        if self.sessions_dir.exists():
            shutil.rmtree(self.sessions_dir, ignore_errors=True)
        await self.new_session()

    async def teardown(self) -> None:
        await self._close_agent()

    async def _close_agent(self) -> None:
        if self._agent is None:
            return
        try:
            await self._agent.aclose()
        except Exception as error:  # noqa: BLE001 - 关闭失败不掩盖主流程结果
            self.close_error = type(error).__name__
        self._agent = None

    # ---- 单轮 ----
    async def send(self, message: str) -> TurnResult:
        if self._agent is None:
            return self._denied()
        started = monotonic()
        marker = len(self._events)
        try:
            response = await asyncio.wait_for(
                self._agent.ask(message), timeout=self.context.turn_timeout_seconds)
        except TimeoutError:
            return TurnResult(text="", elapsed_seconds=monotonic() - started,
                              error="本轮超过超时时间", finish_reason="timeout")
        except Exception as error:  # noqa: BLE001 - 任何异常都要变成可记录的结果
            return TurnResult(text="", elapsed_seconds=monotonic() - started,
                              error=f"{type(error).__name__}: {error}", finish_reason="error")
        events = self._events[marker:]
        diagnostics = {
            "session": self._agent.session_id or "",
            "events": events,
            "compaction": [event for event in events if event.get("type") == "compaction"],
            "prune": [event for event in events if event.get("type") == "prune"],
            "memory_recall": [event for event in events if event.get("type") == "memory_recall"],
            "usage": [event for event in events if event.get("type") == "usage"],
        }
        recall = self._agent.memory.recalled if self._agent.memory.enabled else ""
        if recall:
            diagnostics["recalled_text"] = recall
        return TurnResult(text=response.content, elapsed_seconds=monotonic() - started,
                          finish_reason=response.finish_reason, diagnostics=diagnostics)

    def _on_event(self, event: dict[str, Any]) -> None:
        self._events.append(event)
        if len(self._events) > 5000:
            del self._events[:1000]

    # ---- 配置 ----
    def _tools(self):
        from lynx_agent.tools import ToolRegistry

        if not self.settings.native_tools:
            return ToolRegistry([])
        from lynx_agent.web.setup import create_default_tools

        return create_default_tools(self._config)

    def _build_config(self):
        from lynx_agent.config import Config, load_config

        try:
            base = load_config()
        except Exception as error:
            # 错误信息里不带密钥；只给出异常类型与原因。
            raise AdapterError(f"读取 Lynx 配置失败：{type(error).__name__}：{error}") from error
        if self.env is None or self.workspace is None:
            raise AdapterError("Lynx 适配器尚未拿到运行环境")
        window = int(self.env.align_window)
        output_limit = int(self.context.model.max_tokens)
        if window <= output_limit + SAFETY_TOKENS + 4096:
            raise AdapterError("对齐窗口过小：请提高 align_context_window 或降低 max_tokens")
        reserve = max(output_limit + SAFETY_TOKENS,
                      min(window // 4, base.agent.context.reserve_tokens))
        input_budget = window - reserve
        keep_recent = min(base.agent.context.keep_recent_tokens, max(2000, input_budget // 4))
        summary_max = min(base.agent.context.summary_max_tokens, max(400, input_budget // 16))

        payload = base.model_dump()
        payload["providers"]["deepseek"]["base_url"] = self.context.scope_url(self.env.scope)
        payload["providers"]["deepseek"]["timeout_seconds"] = max(
            base.providers.deepseek.timeout_seconds, self.context.turn_timeout_seconds)
        agent = payload["agent"]
        agent["model"].update({
            "name": self.context.model.name,
            "temperature": self.context.model.temperature,
            "max_tokens": output_limit,
            "reasoning_effort": self.context.model.reasoning_effort or None,
            "context_window_tokens": window,
            "max_output_tokens": None,
        })
        agent["context"].update({
            "reserve_tokens": reserve,
            "safety_tokens": SAFETY_TOKENS,
            "keep_recent_tokens": keep_recent,
            "summary_max_tokens": summary_max,
            "summary_generation_max_tokens": None,
            "summary_input_max_tokens": min(base.agent.summary_input_limit, input_budget),
        })
        agent["tools"].update({
            # 测试环境 MCP 自带沙箱；写入类工具需要显式放开，否则会被执行器拦下。
            "allow_write_tools": True,
            "max_result_chars": min(base.agent.tools.max_result_chars, 60000),
        })
        payload["memory"].update({
            "enabled": bool(self.settings.memory_enabled),
            "executable": self.settings.memory_executable or base.memory.executable,
            "data_dir": str(self.memory_dir),
            "project_id": "bench",
            "auto_save": False,
            "auto_recall": True,
            "allow_write": True,
            "auto_approve_for_bench": True,
            "profile_tokens": min(base.memory.profile_tokens, 1200),
            "recall_tokens": min(base.memory.recall_tokens, 1200),
        })
        payload["mcp"] = {"servers": {"bench_env": {
            "enabled": True,
            "command": sys.executable,
            "args": ["-m", "agent_bench.env_mcp.server"],
            "cwd": str(self.workspace),
            "env": {"BENCH_ENV_DIR": str(self.env.state_dir),
                    "BENCH_WORKSPACE": str(self.workspace)},
            "allowed_tools": list(TOOL_NAMES),
            "read_only_tools": list(READ_ONLY_TOOLS),
            "allow_write": True,
            "call_timeout": 90,
            "max_result_chars": 60000,
        }}}
        # 知识库需要额外的 HTTP MCP，评测环境不提供。
        payload["knowledge"]["enabled"] = False
        try:
            return Config.model_validate(payload)
        except Exception as error:
            raise AdapterError(f"Lynx 评测配置不合法：{error}") from error

    @property
    def memory_dir(self) -> Path:
        return (self.workspace or Path.cwd()) / "memory"

    @property
    def sessions_dir(self) -> Path:
        return (self.workspace or Path.cwd()) / "sessions"

    @property
    def session_id(self) -> str:
        return (self._agent.session_id or "") if self._agent is not None else ""
