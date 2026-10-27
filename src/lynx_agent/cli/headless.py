"""非交互单次运行：复用 AgentRunner 及其回调，把结果投影成稳定事件流。"""

from __future__ import annotations

import asyncio
import json
import sys
from time import monotonic
from typing import Any, TextIO

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.config import ConfigError, load_config
from lynx_agent.providers import ProviderError
from lynx_agent.session import SessionError, SessionStore
from lynx_agent.tools.errors import ToolExecutionError

# 事件类型：session_id、text、tool_call、usage、compaction、prune、memory_recall、
# warning、error、done。新增字段只增不改，供评测框架按类型消费。
EXIT_OK = 0
EXIT_FAILED = 1


class EventWriter:
    """`--json` 时 stdout 只写事件行；人类可读进度一律写 stderr。"""

    def __init__(self, *, json_mode: bool, stdout: TextIO | None = None,
                 stderr: TextIO | None = None) -> None:
        self.json_mode = json_mode
        self.stdout = stdout if stdout is not None else sys.stdout
        self.stderr = stderr if stderr is not None else sys.stderr

    def emit(self, event: dict[str, Any]) -> None:
        if self.json_mode:
            self.stdout.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            self.stdout.flush()
            return
        line = self._human(event)
        if line:
            self.stderr.write(line + "\n")
            self.stderr.flush()

    def answer(self, text: str) -> None:
        """人类可读模式把最终答案写到 stdout；JSON 模式的答案走 text 事件。"""
        if self.json_mode:
            return
        self.stdout.write(text + "\n")
        self.stdout.flush()

    @staticmethod
    def _human(event: dict[str, Any]) -> str:
        kind = event.get("type")
        if kind == "session_id":
            return f"Lynx · 会话 {event.get('session_id') or '创建中'}"
        if kind == "tool_call":
            state = "失败" if event.get("is_error") else "完成"
            return f"工具 {event.get('name')}（{state}，{event.get('content_chars', 0)} 字符）"
        if kind == "compaction":
            return (f"已压缩历史：约 {event.get('before_tokens', 0):,} → "
                    f"{event.get('after_tokens', 0):,} Tokens")
        if kind == "prune":
            return f"已回收 {event.get('count', 0)} 条工具结果"
        if kind == "memory_recall":
            return f"已召回长期记忆约 {event.get('chars', 0)} 字符"
        if kind in {"warning", "error"}:
            return str(event.get("message", ""))
        return ""


async def run_headless(
    task: str, session_id: str | None = None, *, json_mode: bool = False,
    stdout: TextIO | None = None, stderr: TextIO | None = None,
) -> int:
    """执行一个回合；成功返回 0，任何失败返回 1。"""
    writer = EventWriter(json_mode=json_mode, stdout=stdout, stderr=stderr)
    text = (task or "").strip()
    if not text:
        writer.emit({"type": "error", "code": "empty_task", "message": "任务文本为空，未执行任何请求。"})
        writer.emit({"type": "done", "exit_code": EXIT_FAILED, "session_id": session_id or ""})
        return EXIT_FAILED

    started = monotonic()
    current_session = session_id or ""
    try:
        config = load_config()
        runner = AgentRunner.from_config(
            config, session_id=session_id, store=SessionStore(), on_event=writer.emit,
        )
        async with runner as agent:
            response = await agent.ask(text)
            current_session = agent.session_id or ""
            writer.emit({"type": "session_id", "session_id": current_session,
                         "resumed": session_id is not None})
            report = agent.last_usage.report()
            writer.emit({
                "type": "usage", "aggregate": True, "calls": report.calls,
                "reported_calls": report.reported_calls,
                "usage": None if report.usage is None else {
                    "prompt_tokens": report.usage.prompt_tokens,
                    "completion_tokens": report.usage.completion_tokens,
                    "total_tokens": report.usage.total_tokens,
                    "cached_prompt_tokens": report.usage.cached_prompt_tokens,
                },
            })
            if response.finish_reason != "stop":
                writer.emit({"type": "error", "code": "incomplete",
                             "message": f"回答未正常完成（finish_reason={response.finish_reason}）。"})
                writer.emit({"type": "text", "text": response.content})
                writer.answer(response.content)
                writer.emit({"type": "done", "exit_code": EXIT_FAILED, "session_id": current_session,
                             "finish_reason": response.finish_reason,
                             "elapsed_seconds": round(monotonic() - started, 3)})
                return EXIT_FAILED
            writer.emit({"type": "text", "text": response.content})
            writer.answer(response.content)
    except (ConfigError, ProviderError, SessionError, ContextError,
            ToolExecutionError, ValueError, OSError) as error:
        return _fail(writer, str(error), current_session, started)
    except Exception as error:  # noqa: BLE001 - 单次运行入口，未知异常也要给出稳定退出码
        return _fail(writer, f"{type(error).__name__}：{error}", current_session, started)

    writer.emit({"type": "done", "exit_code": EXIT_OK, "session_id": current_session,
                 "finish_reason": "stop", "elapsed_seconds": round(monotonic() - started, 3)})
    return EXIT_OK


def _fail(writer: EventWriter, message: str, session_id: str, started: float) -> int:
    writer.emit({"type": "error", "code": "failed", "message": message})
    writer.emit({"type": "done", "exit_code": EXIT_FAILED, "session_id": session_id,
                 "elapsed_seconds": round(monotonic() - started, 3)})
    return EXIT_FAILED


def read_task(argument: str | None, stdin: TextIO | None = None) -> str:
    """位置参数优先；没有参数时从 stdin 读，空管道不发起任何请求。"""
    if argument and argument.strip():
        return argument
    stream = stdin if stdin is not None else sys.stdin
    if stream is None or stream.isatty():
        return ""
    return stream.read()


def run_cli(task: str | None, session_id: str | None, *, json_mode: bool) -> int:
    return asyncio.run(run_headless(read_task(task), session_id, json_mode=json_mode))
