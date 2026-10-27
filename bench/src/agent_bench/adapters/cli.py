"""通用子进程适配器：命令模板 + 会话续接参数 + 输出解析，全部由配置驱动。

新增一个 CLI 形态的 agent 一般只需要在 bench.toml 里写 command / resume_args /
stdout 解析方式，不需要改代码。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import monotonic
from typing import Any

from agent_bench.adapters.base import AdapterError, BaseAdapter, BenchEnv
from agent_bench.models import TurnResult

MAX_CAPTURE = 400_000


class CliAdapter(BaseAdapter):
    kind = "cli"

    def __init__(self, name: str, settings, context) -> None:
        super().__init__(name, settings, context)
        self._session = ""
        self._process: asyncio.subprocess.Process | None = None
        self._patch_path: Path | None = None

    # ---- 生命周期 ----
    async def setup(self, workspace: Path, env: BenchEnv) -> None:
        await super().setup(workspace, env)
        if self.settings.kind != "cli":
            raise AdapterError("CliAdapter 只接受 cli 类型的配置")
        if not shutil.which(self.settings.command[0]) and not Path(self.settings.command[0]).is_file():
            raise AdapterError(f"找不到可执行程序：{self.settings.command[0]}")
        if self.settings.patch:
            self._patch_path = Path(self.workspace or workspace) / "bench-agent.patch.yml"
            self._patch_path.write_text(self.settings.patch, encoding="utf-8")

    async def new_session(self) -> str:
        self._session = ""
        return ""

    async def reset_memory(self) -> None:
        await super().reset_memory()
        self._session = ""

    async def teardown(self) -> None:
        if self._process is not None and self._process.returncode is None:
            self._process.kill()
            await self._process.wait()
        self._process = None

    # ---- 单轮执行 ----
    async def send(self, message: str) -> TurnResult:
        if self.workspace is None or self.env is None:
            return self._denied()
        argv = self._launch_argv(self._argv(message))
        env = self._environment()
        cwd = self._cwd()
        started = monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE if self.settings.stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            return TurnResult(text="", error=f"无法启动被测程序：{type(error).__name__}",
                              finish_reason="error")
        self._process = process
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(message.encode("utf-8") if self.settings.stdin else None),
                timeout=self.context.turn_timeout_seconds,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return TurnResult(text="", elapsed_seconds=monotonic() - started,
                              error="本轮超过超时时间，已终止被测程序", finish_reason="timeout")
        finally:
            self._process = None
        elapsed = monotonic() - started
        out = self._decode(stdout)[:MAX_CAPTURE]
        err = self._decode(stderr)[-4000:]
        finish = "stop" if process.returncode == 0 else "error"
        text, diagnostics, parsed_finish = self._parse(out)
        error = None
        if process.returncode != 0:
            error = f"退出码 {process.returncode}：{self._redact(err.strip())[:400] or '无 stderr'}"
            finish = "error"
        elif parsed_finish:
            finish = parsed_finish
        return TurnResult(text=text, elapsed_seconds=elapsed, finish_reason=finish,
                          diagnostics=diagnostics, error=error)

    # ---- 解析 ----
    def _parse(self, output: str) -> tuple[str, dict[str, Any], str | None]:
        if self.settings.stdout == "text":
            return output.strip(), {}, None
        text_parts: list[str] = []
        answer = ""
        session = ""
        tools: list[dict[str, Any]] = []
        errors: list[str] = []
        finish: str | None = None
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get(self.settings.event_type_field)
            if self.settings.answer_event_type and kind == self.settings.answer_event_type:
                answer = str(event.get(self.settings.text_field, ""))
            elif kind == "text":
                text_parts.append(str(event.get(self.settings.text_field, "")))
            if self.settings.session_event_type and kind == self.settings.session_event_type:
                session = str(event.get(self.settings.session_field, "") or session)
            if self.settings.tool_event_type and kind == self.settings.tool_event_type:
                tools.append({"tool": event.get("tool") or event.get("name"),
                              "input": event.get("input") or event.get("arguments")})
            if kind == "error":
                errors.append(str(event.get("message", "")))
                finish = "error"
        if session:
            self._session = session
        text = answer or "".join(text_parts)
        diagnostics = {
            "session": self._session,
            "tool_events": tools,
            "error_events": errors,
        }
        return text, diagnostics, finish

    @staticmethod
    def _decode(raw: bytes) -> str:
        for encoding in ("utf-8", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")

    # ---- 命令与环境 ----
    def _argv(self, message: str) -> list[str]:
        """拼装命令行，三个位置各有约定：

        - `extra_args`（全局/启动器选项）紧跟可执行文件，例如 dsh 的 `--patch`；
        - `resume_args`（会话续接选项）插在 `{task}` 之前；
        - `{task}` 展开成任务文本；`stdin = true` 时展开成 stdin 标记（dsh 用 `-`），
          任务走标准输入，避免 Windows 的 .cmd 垫片再解析一遍任务文本。
        """
        extra = [self._expand(item, message) for item in self.settings.extra_args]
        resume = [self._expand(item, message) for item in self.settings.resume_args] \
            if self._session else []
        task = self.settings.stdin_marker if self.settings.stdin else message
        command = list(self.settings.command)
        if any("{task}" in item for item in command):
            parts = self._expand_with_resume(command, resume, task)
        else:
            parts = [self._expand(item, message) for item in command] + resume + [task]
        return [parts[0], *extra, *parts[1:]] if parts else parts

    def _launch_argv(self, argv: list[str]) -> list[str]:
        """Windows 上 npm 装的命令是 .cmd/.ps1 垫片，CreateProcess 起不来。

        .exe 直接执行；.cmd/.bat 交给 cmd.exe；.ps1 交给 PowerShell。
        需要以 stdin 传任务，就是为了避免这些外壳再解析一次任务文本。
        """
        if not argv:
            return argv
        executable = shutil.which(argv[0]) or argv[0]
        suffix = Path(executable).suffix.lower()
        if suffix in {".cmd", ".bat"}:
            line = subprocess.list2cmdline([executable, *argv[1:]])
            return ["cmd.exe", "/d", "/s", "/c", line]
        if suffix == ".ps1":
            shell = shutil.which("pwsh") or shutil.which("powershell") or "powershell.exe"
            return [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                    executable, *argv[1:]]
        return [executable, *argv[1:]]

    def _expand_with_resume(self, command: list[str], resume: list[str], message: str) -> list[str]:
        result: list[str] = []
        for item in command:
            if item == "{task}":
                result.extend(resume)
                result.append(message)
            else:
                result.append(self._expand(item, message))
        return result

    def _expand(self, template: str, message: str) -> str:
        return (template
                .replace("{task}", message)
                .replace("{session}", self._session)
                .replace("{workspace}", str(self.workspace or ""))
                .replace("{patch}", str(self._patch_path or "")))

    def _cwd(self) -> Path:
        base = Path(self.settings.cwd) if self.settings.cwd else (self.workspace or Path.cwd())
        if not base.is_absolute():
            base = (self.workspace or Path.cwd()) / base
        base.mkdir(parents=True, exist_ok=True)
        return base

    def _environment(self) -> dict[str, str]:
        if self.env is None:
            return dict(os.environ)
        scope_url = self.context.scope_url(self.env.scope)
        values = dict(os.environ)
        values[self.settings.base_url_env] = scope_url
        values[self.settings.api_key_env] = self.context.api_key
        values["BENCH_RUN_ID"] = self.context.run_id
        values["BENCH_WORKSPACE"] = str(self.workspace)
        values["BENCH_ENV_DIR"] = str(self.env.state_dir)
        # 供 patch 里的 !!js 表达式引用：用哪个 Python 拉起测试环境 MCP。
        values["BENCH_PYTHON"] = sys.executable
        for key, template in self.settings.extra_env.items():
            values[key] = self._expand(template, "").replace("{base_url}", scope_url) \
                .replace("{api_key}", self.context.api_key)
        return values

    def _redact(self, text: str) -> str:
        secret = self.context.api_key
        return text.replace(secret, "***") if secret else text
