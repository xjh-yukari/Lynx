"""适配器基类与运行环境容器。

只有 `adapters/lynx.py` 允许导入 lynx_agent；其余模块只依赖本文件定义的
协议与数据，从而保证新增 agent 一般只需改 bench.toml。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_bench.config import AgentSettings, ModelSettings
from agent_bench.env_mcp.store import EnvStore
from agent_bench.env_mcp.tools import EnvTools
from agent_bench.models import TurnResult
from agent_bench.proxy.recorder import ProxyRecorder
from agent_bench.proxy.server import RecordingProxy


class AdapterError(Exception):
    """适配器无法启动或运行；错误信息不包含密钥。"""


@dataclass
class BenchEnv:
    """一次用例运行的隔离环境：工作区、测试环境状态与代理作用域。"""

    workspace: Path
    state_dir: Path
    scope: str
    # 本次运行的对齐窗口；用例可以覆盖，报告中会记录实际取值。
    align_window: int = 131_072
    store: EnvStore = field(init=False)
    tools: EnvTools = field(init=False)

    def __post_init__(self) -> None:
        self.store = EnvStore(self.state_dir)
        self.tools = EnvTools(self.store, self.workspace)

    def reset(self, state: dict[str, Any] | None = None,
              control: dict[str, Any] | None = None) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "files").mkdir(parents=True, exist_ok=True)
        self.store.reset(state, control)

    def apply(self, payload: dict[str, Any] | None) -> None:
        if payload:
            self.tools.apply_env(payload)


@dataclass
class RunContext:
    """一次运行的共享依赖；适配器只读。"""

    run_id: str
    model: ModelSettings
    proxy: RecordingProxy
    recorder: ProxyRecorder
    api_key: str
    turn_timeout_seconds: float

    def scope_url(self, scope: str) -> str:
        return self.proxy.scope_url(scope)


class BaseAdapter:
    """所有适配器的公共部分：作用域、工作区与记忆目录清理。"""

    kind = "base"

    def __init__(self, name: str, settings: AgentSettings, context: RunContext) -> None:
        self.name = name
        self.settings = settings
        self.context = context
        self.capabilities: set[str] = settings.default_capabilities()
        self.workspace: Path | None = None
        self.env: BenchEnv | None = None
        self.turn = 0

    # ---- 生命周期 ----
    async def setup(self, workspace: Path, env: BenchEnv) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.env = env
        await self._set_turn(0)

    async def new_session(self) -> str:
        raise NotImplementedError

    async def send(self, message: str) -> TurnResult:
        raise NotImplementedError

    async def reset_memory(self) -> None:
        for relative in self.settings.memory_dirs:
            target = Path(relative)
            if not target.is_absolute():
                base = Path(self.settings.cwd) if self.settings.cwd else (self.workspace or Path.cwd())
                target = base / relative
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)

    async def teardown(self) -> None:
        return None

    # ---- 运行期辅助 ----
    async def begin_turn(self, turn: int) -> None:
        await self._set_turn(turn)

    async def _set_turn(self, turn: int) -> None:
        self.turn = turn
        if self.env is not None:
            await self.context.proxy.set_turn(self.env.scope, turn)

    @property
    def scope(self) -> str:
        return self.env.scope if self.env is not None else ""

    def _denied(self) -> TurnResult:
        return TurnResult(text="", error="适配器尚未完成 setup", finish_reason="error")
