"""运行期公共数据模型：适配器、代理、环境 MCP、判分与统计共用。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

# 适配器可以声明的能力；用例通过 requires 声明前置条件。
CAPABILITIES = frozenset({
    "multi_turn",            # 同一会话可连续多轮
    "cross_session_memory",  # 跨会话保留长期记忆
    "mcp",                   # 可以接入测试环境 MCP
    "tool_result_recall",    # 可以重新取回被回收的工具结果
    "native_tools",          # 自带原生工具（native 赛道）
})


class BenchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def utc_now() -> datetime:
    return datetime.now(UTC)


class TurnResult(BenchModel):
    """一次 send() 的结果；token 与工具调用不从适配器取。"""

    text: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0)
    finish_reason: str = "stop"
    # 仅部分 agent 能提供的内部事件（如压缩摘要、记忆召回），只做诊断展示。
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ModelCallRecord(BenchModel):
    """录制代理看到的单次模型调用。"""

    run_id: str
    agent: str
    case_id: str
    turn: int = Field(default=0, ge=0)
    messages: int = Field(default=0, ge=0)
    request_chars: int = Field(default=0, ge=0)
    tool_definitions: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    cached_prompt_tokens: int | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    status: int = 200
    error: str | None = None
    # 请求未带 stream_options.include_usage 时由代理补注，并在报告中标明。
    usage_injected: bool = False
    retries: int = Field(default=0, ge=0)
    at: datetime = Field(default_factory=utc_now)

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


class ToolCallRecord(BenchModel):
    """测试环境 MCP 记录的一次工具调用（口径对所有 agent 一致）。"""

    seq: int = Field(default=0, ge=0)
    agent: str
    case_id: str
    turn: int = Field(default=0, ge=0)
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_chars: int = Field(default=0, ge=0)
    elapsed_ms: float = Field(default=0.0, ge=0)
    is_error: bool = False
    error_kind: str | None = None
    at: datetime = Field(default_factory=utc_now)


GradeType = Literal[
    "contains", "not_contains", "regex", "exact", "numeric",
    "expect_tool", "expect_state", "length_limit", "llm_judge",
]


class GradeSpec(BenchModel):
    """单条判分规则；value 中的 {fact} 已展开。"""

    type: GradeType
    value: str | int | float | None = None
    # numeric 的容差；length_limit 用 max/min。
    tolerance: float | None = None
    max: int | None = None
    min: int | None = None
    # expect_tool
    name: str | None = None
    called: bool = True
    args: dict[str, Any] | None = None
    # expect_state 的 JSON 子集断言
    state: dict[str, Any] | None = None
    # llm_judge
    rubric: str | None = None
    model: str | None = None
    case_sensitive: bool = False


class GradeResult(BenchModel):
    grader: str
    passed: bool
    detail: str = ""
    expected: str | None = None
    actual: str | None = None


class TurnTrace(BenchModel):
    """traces.jsonl 的一行：一轮的全部可核对证据。"""

    run_id: str
    agent: str
    case_id: str
    suite: str
    track: str
    repeat: int = Field(default=0, ge=0)
    step: int = Field(default=0, ge=0)
    step_kind: str = "user"
    session: str = ""
    turn: int = Field(default=0, ge=0)
    user: str = ""
    answer: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0)
    finish_reason: str = "stop"
    error: str | None = None
    calls: tuple[ModelCallRecord, ...] = ()
    tools: tuple[ToolCallRecord, ...] = ()
    grades: tuple[GradeResult, ...] = ()
    metrics: dict[str, float] = Field(default_factory=dict)
    facts: dict[str, str] = Field(default_factory=dict)
    # 用例标签（如 position:10、window:32768），报告据此绘制热力图与曲线。
    tags: tuple[str, ...] = ()
    # 内部诊断（压缩摘要原文、回收记录、召回的记忆条目等），只展示不计分。
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    at: datetime = Field(default_factory=utc_now)


class MetricSample(BenchModel):
    case_id: str
    suite: str
    metric: str
    value: float
    repeat: int = Field(default=0, ge=0)
    agent: str


class AgentAdapter(Protocol):
    """所有适配器都必须实现的最小接口。"""

    name: str
    capabilities: set[str]

    async def setup(self, workspace: Any, env: Any) -> None: ...

    async def new_session(self) -> str: ...

    async def send(self, message: str) -> TurnResult: ...

    async def reset_memory(self) -> None: ...

    async def teardown(self) -> None: ...
