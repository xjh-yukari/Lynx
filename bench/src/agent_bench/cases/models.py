"""用例模型：YAML 结构、随机事实规格与填充规格。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from agent_bench.models import BenchModel, GradeSpec


class FactSpec(BenchModel):
    """随机事实规格；kind 缺省时按 choice 处理。"""

    kind: Literal["choice", "code", "port", "person", "date", "integer", "text"] = "choice"
    choice: tuple[str, ...] = ()
    distinct_from: str | None = None
    min: int | None = None
    max: int | None = None
    prefix: str = ""

    @model_validator(mode="after")
    def check(self):
        if self.kind == "choice" and len(self.choice) < 2:
            raise ValueError("choice 至少需要两个候选值")
        if self.kind == "integer" and (self.min is None or self.max is None or self.min >= self.max):
            raise ValueError("integer 需要 min < max")
        return self


class FillerSpec(BenchModel):
    """与事实无关的填充对话；按 token 预算生成，内容由种子决定。"""

    tokens: int = Field(default=8000, ge=200)
    turns: int = Field(default=1, ge=1, le=200)
    topic: str = "部署日志"
    tools: bool = False


class ToolExpectation(BenchModel):
    name: str
    called: bool = True
    args: dict[str, Any] | None = None


class StepSpec(BenchModel):
    session: Literal["new", "continue"] = "continue"
    user: str | None = None
    filler: FillerSpec | None = None
    env: dict[str, Any] | None = None
    expect_tool: ToolExpectation | None = None
    expect_state: dict[str, Any] | None = None
    grade: tuple[GradeSpec, ...] = ()
    metric: str | None = None
    max_tokens: int | None = None

    @model_validator(mode="after")
    def check_payload(self):
        if not any((self.user, self.filler, self.env, self.grade, self.expect_tool,
                    self.expect_state)):
            raise ValueError("步骤必须包含 user、filler、env、grade、expect_tool 或 expect_state 之一")
        if self.grade and not self.user:
            raise ValueError("grade 只能挂在有 user 文本的步骤上")
        return self


class BenchCase(BenchModel):
    id: str
    suite: str
    track: Literal["shared", "native"] = "shared"
    requires: tuple[str, ...] = ()
    # 0 表示沿用 bench.toml 里的 run.repeats。
    repeats: int = Field(default=0, ge=0, le=10)
    facts: dict[str, FactSpec] = Field(default_factory=dict)
    steps: tuple[StepSpec, ...]
    metrics: tuple[str, ...]
    description: str = ""
    tags: tuple[str, ...] = ()
    # 需要更长的填充时覆盖默认对齐窗口（仍会记录在报告中）。
    align_context_window: int | None = None

    @model_validator(mode="after")
    def check_shape(self):
        if not self.id or len(self.id) > 120:
            raise ValueError("用例 id 必须是 1-120 字符")
        if any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for ch in self.id):
            raise ValueError("用例 id 只能使用小写字母、数字、点、下划线和连字符")
        if not self.steps:
            raise ValueError("用例至少需要一个步骤")
        if not self.metrics:
            raise ValueError("用例必须声明至少一个指标")
        if self.track == "native" and "native_tools" not in self.requires:
            raise ValueError("native 赛道必须在 requires 中声明 native_tools")
        return self


class ResolvedStep(BenchModel):
    """展开占位符后的步骤；运行期不再引用随机事实。"""

    index: int
    session: Literal["new", "continue"] = "continue"
    user: str | None = None
    filler: FillerSpec | None = None
    env: dict[str, Any] | None = None
    expect_tool: ToolExpectation | None = None
    expect_state: dict[str, Any] | None = None
    grade: tuple[GradeSpec, ...] = ()
    metric: str | None = None
    max_tokens: int | None = None


class ResolvedCase(BenchModel):
    case: BenchCase
    facts: dict[str, str]
    steps: tuple[ResolvedStep, ...]
    seed: int
