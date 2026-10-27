"""与终端无关的运行状态通知。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

AgentPhase = Literal["compacting", "generating", "rewriting", "tooling", "recovering", "pruning", "pruned", "summary_degraded", "output_budget_exhausted", "remembering", "memorizing"]
StatusCallback = Callable[[AgentPhase], None]
TextCallback = Callable[[str], None]


@dataclass(frozen=True)
class ReplyStreamEvent:
    """start 清空上一份显示草稿，delta 追加；不直接修改 Session。"""

    kind: Literal["start", "delta"]
    attempt: int = 0
    text: str = ""


StreamCallback = Callable[[ReplyStreamEvent], None]
