"""终端显示状态，与 Agent 的模型上下文分开。"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from uuid import uuid4


@dataclass
class DisplayMessage:
    role: str
    content: str
    id: str = field(default_factory=lambda: uuid4().hex)
    tools: tuple = ()
    expanded: bool = False
    detail_page: int = 0
    visible_chars: int = 12000


@dataclass
class UIState:
    project: Path
    model: str
    search_backend: str = "未启用"
    permissions: str = "未启用"
    approval: str = "常用操作自动"
    mcp: str = "未配置"
    mcp_status: Callable[[], str] | None = None
    memory_status: Callable[[], str] | None = None
    session_title: str = "新会话（尚未创建）"
    session_id: str | None = None
    status: str = "就绪"
    busy: bool = False
    cancelling: bool = False
    follow_tail: bool = True
    unread_activity: bool = False
    scroll_line: int = 0
    stream_speed: int = 240
    messages: list[DisplayMessage] = field(default_factory=list)
    session_choices: list[tuple[str | None, str]] = field(default_factory=list)
    picker_open: bool = False
    effort_open: bool = False
    permissions_open: bool = False
    permissions_selected: int = 1
    approval_open: bool = False
    approval_selector_open: bool = False
    usage_panel: str = ""
    approval_selected: int = 1
    approval_kind: str = "command"
    approval_supports_batch: bool = False
    batch_running: bool = False
    batch_progress: str = ""
    shell_enabled: bool = False
    effort: str | None = None
    effort_options: tuple[str, ...] = ()
    effort_selected: int = 0
    selected: int = 0
    session_query: str = ""
    session_all_choices: list[tuple[str | None, str]] = field(default_factory=list)
    draft_expanded: bool = False
    visible_messages: int = 80
    history_end_id: str | None = None
    theme: str = "ansi"
    toast: str = ""
    toast_until: float = 0
    selection_available: bool = False
    request_started_at: float | None = None
    last_request_seconds: float | None = None

    def start_request(self) -> None:
        self.request_started_at = monotonic()
        self.last_request_seconds = None

    def finish_request(self) -> None:
        if self.request_started_at is not None:
            self.last_request_seconds = max(0.0, monotonic() - self.request_started_at)
            self.request_started_at = None

    def timing_text(self) -> str:
        if self.request_started_at is not None:
            return f"已用 {max(0.0, monotonic() - self.request_started_at):.1f}s"
        if self.last_request_seconds is not None:
            return f"上次耗时 {self.last_request_seconds:.1f}s"
        return ""
