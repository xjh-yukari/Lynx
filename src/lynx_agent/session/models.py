"""完整会话记录；重置只影响恢复位置，不删除原文。"""

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from lynx_agent.providers import ChatMessage
from lynx_agent.providers.models import (
    ReplyAttempt,
    RequestDiagnostic,
    ToolTrace,
    validate_turn,
)
from lynx_agent.session.recovery import PendingReply
from lynx_agent.session.tool_results import ToolResultReplacement, content_hash


class SessionEvent(BaseModel):
    """一个成功回合或一次清空历史操作。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["turn", "reset", "compact", "attempt", "tool", "prune", "diagnostic", "pending", "pending_clear"]
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    messages: tuple[ChatMessage, ...] = ()
    summary: str = ""
    covered_until: int = Field(default=0, ge=0)
    before_tokens: int = Field(default=0, ge=0)
    after_tokens: int = Field(default=0, ge=0)
    attempt: ReplyAttempt | None = None
    tool: ToolTrace | None = None
    replacements: tuple[ToolResultReplacement, ...] = ()
    prune_before_tokens: int = Field(default=0, ge=0)
    prune_after_tokens: int = Field(default=0, ge=0)
    diagnostic: RequestDiagnostic | None = None
    pending: PendingReply | None = None

    @model_validator(mode="after")
    def validate_messages(self):
        if (self.kind == "diagnostic") != (self.diagnostic is not None):
            raise ValueError("只有 diagnostic 事件携带诊断")
        if (self.kind == "pending") != (self.pending is not None):
            raise ValueError("只有 pending 事件携带未完成任务")
        if (self.kind == "prune") != bool(self.replacements):
            raise ValueError("仅 prune 事件必须携带工具结果替换")
        if self.kind == "prune":
            if self.prune_after_tokens >= self.prune_before_tokens:
                raise ValueError("回收必须减少请求估算 Token")
        elif self.prune_before_tokens or self.prune_after_tokens:
            raise ValueError("只有回收事件允许携带回收用量")
        if (self.kind == "tool") != (self.tool is not None):
            raise ValueError("只有 tool 事件必须携带执行审计")
        if (self.kind == "attempt") != (self.attempt is not None):
            raise ValueError("仅 attempt 事件必须携带截断尝试记录")
        if self.attempt is not None and self.attempt.response.finish_reason != "length":
            raise ValueError("尝试记录必须是 length 截断响应")
        if self.kind == "turn":
            validate_turn(self.messages)
        elif self.messages:
            raise ValueError("非回合事件不能包含上下文消息")
        if self.kind == "compact":
            if not self.summary.strip() or self.covered_until == 0:
                raise ValueError("摘要检查点必须包含摘要和覆盖边界")
            if self.after_tokens >= self.before_tokens:
                raise ValueError("摘要必须减少估算 Token")
        elif self.summary or self.covered_until or self.before_tokens or self.after_tokens:
            raise ValueError("普通回合和重置事件不能携带摘要字段")
        return self


class Session(BaseModel):
    """会话元信息和全部事件。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    title: str = Field(default="", max_length=80)
    memory_project: str | None = None
    id: UUID = Field(default_factory=uuid4)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    events: tuple[SessionEvent, ...] = ()

    @model_validator(mode="after")
    def validate_checkpoints(self):
        boundary = 0
        result_ids = set()
        replaced_ids = set()
        active_traces = {}
        for index, event in enumerate(self.events):
            if event.kind == "tool" and event.tool.result_id is not None:
                if event.tool.result_id in result_ids:
                    raise ValueError("工具存档 ID 重复")
                result_ids.add(event.tool.result_id)
                active_traces[event.tool.result_id] = event.tool
            if event.pending is not None:
                calls = {call.id: call for message in event.pending.messages for call in message.tool_calls}
                for message in event.pending.messages:
                    if message.role == "tool":
                        trace = active_traces.get(message.result_id)
                        if trace is None or trace.call != calls.get(message.tool_call_id):
                            raise ValueError("未完成任务引用了不匹配或已重置的工具记录")
            if event.kind == "prune":
                for replacement in event.replacements:
                    if replacement.result_id not in result_ids or replacement.result_id in replaced_ids:
                        raise ValueError("回收引用不存在或重复")
                    replaced_ids.add(replacement.result_id)
            if event.kind == "reset":
                boundary = index + 1
                active_traces.clear()
            elif event.kind == "compact":
                if not boundary < event.covered_until <= index:
                    raise ValueError("摘要检查点覆盖范围无效")
                if self.events[event.covered_until - 1].kind != "turn":
                    raise ValueError("摘要边界必须位于完整回合之后")
                boundary = event.covered_until
        return self

    def context_boundary(self) -> tuple[str, int]:
        """最后一次重置后的有效摘要及其原文覆盖边界。"""
        summary, boundary = "", 0
        for index, event in enumerate(self.events):
            if event.kind == "reset":
                summary, boundary = "", index + 1
            elif event.kind == "compact":
                summary, boundary = event.summary, event.covered_until
        return summary, boundary

    def pending_reply(self):
        pending = None
        for event in self.events:
            if event.kind in {"turn", "reset", "pending_clear"}:
                pending = None
            elif event.kind == "pending":
                pending = event.pending
        return pending

    def active_turn_indices(self) -> tuple[int, ...]:
        _, boundary = self.context_boundary()
        return tuple(index for index in range(boundary, len(self.events)) if self.events[index].kind == "turn")

    def context_history(self) -> tuple[ChatMessage, ...]:
        replacements = {item.result_id: item for event in self.events for item in event.replacements}
        messages = []
        for index in self.active_turn_indices():
            for message in self.events[index].messages:
                replacement = replacements.get(message.result_id)
                if replacement is not None:
                    if content_hash(message.content) != replacement.source_hash:
                        raise ValueError("回收目标内容不匹配")
                    message = message.model_copy(update={"content": replacement.content})
                messages.append(message)
        return tuple(messages)

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        for event in self.events:
            if event.kind == "turn":
                return " ".join(event.messages[0].content.split())[:24] or "未命名会话"
        return "未命名会话"

    def with_event(self, event: SessionEvent) -> "Session":
        return self.model_copy(update={
            "events": (*self.events, event), "updated_at": event.timestamp,
        })

    def history(self) -> tuple[ChatMessage, ...]:
        messages: list[ChatMessage] = []
        for event in self.events:
            if event.kind == "reset":
                messages.clear()
            else:
                messages.extend(event.messages)
        return tuple(messages)
