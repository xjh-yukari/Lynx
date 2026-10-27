"""未完成任务仅供用户显式 /retry 整理答复，不作为成功历史。"""

from uuid import UUID

from pydantic import Field, model_validator

from lynx_agent.providers.models import ChatMessage, ProviderModel, validate_turn


class PendingReply(ProviderModel):
    turn_id: UUID
    messages: tuple[ChatMessage, ...] = Field(min_length=1, max_length=2000)
    draft: str = Field(default="", max_length=16000)
    recovery_used: int = Field(default=0, ge=0, le=20)
    permission_mode: str = ""
    workspace: str = ""

    @model_validator(mode="after")
    def validate_prefix(self):
        # 加入临时结尾只用于验证闭合前缀，绝不保存为成功回复。
        validate_turn((*self.messages, ChatMessage(role="assistant", content="pending")))
        return self
