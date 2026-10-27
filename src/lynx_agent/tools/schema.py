"""供应商无关的工具定义、调用和执行结果。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ToolModel(BaseModel):
    """验证工具模块边界的数据。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolDefinition(ToolModel):
    """发送给模型的工具说明。"""

    name: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")
    description: str = Field(min_length=1)
    parameters: dict[str, Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """转换成 OpenAI/DeepSeek 兼容的 function schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolCall(ToolModel):
    """Agent 请求执行的一次工具调用。"""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(ToolModel):
    """工具返回给 Agent 的统一结果。"""

    call_id: str
    name: str
    content: str
    is_error: bool = False
    result_id: UUID | None = None
    error_kind: str | None = None
    execution_state: str = "returned"
