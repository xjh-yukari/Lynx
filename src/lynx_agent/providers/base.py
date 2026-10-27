"""参考 Nanobot 的异步 Provider 抽象与资源释放接口。"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from types import TracebackType
from typing import Self

from lynx_agent.providers.models import (
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ToolCallDelta,
)


class ModelProvider(ABC):
    """所有模型实现必须遵守的接口。"""

    supports_streaming = False
    reasoning_efforts: tuple[str, ...] = ()
    reasoning_effort: str | None = None

    def set_reasoning_effort(self, effort: str) -> None:
        raise ValueError("当前 Provider 未声明可调整的推理档位")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """非流式实现的兼容回退；原生流式 Provider 应覆盖此方法。"""
        response = await self.generate(request)
        if response.content:
            yield ModelStreamEvent(kind="delta", text=response.content)
        for index, call in enumerate(response.tool_calls):
            yield ModelStreamEvent(kind="tool_delta", tool_delta=ToolCallDelta(
                index=index, id=call.id, name=call.name, arguments=call.arguments,
            ))
        yield ModelStreamEvent(kind="done", response=response)

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        """发送一次请求，返回统一响应。"""

    @abstractmethod
    async def aclose(self) -> None:
        """释放连接资源。"""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
