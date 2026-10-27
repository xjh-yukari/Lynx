"""最小文本 AgentLoop 测试。"""

import pytest

from lynx_agent.agent import AgentLoop
from lynx_agent.config import AgentConfig, ModelConfig
from lynx_agent.providers import ModelProvider, ModelRequest, ModelResponse


class FakeProvider(ModelProvider):
    """记录请求且不访问网络的测试 Provider。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = iter(replies)
        self.requests: list[ModelRequest] = []
        self.closed = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content=next(self._replies),
            model="fake-model",
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.closed = True


def make_agent_config() -> AgentConfig:
    return AgentConfig(
        model=ModelConfig(
            provider="deepseek",
            name="test-model",
            temperature=0.3,
            max_tokens=128,
        ),
        system_prompt="You are Lynx.",
    )


@pytest.mark.asyncio
async def test_run_turn_calls_provider_and_saves_response() -> None:
    provider = FakeProvider(["你好，我是 Lynx。"])
    loop = AgentLoop(provider, make_agent_config())

    response = await loop.run_turn("你好")

    assert response.content == "你好，我是 Lynx。"
    assert [message.role for message in loop.context.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert provider.requests[0].temperature == 0.3
    assert provider.requests[0].max_tokens == 128


@pytest.mark.asyncio
async def test_second_turn_receives_previous_history() -> None:
    provider = FakeProvider(["第一条回复", "第二条回复"])
    loop = AgentLoop(provider, make_agent_config())

    await loop.run_turn("第一条问题")
    await loop.run_turn("第二条问题")

    second_request = provider.requests[1]
    assert [message.content for message in second_request.messages] == [
        "You are Lynx.",
        "第一条问题",
        "第一条回复",
        "第二条问题",
    ]


@pytest.mark.asyncio
async def test_loop_does_not_close_externally_owned_provider() -> None:
    provider = FakeProvider(["回复"])
    loop = AgentLoop(provider, make_agent_config())

    await loop.run_turn("问题")

    assert provider.closed is False
