"""AgentRunner 组装和生命周期测试。"""

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import ModelProvider, ModelRequest, ModelResponse


class FakeProvider(ModelProvider):
    """不访问网络的 Runner 测试 Provider。"""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.close_count = 0

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(
            content="模拟回复",
            model="fake-model",
            finish_reason="stop",
        )

    async def aclose(self) -> None:
        self.close_count += 1


def make_config() -> Config:
    return Config(
        providers=ProvidersConfig(
            deepseek=ProviderConfig(
                api_key="fake-secret",
                base_url="https://example.invalid",
            )
        ),
        agent=AgentConfig(
            model=ModelConfig(provider="deepseek", name="test-model"),
            system_prompt="You are Lynx.",
        ),
    )


@pytest.mark.asyncio
async def test_runner_asks_and_preserves_context() -> None:
    provider = FakeProvider()

    async with AgentRunner(make_config(), provider) as runner:
        response = await runner.ask("你好")

        assert response.content == "模拟回复"
        assert [message.role for message in runner.context.messages] == [
            "system",
            "user",
            "assistant",
        ]

    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_runner_close_is_idempotent() -> None:
    provider = FakeProvider()
    runner = AgentRunner(make_config(), provider)

    await runner.aclose()
    await runner.aclose()

    assert provider.close_count == 1


@pytest.mark.asyncio
async def test_closed_runner_rejects_new_messages() -> None:
    runner = AgentRunner(make_config(), FakeProvider())
    await runner.aclose()

    with pytest.raises(RuntimeError, match="已经关闭"):
        await runner.ask("不能发送")
