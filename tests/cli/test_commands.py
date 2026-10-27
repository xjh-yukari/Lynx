"""通过真实 Runner 和模拟 Provider 验证终端交互。"""

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import (
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ProviderConnectionError,
    TokenUsage,
)


def make_agent_config():
    return AgentConfig(model=ModelConfig(name="fake"), system_prompt="You are Lynx.")


@pytest.mark.asyncio
async def test_plain_search_switch_commands_do_not_call_model(monkeypatch, capsys, chat_inputs):
    config = Config(agent=make_agent_config(), providers=ProvidersConfig(
        deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))
    provider = FakeProvider([])
    runner = AgentRunner(config, provider)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(["/search off", "/search", "/search on", "/search invalid", "/exit"])
    await commands.chat()
    output = capsys.readouterr().out
    assert "联网搜索：已关闭" in output
    assert "已开启（anysearch）" in output
    assert "用法：/search" in output
    assert not provider.requests
    assert runner.search_enabled


class FakeProvider(ModelProvider):
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []
        self.closed = False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(content=next(self.replies), model="fake", finish_reason="stop")

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_chat_recovers_from_failure_and_clears_history(monkeypatch, capsys, chat_inputs):
    config = Config(
        agent=make_agent_config(),
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")
        ),
    )
    provider = FakeProvider(["回答一", "回答二"])
    runner = AgentRunner(config, provider)
    original_generate = provider.generate
    attempts = 0

    async def generate(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderConnectionError("模拟断网")
        return await original_generate(request)

    monkeypatch.setattr(provider, "generate", generate)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(["", "失败的问题", "你好", "/clear", "新问题", "exit"])

    await commands.chat()

    output = capsys.readouterr().out
    assert "调用失败" in output
    assert "回答一" in output
    assert "对话历史已清空" in output
    assert "回答二" in output
    assert [message.content for message in provider.requests[0].messages] == [
        "You are Lynx.", "你好",
    ]
    assert [message.content for message in provider.requests[1].messages] == [
        "You are Lynx.", "新问题",
    ]
    assert provider.closed is True


@pytest.mark.asyncio
async def test_eof_closes_runner(monkeypatch, chat_inputs):
    config = Config(
        agent=make_agent_config(),
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")
        ),
    )
    provider = FakeProvider([])
    runner = AgentRunner(config, provider)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)

    chat_inputs([EOFError()])

    await commands.chat()

    assert provider.closed is True


@pytest.mark.asyncio
async def test_cli_shows_actual_usage_separate_from_context_estimate(monkeypatch, capsys, chat_inputs):
    config = Config(
        agent=make_agent_config(),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="fake", base_url="https://example.invalid",
        )),
    )
    provider = FakeProvider(["回答"])
    original_generate = provider.generate

    async def generate(request):
        reply = await original_generate(request)
        return reply.model_copy(update={"usage": TokenUsage(
            prompt_tokens=123, completion_tokens=45, total_tokens=168,
        )})

    monkeypatch.setattr(provider, "generate", generate)
    runner = AgentRunner(config, provider)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(["你好", "/tokens", "/exit"])
    await commands.chat()
    output = capsys.readouterr().out
    assert "本地估算（当前准备发送的上下文）" in output
    assert output.count("输入 123，输出 45，合计 168") == 1
    assert "API 实际用量（最近一次模型操作，非当前上下文估算）" in output
    assert len(provider.requests) == 1


@pytest.mark.asyncio
async def test_chat_does_not_show_usage_without_command(monkeypatch, capsys, chat_inputs):
    config = Config(
        agent=make_agent_config(),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="fake", base_url="https://example.invalid",
        )),
    )
    provider = FakeProvider(["回答"])
    runner = AgentRunner(config, provider)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(["你好", "/exit"])
    await commands.chat()
    output = capsys.readouterr().out
    assert "Lynx：回答" in output
    assert "API 实际用量" not in output
    assert "未获得 usage" not in output
    assert runner.last_usage.report().calls == 1


@pytest.mark.asyncio
async def test_plain_chat_rewrites_and_only_prints_final_answer(monkeypatch, capsys, chat_inputs):
    config = Config(agent=make_agent_config(), providers=ProvidersConfig(
        deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid"),
    ))
    provider = FakeProvider(["截断草稿", "精简后的最终回答"])
    generate = provider.generate

    async def respond(request):
        reply = await generate(request)
        return reply.model_copy(update={
            "finish_reason": "length" if len(provider.requests) == 1 else "stop",
        })

    monkeypatch.setattr(provider, "generate", respond)
    runner = AgentRunner(config, provider)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(["问题", "/tokens", "/exit"])
    await commands.chat()
    output = capsys.readouterr().out
    assert "Lynx：精简后的最终回答" in output
    assert "截断草稿" not in output
    assert "回复精简重写" in output
    assert len(provider.requests) == 2
    assert runner.context.history[-1].content == "精简后的最终回答"
