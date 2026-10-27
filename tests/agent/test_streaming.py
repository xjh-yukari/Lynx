"""完整流提交会话，取消或中断不保存半条回复。"""

import asyncio

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import (
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ProviderResponseError,
    TokenUsage,
)
from lynx_agent.session import SessionStore


class StreamProvider(ModelProvider):
    supports_streaming = True

    def __init__(self, *, gate=None, terminal=True):
        self.gate = gate
        self.terminal = terminal
        self.stream_closed = False
        self.closed = False

    async def generate(self, request):
        raise AssertionError("普通聊天应走原生 stream")

    async def stream(self, request):
        try:
            yield ModelStreamEvent(kind="delta", text="第一段\n")
            if self.gate is not None:
                await self.gate.wait()
            yield ModelStreamEvent(kind="delta", text="第二段")
            if self.terminal:
                yield ModelStreamEvent(kind="done", response=ModelResponse(
                    content="第一段\n第二段", model="fake", finish_reason="stop",
                    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                ))
        finally:
            self.stream_closed = True

    async def aclose(self):
        self.closed = True


def config():
    return Config(
        agent=AgentConfig(model=ModelConfig(name="fake")),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="unused", base_url="https://example.invalid",
        )),
    )


@pytest.mark.asyncio
async def test_stream_commits_only_complete_response_and_restores(tmp_path):
    store = SessionStore(tmp_path)
    provider = StreamProvider()
    parts = []
    async with AgentRunner(config(), provider, store) as agent:
        def on_text(text):
            assert agent.context.history == ()
            assert store.load(agent.session_id).events == ()
            parts.append(text)

        reply = await agent.ask("问题", on_text=on_text)
        identifier = agent.session_id
        assert parts == ["第一段\n", "第二段"]
        assert reply.content == "".join(parts)
        assert provider.stream_closed
        assert agent.last_usage.report().usage.total_tokens == 15
    stored = store.load(identifier)
    assert len(stored.events) == 1
    assert stored.history()[-1].content == "第一段\n第二段"
    async with AgentRunner(config(), StreamProvider(), store, stored) as restored:
        assert restored.context.history == stored.history()


@pytest.mark.asyncio
async def test_cancel_stream_discards_partial_and_can_continue(tmp_path):
    gate = asyncio.Event()
    provider = StreamProvider(gate=gate)
    store = SessionStore(tmp_path)
    received = asyncio.Event()
    async with AgentRunner(config(), provider, store) as agent:
        task = asyncio.create_task(agent.ask("取消的问题", on_text=lambda text: received.set()))
        await asyncio.wait_for(received.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.stream_closed
        assert not provider.closed
        assert agent.context.history == ()
        assert store.load(agent.session_id).context_history() == ()
        assert all(event.kind in {"diagnostic", "pending"} for event in store.load(agent.session_id).events)
        assert agent.last_usage.report().usage is None
        provider.gate = None
        assert (await agent.ask("重新提问", on_text=lambda text: None)).content == "第一段\n第二段"


@pytest.mark.asyncio
async def test_missing_final_event_does_not_commit(tmp_path):
    store = SessionStore(tmp_path)
    provider = StreamProvider(terminal=False)
    async with AgentRunner(config(), provider, store) as agent:
        with pytest.raises(ProviderResponseError):
            await agent.ask("问题", on_text=lambda text: None)
        assert provider.stream_closed
        assert agent.context.history == ()
        assert store.load(agent.session_id).context_history() == ()
        assert all(event.kind in {"diagnostic", "pending"} for event in store.load(agent.session_id).events)
