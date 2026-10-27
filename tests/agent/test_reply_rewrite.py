"""截断后的有限重写、窗口预算及恢复；所有请求使用模拟 Provider。"""

import asyncio

import pytest
from pydantic import ValidationError

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.reply_rewrite import rewrite_request
from lynx_agent.agent.tokens import estimate_context_tokens
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    TokenUsage,
)
from lynx_agent.session import SessionError, SessionStore


def make_config(**agent_values):
    return Config(
        agent=AgentConfig(model=ModelConfig(name="fake"), **agent_values),
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")),
    )


def response(content, reason="stop"):
    return ModelResponse(content=content, model="fake", finish_reason=reason,
                         usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))


class ScriptProvider(ModelProvider):
    supports_streaming = True

    def __init__(self, replies, block_rewrite=False):
        self.replies = iter(replies)
        self.requests = []
        self.block_rewrite = block_rewrite
        self.rewriting = asyncio.Event()
        self.streams_closed = 0

    async def generate(self, request):
        self.requests.append(request)
        if self.block_rewrite and len(self.requests) > 1:
            self.rewriting.set()
            await asyncio.Event().wait()
        return next(self.replies)

    async def stream(self, request):
        try:
            reply = await self.generate(request)
            if reply.content:
                yield ModelStreamEvent(kind="delta", text=reply.content)
            yield ModelStreamEvent(kind="done", response=reply)
        finally:
            self.streams_closed += 1

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_default_limit_and_single_success_never_rewrites(tmp_path):
    provider = ScriptProvider([response("完整")])
    async with AgentRunner(make_config(), provider, SessionStore(tmp_path)) as runner:
        assert (await runner.ask("问题")).content == "完整"
        assert provider.requests[0].max_tokens == 8192
        assert len(provider.requests) == 1
        assert runner.last_usage.report("rewrite").calls == 0


@pytest.mark.asyncio
async def test_length_rewrites_with_original_task_and_only_commits_final(tmp_path):
    store = SessionStore(tmp_path)
    provider = ScriptProvider([response("截断原文", "length"), response("精简且完整")])
    async with AgentRunner(make_config(), provider, store) as runner:
        phases = []
        reply = await runner.ask("实现完整功能", on_status=phases.append)
        assert reply.content == "精简且完整"
        assert phases == ["generating", "rewriting"]
        rewrite = provider.requests[1]
        assert rewrite.max_tokens == 8192
        assert rewrite.messages[1].content == "实现完整功能"
        assert "截断原文" in rewrite.messages[-1].content
        assert "6144 Tokens" in rewrite.messages[-1].content
        assert [message.content for message in runner.context.history] == ["实现完整功能", "精简且完整"]
        assert runner.last_usage.report().calls == 2
        assert runner.last_usage.report("rewrite").usage.total_tokens == 15
        identifier = runner.session_id
    session = store.load(identifier)
    assert [event.kind for event in session.events if event.kind != "diagnostic"] == ["attempt", "turn"]
    attempt = next(event.attempt for event in session.events if event.kind == "attempt")
    assert attempt.response.content == "截断原文"
    assert attempt.response.usage.total_tokens == 15
    async with AgentRunner(make_config(), ScriptProvider([]), store, session) as restored:
        assert [message.content for message in restored.context.history] == ["实现完整功能", "精简且完整"]


@pytest.mark.asyncio
async def test_three_lengths_stop_without_committing_and_next_question_is_clean(tmp_path):
    store = SessionStore(tmp_path)
    provider = ScriptProvider([response("草稿", "length")] * 3 + [response("新答案")])
    async with AgentRunner(make_config(), provider, store) as runner:
        reply = await runner.ask("长问题")
        assert reply.finish_reason == "length"
        assert len(provider.requests) == 3
        assert "4096 Tokens" in provider.requests[2].messages[-1].content
        assert runner.context.history == ()
        assert runner.last_usage.report().usage.total_tokens == 45
        attempts = [event for event in store.load(runner.session_id).events if event.kind == "attempt"]
        assert [event.attempt.index for event in attempts] == [0, 1, 2]
        assert len({event.attempt.turn_id for event in attempts}) == 1
        assert store.load(runner.session_id).context_history() == ()
        await runner.ask("新问题")
        assert [message.content for message in provider.requests[-1].messages] == [runner.context.messages[0].content, "新问题"]


@pytest.mark.asyncio
async def test_stream_rewrite_emits_reset_instead_of_concatenating(tmp_path):
    provider = ScriptProvider([response("第一份截断", "length"), response("最终结果")])
    display = ""
    events = []

    def update(event):
        nonlocal display
        events.append((event.kind, event.attempt))
        display = "" if event.kind == "start" else display + event.text

    async with AgentRunner(make_config(), provider, SessionStore(tmp_path)) as runner:
        reply = await runner.ask("问题", on_stream=update)
        assert display == reply.content == "最终结果"
        assert events == [("start", 0), ("delta", 0), ("start", 1), ("delta", 1)]
        assert provider.streams_closed == 2


@pytest.mark.asyncio
async def test_cancel_during_rewrite_preserves_draft_but_not_context(tmp_path):
    store = SessionStore(tmp_path)
    provider = ScriptProvider([response("截断", "length")], block_rewrite=True)
    async with AgentRunner(make_config(), provider, store) as runner:
        task = asyncio.create_task(runner.ask("问题", on_stream=lambda event: None))
        await asyncio.wait_for(provider.rewriting.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runner.context.history == ()
        assert sum(event.kind == "attempt" for event in store.load(runner.session_id).events) == 1
        assert not any(event.kind == "turn" for event in store.load(runner.session_id).events)
        report = runner.last_usage.report()
        assert report.calls == 2
        assert report.reported_calls == 1
        assert provider.streams_closed == 2


@pytest.mark.asyncio
async def test_audit_write_failure_stops_before_more_api_calls(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    provider = ScriptProvider([response("截断", "length"), response("不会调用")])
    async with AgentRunner(make_config(), provider, store) as runner:
        original_save = store.save

        def save(session):
            if session.events:
                raise SessionError("模拟磁盘故障")
            original_save(session)

        monkeypatch.setattr(store, "save", save)
        with pytest.raises(SessionError):
            await runner.ask("问题")
        assert len(provider.requests) == 1
        assert runner.context.history == ()
        assert runner.persistence_error is not None
        monkeypatch.setattr(store, "save", original_save)


@pytest.mark.asyncio
async def test_one_off_limit_is_validated_and_does_not_change_default(tmp_path):
    provider = ScriptProvider([response("小回答"), response("普通回答")])
    async with AgentRunner(make_config(), provider, SessionStore(tmp_path)) as runner:
        with pytest.raises(ValidationError):
            await runner.ask("错误预算", max_tokens=16384)
        assert not provider.requests
        await runner.ask("小问题", max_tokens=1024)
        await runner.ask("正常问题")
        assert [request.max_tokens for request in provider.requests] == [1024, 8192]


@pytest.mark.asyncio
async def test_rewrite_can_be_disabled(tmp_path):
    provider = ScriptProvider([response("截断", "length")])
    config = make_config(reply_rewrite={"enabled": False})
    async with AgentRunner(config, provider, SessionStore(tmp_path)) as runner:
        assert (await runner.ask("问题")).finish_reason == "length"
        assert len(provider.requests) == 1
        assert runner.context.history == ()


def test_large_draft_is_trimmed_to_window_without_mutating_original():
    config = make_config().agent
    original = ModelRequest(messages=(ChatMessage(role="system", content="系统"),
                                      ChatMessage(role="user", content="任务")), max_tokens=8192)
    draft = "超长草稿" * 30000
    request = rewrite_request(original, draft, 1, config)
    assert estimate_context_tokens(request.messages).total <= 65536 - 8192 - 1024
    assert "草稿参考已裁剪" in request.messages[-1].content
    assert len(original.messages) == 2
