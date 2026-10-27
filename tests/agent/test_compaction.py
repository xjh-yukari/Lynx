"""小窗口下验证实际压缩路径和检查点恢复，不访问网络。"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.agent.tokens import estimate_context_tokens
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.config.schema import ContextConfig
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ProviderConnectionError,
    TokenUsage,
)
from lynx_agent.session import SessionError, SessionEvent, SessionStore


def make_config(auto=True):
    return Config(
        agent=AgentConfig(
            model=ModelConfig(name="fake", max_tokens=512, context_window_tokens=4096),
            context=ContextConfig(
                reserve_tokens=1024, keep_recent_tokens=500,
                summary_max_tokens=256, safety_tokens=128, auto_compact=auto,
            ),
            system_prompt="你是 Lynx。",
        ),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="fake", base_url="https://example.invalid",
        )),
    )


def response(text="目标：开发 Lynx。约束：运行数据放 E 盘，保留完整原文。下一步：继续开发。", reason="stop"):
    return ModelResponse(content=text, model="fake", finish_reason=reason)


def assert_history_unchanged(saved, original):
    # 故障诊断允许追加；原事件、摘要边界和模型历史绝不能改变。
    assert saved.events[:len(original.events)] == original.events
    assert all(event.kind == "diagnostic" for event in saved.events[len(original.events):])
    assert saved.context_boundary() == original.context_boundary()
    assert saved.context_history() == original.context_history()


def setup_runner(tmp_path, *, auto=True):
    store = SessionStore(tmp_path)
    session = store.create()
    for index in range(6):
        session = session.with_event(SessionEvent(kind="turn", messages=(
            ChatMessage(role="user", content=f"讨论 {index}：" + "项目讨论，运行数据放 E 盘。" * 80),
            ChatMessage(role="assistant", content="记录要求。" * 30),
        )))
    session = session.with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="下一步完成 CLI"),
        ChatMessage(role="assistant", content="准备实现 CLI"),
    )))
    store.save(session)
    mock = AsyncMock(spec=ModelProvider)
    mock.generate.return_value = response()
    return AgentRunner(make_config(auto), mock, store, session), mock, store, session


@pytest.mark.asyncio
async def test_automatic_compaction_batches_preserves_raw_and_restores(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    recent = runner.context.history[-2:]
    before = estimate_context_tokens(runner.context.messages, "继续").total
    await runner.ask("继续")
    result = runner.last_compaction
    assert result is not None
    assert result.before_tokens == before
    assert result.after_tokens < before
    assert result.after_tokens <= make_config().agent.input_budget
    assert result.kept == recent
    assert result.summary_calls > 1
    for call in mock.generate.call_args_list:
        request = call.args[0]
        assert estimate_context_tokens(request.messages).total <= make_config().agent.input_budget
    normal_request = mock.generate.call_args.args[0]
    assert "历史对话摘要" in normal_request.messages[1].content
    assert normal_request.messages[-3:-1] == recent
    stored = store.load(original.id)
    assert stored.events[:len(original.events)] == original.events
    assert stored.history()[:len(original.history())] == original.history()
    assert stored.events[-2].kind == "compact"
    await runner.aclose()
    restored = AgentRunner(make_config(), mock, store, stored)
    assert restored.context.messages == runner.context.messages
    restored.clear_history()
    loaded = store.load(original.id)
    assert loaded.context_boundary()[0] == ""
    assert loaded.context_history() == ()
    assert len(loaded.events) > len(original.events)
    await restored.aclose()


@pytest.mark.asyncio
async def test_repeated_compaction_uses_previous_summary(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    await runner.compact()
    first_summary = runner.context.summary
    # 新回合足够长，触发新的压缩；先前摘要必须参与下一次合并。
    await runner.ask("新进展。" * 400)
    calls_before = mock.generate.call_count
    for _ in range(4):
        await runner.ask("补充事实。" * 40)
    new_calls = mock.generate.call_args_list[calls_before:]
    assert any(
        first_summary in call.args[0].messages[1].content
        for call in new_calls
        if call.args[0].messages[0].content.startswith("你是对话历史整理器")
    )
    loaded = store.load(original.id)
    assert sum(event.kind == "compact" for event in loaded.events) >= 2
    assert loaded.context_history() == runner.context.history
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_response", [response("", "stop"), response("不完整", "length"), response("长" * 300)])
async def test_bad_summary_leaves_context_and_file_unchanged(tmp_path, bad_response):
    runner, mock, store, original = setup_runner(tmp_path)
    snapshot = runner.context.messages
    mock.generate.return_value = bad_response
    with pytest.raises(ContextError):
        await runner.compact()
    assert runner.context.messages == snapshot
    assert_history_unchanged(store.load(original.id), original)
    await runner.aclose()


@pytest.mark.asyncio
async def test_save_failure_does_not_replace_context(tmp_path, monkeypatch):
    runner, _, store, original = setup_runner(tmp_path)
    snapshot = runner.context.messages
    save = store.save

    def fail_save(session):
        raise SessionError("模拟保存失败")

    monkeypatch.setattr(store, "save", fail_save)
    with pytest.raises(SessionError):
        await runner.compact()
    assert runner.context.messages == snapshot
    assert_history_unchanged(store.load(original.id), original)
    monkeypatch.setattr(store, "save", save)
    await runner.aclose()


@pytest.mark.asyncio
async def test_provider_failure_does_not_replace_context(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    snapshot = runner.context.messages
    mock.generate.side_effect = ProviderConnectionError("模拟断网")
    with pytest.raises(ProviderConnectionError):
        await runner.compact()
    assert runner.context.messages == snapshot
    assert_history_unchanged(store.load(original.id), original)
    await runner.aclose()


@pytest.mark.asyncio
async def test_oversized_current_input_is_rejected_without_api(tmp_path):
    runner, mock, _, _ = setup_runner(tmp_path)
    with pytest.raises(ContextError, match="本次输入"):
        await runner.ask("巨大输入" * 1000)
    mock.generate.assert_not_awaited()
    await runner.aclose()


@pytest.mark.asyncio
async def test_disabled_auto_requires_manual_compaction(tmp_path):
    runner, mock, _, _ = setup_runner(tmp_path, auto=False)
    with pytest.raises(ContextError, match="自动压缩已关闭"):
        await runner.ask("继续")
    mock.generate.assert_not_awaited()
    assert await runner.compact() is not None
    await runner.aclose()


def test_invalid_budget_and_checkpoint_are_rejected():
    with pytest.raises(ValidationError):
        AgentConfig(model=ModelConfig(name="fake", context_window_tokens=1000))
    with pytest.raises(ValidationError):
        SessionEvent(kind="compact", summary="摘要", covered_until=1, before_tokens=10, after_tokens=20)


@pytest.mark.asyncio
async def test_cancelled_summary_preserves_state(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    snapshot = runner.context.messages
    mock.generate.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await runner.compact()
    assert runner.context.messages == snapshot
    assert_history_unchanged(store.load(original.id), original)
    await runner.aclose()


@pytest.mark.asyncio
async def test_nonshrinking_summary_is_not_committed(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create().with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="问"), ChatMessage(role="assistant", content="答"),
    )))
    store.save(session)
    config = make_config()
    config = Config(
        providers=config.providers,
        agent=AgentConfig(
            model=config.agent.model,
            context=config.agent.context.model_copy(update={"keep_recent_tokens": 0}),
        ),
    )
    mock = AsyncMock(spec=ModelProvider)
    mock.generate.return_value = response()
    runner = AgentRunner(config, mock, store, session)
    snapshot = runner.context.messages
    with pytest.raises(ContextError, match="未减少"):
        await runner.compact()
    assert runner.context.messages == snapshot
    assert_history_unchanged(store.load(session.id), session)
    await runner.aclose()


@pytest.mark.asyncio
async def test_cancel_between_summary_batches_preserves_original(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    snapshot = runner.context.messages
    second_batch = asyncio.Event()
    phases = []
    calls = 0

    async def generate(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return response().model_copy(update={"usage": TokenUsage(
                prompt_tokens=100, completion_tokens=20, total_tokens=120,
            )})
        second_batch.set()
        await asyncio.Event().wait()

    mock.generate.side_effect = generate
    task = asyncio.create_task(runner.compact(on_status=phases.append))
    await asyncio.wait_for(second_batch.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert phases == ["compacting"]
    assert runner.context.messages == snapshot
    assert_history_unchanged(store.load(original.id), original)
    usage = runner.last_usage.report("summary")
    assert usage.calls == 2
    assert usage.reported_calls == 1
    assert usage.usage.total_tokens == 120
    await runner.aclose()


@pytest.mark.asyncio
async def test_cancel_generation_keeps_committed_summary_and_can_continue(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    generating = asyncio.Event()
    phases = []

    async def generate(request):
        if request.messages[0].content.startswith("你是对话历史整理器"):
            return response()
        generating.set()
        await asyncio.Event().wait()

    mock.generate.side_effect = generate
    task = asyncio.create_task(runner.ask("取消的问题", on_status=phases.append))
    await asyncio.wait_for(generating.wait(), timeout=3)
    assert phases == ["compacting", "generating"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    loaded = store.load(original.id)
    assert loaded.events[-1].kind == "compact"
    assert loaded.history() == original.history()
    assert runner.context.summary
    assert all(item.content != "取消的问题" for item in runner.context.messages)
    mock.aclose.assert_not_awaited()
    mock.generate.side_effect = None
    mock.generate.return_value = response("取消后继续回复")
    assert (await runner.ask("继续")).content == "取消后继续回复"
    assert store.load(original.id).events[-1].kind == "turn"
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_usage_is_aggregated_and_not_double_counted(tmp_path):
    runner, mock, _, _ = setup_runner(tmp_path)
    phases = []
    summary_usage = TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    chat_usage = TokenUsage(prompt_tokens=200, completion_tokens=30, total_tokens=230)

    async def generate(request):
        is_summary = request.messages[0].content.startswith("你是对话历史整理器")
        assert phases[-1] == ("compacting" if is_summary else "generating")
        return response().model_copy(update={"usage": summary_usage if is_summary else chat_usage})

    mock.generate.side_effect = generate
    await runner.ask("继续", on_status=phases.append)
    count = runner.last_compaction.summary_calls
    assert count > 1
    assert runner.last_usage.report("summary").usage.total_tokens == count * 120
    assert runner.last_usage.report("chat").usage.total_tokens == 230
    assert runner.last_usage.report().usage.total_tokens == count * 120 + 230
    assert runner.last_usage.report().reported_calls == count + 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_auto_compaction_remains_nonstream_then_chat_streams(tmp_path):
    runner, mock, store, original = setup_runner(tmp_path)
    mock.supports_streaming = True
    parts = []

    async def stream(request):
        yield ModelStreamEvent(kind="delta", text="流式回答")
        yield ModelStreamEvent(kind="done", response=response("流式回答"))

    mock.stream = stream

    def on_text(text):
        assert store.load(original.id).events[-1].kind == "compact"
        parts.append(text)

    await runner.ask("继续", on_text=on_text)
    assert parts == ["流式回答"]
    assert mock.generate.call_count == runner.last_compaction.summary_calls
    assert store.load(original.id).events[-1].messages[-1].content == "流式回答"
    await runner.aclose()
