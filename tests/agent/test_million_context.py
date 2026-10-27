"""1M 配置、分批摘要与恢复的离线集成验证；不调用真实 API。"""

import asyncio

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import SummaryError
from lynx_agent.agent.tokens import estimate_context_tokens, estimate_request_tokens
from lynx_agent.cli.formatting import format_tokens
from lynx_agent.cli.metrics import compact_tokens, measure_context
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
    ModelResponse,
    ProviderTimeoutError,
    TokenUsage,
)
from lynx_agent.session import SessionEvent, SessionStore


def settings(*, pruning=True, calibrate=True):
    return Config(
        agent=AgentConfig(
            model=ModelConfig(name="fake", context_window_tokens=1_000_000, max_tokens=8192),
            context={
                "reserve_tokens": 65536, "keep_recent_tokens": 131072,
                "summary_max_tokens": 8192, "summary_generation_max_tokens": 32768,
                "summary_input_max_tokens": 131072, "safety_tokens": 32768,
                "compaction_trigger_ratio": 0.9, "calibrate_tokens": calibrate,
                "tool_result_pruning": {"enabled": pruning, "pressure_ratio": 0.85},
            },
            recovery={"max_seconds": 600},
        ),
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid", timeout_seconds=300)),
    )


class Provider(ModelProvider):
    def __init__(self, fail=None, report_usage=False, summary_usage=False):
        self.summaries = []
        self.requests = []
        self.fail = fail
        self.report_usage = report_usage
        self.summary_usage = summary_usage

    async def generate(self, request):
        self.requests.append(request)
        if request.messages[0].content.startswith("你是对话历史整理器"):
            self.summaries.append(request)
            if len(self.summaries) == 2:
                if self.fail == "timeout":
                    raise ProviderTimeoutError("simulated")
                if self.fail == "cancel":
                    raise asyncio.CancelledError()
                if self.fail == "length":
                    return ModelResponse(content="incomplete", model="fake", finish_reason="length")
            reported = estimate_request_tokens(request) * 2
            usage = TokenUsage(prompt_tokens=reported, completion_tokens=1, total_tokens=reported + 1) if self.summary_usage else None
            return ModelResponse(content="目标：完成任务。约束：保留原文。下一步：继续。", model="fake", finish_reason="stop", usage=usage)
        usage = None
        if self.report_usage:
            reported = estimate_request_tokens(request) * 2
            usage = TokenUsage(prompt_tokens=reported, completion_tokens=1, total_tokens=reported + 1)
        return ModelResponse(content="done", model="fake", finish_reason="stop", usage=usage)

    async def aclose(self):
        pass


def long_session(store, turns=10, tokens_per_turn=90000):
    session = store.create()
    for index in range(turns):
        session = session.with_event(SessionEvent(kind="turn", messages=(
            ChatMessage(role="user", content=f"turn-{index}:" + "abcd" * tokens_per_turn),
            ChatMessage(role="assistant", content="recorded"),
        )))
    store.save(session)
    return session


def test_million_profile_budget_relationships():
    config = settings()
    agent = config.agent
    assert agent.input_budget == 934464
    assert int(agent.input_budget * agent.context.tool_result_pruning.pressure_ratio) == 794294
    assert agent.compaction_trigger == 841017
    assert agent.summary_input_limit == 131072
    assert agent.output_limit == 8192
    assert agent.max_iterations == 8 and agent.recovery.max_extra_requests == 2
    assert agent.tools.max_result_chars == 16000
    assert compact_tokens(agent.model.context_window_tokens) == "1.0M"


@pytest.mark.asyncio
@pytest.mark.parametrize("pruning", [False, True])
async def test_million_history_compacts_in_bounded_batches_and_restores(tmp_path, pruning):
    store, provider, config = SessionStore(tmp_path), Provider(), settings(pruning=pruning)
    original = long_session(store)
    runner = AgentRunner(config, provider, store, original)
    assert 841017 < estimate_context_tokens(runner.context.messages).total < 934464
    response = await runner.ask("continue")
    result = runner.last_compaction
    assert response.content == "done" and result is not None
    assert len(provider.summaries) >= 6
    assert all(estimate_request_tokens(request) <= 131072 for request in provider.summaries)
    assert all(request.max_tokens == 32768 for request in provider.summaries)
    assert result.kept == original.history()[-2:]
    assert result.after_tokens < 131072 + 8192
    assert estimate_request_tokens(provider.requests[-1]) <= config.agent.input_budget
    loaded = store.load(original.id)
    assert loaded.history()[:len(original.history())] == original.history()
    restored = AgentRunner(config, Provider(), store, loaded)
    assert restored.context.messages == runner.context.messages
    reading = measure_context(runner, config)
    assert reading.window == 1_000_000 and reading.trigger == 841017
    assert "934,464" in format_tokens(runner, config)
    await restored.aclose()
    await runner.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail,error", [("timeout", ProviderTimeoutError), ("cancel", asyncio.CancelledError), ("length", SummaryError)])
async def test_failed_second_summary_keeps_million_history(tmp_path, fail, error):
    store = SessionStore(tmp_path)
    original = long_session(store)
    runner = AgentRunner(settings(), Provider(fail=fail), store, original)
    snapshot = runner.context.messages
    with pytest.raises(error):
        await runner.compact()
    assert runner.context.messages == snapshot
    saved = store.load(original.id)
    assert saved.context_history() == original.context_history()
    assert not any(event.kind == "compact" for event in saved.events)
    await runner.aclose()


@pytest.mark.asyncio
async def test_pruning_threshold_does_not_force_early_summary(tmp_path):
    store, provider = SessionStore(tmp_path), Provider()
    # 810K 高于回收线，低于独立摘要线；即使回收无候选，也不应提前摘要。
    runner = AgentRunner(settings(pruning=False), provider, store, long_session(store, turns=9))
    await runner.ask("continue")
    assert not provider.summaries and runner.last_compaction is None
    await runner.aclose()


@pytest.mark.asyncio
async def test_summary_calibration_rechecks_recent_turns_and_later_batch_sizes(tmp_path):
    store, provider = SessionStore(tmp_path), Provider(summary_usage=True)
    original = long_session(store)
    runner = AgentRunner(settings(), provider, store, original)
    result = await runner.compact()
    assert runner.token_estimate_factor == 2.1
    assert result is not None and not result.kept  # 单个近期回合校正后已超过 128K。
    assert result.summary_calls == len(provider.summaries) <= 128
    assert all(runner._loop.token_budget.request(request) <= 131072 for request in provider.summaries[1:])
    assert result.after_tokens <= runner._config.agent.input_budget
    assert store.load(original.id).history() == original.history()
    await runner.aclose()


@pytest.mark.asyncio
async def test_usage_calibration_is_visible_and_resets_at_session_boundaries(tmp_path):
    provider, config, store = Provider(report_usage=True), settings(), SessionStore(tmp_path)
    runner = AgentRunner(config, provider, store)
    await runner.ask("test")
    assert runner.token_estimate_factor == 2.1
    assert any(item.error_kind == "token_estimate_low" for item in runner.diagnostics)
    usage = runner.last_usage.latest("chat")
    assert usage.prompt_tokens == estimate_request_tokens(provider.requests[-1]) * 2
    assert measure_context(runner, config).estimate_factor == 2.1
    assert "2.100" in format_tokens(runner, config)
    runner.switch_session()
    assert runner.token_estimate_factor == 1
    await runner.ask("test again")
    assert runner.token_estimate_factor == 2.1
    runner.clear_history()
    assert runner.token_estimate_factor == 1
    await runner.aclose()
