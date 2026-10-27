"""服务端用量与本地估算分离，缺失和取消不按零计算。"""

from unittest.mock import AsyncMock

import pytest

from lynx_agent.agent import AgentLoop
from lynx_agent.agent.usage import UsageTracker
from lynx_agent.cli.commands import show_api_usage
from lynx_agent.config import AgentConfig, ModelConfig
from lynx_agent.providers import ModelProvider, ModelResponse, TokenUsage


def test_tracker_separates_chat_summary_and_missing_usage(capsys):
    tracker = UsageTracker()
    first = tracker.start("summary")
    tracker.finish(first, TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120))
    tracker.start("summary")
    third = tracker.start("chat")
    tracker.finish(third, TokenUsage(prompt_tokens=40, completion_tokens=10, total_tokens=50))
    report = tracker.report()
    assert report.calls == 3
    assert report.reported_calls == 2
    assert report.usage.total_tokens == 170
    assert tracker.report("summary").usage.total_tokens == 120
    assert tracker.report("chat").usage.total_tokens == 50
    show_api_usage(tracker)
    output = capsys.readouterr().out
    assert "输入 40，输出 10，合计 50" in output
    assert "输入 100，输出 20，合计 120" in output
    assert "统计不完整" in output


def test_missing_usage_is_not_estimated_or_shown_as_zero(capsys):
    tracker = UsageTracker()
    tracker.finish(tracker.start("chat"), None)
    assert tracker.report().usage is None
    show_api_usage(tracker)
    output = capsys.readouterr().out
    assert "未获得 usage" in output
    assert "合计 0" not in output


def test_cache_breakdown_is_weighted_and_unknown_is_not_a_miss():
    tracker = UsageTracker()
    for prompt, cached in ((100, 90), (900, 90)):
        tracker.finish(tracker.start("chat"), TokenUsage(
            prompt_tokens=prompt, completion_tokens=10, total_tokens=prompt + 10,
            cached_prompt_tokens=cached,
        ))
    total = tracker.breakdown()
    assert total.cached_tokens == 180 and total.uncached_tokens == 820
    assert total.cache_percent == 18  # 不是两个请求命中率的平均值 50%。
    assert total.total_tokens == 1020 and not total.incomplete
    tracker.start("summary")
    assert tracker.breakdown().incomplete and tracker.breakdown().pending == 1
    tracker.finish(tracker.start("rewrite"), TokenUsage(prompt_tokens=50, completion_tokens=5, total_tokens=55))
    assert tracker.breakdown().cached_tokens is None
    assert tracker.breakdown().uncached_tokens is None
    assert tracker.breakdown().cache_percent is None


def test_zero_input_does_not_require_a_cache_field():
    tracker = UsageTracker()
    tracker.finish(tracker.start("chat"), TokenUsage(prompt_tokens=0, completion_tokens=3, total_tokens=3))
    assert tracker.breakdown().cached_tokens == tracker.breakdown().uncached_tokens == 0
    assert tracker.breakdown().cache_percent is None
    assert not tracker.breakdown().incomplete


@pytest.mark.asyncio
async def test_noop_compact_keeps_latest_usage_and_next_turn_resets_it():
    provider = AsyncMock(spec=ModelProvider)
    provider.generate.return_value = ModelResponse(
        content="回答", model="fake", finish_reason="stop",
        usage=TokenUsage(prompt_tokens=30, completion_tokens=10, total_tokens=40),
    )
    loop = AgentLoop(provider, AgentConfig(model=ModelConfig(name="fake")))
    await loop.run_turn("你好")
    latest = loop.last_usage
    assert await loop.compact() is None
    assert loop.last_usage is latest
    await loop.run_turn("继续")
    assert loop.last_usage is not latest
    assert loop.last_usage.report().calls == 1
    assert loop.last_usage.report().usage.total_tokens == 40
