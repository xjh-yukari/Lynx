"""固定时钟与官方返回用量，不调用付费服务。"""

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.usage import UsageRecord, UsageTracker
from lynx_agent.billing import CostLedger
from lynx_agent.billing.pricing import (
    BEIJING,
    official_endpoint,
    period_at,
    price_components,
    price_usage,
)
from lynx_agent.config import Config
from lynx_agent.providers import ModelProvider, ModelResponse, TokenUsage
from lynx_agent.session import SessionStore


def moment(day=19, hour=10, minute=0, year=2026):
    return datetime(year, 10, day, hour, minute, tzinfo=BEIJING)


def sample():
    return TokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, cached_prompt_tokens=800)


@pytest.mark.parametrize(("hour", "minute", "expected"), [
    (8, 59, "off_peak"), (9, 0, "peak"), (11, 59, "peak"), (12, 0, "off_peak"),
    (13, 59, "off_peak"), (14, 0, "peak"), (17, 59, "peak"), (18, 0, "off_peak"),
])
def test_boundaries(hour, minute, expected):
    assert period_at(moment(hour=hour, minute=minute)) == expected
    assert period_at(moment(hour=hour, minute=minute).astimezone(UTC)) == expected


def test_holidays_weekends_and_unknown_calendar():
    assert period_at(moment(day=1)) == "off_peak"
    assert period_at(moment(day=24)) == "off_peak"
    assert period_at(datetime(2026, 10, 10, 10, tzinfo=BEIJING)) == "off_peak"
    assert period_at(datetime(2027, 10, 18, 10, tzinfo=BEIJING)) is None
    with pytest.raises(ValueError, match="时区"):
        period_at(datetime(2026, 10, 19))


def test_cached_input_output_and_half_price():
    assert price_usage("deepseek-flash", sample(), moment()) == Decimal("0.001232")
    assert price_usage("deepseek-v4-flash", sample(), moment(hour=20)) == Decimal("0.000616")
    assert price_usage("deepseek-v4-pro", sample(), moment()) == Decimal("0.00474")


def test_cost_components_and_session_usage_are_deduplicated():
    parts = price_components("deepseek-flash", sample(), moment())
    assert parts == (Decimal("0.000032"), Decimal("0.0004"), Decimal("0.0008"))
    ledger = CostLedger("deepseek", "https://api.deepseek.com")
    record = UsageRecord("chat", usage=sample(), started_at=moment(), finished=True)
    for _ in range(3):
        ledger.observe("a", "deepseek-flash", record)
    assert ledger.usage("a").total_tokens == 1100
    assert ledger.usage("a").cached_tokens == 800
    assert "缓存命中输入 ¥0.00003200" in ledger.details("a")
    assert "未命中输入 ¥0.00040000" in ledger.details("a")
    assert "输出 ¥0.00080000" in ledger.details("a")
    assert ledger.usage("b").calls == 0


def test_unknown_cost_never_becomes_zero():
    assert price_usage("other", sample(), moment()) is None
    assert price_usage("deepseek-flash", None, moment()) is None
    assert price_usage("deepseek-flash", sample(), moment(), official=False) is None
    missing = sample().model_copy(update={"cached_prompt_tokens": None})
    assert price_usage("deepseek-flash", missing, moment()) is None
    assert not official_endpoint("https://api.deepseek.com.example.org/v1")
    assert not official_endpoint("http://api.deepseek.com")
    assert official_endpoint("https://api.deepseek.com/v1/")


def test_dedup_categories_sessions_and_missing(monkeypatch):
    ledger = CostLedger("deepseek", "https://api.deepseek.com")
    record = UsageRecord("chat", started_at=moment())
    ledger.observe("a", "deepseek-flash", record)
    assert "*" in ledger.label("a")
    done = replace(record, usage=sample(), finished=True)
    ledger.observe("a", "deepseek-flash", done)
    ledger.observe("a", "deepseek-flash", done)
    monkeypatch.setattr("lynx_agent.billing.pricing.utc_now", lambda: moment(hour=20))
    assert ledger.report("a").amount == Decimal("0.001232")
    assert ledger.report("a").calls == 1
    assert ledger.label("a").endswith("谷")
    tracker = UsageTracker(lambda rec: ledger.observe("a", "deepseek-flash", rec))
    for kind in ("summary", "rewrite", "memory"):
        tracker.finish(tracker.start(kind), sample())
        assert ledger.report("a", kind).calls == 1
    tracker.finish(tracker.start("chat"), None)
    tracker.start("chat")  # cancellation/timeout without usage
    assert ledger.report("a").unpriced == 1
    assert ledger.report("a").pending == 1
    assert "*" in ledger.label("a")
    assert ledger.report("b").calls == 0
    assert "非账单" in ledger.details("a")


@pytest.mark.asyncio
async def test_runner_accumulates_and_switches_without_recount(tmp_path):
    cfg = Config.model_validate({"agent": {"model": {"name": "deepseek-flash"}},
                                 "providers": {"deepseek": {"api_key": "fake", "base_url": "https://api.deepseek.com"}}})
    provider = AsyncMock(spec=ModelProvider)
    provider.supports_streaming = False
    provider.generate.return_value = ModelResponse(content="好", model="deepseek-flash", finish_reason="stop", usage=sample())
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(cfg, provider, store) as runner:
        await runner.ask("第一句")
        first = runner.session_id
        await runner.ask("第二句")
        assert runner.costs.report(first).calls == 2
        runner.switch_session()
        await runner.ask("新的会话")
        assert runner.costs.report(runner.session_id).calls == 1
        runner.switch_session(first)
        assert runner.costs.report(first).calls == 2
        await runner.ask("继续")
        assert runner.costs.report(first).calls == 3
