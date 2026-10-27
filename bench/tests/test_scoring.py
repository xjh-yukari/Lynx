"""统计：bootstrap 区间、pass^k、维度分与跨 agent 差值。"""

import random

import pytest
from agent_bench.config import PriceSettings, ScoringSettings
from agent_bench.models import MetricSample
from agent_bench.scoring import (
    AgentRunData,
    bootstrap_ci,
    build_report,
    diff_metric,
    mean_std,
    pass_k,
    score_agent,
)


def sample(metric, value, case_id="case", repeat=0, agent="fake"):
    return MetricSample(agent=agent, case_id=case_id, suite=metric.split(".")[0],
                        metric=metric, value=value, repeat=repeat)


def settings(**kwargs):
    payload = {"bootstrap_iterations": 200}
    payload.update(kwargs)
    return ScoringSettings(**payload)


def test_bootstrap_ci_brackets_the_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    low, high = bootstrap_ci(values, iterations=400, rng=random.Random(1))
    average, _ = mean_std(values)
    assert low <= average <= high
    assert high - low > 0


def test_bootstrap_ci_handles_single_value():
    assert bootstrap_ci([0.5]) == (0.5, 0.5)
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_pass_k_counts_repeats():
    samples = []
    for repeat in range(3):
        samples.append(sample("memory.cross_session_recall", 1.0, repeat=repeat))
    samples.append(sample("memory.cross_session_recall", 0.0, case_id="other", repeat=0))
    ratio, repeats = pass_k(samples, {"case": ("memory.cross_session_recall",),
                                      "other": ("memory.cross_session_recall",)}, k=3)
    assert repeats == 3
    assert ratio == 0.5


def test_dimension_and_composite_scores():
    data = AgentRunData(
        name="fake",
        samples=[sample("context.retention", 1.0), sample("context.retention", 0.0),
                 sample("memory.update_accuracy", 0.5)],
    )
    score = score_agent(data, settings=settings(), price=PriceSettings(), cases={})
    assert score.dimensions["context"].score == 50.0
    assert score.dimensions["memory"].score == 50.0
    assert score.dimensions["tools"].score is None
    # 稳定性（pass^k）是自动补充的维度，也进入综合分：本题有一次未通过 → 0 分。
    assert score.dimensions["stability"].score == 0.0
    assert score.composite == pytest.approx((50 + 50 + 0) / 3)
    assert set(score.composite_dimensions) == {"context", "memory", "stability"}
    assert score.supported is True


def test_unsupported_dimension_is_not_zero():
    data = AgentRunData(name="fake", samples=[sample("context.retention", 1.0)])
    score = score_agent(data, settings=settings(), price=PriceSettings(), cases={})
    assert score.composite == 100.0
    assert score.dimensions["memory"].score is None


def test_diff_marks_overlapping_interval_as_not_significant():
    delta, low, high, significant = diff_metric([0.5, 0.6], [0.5, 0.6], iterations=300,
                                                confidence=0.95, rng=random.Random(2))
    assert abs(delta) < 1e-9
    assert low <= 0 <= high
    assert significant is False


def test_build_report_uses_shared_dimensions_only():
    left = AgentRunData(name="a", samples=[sample("context.retention", 1.0),
                                           sample("memory.update_accuracy", 1.0)])
    right = AgentRunData(name="b", samples=[sample("context.retention", 0.0)])
    report = build_report("run", [left, right], settings=settings(), price=PriceSettings(),
                          case_metrics={"case": ("context.retention",)})
    # stability（pass^k）两边都有，因此也是共同维度；memory 只有左边有。
    assert "memory" not in report.shared_dimensions
    assert "context" in report.shared_dimensions
    assert report.agents["a"].dimensions["memory"].score == 100.0
    assert report.agents["b"].dimensions["memory"].score is None
    assert any(item.metric == "context.retention" and item.significant
               for item in report.diffs)


def test_efficiency_summary_is_computed_from_records():
    from agent_bench.models import ModelCallRecord

    records = [ModelCallRecord(run_id="r", agent="a", case_id="c", prompt_tokens=100,
                               completion_tokens=10, cached_prompt_tokens=40,
                               latency_ms=120.0)
               for _ in range(3)]
    data = AgentRunData(name="a", samples=[sample("context.retention", 1.0)],
                        records=records)
    score = score_agent(data, settings=settings(), price=PriceSettings(input_per_million=2.0,
                                                                     output_per_million=8.0),
                        cases={})
    assert score.efficiency.calls == 3
    assert score.efficiency.prompt_tokens == 300
    assert score.efficiency.cached_ratio is not None
    assert score.efficiency.cost_cny > 0
