"""聚合统计：均值、标准差、bootstrap 置信区间、pass^k、维度分与对比。

所有指标样本都是 0~1 的得分；维度分是维度内指标均值 ×100，综合分只在
双方都支持的维度上计算，并显式列出参与的维度。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from statistics import mean, pstdev

from pydantic import Field

from agent_bench.config import PriceSettings, ScoringSettings
from agent_bench.metrics import DIMENSIONS, METRICS, dimension_of
from agent_bench.models import (
    BenchModel,
    MetricSample,
    ModelCallRecord,
    TurnTrace,
    utc_now,
)
from agent_bench.proxy.recorder import cost_of

# 不参与 pass^k 的维度：这两个维度衡量代价而不是“答对”。
PASS_K_EXCLUDED = {"efficiency", "stability"}


class MetricStat(BenchModel):
    metric: str
    dimension: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    n: int


class DimensionScore(BenchModel):
    dimension: str
    score: float | None = None
    metrics: tuple[str, ...] = ()
    n: int = 0


class EfficiencySummary(BenchModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cached_ratio: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    retries: int = 0
    usage_missing_calls: int = 0
    tokens_per_turn: float | None = None
    turns: int = 0
    cost_cny: float = 0.0
    cost_per_case_cny: float = 0.0


class AgentScore(BenchModel):
    agent: str
    capabilities: tuple[str, ...] = ()
    supported: bool = True
    metrics: dict[str, MetricStat] = Field(default_factory=dict)
    dimensions: dict[str, DimensionScore] = Field(default_factory=dict)
    composite: float | None = None
    composite_dimensions: tuple[str, ...] = ()
    pass_k: float | None = None
    pass_k_repeats: int = 1
    unsupported_cases: tuple[str, ...] = ()
    failed_cases: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    efficiency: EfficiencySummary = EfficiencySummary()


class PairDiff(BenchModel):
    left: str
    right: str
    metric: str | None = None
    dimension: str | None = None
    delta: float
    ci_low: float
    ci_high: float
    significant: bool
    note: str = ""


class ScoreReport(BenchModel):
    run_id: str
    generated_at: object = utc_now()
    agents: dict[str, AgentScore] = Field(default_factory=dict)
    diffs: tuple[PairDiff, ...] = ()
    composite_diffs: tuple[PairDiff, ...] = ()
    shared_dimensions: tuple[str, ...] = ()
    bootstrap_iterations: int = 2000
    confidence: float = 0.95
    seed: int = 0
    notes: tuple[str, ...] = ()


@dataclass
class AgentRunData:
    """一个 agent 在一次运行中的全部可计分证据。"""

    name: str
    capabilities: set[str] = field(default_factory=set)
    samples: list[MetricSample] = field(default_factory=list)
    traces: list[TurnTrace] = field(default_factory=list)
    records: list[ModelCallRecord] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(mean(values)), float(pstdev(values))


def bootstrap_ci(values: list[float], *, iterations: int = 2000,
                 confidence: float = 0.95, rng: random.Random | None = None) -> tuple[float, float]:
    """对均值做非参数 bootstrap；样本少于 2 个时退化为点估计。"""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), float(values[0])
    generator = rng or random.Random(0)
    size = len(values)
    estimates = []
    for _ in range(max(100, iterations)):
        estimates.append(mean(generator.choices(values, k=size)))
    estimates.sort()
    alpha = (1 - confidence) / 2
    low_index = max(0, min(len(estimates) - 1, int(alpha * len(estimates))))
    high_index = max(0, min(len(estimates) - 1, int((1 - alpha) * len(estimates)) - 1))
    return float(estimates[low_index]), float(estimates[high_index])


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def pass_k(samples: list[MetricSample], cases: dict[str, tuple[str, ...]],
           k: int) -> tuple[float | None, int]:
    """同一题 k 次重复全部通过的比例；只统计质量指标。"""
    grouped: dict[tuple[str, int], dict[str, float]] = {}
    for sample in samples:
        if METRICS.get(sample.metric) and dimension_of(sample.metric) in PASS_K_EXCLUDED:
            continue
        grouped.setdefault((sample.case_id, sample.repeat), {})[sample.metric] = sample.value
    if not grouped:
        return None, 0
    per_case: dict[str, list[bool]] = {}
    for (case_id, _repeat), metrics in grouped.items():
        metrics_for_case = cases.get(case_id) or tuple(metrics)
        relevant = [metrics[name] for name in metrics_for_case
                    if name in metrics and dimension_of(name) not in PASS_K_EXCLUDED]
        if not relevant:
            continue
        per_case.setdefault(case_id, []).append(all(value >= 0.999 for value in relevant))
    if not per_case:
        return None, 0
    hits = []
    for results in per_case.values():
        window = min(k, len(results))
        hits.append(1.0 if results[:window] and all(results[:window]) else 0.0)
    return float(mean(hits)), min(k, max(len(item) for item in per_case.values()))


def efficiency_summary(records: list[ModelCallRecord], traces: list[TurnTrace],
                       price: PriceSettings, cases: int) -> EfficiencySummary:
    latencies = sorted(record.latency_ms for record in records if record.status < 400)
    prompt = sum(record.prompt_tokens or 0 for record in records)
    cached = sum(record.cached_prompt_tokens or 0 for record in records)
    completion = sum(record.completion_tokens or 0 for record in records)
    turns = len(traces)
    cost = cost_of(records, price)
    return EfficiencySummary(
        calls=len(records),
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        cached_ratio=(cached / prompt) if prompt else None,
        latency_p50_ms=latencies[len(latencies) // 2] if latencies else None,
        latency_p95_ms=latencies[int((len(latencies) - 1) * 0.95)] if latencies else None,
        retries=sum(record.retries for record in records),
        usage_missing_calls=sum(1 for record in records if record.prompt_tokens is None),
        tokens_per_turn=(prompt / turns) if turns else None,
        turns=turns,
        cost_cny=cost,
        cost_per_case_cny=(cost / cases) if cases else 0.0,
    )


def score_agent(data: AgentRunData, *, settings: ScoringSettings, price: PriceSettings,
                cases: dict[str, tuple[str, ...]], seed: int = 0) -> AgentScore:
    rng = random.Random(seed)
    by_metric: dict[str, list[float]] = {}
    for sample in data.samples:
        by_metric.setdefault(sample.metric, []).append(_clip(sample.value))
    stats: dict[str, MetricStat] = {}
    for metric, values in sorted(by_metric.items()):
        definition = METRICS.get(metric)
        if definition is None:
            continue
        average, deviation = mean_std(values)
        low, high = bootstrap_ci(values, iterations=settings.bootstrap_iterations,
                                 confidence=settings.confidence, rng=rng)
        stats[metric] = MetricStat(metric=metric, dimension=definition.dimension, mean=average,
                                   std=deviation, ci_low=low, ci_high=high, n=len(values))

    dimensions: dict[str, DimensionScore] = {}
    for dimension in DIMENSIONS:
        names = tuple(name for name, stat in stats.items() if stat.dimension == dimension)
        if not names:
            dimensions[dimension] = DimensionScore(dimension=dimension)
            continue
        values = [stats[name].mean for name in names]
        dimensions[dimension] = DimensionScore(
            dimension=dimension, score=100 * mean(values), metrics=names,
            n=sum(stats[name].n for name in names),
        )

    # 稳定性：没有显式样本时用同一题 k 次重复全部通过的比例补一个维度分。
    stability = [sample.value for sample in data.samples if sample.metric == "stability.pass_k"]
    if not stability:
        ratio, repeats = pass_k(data.samples, cases, settings.pass_k)
        if ratio is not None:
            dimensions["stability"] = DimensionScore(
                dimension="stability", score=100 * ratio,
                metrics=("stability.pass_k",), n=repeats)

    available = [dimension for dimension, item in dimensions.items() if item.score is not None]
    # 维度分已经是 0~100，综合分直接取这些维度分的均值。
    composite = mean([dimensions[item].score for item in available]) if available else None
    composite = None if composite is None else float(composite)
    composite_dims = tuple(item for item in available)
    return AgentScore(
        agent=data.name,
        capabilities=tuple(sorted(data.capabilities)),
        supported=bool(data.samples) or bool(data.traces),
        metrics=stats,
        dimensions=dimensions,
        composite=composite,
        composite_dimensions=composite_dims,
        pass_k=(mean(stability) if stability else pass_k(data.samples, cases, settings.pass_k)[0]),
        pass_k_repeats=settings.pass_k,
        unsupported_cases=tuple(sorted(set(data.unsupported))),
        failed_cases=tuple(sorted(set(data.failed))),
        errors=tuple(data.errors),
        efficiency=efficiency_summary(data.records, data.traces, price,
                                      cases=max(1, len({trace.case_id for trace in data.traces}))),
    )


def diff_metric(left_values: list[float], right_values: list[float], *,
                iterations: int, confidence: float, rng: random.Random) -> tuple[float, float, float, bool]:
    """差值（left - right）与 bootstrap 置信区间；跨 0 视为无显著差异。"""
    if not left_values or not right_values:
        return 0.0, 0.0, 0.0, False
    delta = mean(left_values) - mean(right_values)
    generator = rng
    samples = []
    for _ in range(max(100, iterations)):
        a = mean(generator.choices(left_values, k=len(left_values)))
        b = mean(generator.choices(right_values, k=len(right_values)))
        samples.append(a - b)
    samples.sort()
    alpha = (1 - confidence) / 2
    low = samples[max(0, min(len(samples) - 1, int(alpha * len(samples))))]
    high = samples[max(0, min(len(samples) - 1, int((1 - alpha) * len(samples)) - 1))]
    significant = not (low <= 0 <= high)
    return float(delta), float(low), float(high), significant


def build_report(run_id: str, agents: list[AgentRunData], *, settings: ScoringSettings,
                 price: PriceSettings, case_metrics: dict[str, tuple[str, ...]] | None = None,
                 seed: int = 0) -> ScoreReport:
    metrics_by_case = dict(case_metrics or {})
    scores = {agent.name: score_agent(agent, settings=settings, price=price,
                                      cases=metrics_by_case, seed=seed)
              for agent in agents}
    by_name = {agent.name: agent for agent in agents}

    rng = random.Random(seed ^ 0x5EED)
    names = list(scores)
    diffs: list[PairDiff] = []
    composite_diffs: list[PairDiff] = []
    shared_all: tuple[str, ...] | None = None
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            left, right = scores[left_name], scores[right_name]
            shared = tuple(dimension for dimension in DIMENSIONS
                           if left.dimensions.get(dimension) is not None
                           and right.dimensions.get(dimension) is not None
                           and left.dimensions[dimension].score is not None
                           and right.dimensions[dimension].score is not None)
            shared_all = shared if shared_all is None else tuple(
                item for item in shared_all if item in shared)
            left_metric: dict[str, list[float]] = {}
            right_metric: dict[str, list[float]] = {}
            for sample in by_name[left_name].samples:
                left_metric.setdefault(sample.metric, []).append(_clip(sample.value))
            for sample in by_name[right_name].samples:
                right_metric.setdefault(sample.metric, []).append(_clip(sample.value))
            for metric in sorted(set(left_metric) & set(right_metric)):
                delta, low, high, significant = diff_metric(
                    left_metric[metric], right_metric[metric],
                    iterations=settings.bootstrap_iterations,
                    confidence=settings.confidence, rng=rng)
                diffs.append(PairDiff(left=left_name, right=right_name, metric=metric,
                                      delta=delta, ci_low=low, ci_high=high,
                                      significant=significant,
                                      note="" if significant else "无显著差异"))
            per_dimension: dict[str, tuple[list[float], list[float]]] = {}
            for metric, values in left_metric.items():
                definition = METRICS.get(metric)
                if definition and definition.dimension in shared:
                    per_dimension.setdefault(definition.dimension, ([], []))[0].extend(values)
            for metric, values in right_metric.items():
                definition = METRICS.get(metric)
                if definition and definition.dimension in shared:
                    per_dimension.setdefault(definition.dimension, ([], []))[1].extend(values)
            for dimension in shared:
                left_values, right_values = per_dimension.get(dimension, ([], []))
                delta, low, high, significant = diff_metric(
                    left_values, right_values, iterations=settings.bootstrap_iterations,
                    confidence=settings.confidence, rng=rng)
                composite_diffs.append(PairDiff(
                    left=left_name, right=right_name, dimension=dimension,
                    delta=100 * delta, ci_low=100 * low, ci_high=100 * high,
                    significant=significant, note="" if significant else "无显著差异"))
            if left.composite is not None and right.composite is not None and shared:
                left_dims = [left.dimensions[item].score for item in shared
                             if left.dimensions[item].score is not None]
                right_dims = [right.dimensions[item].score for item in shared
                              if right.dimensions[item].score is not None]
                delta, low, high, significant = diff_metric(
                    left_dims, right_dims, iterations=settings.bootstrap_iterations,
                    confidence=settings.confidence, rng=rng)
                composite_diffs.append(PairDiff(
                    left=left_name, right=right_name, dimension="综合分",
                    delta=delta, ci_low=low, ci_high=high, significant=significant,
                    note="" if significant else "无显著差异（综合分按维度均值重采样）"))

    notes = []
    if any(score.efficiency.usage_missing_calls for score in scores.values()):
        notes.append("部分模型调用没有返回 usage；这些调用不计入 Token 与费用统计。")
    if len(names) < 2:
        notes.append("只有 1 个 agent，未计算横向差值。")
    return ScoreReport(
        run_id=run_id, agents=scores, diffs=tuple(diffs), composite_diffs=tuple(composite_diffs),
        shared_dimensions=shared_all or (),
        bootstrap_iterations=settings.bootstrap_iterations,
        confidence=settings.confidence, seed=seed, notes=tuple(notes),
    )
