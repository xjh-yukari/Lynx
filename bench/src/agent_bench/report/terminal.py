"""终端成绩单：按维度列出各 agent 的均值 ± 标准差、差值与显著性。"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from agent_bench.metrics import DIMENSION_LABELS, DIMENSIONS
from agent_bench.scoring import ScoreReport

NONE = "不支持"


def _cell(score: float | None) -> str:
    return NONE if score is None else f"{score:.1f}"


def _label(dimension: str) -> str:
    return DIMENSION_LABELS.get(dimension, dimension)


def dimension_table(report: ScoreReport) -> Table:
    table = Table(title="维度分（0~100，越高越好；不支持的能力不记 0 分）", expand=False)
    table.add_column("维度", style="bold")
    names = list(report.agents)
    for name in names:
        table.add_column(name, justify="right")
    table.add_column("支持的 agent", justify="right", style="dim")
    rows = [dimension for dimension in DIMENSIONS
            if any(report.agents[name].dimensions.get(dimension) is not None
                   and report.agents[name].dimensions[dimension].score is not None
                   for name in names)]
    for dimension in rows:
        row = [f"{_label(dimension)}（{dimension}）"]
        for name in names:
            item = report.agents[name].dimensions.get(dimension)
            row.append(_cell(item.score if item else None))
        supported = [name for name in names
                     if report.agents[name].dimensions.get(dimension) is not None
                     and report.agents[name].dimensions[dimension].score is not None]
        row.append(", ".join(supported))
        table.add_row(*row)
    composite = ["综合分（共同维度）"]
    for name in names:
        composite.append(_cell(report.agents[name].composite))
    composite.append(", ".join(report.shared_dimensions))
    table.add_row(*composite, style="bold")
    pass_k = ["pass^k"]
    for name in names:
        value = report.agents[name].pass_k
        pass_k.append(NONE if value is None else f"{value:.2f}")
    pass_k.append(f"k={report.agents[names[0]].pass_k_repeats if names else '—'}")
    table.add_row(*pass_k)
    return table


def metric_table(report: ScoreReport) -> Table:
    table = Table(title="指标明细（均值 ± 标准差，括号内为样本数）", expand=False)
    table.add_column("指标", style="bold")
    table.add_column("维度", style="dim")
    for name in report.agents:
        table.add_column(name, justify="right")
    metrics = sorted({metric for score in report.agents.values() for metric in score.metrics})
    for metric in metrics:
        definition = next((score.metrics[metric] for score in report.agents.values()
                           if metric in score.metrics), None)
        row = [metric, _label(definition.dimension if definition else "")]
        for name in report.agents:
            stat = report.agents[name].metrics.get(metric)
            row.append(NONE if stat is None else f"{stat.mean:.2f} ± {stat.std:.2f} ({stat.n})")
        table.add_row(*row)
    return table


def diff_table(report: ScoreReport) -> Table | None:
    if not report.composite_diffs and not report.diffs:
        return None
    table = Table(title=f"差值（{report.confidence:.0%} bootstrap 置信区间，跨 0 记为无显著差异）",
                  expand=False)
    table.add_column("对比", style="bold")
    table.add_column("维度/指标")
    table.add_column("差值", justify="right")
    table.add_column("置信区间", justify="right")
    table.add_column("结论")
    for item in (*report.composite_diffs, *report.diffs):
        conclusion = "显著" if item.significant else "无显著差异"
        table.add_row(f"{item.left} − {item.right}", str(item.dimension or item.metric or ""),
                      f"{item.delta:+.2f}", f"[{item.ci_low:+.2f}, {item.ci_high:+.2f}]",
                      conclusion)
    return table


def efficiency_table(report: ScoreReport) -> Table:
    table = Table(title="效率与成本（原始值，来自录制代理）", expand=False)
    table.add_column("agent", style="bold")
    for column in ("调用", "输入 Token", "输出 Token", "缓存命中", "P50 延迟", "费用 CNY",
                   "每用例 CNY", "重试"):
        table.add_column(column, justify="right")
    for name, score in report.agents.items():
        item = score.efficiency
        table.add_row(
            name, str(item.calls), f"{item.prompt_tokens:,}", f"{item.completion_tokens:,}",
            "—" if item.cached_ratio is None else f"{item.cached_ratio:.0%}",
            "—" if item.latency_p50_ms is None else f"{item.latency_p50_ms:.0f} ms",
            f"{item.cost_cny:.3f}", f"{item.cost_per_case_cny:.4f}", str(item.retries),
        )
    return table


def unsupported_table(report: ScoreReport) -> Table | None:
    rows = [(name, score.unsupported_cases) for name, score in report.agents.items()
            if score.unsupported_cases]
    if not rows:
        return None
    table = Table(title="不支持的用例（能力不足，不记 0 分）", expand=False)
    table.add_column("agent", style="bold")
    table.add_column("用例")
    for name, cases in rows:
        table.add_row(name, ", ".join(cases))
    return table


def print_report(report: ScoreReport, *, console: Console | None = None,
                 include_metrics: bool = True) -> None:
    console = console or Console()
    console.print(dimension_table(report))
    if include_metrics:
        console.print(metric_table(report))
    for table in (diff_table(report), efficiency_table(report), unsupported_table(report)):
        if table is not None:
            console.print(table)
    failures = [(name, score.failed_cases) for name, score in report.agents.items()
                if score.failed_cases]
    if failures:
        console.print("[red]失败用例：[/red]" + "；".join(
            f"{name}: {', '.join(cases)}" for name, cases in failures))
    for note in report.notes:
        console.print(f"[dim]说明：{note}[/dim]")
