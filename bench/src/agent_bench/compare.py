"""两次运行之间的对比：同一 agent 的两个版本，或两个不同 agent。

做法是把两次运行的同名证据重命名成 `name@run_id`，复用同一套
bootstrap 统计，因此差值与显著性的口径与成绩单完全一致。
"""

from __future__ import annotations

from pathlib import Path

from agent_bench.config import BenchConfig
from agent_bench.runner import BenchRunner, RunPaths, results_root
from agent_bench.scoring import PairDiff, ScoreReport, build_report


class CompareError(Exception):
    """无法对比：运行不存在或没有可比数据。"""


def _rename(data, run_id: str):
    data.name = f"{data.name}@{run_id}"
    return data


def compare_runs(config: BenchConfig, left: str, right: str, *,
                 repo_root: Path) -> ScoreReport:
    runner = BenchRunner(config, repo_root=repo_root)
    root = results_root(config, repo_root)
    left_paths = RunPaths(root / left)
    right_paths = RunPaths(root / right)
    for name, paths in (("left", left_paths), ("right", right_paths)):
        if not paths.traces.exists():
            raise CompareError(f"{name} 运行不存在或没有 traces：{paths.root}")
    agents = list(config.agents)
    left_data = [_rename(item, left) for item in runner.collect(left_paths, agents)]
    right_data = [_rename(item, right) for item in runner.collect(right_paths, agents)]
    data = [item for item in left_data + right_data if item.samples]
    if len(data) < 2:
        raise CompareError("两次运行没有可比的指标样本")
    cases = runner.select_cases(None)
    case_metrics = {case.id: tuple(case.metrics) for case in cases}
    return build_report(f"{left}-vs-{right}", data, settings=config.scoring,
                        price=config.price, case_metrics=case_metrics,
                        seed=config.run.seed)


def regressions(report: ScoreReport) -> list[PairDiff]:
    return [item for item in report.composite_diffs
            if item.significant and item.delta < 0]


def save(report: ScoreReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
