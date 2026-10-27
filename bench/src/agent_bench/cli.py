"""`bench` 命令入口：列表、校验、运行、续跑、报告、对比、基线、追踪。"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.table import Table

from agent_bench.adapters import AdapterError
from agent_bench.cases import CaseError, load_cases, validate_cases
from agent_bench.compare import CompareError, compare_runs, regressions, save
from agent_bench.config import BenchConfig, ConfigError, load_bench_config
from agent_bench.report import print_report, render_html
from agent_bench.runner import BenchRunner, RunPaths, load_traces, results_root

DEFAULT_CONFIG = "bench.toml"
BASELINE_FILE = "baseline.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description="agent 量化评测框架")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径（默认 bench.toml）")
    parser.add_argument("--results-dir", default=None, help="覆盖配置中的 results 目录")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出套件、用例与所需能力")

    sub.add_parser("validate", help="校验用例结构、占位符与指标名")

    run = sub.add_parser("run", help="运行评测（真实 API）")
    run.add_argument("--agents", default=None, help="逗号分隔的 agent 名")
    run.add_argument("--suites", default=None, help="逗号分隔的套件名")
    run.add_argument("--cases", default=None, help="逗号分隔的用例 id")
    run.add_argument("--repeats", type=int, default=None, help="覆盖每个用例的重复次数")
    run.add_argument("--estimate", action="store_true", help="只估算费用，不调用 API")
    run.add_argument("--against", default=None, help="与基线对比，显著退步时退出码非 0")

    resume = sub.add_parser("resume", help="续跑未完成的运行")
    resume.add_argument("run_id")

    report = sub.add_parser("report", help="重新生成报告并打开")
    report.add_argument("run_id")
    report.add_argument("--no-open", action="store_true", help="不自动打开浏览器")

    compare = sub.add_parser("compare", help="对比两次运行")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.add_argument("--no-open", action="store_true", help="不自动打开浏览器")

    baseline = sub.add_parser("baseline", help="基线管理")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", required=True)
    baseline_set = baseline_sub.add_parser("set", help="把某次运行设为基线")
    baseline_set.add_argument("run_id")

    trace = sub.add_parser("trace", help="查看某个用例的逐轮证据")
    trace.add_argument("run_id")
    trace.add_argument("case_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_bench_config(args.config)
        if args.results_dir:
            config = config.model_copy(update={
                "run": config.run.model_copy(update={"results_dir": args.results_dir})})
        handler = {
            "list": cmd_list,
            "validate": cmd_validate,
            "run": cmd_run,
            "resume": cmd_resume,
            "report": cmd_report,
            "compare": cmd_compare,
            "baseline": cmd_baseline,
            "trace": cmd_trace,
        }[args.command]
        return handler(args, config)
    except (ConfigError, CaseError, CompareError, AdapterError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


# ----------------------------------------------------------------- 子命令
def cmd_list(args, config: BenchConfig) -> int:
    cases = load_cases()
    table = Table(title=f"用例 {len(cases)} 个")
    for column in ("套件", "用例", "赛道", "重复", "所需能力", "指标"):
        table.add_column(column)
    for case in cases:
        table.add_row(case.suite, case.id, case.track, str(case.repeats or config.run.repeats),
                      ", ".join(case.requires) or "—", ", ".join(case.metrics))
    Console().print(table)
    return 0


def cmd_validate(args, config: BenchConfig) -> int:
    cases = load_cases()
    problems = validate_cases(cases)
    if problems:
        print(f"发现 {len(problems)} 个问题：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"校验通过：{len(cases)} 个用例。")
    return 0


def cmd_run(args, config: BenchConfig) -> int:
    runner = BenchRunner(config, repo_root=Path.cwd(), config_path=Path(args.config))
    agents = _split(args.agents)
    suites = _split(args.suites)
    cases = _split(args.cases)
    if args.estimate:
        selected = runner.select_cases(suites)
        if cases:
            wanted = set(cases)
            selected = [case for case in selected if case.id in wanted]
        estimate = runner.estimate(selected, agents=runner.select_agents(agents),
                                   repeats=args.repeats)
        runner._print_estimate(estimate, selected, runner.select_agents(agents))
        return 0
    code = asyncio.run(runner.run(agents=agents, suites=suites, repeats=args.repeats,
                                  only_ids=cases))
    if code != 0:
        return code
    return _check_baseline(args, config, runner)


def _check_baseline(args, config: BenchConfig, runner: BenchRunner) -> int:
    if args.against != "baseline":
        return 0
    baseline = _read_baseline(config)
    if baseline is None:
        print("没有找到基线，请先运行 bench baseline set <run_id>", file=sys.stderr)
        return 2
    try:
        report = compare_runs(config, baseline, runner._run_id, repo_root=Path.cwd())
    except CompareError as error:
        print(f"基线对比失败：{error}", file=sys.stderr)
        return 2
    worse = regressions(report)
    if not worse:
        print(f"与基线 {baseline} 相比没有显著退步。")
        return 0
    print(f"与基线 {baseline} 相比出现 {len(worse)} 项显著退步：", file=sys.stderr)
    for item in worse:
        print(f"  - {item.dimension}：{item.delta:+.2f} "
              f"[{item.ci_low:+.2f}, {item.ci_high:+.2f}]", file=sys.stderr)
    return 1


def cmd_resume(args, config: BenchConfig) -> int:
    paths = _paths(config, args.run_id)
    if not paths.run_json.exists():
        print(f"运行不存在：{paths.root}", file=sys.stderr)
        return 2
    payload = json.loads(paths.run_json.read_text(encoding="utf-8"))
    case_ids = [item["id"] for item in payload.get("cases", [])]
    agents = [name for name in payload.get("agents", {})]
    runner = BenchRunner(config, repo_root=Path.cwd(), config_path=Path(args.config))
    return asyncio.run(runner.run(agents=agents, resume=args.run_id, only_ids=case_ids))


def cmd_report(args, config: BenchConfig) -> int:
    paths = _paths(config, args.run_id)
    if not paths.traces.exists():
        print(f"运行不存在或没有 traces：{paths.root}", file=sys.stderr)
        return 2
    runner = BenchRunner(config, repo_root=Path.cwd(), config_path=Path(args.config))
    meta = _run_meta(paths)
    if meta.get("status") != "finished":
        print(f"注意：这次运行没有正常结束（status={meta.get('status', '未知')}），"
              f"成绩只覆盖已完成的用例；可用 bench resume {args.run_id} 补齐。",
              file=sys.stderr)
    names = sorted({trace.agent for trace in load_traces(paths)}) or None
    report = runner.score(paths, names=names)
    print_report(report)
    output = render_html(report=report, traces=load_traces(paths),
                         run_meta=_run_meta(paths), output=paths.report)
    print(f"\nHTML 报告：{output}")
    if not args.no_open:
        webbrowser.open(output.resolve().as_uri())
    return 0


def cmd_compare(args, config: BenchConfig) -> int:
    report = compare_runs(config, args.run_a, args.run_b, repo_root=Path.cwd())
    print_report(report)
    right = _paths(config, args.run_b)
    output = right.root / f"compare-{args.run_a}.json"
    save(report, output)
    html_path = right.root / "compare.html"
    render_html(report=report, traces=load_traces(right),
                run_meta=_run_meta(right), output=html_path)
    print(f"\n对比结果：{output}\n对比报告：{html_path}")
    if not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


def cmd_baseline(args, config: BenchConfig) -> int:
    paths = _paths(config, args.run_id)
    if not paths.scores.exists():
        print(f"这次运行还没有 scores.json：{paths.root}", file=sys.stderr)
        return 2
    target = _baseline_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"run_id": args.run_id}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"基线已设为 {args.run_id}（{target}）")
    return 0


def cmd_trace(args, config: BenchConfig) -> int:
    paths = _paths(config, args.run_id)
    traces = [trace for trace in load_traces(paths) if trace.case_id == args.case_id]
    if not traces:
        print(f"没有找到用例 {args.case_id} 的记录", file=sys.stderr)
        return 2
    for trace in traces:
        failed = any(not item.passed for item in trace.grades)
        flag = "未通过" if failed else "通过"
        tokens = sum((call.prompt_tokens or 0) + (call.completion_tokens or 0)
                     for call in trace.calls)
        print(f"[{trace.agent} · 重复 {trace.repeat} · 第 {trace.turn} 轮 · {flag}] "
              f"输入输出合计 {tokens} Tokens，耗时 {trace.elapsed_seconds:.1f}s")
        print(f"  用户：{_clip(trace.user)}")
        print(f"  回答：{_clip(trace.answer)}")
        for item in trace.grades:
            print(f"  判分 {item.grader}：{'通过' if item.passed else '未通过'} — {item.detail}")
        if trace.facts:
            print("  事实：" + "；".join(f"{key}={value}" for key, value in trace.facts.items()))
        if trace.error:
            print(f"  错误：{trace.error}")
        print()
    return 0


# ----------------------------------------------------------------- 辅助
def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _paths(config: BenchConfig, run_id: str) -> RunPaths:
    return RunPaths(results_root(config, Path.cwd()) / run_id)


def _baseline_path(config: BenchConfig) -> Path:
    return results_root(config, Path.cwd()) / BASELINE_FILE


def _read_baseline(config: BenchConfig) -> str | None:
    path = _baseline_path(config)
    if not path.exists():
        return None
    try:
        return str(json.loads(path.read_text(encoding="utf-8"))["run_id"])
    except (OSError, ValueError, KeyError):
        return None


def _run_meta(paths: RunPaths) -> dict:
    if not paths.run_json.exists():
        return {}
    try:
        return json.loads(paths.run_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"
