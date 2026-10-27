"""运行调度：能力过滤、预算中断、断点续跑与落盘（假适配器，无网络）。"""

import json
import os

import pytest
from agent_bench.adapters.base import BaseAdapter
from agent_bench.cases.models import BenchCase
from agent_bench.config import BenchConfig
from agent_bench.models import TurnResult
from agent_bench.runner import BenchRunner, RunPaths, load_traces


class FakeAdapter(BaseAdapter):
    """按用户消息返回固定回答；可选地模拟一次工具调用。"""

    def __init__(self, name, settings, context):
        super().__init__(name, settings, context)
        self.capabilities = {"multi_turn", "mcp", "cross_session_memory"}
        self.sessions = 0

    async def new_session(self) -> str:
        self.sessions += 1
        return f"s{self.sessions}"

    async def reset_memory(self) -> None:
        self.sessions = 0

    async def send(self, message: str) -> TurnResult:
        if self.env is not None:
            self.env.store.append_record({
                "seq": self.env.store.next_seq(), "tool": "notes_set",
                "arguments": {"key": "k", "value": message[:20]},
                "result_chars": 10, "elapsed_ms": 1.0, "is_error": False,
            })
        if "超时" in message:
            return TurnResult(text="", error="模拟超时", finish_reason="timeout")
        return TurnResult(text=f"回答：{message[:40]}", elapsed_seconds=0.01)

    @property
    def session_id(self) -> str:
        return f"s{self.sessions}"


def make_case(case_id="context.fake", suite="context", metrics=("context.retention",),
              requires=("multi_turn",), steps=None):
    return BenchCase.model_validate({
        "id": case_id, "suite": suite, "metrics": list(metrics),
        "requires": list(requires), "repeats": 1,
        "steps": steps or [
            {"session": "new", "user": "你好", "grade": [{"type": "contains", "value": "回答"}]},
            {"user": "再问一次", "grade": [{"type": "contains", "value": "回答"}]},
        ],
    })


def make_config(tmp_path, *, budget=100.0, agents=None, cases_repeats=1):
    return BenchConfig.model_validate({
        "budget": {"max_cost_cny": budget},
        "run": {"results_dir": str(tmp_path / "results"), "seed": 5,
                "align_context_window": 32768,
                "repeats": cases_repeats, "turn_timeout_seconds": 10},
        "agents": agents or {
            "fake": {"kind": "cli", "command": ["python", "-c", "pass"],
                     "capabilities": ["multi_turn", "mcp", "cross_session_memory"]},
        },
    })


@pytest.fixture
def patch_adapter(monkeypatch):
    def install(factory):
        monkeypatch.setattr("agent_bench.runner.create_adapter", factory)
    return install


@pytest.fixture
def patch_cases(monkeypatch):
    def install(cases):
        monkeypatch.setattr(BenchRunner, "select_cases", lambda self, suites: list(cases))
    return install


@pytest.mark.asyncio
async def test_run_writes_traces_scores_and_run_json(tmp_path, monkeypatch, patch_adapter,
                                                    patch_cases):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    patch_adapter(lambda name, settings, context: FakeAdapter(name, settings, context))
    patch_cases([make_case()])
    runner = BenchRunner(make_config(tmp_path), repo_root=tmp_path)
    code = await runner.run(agents=["fake"])
    assert code == 0
    root = tmp_path / "results"
    run_dir = next(root.iterdir())
    paths = RunPaths(run_dir)
    assert paths.run_json.exists() and paths.scores.exists()
    payload = json.loads(paths.run_json.read_text(encoding="utf-8"))
    assert payload["status"] == "finished"
    assert payload["model"]["name"] and payload["seed"] == 5
    traces = load_traces(paths)
    assert len(traces) == 2
    assert traces[0].metrics["context.retention"] == 1.0
    assert traces[0].calls == ()  # 假适配器不经过代理
    assert len(traces[0].tools) == 1
    scores = json.loads(paths.scores.read_text(encoding="utf-8"))
    assert "fake" in scores["agents"]
    assert scores["agents"]["fake"]["dimensions"]["context"]["score"] == 100.0


@pytest.mark.asyncio
async def test_capability_gate_marks_unsupported_without_scoring(tmp_path, monkeypatch,
                                                                patch_adapter, patch_cases):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    patch_adapter(lambda name, settings, context: FakeAdapter(name, settings, context))
    patch_cases([make_case(case_id="memory.fake", suite="memory",
                           metrics=("memory.cross_session_recall",),
                           requires=("cross_session_memory", "tool_result_recall"))])
    runner = BenchRunner(make_config(tmp_path), repo_root=tmp_path)
    await runner.run(agents=["fake"])
    paths = RunPaths(next((tmp_path / "results").iterdir()))
    assert paths.traces.read_text(encoding="utf-8").strip() == ""
    progress = [json.loads(line) for line in paths.progress.read_text(encoding="utf-8").splitlines()]
    assert progress and progress[0]["unsupported"] is True
    scores = json.loads(paths.scores.read_text(encoding="utf-8"))
    assert scores["agents"]["fake"]["unsupported_cases"] == ["memory.fake"]


@pytest.mark.asyncio
async def test_budget_below_estimate_refuses_to_start(tmp_path, monkeypatch, patch_adapter,
                                                      patch_cases):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    patch_adapter(lambda name, settings, context: FakeAdapter(name, settings, context))
    patch_cases([make_case()])
    runner = BenchRunner(make_config(tmp_path, budget=0.0001), repo_root=tmp_path)
    code = await runner.run(agents=["fake"])
    assert code == 2
    assert not (tmp_path / "results").exists()


@pytest.mark.asyncio
async def test_budget_tracker_stops_further_work():
    from agent_bench.runner import BudgetTracker

    tracker = BudgetTracker(1.0)
    await tracker.add(0.4)
    assert tracker.exceeded is False
    await tracker.add(0.7)
    assert tracker.exceeded is True
    assert tracker.snapshot()["spent_cny"] == 1.1


@pytest.mark.asyncio
async def test_resume_skips_completed_cases(tmp_path, monkeypatch, patch_adapter, patch_cases):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    calls = []

    def factory(name, settings, context):
        adapter = FakeAdapter(name, settings, context)
        calls.append(name)
        return adapter

    patch_adapter(factory)
    patch_cases([make_case()])
    config = make_config(tmp_path)
    runner = BenchRunner(config, repo_root=tmp_path)
    await runner.run(agents=["fake"])
    run_id = runner._run_id
    calls.clear()
    resumed = BenchRunner(config, repo_root=tmp_path)
    await resumed.run(agents=["fake"], resume=run_id)
    assert calls == []  # 已完成的用例与重复被跳过
    paths = RunPaths(tmp_path / "results" / run_id)
    assert len(load_traces(paths)) == 2


def test_estimate_scales_with_repeats_and_agents(tmp_path):
    runner = BenchRunner(make_config(tmp_path, budget=1000.0), repo_root=tmp_path)
    cases = [make_case()]
    one = runner.estimate(cases, agents=["fake"])
    two = runner.estimate(cases, agents=["fake", "fake"])
    assert two.input_tokens == one.input_tokens * 2
    assert runner.estimate(cases, agents=["fake"], repeats=3).turns == one.turns * 3
    assert runner.rounds_for(cases[0], None) == 1


@pytest.mark.asyncio
async def test_timeout_turn_is_recorded_as_failure(tmp_path, monkeypatch, patch_adapter,
                                                   patch_cases):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    patch_adapter(lambda name, settings, context: FakeAdapter(name, settings, context))
    patch_cases([make_case(steps=[
        {"session": "new", "user": "请超时", "grade": [{"type": "contains", "value": "回答"}]},
    ])])
    runner = BenchRunner(make_config(tmp_path), repo_root=tmp_path)
    await runner.run(agents=["fake"])
    paths = RunPaths(next((tmp_path / "results").iterdir()))
    traces = load_traces(paths)
    assert traces[0].error and traces[0].finish_reason == "timeout"
    assert traces[0].metrics["context.retention"] == 0.0
    scores = json.loads(paths.scores.read_text(encoding="utf-8"))
    assert scores["agents"]["fake"]["failed_cases"] == ["context.fake"]


@pytest.mark.asyncio
async def test_report_command_renders_html(tmp_path, monkeypatch, patch_adapter, patch_cases):
    from agent_bench.cli import main

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    patch_adapter(lambda name, settings, context: FakeAdapter(name, settings, context))
    patch_cases([make_case()])
    config_path = tmp_path / "bench.toml"
    config_path.write_text(
        "[budget]\nmax_cost_cny = 50\n"
        "[run]\nresults_dir = 'results'\n"
        "[agents.fake]\nkind = 'cli'\ncommand = ['python', '-c', 'pass']\n"
        "capabilities = ['multi_turn', 'mcp']\n",
        encoding="utf-8",
    )
    runner = BenchRunner(make_config(tmp_path), repo_root=tmp_path)
    await runner.run(agents=["fake"])
    run_id = next((tmp_path / "results").iterdir()).name
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(config_path), "report", run_id, "--no-open"])
    assert code == 0
    html = (tmp_path / "results" / run_id / "report.html").read_text(encoding="utf-8")
    assert "<svg" in html and "维度分" in html
    assert "失败用例" in html


@pytest.mark.asyncio
async def test_compare_runs_detects_regressions(tmp_path, monkeypatch, patch_adapter, patch_cases):
    from agent_bench.compare import compare_runs, regressions

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    patch_adapter(lambda name, settings, context: FakeAdapter(name, settings, context))
    patch_cases([make_case()])
    config = make_config(tmp_path)
    runner = BenchRunner(config, repo_root=tmp_path)
    await runner.run(agents=["fake"])
    root = tmp_path / "results"
    first = next(root.iterdir())
    first.rename(root / "run-a")
    # 复制一份并把所有指标改差，模拟“退步的版本”。
    import shutil

    shutil.copytree(root / "run-a", root / "run-b")
    traces = root / "run-b" / "traces.jsonl"
    lines = []
    for line in traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        entry["metrics"] = {key: 0.0 for key in entry.get("metrics", {})}
        lines.append(json.dumps(entry, ensure_ascii=False))
    traces.write_text("\n".join(lines) + "\n", encoding="utf-8")

    forward = compare_runs(config, "run-a", "run-b", repo_root=tmp_path)
    assert forward.agents.keys() == {"fake@run-a", "fake@run-b"}
    assert any(item.delta > 0 for item in forward.composite_diffs)
    backward = compare_runs(config, "run-b", "run-a", repo_root=tmp_path)
    assert regressions(backward)


def test_baseline_set_and_missing_baseline(tmp_path, monkeypatch, capsys):
    from agent_bench.cli import _read_baseline, cmd_baseline

    config_path = tmp_path / "bench.toml"
    config_path.write_text(
        "[budget]\nmax_cost_cny = 5\n[run]\nresults_dir = 'results'\n"
        "[agents.fake]\nkind = 'cli'\ncommand = ['python', '-c', 'pass']\n",
        encoding="utf-8",
    )
    from agent_bench.config import load_bench_config

    config = load_bench_config(config_path)
    run_dir = tmp_path / "results" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "scores.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert cmd_baseline(argparse_namespace("run-a"), config) == 0
    assert _read_baseline(config) == "run-a"
    assert cmd_baseline(argparse_namespace("missing"), config) == 2


def argparse_namespace(run_id):
    from types import SimpleNamespace

    return SimpleNamespace(run_id=run_id)


def test_trace_command_prints_grades(tmp_path, monkeypatch, capsys):
    from agent_bench.cli import main
    from agent_bench.models import GradeResult, TurnTrace

    config_path = tmp_path / "bench.toml"
    config_path.write_text(
        "[budget]\nmax_cost_cny = 5\n[run]\nresults_dir = 'results'\n"
        "[agents.fake]\nkind = 'cli'\ncommand = ['python', '-c', 'pass']\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "results" / "run-a"
    run_dir.mkdir(parents=True)
    trace = TurnTrace(run_id="run-a", agent="fake", case_id="context.fake", suite="context",
                      track="shared", turn=1, user="问题", answer="回答",
                      grades=[GradeResult(grader="contains", passed=False, detail="缺少期望内容",
                                          expected="期望", actual="回答")],
                      metrics={"context.retention": 0.0})
    (run_dir / "traces.jsonl").write_text(trace.model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path), "trace", "run-a", "context.fake"]) == 0
    output = capsys.readouterr().out
    assert "未通过" in output and "判分 contains" in output


def test_report_infers_compaction_for_black_box_agents(tmp_path):
    from agent_bench.models import ModelCallRecord, TurnTrace
    from agent_bench.report import render_html
    from agent_bench.scoring import AgentRunData, MetricSample, build_report

    config = BenchConfig.model_validate(
        {"budget": {"max_cost_cny": 5},
         "agents": {"dsh": {"kind": "cli", "command": ["x"]}}})
    traces = []
    for turn, tokens in ((1, 1000), (2, 4000), (3, 8000), (4, 2000)):
        traces.append(TurnTrace(
            run_id="r", agent="dsh", case_id="context.retention.basic", suite="context",
            track="shared", turn=turn, user="问题", answer="回答",
            metrics={"context.retention": 1.0},
            calls=(ModelCallRecord(run_id="r", agent="dsh",
                                   case_id="context.retention.basic", turn=turn,
                                   prompt_tokens=tokens, completion_tokens=10),),
        ))
    samples = [MetricSample(agent="dsh", case_id="context.retention.basic", suite="context",
                            metric="context.retention", value=1.0)]
    report = build_report("r", [AgentRunData(name="dsh", samples=samples, traces=traces)],
                          settings=config.scoring, price=config.price)
    html = render_html(report=report, traces=traces, run_meta={},
                       output=tmp_path / "report.html").read_text(encoding="utf-8")
    assert "推断压缩" in html and "非 agent 自报" in html


def test_compare_command_writes_artifacts(tmp_path, monkeypatch, capsys):
    from agent_bench.cli import main

    config_path = tmp_path / "bench.toml"
    config_path.write_text(
        "[budget]\nmax_cost_cny = 5\n[run]\nresults_dir = 'results'\n"
        "[agents.fake]\nkind = 'cli'\ncommand = ['python', '-c', 'pass']\n",
        encoding="utf-8",
    )
    for run_id, value in (("run-a", 1.0), ("run-b", 0.0)):
        run_dir = tmp_path / "results" / run_id
        run_dir.mkdir(parents=True)
        trace = {
            "run_id": run_id, "agent": "fake", "case_id": "context.fake",
            "suite": "context", "track": "shared", "turn": 1, "user": "问题",
            "answer": "回答", "metrics": {"context.retention": value},
        }
        (run_dir / "traces.jsonl").write_text(json.dumps(trace) + "\n", encoding="utf-8")
        (run_dir / "records.jsonl").write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    code = main(["--config", str(config_path), "compare", "run-a", "run-b", "--no-open"])
    assert code == 0
    artifacts = tmp_path / "results" / "run-b"
    assert (artifacts / "compare-run-a.json").exists()
    assert (artifacts / "compare.html").exists()


def test_loopback_bypass_is_added_without_clobbering(monkeypatch):
    from agent_bench.runner import ensure_loopback_bypass

    monkeypatch.setenv("NO_PROXY", "example.com")
    added = ensure_loopback_bypass()
    assert "127.0.0.1" in os.environ["NO_PROXY"]
    assert "example.com" in os.environ["NO_PROXY"]  # 用户原有条目不能被覆盖
    assert "127.0.0.1" in added
    # 再调一次不会重复追加
    assert ensure_loopback_bypass() == []
    assert os.environ["NO_PROXY"].count("127.0.0.1") == 1
