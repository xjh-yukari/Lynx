"""运行调度：预算、并发、断点续跑、判分与落盘。

所有结果都写进 results/<run_id>/，报告只从 traces.jsonl 与 records.jsonl
渲染，不再重新调用 API。
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

from agent_bench.adapters import (
    AdapterError,
    BaseAdapter,
    BenchEnv,
    RunContext,
    create_adapter,
)
from agent_bench.cases import FactGenerator, load_cases, resolve_case
from agent_bench.cases.filler import filler_rng, generate_filler
from agent_bench.cases.models import BenchCase, ResolvedCase
from agent_bench.config import (
    BenchConfig,
    config_fingerprint,
    model_parameters,
    resolve_api_key,
)
from agent_bench.graders import GradeContext, LlmJudge, grade
from agent_bench.metrics import METRICS
from agent_bench.models import (
    GradeSpec,
    MetricSample,
    ModelCallRecord,
    ToolCallRecord,
    TurnTrace,
    utc_now,
)
from agent_bench.proxy.recorder import ProxyRecorder, cost_of, estimate_cost
from agent_bench.proxy.server import RecordingProxy, make_scope
from agent_bench.scoring import AgentRunData, ScoreReport, build_report

# 估算参数集中放在这里，便于用真实运行校准。
ESTIMATE_SYSTEM_TOKENS = 900
ESTIMATE_ANSWER_TOKENS = 320
ESTIMATE_TOOL_OVERHEAD = 0.35
ESTIMATE_CACHED_RATIO = 0.0


class BudgetExceeded(Exception):
    """预算耗尽：立即停止调度，但保留已完成的部分。"""


LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def ensure_loopback_bypass() -> list[str]:
    """让本机回环绕过系统/环境代理。

    Windows 上开着系统代理（WinINET 注册表）时，不少 HTTP 客户端会把发往
    127.0.0.1 的请求也交给代理，代理再回 502，表现为「一条模型调用都没到上游、
    也没产生费用」。这里只补 NO_PROXY，不改用户已有的条目。
    """
    added: list[str] = []
    # Windows 的变量名不区分大小写，同时写 NO_PROXY 与 no_proxy 会互相覆盖。
    names = ("NO_PROXY",) if os.name == "nt" else ("NO_PROXY", "no_proxy")
    for name in names:
        current = os.environ.get(name, "")
        entries = [item.strip() for item in current.split(",") if item.strip()]
        lowered = {item.lower() for item in entries}
        missing = [host for host in LOOPBACK_HOSTS if host.lower() not in lowered]
        if missing:
            os.environ[name] = ",".join([*entries, *missing])
            added.extend(missing)
    return added


def results_root(config: BenchConfig, repo_root: Path) -> Path:
    """结果目录：相对路径按仓库根解析，避免工作目录不同导致产物散落。"""
    root = Path(config.run.results_dir)
    return root if root.is_absolute() else repo_root / root


@dataclass
class RunPaths:
    root: Path

    @property
    def run_json(self) -> Path:
        return self.root / "run.json"

    @property
    def traces(self) -> Path:
        return self.root / "traces.jsonl"

    @property
    def scores(self) -> Path:
        return self.root / "scores.json"

    @property
    def records(self) -> Path:
        return self.root / "records.jsonl"

    @property
    def progress(self) -> Path:
        return self.root / "progress.jsonl"

    @property
    def report(self) -> Path:
        return self.root / "report.html"

    @property
    def workspaces(self) -> Path:
        return self.root / "workspaces"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in (self.traces, self.records, self.progress):
            if not path.exists():
                path.write_text("", encoding="utf-8")


@dataclass
class Task:
    agent: str
    case: BenchCase
    repeat: int


@dataclass
class CaseOutcome:
    case_id: str
    agent: str
    repeat: int
    traces: list[TurnTrace] = field(default_factory=list)
    unsupported: bool = False
    failed: bool = False
    error: str | None = None
    completed: bool = False


class BudgetTracker:
    def __init__(self, limit: float) -> None:
        self.limit = limit
        self.spent = 0.0
        self.exceeded = False
        self._lock = asyncio.Lock()

    async def add(self, amount: float) -> None:
        async with self._lock:
            self.spent += max(0.0, amount)
            if self.spent >= self.limit:
                self.exceeded = True

    def snapshot(self) -> dict[str, Any]:
        return {"limit_cny": self.limit, "spent_cny": round(self.spent, 4),
                "exceeded": self.exceeded}


@dataclass
class Estimate:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_cny: float = 0.0
    cases: int = 0
    turns: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "cost_cny": round(self.cost_cny, 4),
            "cases": self.cases,
            "turns": self.turns,
        }


class BenchRunner:
    def __init__(self, config: BenchConfig, *, repo_root: Path | None = None,
                 config_path: Path | None = None) -> None:
        self.config = config
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.config_path = config_path
        self.facts = FactGenerator(config.run.seed)
        self.budget = BudgetTracker(config.budget.max_cost_cny)
        self._judge: LlmJudge | None = None
        self._stop = asyncio.Event()
        self._write_lock = asyncio.Lock()
        self._run_id = ""
        self.started_at = utc_now().isoformat()

    # ------------------------------------------------------------------ 估算
    def rounds_for(self, case: BenchCase, repeats: int | None = None) -> int:
        return repeats or case.repeats or self.config.run.repeats

    def estimate(self, cases: list[BenchCase], *, agents: list[str] | None = None,
                 repeats: int | None = None) -> Estimate:
        names = agents if agents is not None else self.select_agents(None)
        total = Estimate()
        for case in cases:
            rounds = self.rounds_for(case, repeats)
            tokens_in, tokens_out, calls, turns = self._estimate_case(case)
            factor = rounds * len(names)
            total.input_tokens += tokens_in * factor
            total.output_tokens += tokens_out * factor
            total.calls += calls * factor
            total.turns += turns * factor
            total.cases += factor
        total.cost_cny = estimate_cost(total.input_tokens, total.output_tokens,
                                       self.config.price, cached_ratio=ESTIMATE_CACHED_RATIO)
        return total

    @staticmethod
    def _estimate_case(case: BenchCase) -> tuple[int, int, int, int]:
        """返回 (输入 token, 输出 token, 模型调用次数, 轮次)。"""
        fillers = 0
        turns = 0
        for step in case.steps:
            if step.filler:
                fillers += step.filler.tokens
                turns += step.filler.turns
            if step.user:
                turns += 1
        turns = max(1, turns)
        context = ESTIMATE_SYSTEM_TOKENS
        input_tokens = 0
        per_turn_filler = fillers / turns
        for _ in range(turns):
            context += int(per_turn_filler) + ESTIMATE_ANSWER_TOKENS
            input_tokens += context
        output_tokens = int(turns * ESTIMATE_ANSWER_TOKENS * (1 + ESTIMATE_TOOL_OVERHEAD))
        calls = int(turns * (1 + ESTIMATE_TOOL_OVERHEAD)) + 1
        return input_tokens, output_tokens, calls, turns

    # ------------------------------------------------------------------ 运行
    async def run(self, *, agents: list[str] | None = None, suites: list[str] | None = None,
                  repeats: int | None = None, resume: str | None = None,
                  estimate_only: bool = False, only_ids: list[str] | None = None) -> int:
        # 先把本机回环绕过代理：否则系统代理会劫持发往录制代理的请求。
        ensure_loopback_bypass()
        cases = self.select_cases(suites)
        if only_ids:
            wanted = set(only_ids)
            cases = [case for case in cases if case.id in wanted]
        names = self.select_agents(agents)
        if not cases:
            print("没有匹配的用例。", file=sys.stderr)
            return 2
        if not names:
            print("没有启用的 agent。", file=sys.stderr)
            return 2
        estimate = self.estimate(cases, agents=names, repeats=repeats)
        if estimate_only:
            self._print_estimate(estimate, cases, names)
            return 0
        if estimate.cost_cny > self.config.budget.max_cost_cny:
            print(f"预计费用 {estimate.cost_cny:.2f} CNY 超过预算上限 "
                  f"{self.config.budget.max_cost_cny:.2f} CNY，拒绝启动。"
                  f"可先用 --estimate 查看明细，或调整用例与预算。", file=sys.stderr)
            return 2

        api_key = resolve_api_key(self.config, root=self.repo_root)
        run_id = resume or self._new_run_id()
        self._run_id = run_id
        paths = RunPaths(self.results_root / run_id)
        paths.ensure()
        self._judge = self._build_judge(api_key)
        recorder = ProxyRecorder(run_id, paths.records, truncate=resume is None)
        proxy = RecordingProxy(run_id=run_id, proxy=self.config.proxy, model=self.config.model,
                               run=self.config.run, api_key=api_key, recorder=recorder)
        await proxy.start()
        try:
            context = RunContext(run_id=run_id, model=self.config.model, proxy=proxy,
                                 recorder=recorder, api_key=api_key,
                                 turn_timeout_seconds=self.config.run.turn_timeout_seconds)
            done = self._load_progress(paths) if resume else set()
            tasks = self._build_tasks(cases, names, repeats, done)
            self._write_run_json(paths, cases, names, repeats, proxy, status="running")
            await self._execute(tasks, paths, context)
            self._write_run_json(paths, cases, names, repeats, proxy, status="finished")
        finally:
            await proxy.stop()
        report = self.score(paths, cases, names)
        print(self._summary_line(report, paths))
        return 0

    @property
    def results_root(self) -> Path:
        return results_root(self.config, self.repo_root)

    def select_cases(self, suites: list[str] | None = None) -> list[BenchCase]:
        selected = load_cases(suites=tuple(suites) if suites else None)
        if suites:
            wanted = set(suites)
            selected = [case for case in selected if case.suite in wanted]
        return selected

    def select_agents(self, agents: list[str] | None) -> list[str]:
        enabled = [name for name, item in self.config.agents.items() if item.enabled]
        if not agents:
            return enabled
        missing = [name for name in agents if name not in self.config.agents]
        if missing:
            raise AdapterError(f"配置里没有这些 agent：{', '.join(missing)}")
        return [name for name in agents if self.config.agents[name].enabled]

    @staticmethod
    def _new_run_id() -> str:
        return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    def _build_tasks(self, cases: list[BenchCase], names: list[str], repeats: int | None,
                     done: set[tuple[str, str, int]]) -> list[Task]:
        tasks = []
        for agent in names:
            for case in cases:
                for repeat in range(self.rounds_for(case, repeats)):
                    if (agent, case.id, repeat) in done:
                        continue
                    tasks.append(Task(agent=agent, case=case, repeat=repeat))
        return tasks

    # ------------------------------------------------------------------ 调度
    async def _execute(self, tasks: list[Task], paths: RunPaths, context: RunContext) -> None:
        semaphore = asyncio.Semaphore(self.config.run.concurrency)

        async def worker(task: Task) -> None:
            async with semaphore:
                if self._stop.is_set():
                    return
                await self._run_task(task, paths, context)

        await asyncio.gather(*(worker(task) for task in tasks))

    async def _run_task(self, task: Task, paths: RunPaths, context: RunContext) -> CaseOutcome:
        outcome = CaseOutcome(case_id=task.case.id, agent=task.agent, repeat=task.repeat)
        try:
            outcome = await self._execute_case(task, paths, context, memory=True, persist=True)
        except BudgetExceeded:
            outcome.error = "预算耗尽，未开始或未完成本用例"
            self._stop.set()
        except AdapterError as error:
            outcome.error = f"适配器错误：{error}"
            outcome.failed = True
        except Exception as error:  # noqa: BLE001 - 单用例失败不能终止整个运行
            outcome.error = f"{type(error).__name__}: {error}"
            outcome.failed = True
        finally:
            await self._append_progress(paths, task, outcome)
        return outcome

    async def _execute_case(self, task: Task, paths: RunPaths, context: RunContext, *,
                            memory: bool, persist: bool) -> CaseOutcome:
        settings = self.config.agents[task.agent]
        case = task.case
        outcome = CaseOutcome(case_id=case.id, agent=task.agent, repeat=task.repeat)
        resolved = resolve_case(case, self.facts, task.repeat)
        window = case.align_context_window or self.config.run.align_context_window
        suffix = "" if memory else "-nomem"
        workspace = (paths.workspaces / _safe(task.agent) / _safe(case.id)
                     / f"r{task.repeat}{suffix}")
        scope = make_scope(task.agent, _safe(case.id) + f"-r{task.repeat}" + suffix)
        env = BenchEnv(workspace=workspace, state_dir=workspace / "env", scope=scope,
                       align_window=window)
        env.reset()
        if not memory:
            settings = settings.model_copy(update={"memory_enabled": False})
        adapter = create_adapter(task.agent, settings, context)
        turn = 0
        started = monotonic()
        try:
            await adapter.setup(workspace, env)
            missing = set(case.requires) - adapter.capabilities
            if missing:
                outcome.unsupported = True
                return outcome
            if case.track == "native" and not settings.native_tools:
                outcome.unsupported = True
                return outcome
            await adapter.reset_memory()
            case_tools: list[ToolCallRecord] = []
            for step in resolved.steps:
                if self.budget.exceeded:
                    raise BudgetExceeded
                if step.env:
                    env.apply(step.env)
                    continue
                if step.session == "new":
                    await adapter.new_session()
                for payload in self._step_messages(step, resolved, case):
                    if self.budget.exceeded:
                        raise BudgetExceeded
                    turn += 1
                    await adapter.begin_turn(turn)
                    tools_before = len(env.store.records())
                    calls_before = len(context.recorder.records)
                    result = await adapter.send(payload)
                    calls = self._calls_for(context.recorder.records[calls_before:], task, scope)
                    tools = self._tools_after(env, tools_before, task, turn)
                    case_tools.extend(tools)
                    trace = await self._grade_turn(
                        task, resolved, step, payload, result, calls, tools, case_tools,
                        env, adapter, turn,
                    )
                    outcome.traces.append(trace)
                    await self.budget.add(cost_of(calls, self.config.price))
                    if persist:
                        await self._append_trace(paths, trace)
                    if result.error:
                        outcome.failed = True
            if memory and "memory.cost" in case.metrics and settings.kind == "lynx":
                sample = await self._memory_cost_trace(task, paths, context, outcome)
                outcome.traces.append(sample)
                if persist:
                    await self._append_trace(paths, sample)
            outcome.completed = True
        finally:
            await adapter.teardown()
            if not self.config.run.keep_workspaces:
                shutil.rmtree(workspace, ignore_errors=True)
        if monotonic() - started > self.config.run.case_timeout_seconds:
            outcome.error = "用例超过时间上限"
            outcome.completed = False
        return outcome

    def _step_messages(self, step, resolved: ResolvedCase, case: BenchCase) -> list[str]:
        if step.filler:
            spec = step.filler
            return [generate_filler(spec, filler_rng(resolved.seed, case.id, index), turn=index)
                    for index in range(spec.turns)]
        if step.user:
            return [step.user]
        return []

    @staticmethod
    def _calls_for(records: tuple[ModelCallRecord, ...], task: Task,
                   scope: str) -> list[ModelCallRecord]:
        case_scope = scope.partition("~")[2]
        return [record for record in records
                if record.agent == task.agent and record.case_id == case_scope]

    @staticmethod
    def _tools_after(env: BenchEnv, before: int, task: Task, turn: int) -> list[ToolCallRecord]:
        tools = []
        for entry in env.store.records()[before:]:
            tools.append(ToolCallRecord(
                seq=int(entry.get("seq", 0)), agent=task.agent, case_id=task.case.id, turn=turn,
                tool=str(entry.get("tool", "")),
                arguments=entry.get("arguments") or {},
                result_chars=int(entry.get("result_chars", 0)),
                elapsed_ms=float(entry.get("elapsed_ms", 0)),
                is_error=bool(entry.get("is_error")),
                error_kind=entry.get("error_kind"),
            ))
        return tools

    async def _grade_turn(self, task: Task, resolved: ResolvedCase, step, payload: str,
                          result, calls: list[ModelCallRecord], tools: list[ToolCallRecord],
                          case_tools: list[ToolCallRecord], env: BenchEnv, adapter,
                          turn: int) -> TurnTrace:
        specs: list[GradeSpec] = list(step.grade)
        if step.expect_tool is not None:
            specs.append(GradeSpec(type="expect_tool", name=step.expect_tool.name,
                                   called=step.expect_tool.called, args=step.expect_tool.args))
        if step.expect_state is not None:
            specs.append(GradeSpec(type="expect_state", state=step.expect_state))
        context = GradeContext(answer=result.text, tools=tuple(tools),
                               case_tools=tuple(case_tools), state=env.tools.state_view(),
                               judge=self._judge, question=payload)
        grades = await grade(tuple(specs), context) if specs else []
        metrics: dict[str, float] = {}
        if grades:
            scored = [item for item in grades if item.grader != "llm_judge_skipped"]
            if scored:
                metrics[self._step_metric(task.case, step)] = (
                    sum(1 for item in scored if item.passed) / len(scored))
        diagnostics = dict(result.diagnostics or {})
        events = diagnostics.get("events")
        if isinstance(events, list) and len(events) > 200:
            diagnostics["events"] = events[:200]
            diagnostics["events_truncated"] = len(events) - 200
        diagnostics["step_index"] = step.index
        diagnostics["step_kind"] = "filler" if step.filler else "user"
        diagnostics["align_window"] = env.align_window
        return TurnTrace(
            run_id=self._run_id, agent=task.agent, case_id=task.case.id, suite=task.case.suite,
            track=task.case.track, repeat=task.repeat, step=step.index,
            step_kind=diagnostics["step_kind"], session=_session_of(adapter), turn=turn,
            user=payload if not step.filler else f"[填充对话 {len(payload)} 字符]",
            answer=result.text, elapsed_seconds=result.elapsed_seconds,
            finish_reason=result.finish_reason, error=result.error,
            calls=tuple(calls), tools=tuple(tools), grades=tuple(grades), metrics=metrics,
            facts=resolved.facts, tags=task.case.tags, diagnostics=diagnostics,
        )

    @staticmethod
    def _step_metric(case: BenchCase, step) -> str:
        if step.metric:
            return step.metric
        if len(case.metrics) == 1:
            return case.metrics[0]
        preferred = [name for name in case.metrics
                     if METRICS.get(name) and METRICS[name].dimension == case.suite]
        return (preferred or list(case.metrics))[0]

    async def _memory_cost_trace(self, task: Task, paths: RunPaths, context: RunContext,
                                 outcome: CaseOutcome) -> TurnTrace:
        """有记忆 / 无记忆各跑一遍，用额外 Token 比例折算得分。"""
        baseline = await self._execute_case(task, paths, context, memory=False, persist=False)
        with_memory = sum(call.prompt_tokens or 0
                          for trace in outcome.traces for call in trace.calls)
        without = sum(call.prompt_tokens or 0
                      for trace in baseline.traces for call in trace.calls)
        ratio = 0.0 if without <= 0 else max(0.0, (with_memory - without) / without)
        return TurnTrace(
            run_id=self._run_id, agent=task.agent, case_id=task.case.id,
            suite=task.case.suite, track=task.case.track, repeat=task.repeat,
            step=999, step_kind="case", turn=0, user="[对照运行]",
            answer=f"有记忆 {with_memory} / 无记忆 {without} Tokens",
            metrics={"memory.cost": max(0.0, min(1.0, 1.0 / (1.0 + ratio)))},
            diagnostics={"baseline_input_tokens": without,
                         "memory_input_tokens": with_memory, "extra_ratio": ratio},
        )

    # ------------------------------------------------------------------ 落盘
    async def _append_trace(self, paths: RunPaths, trace: TurnTrace) -> None:
        line = trace.model_dump_json() + "\n"
        async with self._write_lock:
            with paths.traces.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())

    async def _append_progress(self, paths: RunPaths, task: Task, outcome: CaseOutcome) -> None:
        entry = {
            "type": "case_done", "agent": task.agent, "case_id": task.case.id,
            "repeat": task.repeat, "completed": outcome.completed,
            "unsupported": outcome.unsupported, "failed": outcome.failed,
            "error": outcome.error, "at": utc_now().isoformat(),
        }
        async with self._write_lock:
            with paths.progress.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                stream.flush()

    def _load_progress(self, paths: RunPaths) -> set[tuple[str, str, int]]:
        done: set[tuple[str, str, int]] = set()
        if not paths.progress.exists():
            return done
        for line in paths.progress.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("type") == "case_done" and (entry.get("completed") or entry.get("unsupported")):
                done.add((str(entry["agent"]), str(entry["case_id"]), int(entry["repeat"])))
        return done

    def _write_run_json(self, paths: RunPaths, cases: list[BenchCase], names: list[str],
                        repeats: int | None, proxy: RecordingProxy, *, status: str) -> None:
        payload = {
            "run_id": paths.root.name,
            "status": status,
            "updated_at": utc_now().isoformat(),
            "started_at": self.started_at,
            "model": model_parameters(self.config),
            "seed": self.config.run.seed,
            "price": self.config.price.model_dump(mode="json"),
            "budget": self.budget.snapshot(),
            "config_hash": config_fingerprint(self.config),
            "config_path": str(self.config_path) if self.config_path else None,
            "context_alignment": {
                "align_context_window": self.config.run.align_context_window,
                "context_fill_ratio": self.config.run.context_fill_ratio,
                "note": "填充长度按本窗口计算；各 agent 的本地窗口配置见 agents[*]",
            },
            "agents": {name: self._agent_snapshot(name) for name in names},
            "cases": [{"id": case.id, "suite": case.suite, "track": case.track,
                       "requires": list(case.requires), "metrics": list(case.metrics),
                       "repeats": self.rounds_for(case, repeats)}
                      for case in cases],
            "environment": self._environment_snapshot(),
            "loopback_no_proxy": ",".join(LOOPBACK_HOSTS),
            "proxy": {"host": proxy.settings.host, "port": proxy.port,
                      "stream_usage_note": "请求未带 stream_options.include_usage 时由代理补注，"
                                           "并把多出来的 usage-only 分片从返回流中剥离"},
            "llm_judge": {"enabled": self.config.llm_judge.enabled,
                          "model": self.config.llm_judge.model if self.config.llm_judge.enabled else None},
        }
        paths.run_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    def _agent_snapshot(self, name: str) -> dict[str, Any]:
        settings = self.config.agents[name]
        data = settings.model_dump(mode="json")
        data["extra_env"] = {key: "***" for key in settings.extra_env}
        data["label"] = settings.label
        data["capabilities"] = sorted(settings.default_capabilities())
        data["patch"] = "已写入工作区" if settings.patch else None
        return data

    def _environment_snapshot(self) -> dict[str, Any]:
        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "agent_bench": _package_version("agent-bench"),
            "lynx_agent": _package_version("lynx-agent"),
            "git_commit": _git_commit(self.repo_root),
        }

    def _build_judge(self, api_key: str) -> LlmJudge | None:
        if not self.config.llm_judge.enabled:
            return None
        return LlmJudge(base_url=self.config.model.base_url, api_key=api_key,
                        model=self.config.llm_judge.model, tested_model=self.config.model.name,
                        max_tokens=self.config.llm_judge.max_tokens)

    # ------------------------------------------------------------------ 打分
    def collect(self, paths: RunPaths, names: list[str] | None = None) -> list[AgentRunData]:
        """把 traces.jsonl 与 records.jsonl 还原成可打分的证据；不调用 API。"""
        names = names or self.select_agents(None)
        traces = load_traces(paths)
        records = load_records(paths)
        flags = self._progress_flags(paths)
        data = []
        for name in names:
            agent_traces = [trace for trace in traces if trace.agent == name]
            samples = []
            for trace in agent_traces:
                for metric, value in trace.metrics.items():
                    samples.append(MetricSample(case_id=trace.case_id, suite=trace.suite,
                                                metric=metric, value=value,
                                                repeat=trace.repeat, agent=name))
            samples.extend(self.case_level_samples(agent_traces))
            unsupported, failed = flags.get(name, (set(), set()))
            data.append(AgentRunData(
                name=name,
                capabilities=set(self.config.agents[name].default_capabilities())
                if name in self.config.agents else set(),
                samples=samples, traces=agent_traces,
                unsupported=sorted(unsupported), failed=sorted(failed),
                # 对照运行（-nomem）只用于算 memory.cost，不计入该 agent 的用量。
                records=[record for record in records
                         if record.agent == name and not record.case_id.endswith("-nomem")],
            ))
        return data

    @staticmethod
    def _progress_flags(paths: RunPaths) -> dict[str, tuple[set[str], set[str]]]:
        """从 progress.jsonl 还原“不支持”与“失败”用例，供报告和续跑使用。"""
        flags: dict[str, tuple[set[str], set[str]]] = {}
        if not paths.progress.exists():
            return flags
        for line in paths.progress.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if entry.get("type") != "case_done":
                continue
            agent = str(entry.get("agent", ""))
            unsupported, failed = flags.setdefault(agent, (set(), set()))
            if entry.get("unsupported"):
                unsupported.add(str(entry.get("case_id")))
            if entry.get("failed") or entry.get("error"):
                failed.add(str(entry.get("case_id")))
        return flags

    def score(self, paths: RunPaths, cases: list[BenchCase] | None = None,
              names: list[str] | None = None) -> ScoreReport:
        cases = cases or self.select_cases(None)
        names = names or self.select_agents(None)
        case_metrics = {case.id: tuple(case.metrics) for case in cases}
        report = build_report(paths.root.name, self.collect(paths, names),
                              settings=self.config.scoring, price=self.config.price,
                              case_metrics=case_metrics, seed=self.config.run.seed)
        paths.scores.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return report

    def case_level_samples(self, traces: list[TurnTrace]) -> list[MetricSample]:
        """从 trace 重建用例级指标，保证重新打分与断点续跑口径一致。"""
        grouped: dict[tuple[str, int], list[TurnTrace]] = {}
        for trace in traces:
            grouped.setdefault((trace.case_id, trace.repeat), []).append(trace)
        targets = self.config.scoring.efficiency
        samples = []
        for (case_id, repeat), items in grouped.items():
            calls = [call for item in items for call in item.calls]
            turns = max(1, len([item for item in items if item.step_kind != "case"]))
            prompt = sum(call.prompt_tokens or 0 for call in calls)
            cached = sum(call.cached_prompt_tokens or 0 for call in calls)
            cost = cost_of(calls, self.config.price)
            latencies = sorted(call.latency_ms for call in calls if call.status < 400)
            p50 = latencies[len(latencies) // 2] if latencies else 0.0
            failures = sum(1 for item in items if item.error)
            values = {
                "efficiency.input_tokens_per_turn": _ratio_score(
                    prompt / turns, targets.max_input_tokens_per_turn),
                "efficiency.cached_ratio": _target_score(
                    (cached / prompt) if prompt else 0.0, targets.min_cached_ratio),
                "efficiency.latency_p50": _ratio_score(p50, targets.max_latency_ms),
                "efficiency.cost_per_case": _ratio_score(cost, targets.max_cost_cny_per_case),
                "stability.turn_failures": 1.0 - failures / turns,
                "context.overflow_errors": 1.0 / (
                    1.0 + sum(1 for call in calls if call.status in {400, 413})),
            }
            for metric, value in values.items():
                samples.append(MetricSample(
                    case_id=case_id, suite=items[0].suite, metric=metric,
                    value=max(0.0, min(1.0, value)), repeat=repeat, agent=items[0].agent))
        return samples

    def _summary_line(self, report: ScoreReport, paths: RunPaths) -> str:
        parts = []
        for name, score in report.agents.items():
            composite = "不支持" if score.composite is None else f"{score.composite:.1f}"
            parts.append(f"{name}={composite}")
        return (f"运行完成：{paths.root}（{', '.join(parts)}）；"
                f"费用 {self.budget.spent:.3f} / {self.budget.limit:.2f} CNY")

    def _print_estimate(self, estimate: Estimate, cases: list[BenchCase],
                        names: list[str]) -> None:
        print(f"用例 {len(cases)} 个，agent {len(names)} 个（{', '.join(names)}）")
        print(f"预计轮次 {estimate.turns}，模型调用 {estimate.calls} 次")
        print(f"预计输入 {estimate.input_tokens:,} Tokens，输出 {estimate.output_tokens:,} Tokens")
        print(f"预计费用 {estimate.cost_cny:.2f} CNY（预算上限 "
              f"{self.config.budget.max_cost_cny:.2f} CNY，价格来自 bench.toml）")
        print("说明：估算按填充 Token 与历史增长建模，未计缓存命中；实际值以代理记录为准。")


def load_traces(paths: RunPaths) -> list[TurnTrace]:
    if not paths.traces.exists():
        return []
    traces = []
    for line in paths.traces.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            traces.append(TurnTrace.model_validate_json(line))
        except ValueError:
            continue
    return traces


def load_records(paths: RunPaths) -> list[ModelCallRecord]:
    if not paths.records.exists():
        return []
    records = []
    for line in paths.records.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(ModelCallRecord.model_validate_json(line))
        except ValueError:
            continue
    return records


def load_scores(paths: RunPaths) -> ScoreReport | None:
    if not paths.scores.exists():
        return None
    try:
        return ScoreReport.model_validate_json(paths.scores.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _ratio_score(value: float, target: float) -> float:
    """越低越好的量：达到目标即 1，超标按 目标/实际 折算，落在 0~1。"""
    if target <= 0 or value <= 0:
        return 1.0
    return min(1.0, target / value)


def _target_score(value: float, target: float) -> float:
    """越高越好的量：达到目标即 1，不足按 实际/目标 折算，落在 0~1。"""
    if target <= 0:
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, value / target))


def _session_of(adapter: BaseAdapter) -> str:
    value = getattr(adapter, "session_id", "")
    return value if isinstance(value, str) else ""


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - 版本信息缺失不影响评测
        return None


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
