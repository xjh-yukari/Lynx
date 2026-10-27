"""显式开启的离线压力测试：LYNX_PRESSURE_TEST=1，不调用真实 API。"""

import asyncio
import json
import os
import statistics
import tracemalloc
from time import perf_counter
from uuid import uuid4

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.tokens import estimate_request_tokens
from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ModelToolCall,
)
from lynx_agent.providers.errors import ProviderConnectionError
from lynx_agent.session import Session, SessionError, SessionEvent, SessionStore
from lynx_agent.tools import Tool, ToolRegistry
from lynx_agent.tools.builtin.filesystem import create_file_tools
from lynx_agent.tools.errors import ToolExecutionError

pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(
    os.environ.get("LYNX_PRESSURE_TEST") != "1", reason="离线压力测试需显式设置 LYNX_PRESSURE_TEST=1")]
SCALE = int(os.environ.get("LYNX_PRESSURE_SCALE", "1"))
if not 1 <= SCALE <= 5:
    raise ValueError("LYNX_PRESSURE_SCALE 必须为 1–5，避免误启动无界负载")


def settings():
    return Config(agent=AgentConfig(model=ModelConfig(name="offline-pressure", max_tokens=512, context_window_tokens=8192),
        context={"reserve_tokens": 2048, "keep_recent_tokens": 1000, "summary_max_tokens": 256,
                 "safety_tokens": 128, "tool_result_pruning": {"pressure_ratio": 0.65, "threshold_chars": 2048,
                                                               "head_chars": 512, "tail_chars": 128}}),
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake-offline", base_url="https://example.invalid")))


def answer(text="completed", reason="stop", *, reasoning=None):
    return ModelResponse(content=text, model="offline-pressure", finish_reason=reason, reasoning_content=reasoning)


def call(name="probe", **arguments):
    return ModelToolCall(id=uuid4().hex, name=name, arguments=json.dumps(arguments))


def invocation(name="probe", **arguments):
    return ModelResponse(content="", model="offline-pressure", finish_reason="tool_calls", tool_calls=(call(name, **arguments),))


def report(case, started, **metrics):
    print("\nPRESSURE " + json.dumps({"case": case, "seconds": round(perf_counter() - started, 3), **metrics}, ensure_ascii=False))


class QueueProvider(ModelProvider):
    def __init__(self):
        self.replies = []
        self.calls = 0
        self.summaries = 0

    async def generate(self, request):
        self.calls += 1
        assert estimate_request_tokens(request) <= settings().agent.input_budget
        if "最终摘要目标不超过" in request.messages[0].content:
            self.summaries += 1
            return answer("[测试摘要] 保留既有权限边界，历史任务已处理；仅用于离线负载测试，不评价语义质量。")
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def aclose(self):
        pass


class Probe(Tool):
    name = "probe"
    description = "offline probe"

    def __init__(self, *, writes=False, fail=False, slow=False):
        self.writes = writes
        self.fail = fail
        self.slow = slow
        self.calls = 0
        self.closed = 0

    @property
    def read_only(self):
        return not self.writes

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **arguments):
        self.calls += 1
        try:
            if self.slow:
                await asyncio.Event().wait()
            if self.fail:
                raise ToolExecutionError("injected failure")
            return "bounded original evidence"
        finally:
            self.closed += 1


class LargeResult(Probe):
    name = "web_fetch"

    async def execute(self, **arguments):
        self.calls += 1
        return json.dumps({"url": "https://example.invalid/offline", "title": "fake source",
                           "content": f"snapshot-{self.calls}:" + "X" * 12000, "truncated": False})


async def test_long_conversation_pressure(tmp_path):
    started = perf_counter()
    rounds = 60 * SCALE
    provider = QueueProvider()
    probe = LargeResult()
    store = SessionStore(tmp_path / "sessions")
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
    durations = []
    tracemalloc.start()
    try:
        for index in range(rounds):
            provider.replies.extend([invocation("web_fetch"), answer(f"round-{index}")])
            tick = perf_counter()
            await runner.ask(f"read source {index}; reply briefly")
            durations.append(perf_counter() - tick)
            if (index + 1) % 15 == 0:
                saved = store.load(runner.session_id)
                before = runner.context.history
                await runner.aclose()
                runner = AgentRunner(settings(), provider, store, saved, tools=ToolRegistry([probe]))
                assert runner.context.history == before
        saved = store.load(runner.session_id)
        prune_count = sum(len(event.replacements) for event in saved.events)
        assert prune_count > 0 and provider.summaries > 0
        assert sum(event.kind == "turn" for event in saved.events) == rounds
        assert sum(event.kind == "tool" for event in saved.events) == rounds
        assert probe.calls == rounds and not provider.replies
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
        await runner.aclose()
    p95 = sorted(durations)[int((len(durations) - 1) * 0.95)]
    report("long_conversation", started, turns=rounds, model_calls=provider.calls, summaries=provider.summaries,
           pruned_results=prune_count, reloads=rounds // 15, median_ms=round(statistics.median(durations) * 1000, 2),
           p95_ms=round(p95 * 1000, 2), python_peak_mib=round(peak / 1048576, 2),
           session_mib=round(store._path(saved.id).stat().st_size / 1048576, 2))


async def test_repeated_recovery_pressure(tmp_path):
    started = perf_counter()
    provider = QueueProvider()
    probe = Probe()
    store = SessionStore(tmp_path / "sessions")
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
    rounds = 60 * SCALE
    for index in range(rounds):
        provider.replies.append(invocation())
        if index % 3 == 0:
            provider.replies.append(ProviderConnectionError("injected connection failure"))
            provider.replies.append(answer("recovered"))
            with pytest.raises(ProviderConnectionError):
                await runner.ask(f"task-{index}")
        elif index % 3 == 1:
            provider.replies.extend([answer("", "length", reasoning="fake reasoning" * 100), answer("recovered")])
            assert (await runner.ask(f"task-{index}")).finish_reason == "length"
        else:
            provider.replies.extend([answer("draft", "length")] * 3 + [answer("recovered")])
            assert (await runner.ask(f"task-{index}")).finish_reason == "length"
        assert runner.can_retry
        if (index + 1) % 10 == 0:
            saved = store.load(runner.session_id)
            await runner.aclose()
            runner = AgentRunner(settings(), provider, store, saved, tools=ToolRegistry([probe]))
        assert (await runner.retry()).finish_reason == "stop"
        assert probe.calls == index + 1  # retries must never execute the tool again
        assert not runner.can_retry and runner._loop._recovery.used <= 2
    saved = store.load(runner.session_id)
    assert sum(event.kind == "turn" for event in saved.events) == rounds
    assert saved.pending_reply() is None and not provider.replies
    await runner.aclose()
    report("repeated_recovery", started, failed_tasks=rounds, recovered_tasks=rounds, tool_executions=probe.calls,
           model_calls=provider.calls, summaries=provider.summaries, reloads=rounds // 10)


async def test_duplicate_and_timeout_pressure():
    started = perf_counter()
    failing = Probe(fail=True)
    reader = ToolExecutor(ToolRegistry([failing]), settings().agent.tools)
    for index in range(1000 * SCALE):
        result = await reader.execute(call(), uuid4(), 1000)
        if index >= 2:
            assert result.error_kind == "repeated_call_blocked"
    assert failing.calls == 2
    writing = Probe(writes=True)
    writer = ToolExecutor(ToolRegistry([writing]), settings().agent.tools.model_copy(update={"allow_write_tools": True}))
    for _ in range(1000 * SCALE):
        await writer.execute(call(), uuid4(), 1000)
    assert writing.calls == 1
    slow = Probe(slow=True)
    timed = ToolExecutor(ToolRegistry([slow]), settings().agent.tools.model_copy(update={"timeout_seconds": 0.001}))
    for _ in range(25 * SCALE):
        timed.start_turn()
        for _ in range(10):
            result = await timed.execute(call(), uuid4(), 1000)
            assert result.is_error
    assert slow.calls == slow.closed == 50 * SCALE
    report("duplicate_timeout", started, requested_calls=2250 * SCALE, repeated_read_executions=failing.calls,
           repeated_write_executions=writing.calls, timeouts=slow.calls, timeout_cleanup=slow.closed)


async def test_concurrent_session_isolation_pressure(tmp_path):
    started = perf_counter()
    root = tmp_path / "workspace"
    root.mkdir()
    store = SessionStore(tmp_path / "sessions")
    semaphore = asyncio.Semaphore(4)
    results = []

    async def prepare(index):
        async with semaphore:
            path = root / f"source-{index}.txt"
            path.write_text(f"session-{index}-only", encoding="utf-8")
            provider = QueueProvider()
            provider.replies.extend([invocation("read_file", path=path.name), ProviderConnectionError("injected")])
            runner = AgentRunner(settings(), provider, store, tools=ToolRegistry(create_file_tools(FileAccess(root))))
            with pytest.raises(ProviderConnectionError):
                await runner.ask(f"read unique source {index}")
            saved = store.load(runner.session_id)
            identifier = next(event.tool.result_id for event in saved.events if event.tool)
            await runner.aclose()
            results.append((saved, identifier))

    count = 20 * SCALE
    await asyncio.gather(*(prepare(index) for index in range(count)))
    assert len({session.id for session, _ in results}) == count
    rejected = 0
    for session, own_id in results:
        provider = QueueProvider()
        provider.replies.append(answer("recovered own session"))
        runner = AgentRunner(settings(), provider, store, session, tools=ToolRegistry(create_file_tools(FileAccess(root))))
        for other, identifier in results:
            if other.id != session.id:
                with pytest.raises(ToolExecutionError):
                    runner._archive.read(identifier)
                rejected += 1
        await runner.retry()
        runner.clear_history()
        with pytest.raises(ToolExecutionError):
            runner._archive.read(own_id)
        assert not runner.can_retry
        await runner.aclose()
    report("session_isolation", started, sessions=count, concurrency=4, cross_session_rejections=rejected,
           recovered_sessions=count, clear_rejections=count)


async def test_save_failure_cycles_pressure(tmp_path):
    started = perf_counter()
    class FaultStore(SessionStore):
        fail_kind = None
        failures = 0
        def save(self, session):
            event = session.events[-1] if session.events else None
            # 本场景专门注入工具执行后的故障；维护步骤故障另设场景验证。
            eligible = event is not None and (event.kind != "diagnostic" or event.diagnostic.stage != "summary")
            if event is not None and event.kind == self.fail_kind and eligible:
                self.fail_kind = None
                self.failures += 1
                raise SessionError("injected disk failure")
            super().save(session)
    store = FaultStore(tmp_path / "sessions")
    provider = QueueProvider()
    probe = Probe()
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
    rounds = 30 * SCALE
    for index in range(rounds):
        provider.replies.extend([invocation(), ProviderConnectionError("injected connection"), answer("recovered")])
        store.fail_kind = "pending" if index % 2 else "diagnostic"
        with pytest.raises(SessionError):
            await runner.ask(f"task-{index}")
        assert runner.can_retry and runner.persistence_error
        assert any(message.role == "tool" for message in runner._pending.messages)
        with pytest.raises(ValueError):
            await runner.retry()
        runner.save_session()
        assert store.load(runner.session_id).pending_reply() is not None
        if index % 3 == 0:
            store.fail_kind = "turn"
        await runner.retry()
        if runner.persistence_error:
            runner.save_session()
        assert probe.calls == index + 1
        assert store.load(runner.session_id).pending_reply() is None
    saved = store.load(runner.session_id)
    assert sum(event.kind == "turn" for event in saved.events) == rounds
    await runner.aclose()
    report("save_fault_cycles", started, tasks=rounds, injected_save_failures=store.failures,
           tool_executions=probe.calls, completed_turns=rounds)


async def test_maintenance_save_failure_before_tools_pressure(tmp_path):
    started = perf_counter()
    class FaultStore(SessionStore):
        armed = True
        def save(self, session):
            if self.armed and session.events and session.events[-1].kind == "diagnostic":
                self.armed = False
                raise SessionError("injected summary diagnostic save failure")
            super().save(session)
    rounds = 10 * SCALE
    probe = Probe()
    for index in range(rounds):
        store = FaultStore(tmp_path / f"maintenance-{index}")
        session = Session().with_event(SessionEvent(kind="turn", messages=(
            ChatMessage(role="user", content="old source " + "X" * 16000),
            ChatMessage(role="assistant", content="noted"))))
        store.save(session)
        provider = QueueProvider()
        runner = AgentRunner(settings(), provider, store, session, tools=ToolRegistry([probe]))
        with pytest.raises(SessionError):
            await runner.ask("new task requiring a tool")
        assert runner.can_retry and runner.persistence_error
        assert not any(message.role == "tool" for message in runner._pending.messages)
        assert probe.calls == 0
        runner.save_session()
        provider.replies.append(answer("工具尚未执行；这里只能整理现有资料，不能声称本轮工具任务已完成。"))
        await runner.retry()
        assert provider.calls == 2  # one maintenance model call + one explicit finalization
        assert probe.calls == 0 and not runner.can_retry
        assert provider.replies == []
        await runner.aclose()
    report("maintenance_save_fault", started, paused_before_tools=rounds, tool_executions=probe.calls,
           finalization_only_recoveries=rounds)


async def test_cancellation_cycles_pressure(tmp_path):
    started = perf_counter()
    class WaitingProvider(QueueProvider):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()
            self.cleaned = False
        async def generate(self, request):
            if self.replies:
                return await super().generate(request)
            self.calls += 1
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned = True
    probe = Probe()
    rounds = 30 * SCALE
    for index in range(rounds):
        provider = WaitingProvider()
        provider.replies.append(invocation())
        store = SessionStore(tmp_path / f"run-{index}")
        runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
        task = asyncio.create_task(runner.ask("will cancel after tool"))
        await asyncio.wait_for(provider.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.cleaned and not runner._operation_running
        assert runner.context.history == () and not runner.can_retry
        provider.replies.extend([invocation(), answer("fresh task done")])
        await runner.ask("new explicit task")
        assert runner.context.history[-1].content == "fresh task done"
        saved = store.load(runner.session_id)
        assert sum(event.kind == "tool" for event in saved.events) == 2
        assert sum(event.kind == "turn" for event in saved.events) == 1
        await runner.aclose()
    assert probe.calls == rounds * 2
    report("cancel_cycles", started, cancellations=rounds, subsequent_successes=rounds, tool_executions=probe.calls)
