"""自包含的故障注入测试：不联网、不依赖其他测试模块导入。"""

import asyncio
import json
from time import monotonic
from uuid import uuid4

import pytest
from pydantic import ValidationError

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import SummaryError
from lynx_agent.agent.recovery import RecoveryBudget
from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.cli.formatting import format_diagnostics
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
    RecoveryConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ModelToolCall,
    TokenUsage,
)
from lynx_agent.providers.errors import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from lynx_agent.session import Session, SessionError, SessionEvent, SessionStore
from lynx_agent.session.recovery import PendingReply
from lynx_agent.tools import Tool, ToolRegistry
from lynx_agent.tools.builtin.filesystem import create_file_tools
from lynx_agent.tools.errors import ToolExecutionError


def settings(*, extra=2, timeout=90, output=512, summary_generation=None):
    return Config(agent=AgentConfig(model=ModelConfig(name="fake", max_tokens=output, context_window_tokens=8192),
        context={"reserve_tokens": 2048, "keep_recent_tokens": 0, "summary_max_tokens": 256,
                 "summary_generation_max_tokens": summary_generation, "safety_tokens": 128,
                 "compaction_trigger_ratio": 0.5,
                 "tool_result_pruning": {"pressure_ratio": 0.5}},
        recovery={"max_extra_requests": extra, "max_seconds": timeout}),
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))


def reply(text="done", reason="stop", *, reasoning=None, calls=()):
    return ModelResponse(content=text, model="fake", finish_reason=reason, tool_calls=calls, reasoning_content=reasoning,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=512 if reason == "length" else 5, total_tokens=522 if reason == "length" else 15))


def tool_call(name="probe", identifier="call-1", **arguments):
    return ModelToolCall(id=identifier, name=name, arguments=json.dumps(arguments))


class ScriptProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        value = next(self.responses)
        if isinstance(value, BaseException):
            raise value
        return value

    async def aclose(self):
        pass


class ProbeTool(Tool):
    name = "probe"
    description = "offline probe"

    def __init__(self, *, writes=False, fail=False, slow=False):
        self.writes = writes
        self.fail = fail
        self.slow = slow
        self.count = 0
        self.cancelled = False

    @property
    def read_only(self):
        return not self.writes

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **arguments):
        self.count += 1
        if self.slow:
            try:
                await asyncio.Event().wait()
            finally:
                self.cancelled = True
        if self.fail:
            raise ToolExecutionError("failed safely")
        return "original tool evidence"


def historical_session(store, chars=12000):
    session = Session().with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="keep important condition " + "A" * chars),
        ChatMessage(role="assistant", content="noted"))))
    store.save(session)
    return session


@pytest.mark.asyncio
async def test_soft_summary_length_preserves_history_and_continues(tmp_path):
    store = SessionStore(tmp_path)
    original = historical_session(store)
    provider = ScriptProvider([reply("", "length", reasoning="internal" * 200), reply("answer")])
    runner = AgentRunner(settings(), provider, store, original)
    phases = []
    response = await runner.ask("continue", on_status=phases.append)
    assert response.content == "answer"
    assert "summary_degraded" in phases
    assert len(provider.requests) == 2 and runner.last_usage.report("summary").calls == 1
    assert provider.requests[0].max_tokens == 512
    assert any("keep important condition" in message.content for message in provider.requests[1].messages)
    saved = store.load(original.id)
    assert saved.events[0] == original.events[0]
    assert not any(event.kind == "compact" for event in saved.events)
    diagnostic = next(event.diagnostic for event in saved.events if event.diagnostic and event.diagnostic.finish_reason == "length")
    assert diagnostic.reasoning_chars > 0 and diagnostic.usage.completion_tokens == 512
    assert "internal" not in diagnostic.model_dump_json()
    assert "kept_history_continue" in format_diagnostics(runner)


@pytest.mark.asyncio
async def test_soft_summary_is_not_retried_between_tool_steps(tmp_path):
    store = SessionStore(tmp_path)
    session = historical_session(store)
    probe = ProbeTool()
    provider = ScriptProvider([reply("", "length"), reply("", "tool_calls", calls=(tool_call(),)), reply("done")])
    runner = AgentRunner(settings(), provider, store, session, tools=ToolRegistry([probe]))
    await runner.ask("continue")
    assert len(provider.requests) == 3
    assert runner.last_usage.report("summary").calls == 1
    assert probe.count == 1


@pytest.mark.asyncio
async def test_hard_summary_failure_stops_without_chat_request(tmp_path):
    store = SessionStore(tmp_path)
    session = historical_session(store, chars=28000)
    provider = ScriptProvider([reply("", "length")])
    runner = AgentRunner(settings(), provider, store, session)
    with pytest.raises(SummaryError):
        await runner.ask("continue")
    assert len(provider.requests) == 1
    assert runner.context.history == session.context_history()
    assert runner.can_retry


@pytest.mark.asyncio
async def test_auth_failure_not_softly_swallowed(tmp_path):
    store = SessionStore(tmp_path)
    session = historical_session(store)
    provider = ScriptProvider([ProviderAuthenticationError("safe auth error")])
    runner = AgentRunner(settings(), provider, store, session)
    with pytest.raises(ProviderAuthenticationError):
        await runner.ask("continue")
    assert len(provider.requests) == 1
    assert not any(item.action == "kept_history_continue" for item in runner.diagnostics)


@pytest.mark.asyncio
async def test_explicit_compact_still_reports_failure(tmp_path):
    store = SessionStore(tmp_path)
    session = historical_session(store)
    runner = AgentRunner(settings(), ScriptProvider([reply("", "length")]), store, session)
    with pytest.raises(SummaryError):
        await runner.compact()
    assert runner.context.history == session.context_history()


@pytest.mark.asyncio
async def test_zero_text_reasoning_exhaustion_does_not_rewrite_empty_draft(tmp_path):
    provider = ScriptProvider([reply("", "length", reasoning="hidden" * 200)])
    runner = AgentRunner(settings(), provider, SessionStore(tmp_path))
    response = await runner.ask("question")
    assert response.finish_reason == "length" and len(provider.requests) == 1
    assert runner.can_retry and runner.context.history == ()
    assert runner.diagnostics[-1].error_kind == "generation_budget_exhausted"
    assert "hidden" not in format_diagnostics(runner)


@pytest.mark.asyncio
async def test_recovery_budget_shared_between_markup_and_length(tmp_path):
    provider = ScriptProvider([reply("<|DSML|calls>"), reply("long", "length"), reply("long", "length")])
    runner = AgentRunner(settings(extra=2), provider, SessionStore(tmp_path))
    response = await runner.ask("question")
    assert response.finish_reason == "length"
    assert len(provider.requests) == 3  # one normal + two shared recoveries, not 1+1+2
    assert runner._loop._recovery.used == 2


def test_recovery_budget_time_and_count_limits():
    budget = RecoveryBudget(RecoveryConfig(max_extra_requests=1, max_seconds=1))
    assert budget.consume()
    assert not budget.consume()
    budget = RecoveryBudget(RecoveryConfig())
    budget.started = monotonic() - 100
    assert not budget.consume()


@pytest.mark.asyncio
async def test_retry_restores_after_restart_without_reexecuting_tool(tmp_path):
    probe = ProbeTool()
    provider = ScriptProvider([reply("", "tool_calls", calls=(tool_call(),)), ProviderConnectionError("connection")])
    store = SessionStore(tmp_path)
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
    with pytest.raises(ProviderConnectionError):
        await runner.ask("task")
    assert probe.count == 1 and runner.can_retry
    saved = store.load(runner.session_id)
    assert saved.pending_reply() is not None and saved.context_history() == ()
    resumed_provider = ScriptProvider([reply("completed using existing evidence")])
    restored = AgentRunner(settings(output=1024), resumed_provider, store, saved, tools=ToolRegistry([probe]))
    response = await restored.retry()
    assert response.finish_reason == "stop" and probe.count == 1
    request = resumed_provider.requests[0]
    assert request.tools == () and request.max_tokens == 1024
    assert "original tool evidence" in request.messages[-1].content
    assert restored.context.history[-1].content == response.content
    assert not restored.can_retry and store.load(restored.session_id).pending_reply() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear", "new", "new_question"])
async def test_pending_does_not_leak_across_boundaries(tmp_path, action):
    provider = ScriptProvider([ProviderConnectionError("connection"), reply("fresh")])
    runner = AgentRunner(settings(), provider, SessionStore(tmp_path))
    with pytest.raises(ProviderConnectionError):
        await runner.ask("failed task")
    if action == "clear":
        runner.clear_history()
    elif action == "new":
        runner.switch_session()
    else:
        await runner.ask("new task")
    assert not runner.can_retry
    with pytest.raises(ValueError):
        await runner.retry()


@pytest.mark.asyncio
async def test_permission_change_invalidates_pending(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("evidence", encoding="utf-8")
    files = FileAccess(root)
    provider = ScriptProvider([reply("", "tool_calls", calls=(tool_call("read_file", path="a.txt"),)), ProviderConnectionError("connection")])
    runner = AgentRunner(settings(), provider, SessionStore(tmp_path / "sessions"), tools=ToolRegistry(create_file_tools(files)))
    with pytest.raises(ProviderConnectionError):
        await runner.ask("read file")
    assert runner.can_retry
    runner.set_file_mode("read")
    assert not runner.can_retry
    with pytest.raises(ValueError):
        await runner.retry()


@pytest.mark.asyncio
async def test_sensitive_permission_rechecked_on_restart(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    secret = root / ".env"
    secret.write_text("fake-placeholder-only", encoding="utf-8")
    files = FileAccess(root)
    files.policy.sensitive_grants.add(str(secret.resolve()))
    provider = ScriptProvider([reply("", "tool_calls", calls=(tool_call("read_file", path=".env"),)), ProviderConnectionError("connection")])
    store = SessionStore(tmp_path / "sessions")
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry(create_file_tools(files)))
    with pytest.raises(ProviderConnectionError):
        await runner.ask("read")
    restored_provider = ScriptProvider([])
    restored = AgentRunner(settings(), restored_provider, store, store.load(runner.session_id),
                           tools=ToolRegistry(create_file_tools(FileAccess(root))))
    with pytest.raises(ToolExecutionError, match="敏感"):
        await restored.retry()
    assert restored_provider.requests == []


@pytest.mark.asyncio
async def test_read_failures_blocked_after_limit_but_new_turn_resets():
    probe = ProbeTool(fail=True)
    executor = ToolExecutor(ToolRegistry([probe]), settings().agent.tools)
    result = None
    for i in range(3):
        result = await executor.execute(tool_call(identifier=f"id-{i}"), uuid4(), 1000)
    assert probe.count == 2
    assert result.error_kind == "repeated_call_blocked" and result.execution_state == "not_executed"
    executor.start_turn()
    await executor.execute(tool_call(), uuid4(), 1000)
    assert probe.count == 3


@pytest.mark.asyncio
async def test_successful_write_is_not_reexecuted_by_new_call_id():
    probe = ProbeTool(writes=True)
    config = settings().agent.tools.model_copy(update={"allow_write_tools": True})
    executor = ToolExecutor(ToolRegistry([probe]), config)
    first = await executor.execute(tool_call(), uuid4(), 1000)
    second = await executor.execute(tool_call(identifier="another-id"), uuid4(), 1000)
    assert not first.is_error and second.is_error
    assert probe.count == 1


@pytest.mark.asyncio
async def test_tool_timeout_is_feedback_not_whole_turn_failure(tmp_path):
    probe = ProbeTool(slow=True)
    config = settings()
    config = config.model_copy(update={"agent": config.agent.model_copy(update={
        "tools": config.agent.tools.model_copy(update={"timeout_seconds": 0.01})})})
    provider = ScriptProvider([reply("", "tool_calls", calls=(tool_call(),)), reply("tool timed out; evidence unavailable")])
    store = SessionStore(tmp_path)
    runner = AgentRunner(config, provider, store, tools=ToolRegistry([probe]))
    response = await runner.ask("task")
    assert response.finish_reason == "stop" and probe.cancelled
    error = json.loads(provider.requests[1].messages[-1].content)
    assert error["error_kind"] == "timeout" and error["execution_state"] == "unknown"
    trace = next(event.tool for event in store.load(runner.session_id).events if event.tool)
    assert trace.error_kind == "timeout" and trace.execution_state == "unknown"


@pytest.mark.asyncio
async def test_pending_save_failure_cannot_offer_durable_resume(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    runner = AgentRunner(settings(), ScriptProvider([ProviderConnectionError("connection")]), store)
    save = store.save
    def fail(session):
        if session.events and session.events[-1].kind == "pending":
            raise SessionError("disk failure")
        save(session)
    monkeypatch.setattr(store, "save", fail)
    with pytest.raises(SessionError):
        await runner.ask("task")
    assert runner.can_retry and runner.persistence_error is not None
    with pytest.raises(ValueError, match="保存"):
        await runner.retry()
    monkeypatch.setattr(store, "save", save)
    runner.save_session()
    assert runner.persistence_error is None
    assert store.load(runner.session_id).pending_reply() is not None


def test_pending_rejects_unpaired_tools_and_summary_budget_validation():
    with pytest.raises(ValidationError):
        PendingReply(turn_id=uuid4(), messages=(ChatMessage(role="user", content="task"),
            ChatMessage(role="assistant", tool_calls=(tool_call(),))))
    with pytest.raises(ValidationError):
        settings(summary_generation=3000)  # reserve=2048 cannot cover generation + safety


@pytest.mark.asyncio
async def test_retry_rejects_new_tool_requests_without_executing(tmp_path):
    probe = ProbeTool()
    provider = ScriptProvider([reply("", "tool_calls", calls=(tool_call(),)), ProviderConnectionError("connection"),
                               reply("", "tool_calls", calls=(tool_call(identifier="new-call"),))])
    runner = AgentRunner(settings(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe]))
    with pytest.raises(ProviderConnectionError):
        await runner.ask("task")
    with pytest.raises(ProviderResponseError, match="恢复只允许"):
        await runner.retry()
    assert probe.count == 1 and runner.can_retry


@pytest.mark.asyncio
async def test_write_error_reports_uncertainty_and_blocks_repeat():
    probe = ProbeTool(writes=True, fail=True)
    executor = ToolExecutor(ToolRegistry([probe]), settings().agent.tools.model_copy(update={"allow_write_tools": True}))
    first = await executor.execute(tool_call(), uuid4(), 1000)
    assert json.loads(first.content)["execution_state"] == "unknown"
    second = await executor.execute(tool_call(identifier="new-id"), uuid4(), 1000)
    assert second.error_kind == "repeated_call_blocked" and probe.count == 1


@pytest.mark.asyncio
async def test_tui_retry_and_diagnostics_commands(tmp_path):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from lynx_agent.cli.app import LynxTUI
    from lynx_agent.cli.layout import transcript_text

    probe = ProbeTool()
    provider = ScriptProvider([reply("", "tool_calls", calls=(tool_call(),)), ProviderConnectionError("connection"), reply("recovered")])
    store = SessionStore(tmp_path)
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
    with create_pipe_input() as pipe:
        ui = LynxTUI(runner, settings(), store, input_stream=pipe, output_stream=DummyOutput())
        await ui._guard(lambda: ui._dispatch("task"))
        assert runner.can_retry
        assert "/retry" in transcript_text(ui.state)
        calls = len(provider.requests)
        await ui._dispatch("/diagnostics")
        assert len(provider.requests) == calls
        assert "ProviderConnectionError" in transcript_text(ui.state)
        await ui._guard(lambda: ui._dispatch("/retry"))
        assert not runner.can_retry and probe.count == 1
        assert runner.context.history[-1].content == "recovered"
        assert "recovered" in transcript_text(ui.state)


@pytest.mark.asyncio
async def test_recovery_wait_is_bounded_and_preserves_task(tmp_path):
    class WaitingProvider(ScriptProvider):
        async def generate(self, request):
            if not self.requests:
                self.requests.append(request)
                return reply("partial", "length")
            self.requests.append(request)
            await asyncio.Event().wait()
    provider = WaitingProvider([])
    runner = AgentRunner(settings(timeout=0.02), provider, SessionStore(tmp_path))
    with pytest.raises(ProviderTimeoutError, match="自动恢复"):
        await asyncio.wait_for(runner.ask("task"), 2)
    assert len(provider.requests) == 2 and runner.can_retry
    assert runner._pending.recovery_used == 1


@pytest.mark.asyncio
async def test_cancelled_retry_does_not_reset_consumed_recovery_budget(tmp_path):
    class WaitingProvider(ScriptProvider):
        def __init__(self):
            super().__init__([])
            self.started = asyncio.Event()
        async def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ProviderConnectionError("connection")
            if len(self.requests) == 2:
                return reply("partial", "length")
            self.started.set()
            await asyncio.Event().wait()
    provider = WaitingProvider()
    store = SessionStore(tmp_path)
    runner = AgentRunner(settings(), provider, store)
    with pytest.raises(ProviderConnectionError):
        await runner.ask("task")
    task = asyncio.create_task(runner.retry())
    await asyncio.wait_for(provider.started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner._pending.recovery_used == 1
    assert store.load(runner.session_id).pending_reply().recovery_used == 1
