"""离线覆盖结构化回收、存档恢复、近期保护、权限与失败恢复。"""

import asyncio
import json
from uuid import uuid4

import httpx2
import pytest
from openai import BadRequestError
from pydantic import ValidationError

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.agent.tokens import estimate_request_tokens
from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.agent.tool_result_adapters import bound_structured_result, prune_content
from lynx_agent.agent.tool_result_pruner import plan_pruning
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.config.loader import load_config
from lynx_agent.config.schema import ToolResultPruningConfig
from lynx_agent.filesystem.service import FileAccess
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    TokenUsage,
)
from lynx_agent.providers.deepseek import DeepSeekProvider, _status_error
from lynx_agent.providers.errors import (
    ProviderConnectionError,
    ProviderContextOverflowError,
)
from lynx_agent.providers.models import ToolTrace
from lynx_agent.session import Session, SessionError, SessionEvent, SessionStore
from lynx_agent.session.tool_results import (
    ToolResultArchive,
    ToolResultReplacement,
    content_hash,
)
from lynx_agent.tools import Tool, ToolCall, ToolRegistry
from lynx_agent.tools.builtin.filesystem import create_file_tools
from lynx_agent.tools.builtin.tool_results import ReadToolResultTool
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError


def call(identifier="c1", name="calculator", arguments='{"expression":"(12+3)*4"}'):
    return ModelToolCall(id=identifier, name=name, arguments=arguments)


def tools(*calls):
    return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=calls,
                         usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))


def answer(text="done", reason="stop"):
    return ModelResponse(content=text, model="fake", finish_reason=reason,
                         usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))


class ScriptProvider(ModelProvider):
    """本模块自包含的离线 Provider，不依赖其他测试模块的导入路径。"""

    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return next(self.replies)

    async def aclose(self):
        pass


def settings(**pruning):
    return Config(agent=AgentConfig(
        model=ModelConfig(name="fake", max_tokens=512, context_window_tokens=8192),
        context={"reserve_tokens": 1024, "keep_recent_tokens": 1000, "summary_max_tokens": 256,
                 "safety_tokens": 128, "tool_result_pruning": {
                     "pressure_ratio": 0.5, "threshold_chars": 2048, "head_chars": 512,
                     "tail_chars": 128, **pruning}},
    ), providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))


class FetchProbe(Tool):
    name = "web_fetch"
    description = "离线网页结果测试工具"
    read_only = True

    def __init__(self, *, text="done"):
        self.text = text

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **arguments):
        return self.text


def fetch(text):
    return json.dumps({"url": "https://example.com/", "title": "source", "content": text,
                       "truncated": True, "published_at": None}, ensure_ascii=False)


def archived(content=None, name="web_fetch", **fields):
    return ToolTrace(turn_id=uuid4(), call=call(name=name, arguments="{}"),
                     content=content or fetch("A" * 12000), result_id=uuid4(), original_complete=True, **fields)


def request_for(trace, *, content=None):
    return ModelRequest(messages=(ChatMessage(role="system", content="system"),
                        ChatMessage(role="tool", tool_call_id=trace.call.id,
                                    result_id=trace.result_id, content=trace.content if content is None else content)))


@pytest.mark.parametrize("name", ["web_fetch", "read_file"])
def test_body_pruning_preserves_metadata_and_is_idempotent(name):
    original = json.dumps({"path": "file.py", "url": "https://example.com/", "truncated": True,
                           "total_lines": 400, "next_offset": 201,
                           "content": "\n".join(f"{i}: " + "X" * 80 for i in range(200))})
    policy = ToolResultPruningConfig()
    identifier = uuid4()
    result = prune_content(name, original, identifier, policy)
    parsed = json.loads(result)
    assert parsed["truncated"] is True and parsed["next_offset"] == 201
    assert parsed["path"] == "file.py" and parsed["url"] == "https://example.com/"
    assert parsed["recovery"]["result_id"] == str(identifier)
    assert "已回收" in parsed["content"]
    assert len(result) < policy.threshold_chars
    assert prune_content(name, result, identifier, policy) is None


@pytest.mark.parametrize("name", ["web_search", "search_files", "find_files", "list_dir"])
def test_lists_keep_legal_records_and_report_omissions(name):
    original = json.dumps({"query": "q", "truncated": True, "results": [
        {"title": f"title-{i}", "url": f"https://example.com/{i}", "published_at": None,
         "path": f"file-{i}.py", "line": i + 1, "snippet": "A" * 3000,
         "text": "B" * 500} for i in range(20)]})
    result = json.loads(prune_content(name, original, uuid4(), ToolResultPruningConfig()))
    assert result["query"] == "q" and result["truncated"] is True
    assert result["returned_count_before_recycling"] == 20
    assert 0 < len(result["results"]) < 20
    assert result["results"][0]["url"] == "https://example.com/0"
    assert result["results"][0]["line"] == 1


@pytest.mark.parametrize("name", ["calculator", "edit_file", "write_file", "read_tool_result", "unknown"])
def test_non_candidate_tools_untouched(name):
    assert prune_content(name, fetch("A" * 16000), uuid4(), ToolResultPruningConfig()) is None


def test_malformed_or_small_or_protected_or_error_results_untouched():
    policy = ToolResultPruningConfig()
    for original in ("bad" * 10000, "[]", "{}", json.dumps({"content": "A" * 10000, "pending_approval": "id"})):
        assert prune_content("web_fetch", original, uuid4(), policy) is None
    trace = archived()
    archive = ToolResultArchive()
    archive.add(trace)
    request = request_for(trace)
    assert not plan_pruning(request, settings().agent, archive, (trace.result_id,), force=True).replacements
    bad = trace.model_copy(update={"is_error": True})
    archive.records[trace.result_id] = bad
    assert not plan_pruning(request, settings().agent, archive, force=True).replacements


def test_pressure_then_remeasure_and_single_convergence():
    trace = archived()
    archive = ToolResultArchive()
    archive.add(trace)
    request = request_for(trace)
    assert not plan_pruning(request, settings().agent, archive).replacements  # 低于压力线
    result = plan_pruning(request, settings().agent, archive, force=True)
    assert len(result.replacements) == 1
    assert result.after_tokens < result.before_tokens
    request = request_for(trace, content=result.replacements[0].content)
    assert not plan_pruning(request, settings().agent, archive, force=True).replacements
    assert not plan_pruning(request_for(trace), settings(enabled=False).agent, archive, force=True).replacements


def test_initial_limit_keeps_json_and_archive_reference():
    identifier = uuid4()
    result = bound_structured_result(fetch("资料" * 20000), identifier, 1200, 600)
    parsed = json.loads(result)
    assert len(result) <= 1200
    assert parsed["initial_truncation"]["result_id"] == str(identifier)
    assert parsed["truncated"] is True


@pytest.mark.asyncio
async def test_executor_archives_before_clipping_and_stops_on_save_failure():
    traces = []
    probe = FetchProbe(text=fetch("A" * 16000))
    executor = ToolExecutor(ToolRegistry([probe]), settings().agent.tools, traces.append)
    result = await executor.execute(call(name="web_fetch", arguments="{}"), uuid4(), 1000)
    assert traces[0].content == probe.text and traces[0].original_complete
    assert len(result.content) < len(traces[0].content)
    assert result.result_id == traces[0].result_id
    json.loads(result.content)

    def fail(trace):
        raise SessionError("disk failure")
    executor.on_trace = fail
    with pytest.raises(SessionError):
        await executor.execute(call(name="web_fetch", arguments="{}"), uuid4(), 1000)


@pytest.mark.asyncio
async def test_chain_protects_latest_batch_prunes_old_and_restores_raw(tmp_path):
    probe = FetchProbe(text=fetch("A" * 12000))
    provider = ScriptProvider([tools(call("first", "web_fetch", "{}")),
                               tools(call("second", "web_fetch", "{}")), answer("done")])
    registry = ToolRegistry([probe])
    store = SessionStore(tmp_path)
    runner = AgentRunner(settings(), provider, store, tools=registry)
    phases = []
    await runner.ask("test", on_status=phases.append)
    assert len(provider.requests) == 3
    assert "recovery" not in json.loads(provider.requests[1].messages[-1].content)
    first, second = [message for message in provider.requests[2].messages if message.role == "tool"]
    assert "recovery" in json.loads(first.content)
    assert "recovery" not in json.loads(second.content)
    assert "pruning" in phases and "pruned" in phases
    assert all(estimate_request_tokens(request) <= settings().agent.input_budget for request in provider.requests)
    assert runner.last_usage.report("summary").calls == 0
    saved = store.load(runner.session_id)
    assert any(event.kind == "prune" for event in saved.events)
    original = next(message for message in saved.history() if message.role == "tool")
    assert "recovery" not in json.loads(original.content)
    assert json.loads(runner._archive.read(first.result_id))["content"] == probe.text[:4000]
    restored = AgentRunner(settings(), ScriptProvider([]), store, saved, tools=registry)
    assert restored.context.history == runner.context.history
    assert json.loads(restored._archive.read(first.result_id))["total_chars"] == len(probe.text)
    restored.switch_session()
    with pytest.raises(ToolExecutionError):
        restored._archive.read(first.result_id)
    await runner.aclose()
    await restored.aclose()


@pytest.mark.asyncio
async def test_prune_failure_does_not_apply_replacements(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    probe = FetchProbe(text=fetch("A" * 12000))
    provider = ScriptProvider([tools(call("first", "web_fetch", "{}")), tools(call("second", "web_fetch", "{}"))])
    runner = AgentRunner(settings(), provider, store, tools=ToolRegistry([probe]))
    save = store.save
    def fail(session):
        if session.events and session.events[-1].kind == "prune":
            raise SessionError("disk failure")
        save(session)
    monkeypatch.setattr(store, "save", fail)
    with pytest.raises(SessionError):
        await runner.ask("test")
    assert runner.context.replacements == {}
    assert runner.context.history == ()
    assert len(provider.requests) == 2
    await runner.aclose()


@pytest.mark.asyncio
async def test_file_archive_permissions_and_sensitive_grants_rechecked(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    inside = root / "inside.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    secret = root / ".env"
    secret.write_text("test-placeholder", encoding="utf-8")
    files = FileAccess(root)
    archive = ToolResultArchive(files)
    executor = ToolExecutor(ToolRegistry(create_file_tools(files)), settings().agent.tools, archive.add)
    files.set_mode("full")
    result = await executor.execute(call(name="read_file", arguments=json.dumps({"path": str(outside)})), uuid4(), 4000)
    assert "outside" in archive.read(result.result_id)
    files.set_mode("read")
    with pytest.raises(ToolExecutionError, match="工作区"):
        archive.read(result.result_id)
    result = await executor.execute(call(name="read_file", arguments=json.dumps({"path": str(inside)})), uuid4(), 4000)
    assert "inside" in archive.read(result.result_id)
    files.policy.sensitive_grants.add(str(secret.resolve()))
    result = await executor.execute(call(name="read_file", arguments=json.dumps({"path": str(secret)})), uuid4(), 4000)
    assert "test-placeholder" in archive.read(result.result_id)
    files.set_mode("read")
    with pytest.raises(ToolExecutionError, match="敏感"):
        archive.read(result.result_id)


@pytest.mark.asyncio
async def test_read_archive_schema_and_paging():
    archive = ToolResultArchive()
    trace = archived()
    archive.add(trace)
    tool = ReadToolResultTool(archive)
    with pytest.raises(ToolValidationError):
        await tool.execute(result_id=str(trace.result_id), path="file")
    with pytest.raises(ToolValidationError):
        await tool.execute(result_id=str(trace.result_id), offset=True)
    page = json.loads(await tool.execute(result_id=str(trace.result_id), offset=4000, limit=10))
    assert page["content"] == trace.content[4000:4010]
    assert page["next_offset"] == 4010
    with pytest.raises(ToolExecutionError):
        await tool.execute(result_id=str(uuid4()))


@pytest.mark.parametrize("code,status,overflow", [("context_length_exceeded", 400, True),
    ("context_window_exceeded", 413, True), ("invalid_request_error", 400, False),
    ("context_length_exceeded", 500, False)])
def test_only_confirmed_overflow_has_recovery_type(code, status, overflow):
    error = BadRequestError("private", response=httpx2.Response(status, request=httpx2.Request("POST", "https://example.com")),
                            body={"error": {"code": code, "message": "private"}})
    mapped = _status_error(error)
    assert isinstance(mapped, ProviderContextOverflowError) is overflow
    assert "private" not in str(mapped)


@pytest.mark.asyncio
async def test_overflow_does_not_retry_unchanged_request(tmp_path):
    class Overflow(ScriptProvider):
        async def generate(self, request):
            self.requests.append(request)
            raise ProviderContextOverflowError("overflow")
    provider = Overflow([])
    runner = AgentRunner(settings(), provider, SessionStore(tmp_path))
    with pytest.raises(ContextError, match="未原样重试"):
        await runner.ask("test")
    assert len(provider.requests) == 1
    await runner.aclose()


def test_local_result_id_not_sent_to_provider():
    trace = archived()
    wire = DeepSeekProvider._messages(request_for(trace))
    assert "result_id" not in wire[-1]
    assert wire[-1]["tool_call_id"] == trace.call.id


def test_prune_config_validation_and_environment(tmp_path):
    with pytest.raises(ValidationError):
        ToolResultPruningConfig(head_chars=8192)
    env = tmp_path / "config.env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\nTOOL_PRUNING_ENABLED=false\n"
                   "TOOL_PRUNING_PRESSURE_RATIO=0.8", encoding="utf-8")
    policy = load_config(env).agent.context.tool_result_pruning
    assert not policy.enabled and policy.pressure_ratio == 0.8


def test_replacement_requires_valid_hash_and_prune_event_requires_reduction():
    with pytest.raises(ValidationError):
        ToolResultReplacement(result_id=uuid4(), source_hash="invalid", content="short")
    replacement = ToolResultReplacement(result_id=uuid4(), source_hash=content_hash("original"), content="short")
    with pytest.raises(ValidationError):
        SessionEvent(kind="prune", replacements=(replacement,))


def saved_tool_turn(store, content):
    trace = archived(content)
    messages = (
        ChatMessage(role="user", content="old question"),
        ChatMessage(role="assistant", tool_calls=(trace.call,)),
        ChatMessage(role="tool", content=trace.content, tool_call_id=trace.call.id, result_id=trace.result_id),
        ChatMessage(role="assistant", content="old answer"),
    )
    session = Session().with_event(SessionEvent(kind="tool", tool=trace))
    session = session.with_event(SessionEvent(kind="turn", messages=messages))
    store.save(session)
    return session, trace


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_again", [False, True])
async def test_confirmed_overflow_retries_once_only_after_change(tmp_path, fail_again):
    store = SessionStore(tmp_path)
    session, trace = saved_tool_turn(store, fetch("A" * 10000))
    class OverflowOnce(ScriptProvider):
        async def generate(self, request):
            self.requests.append(request)
            if len(self.requests) == 1 or fail_again:
                raise ProviderContextOverflowError("overflow")
            return answer("answer based on available source")
    provider = OverflowOnce([])
    runner = AgentRunner(settings(), provider, store, session, tools=ToolRegistry([FetchProbe()]))
    if fail_again:
        with pytest.raises(ProviderContextOverflowError):
            await runner.ask("test")
    else:
        await runner.ask("test")
    assert len(provider.requests) == 2
    assert provider.requests[0].messages != provider.requests[1].messages
    assert estimate_request_tokens(provider.requests[1]) < estimate_request_tokens(provider.requests[0])
    assert runner.last_usage.report().calls == 2
    saved = store.load(runner.session_id)
    assert any(event.kind == "prune" for event in saved.events)
    assert runner._archive.records[trace.result_id].content == trace.content
    await runner.aclose()


@pytest.mark.asyncio
async def test_network_failure_is_not_overflow_retry(tmp_path):
    class Connection(ScriptProvider):
        async def generate(self, request):
            self.requests.append(request)
            raise ProviderConnectionError("connection")
    provider = Connection([])
    runner = AgentRunner(settings(), provider, SessionStore(tmp_path))
    with pytest.raises(ProviderConnectionError):
        await runner.ask("test")
    assert len(provider.requests) == 1
    await runner.aclose()


@pytest.mark.asyncio
async def test_cancellation_after_durable_pruning_keeps_checkpoint(tmp_path):
    store = SessionStore(tmp_path)
    session, trace = saved_tool_turn(store, fetch("A" * 10000))
    class Waiting(ScriptProvider):
        def __init__(self):
            super().__init__([])
            self.started = asyncio.Event()
            self.waiting = asyncio.Event()
        async def generate(self, request):
            self.requests.append(request)
            self.started.set()
            await self.waiting.wait()
    provider = Waiting()
    runner = AgentRunner(settings(), provider, store, session, tools=ToolRegistry([FetchProbe()]))
    task = asyncio.create_task(runner.ask("test " + "X" * 6000))
    await asyncio.wait_for(provider.started.wait(), 2)
    assert runner.last_pruning is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    saved = store.load(runner.session_id)
    assert len([event for event in saved.events if event.kind == "turn"]) == 1
    restored = AgentRunner(settings(), ScriptProvider([]), store, saved, tools=ToolRegistry([FetchProbe()]))
    assert "recovery" in json.loads(next(message for message in restored.context.history if message.role == "tool").content)
    assert restored._archive.records[trace.result_id].content == trace.content
    await runner.aclose()
    await restored.aclose()


@pytest.mark.asyncio
async def test_clear_hides_old_archives_after_restart(tmp_path):
    store = SessionStore(tmp_path)
    session, trace = saved_tool_turn(store, fetch("A" * 10000))
    runner = AgentRunner(settings(), ScriptProvider([]), store, session, tools=ToolRegistry([FetchProbe()]))
    runner.clear_history()
    with pytest.raises(ToolExecutionError):
        runner._archive.read(trace.result_id)
    saved = store.load(runner.session_id)
    restored = AgentRunner(settings(), ScriptProvider([]), store, saved, tools=ToolRegistry([FetchProbe()]))
    with pytest.raises(ToolExecutionError):
        restored._archive.read(trace.result_id)
    assert saved.events[0].tool.content == trace.content  # 原始审计未删除
    await runner.aclose()
    await restored.aclose()


def test_corrupt_replacement_target_rejected_on_load(tmp_path):
    store = SessionStore(tmp_path)
    session, trace = saved_tool_turn(store, fetch("A" * 10000))
    replacement = ToolResultReplacement(result_id=trace.result_id, source_hash=content_hash("wrong source"), content="short")
    session = session.with_event(SessionEvent(kind="prune", replacements=(replacement,), prune_before_tokens=100, prune_after_tokens=20))
    store.save(session)
    with pytest.raises(SessionError, match="损坏"):
        store.load(session.id)


@pytest.mark.asyncio
async def test_tui_shows_pruning_notice_but_usage_only_on_tokens(tmp_path):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from lynx_agent.cli.app import LynxTUI
    from lynx_agent.cli.layout import transcript_text

    store = SessionStore(tmp_path)
    session, _ = saved_tool_turn(store, fetch("A" * 10000))
    runner = AgentRunner(settings(), ScriptProvider([answer("done")]), store, session, tools=ToolRegistry([FetchProbe()]))
    with create_pipe_input() as pipe:
        ui = LynxTUI(runner, settings(), store, input_stream=pipe, output_stream=DummyOutput())
        ui._status("pruning")
        assert ui.state.status == "正在回收工具结果…"
        await ui._ask("test " + "X" * 6000)
        text = transcript_text(ui.state)
        assert "已回收 1 条工具结果" in text
        assert "API 实际用量" not in text
        await ui._dispatch("/tokens")
        text = transcript_text(ui.state)
        assert "最近一次回收：1 条工具结果" in text
        assert "非 API 用量" in text and "API 实际用量" in text
    await runner.aclose()


@pytest.mark.asyncio
async def test_full_file_access_cannot_bypass_archive_scope_via_jsonl(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    store = SessionStore(tmp_path / "sessions")
    session, _ = saved_tool_turn(store, fetch("private-history"))
    files = FileAccess(root)
    files.set_mode("full")
    runner = AgentRunner(settings(), ScriptProvider([]), store, session,
                         tools=ToolRegistry(create_file_tools(files)))
    result = await runner.tools.execute(ToolCall(
        id="bypass", name="read_file", arguments={"path": str(store.directory / f"{session.id}.jsonl")},
    ))
    assert result.is_error and "存档目录" in result.content
    assert "private-history" not in result.content
    result = await runner.tools.execute(ToolCall(
        id="bypass-scan", name="search_files", arguments={"path": str(tmp_path), "query": "private-history"},
    ))
    assert not result.is_error and json.loads(result.content)["results"] == []
    await runner.aclose()


@pytest.mark.asyncio
async def test_oversize_original_marked_incomplete_not_recoverable():
    archive = ToolResultArchive()
    probe = FetchProbe(text=fetch("A" * 1100000))
    executor = ToolExecutor(ToolRegistry([probe]), settings().agent.tools, archive.add)
    result = await executor.execute(call(name="web_fetch", arguments="{}"), uuid4(), 2000)
    trace = archive.records[result.result_id]
    assert len(trace.content.encode("utf-8")) <= 1048576
    assert not trace.original_complete
    with pytest.raises(ToolExecutionError, match="完整存档"):
        archive.read(result.result_id)


@pytest.mark.asyncio
async def test_summary_failure_does_not_undo_durable_pruning(tmp_path):
    store = SessionStore(tmp_path)
    session, trace = saved_tool_turn(store, fetch("A" * 10000))
    session = session.with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="资料" * 2500), ChatMessage(role="assistant", content="answer"))))
    store.save(session)
    provider = ScriptProvider([answer("", "length")])
    runner = AgentRunner(settings(), provider, store, session, tools=ToolRegistry([FetchProbe()]))
    with pytest.raises(ContextError, match="摘要"):
        await runner.ask("test")
    saved = store.load(runner.session_id)
    assert any(event.kind == "prune" for event in saved.events)
    assert not any(event.kind == "compact" for event in saved.events)
    assert runner._archive.records[trace.result_id].content == trace.content
    await runner.aclose()
