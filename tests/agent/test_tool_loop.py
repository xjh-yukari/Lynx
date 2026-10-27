"""离线验证调用、执行、回填、预算和持久化工具链。"""

import asyncio
import json

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.agent.loop import AgentLoop
from lynx_agent.agent.tokens import estimate_request_tokens
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelToolCall,
    ProviderResponseError,
    TokenUsage,
    ToolCallDelta,
)
from lynx_agent.session import SessionError, SessionEvent, SessionStore
from lynx_agent.tools import Tool, ToolRegistry


def config(**agent):
    return Config(agent=AgentConfig(model=ModelConfig(name="fake"), **agent),
                  providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))


def call(identifier="c1", name="calculator", arguments='{"expression":"(12+3)*4"}'):
    return ModelToolCall(id=identifier, name=name, arguments=arguments)


def tools(*calls, text="", reason="tool_calls"):
    return ModelResponse(content=text, model="fake", finish_reason=reason, tool_calls=calls,
                         usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))


def answer(text="结果是 60", reason="stop"):
    return ModelResponse(content=text, model="fake", finish_reason=reason,
                         usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))


class ScriptProvider(ModelProvider):
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return next(self.replies)

    async def aclose(self):
        pass


class ProbeTool(Tool):
    def __init__(self, *, writable=False, slow=False, text="done", error=False):
        self.writable = writable
        self.slow = slow
        self.text = text
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.cleaned = False

    name = "probe"
    description = "测试工具"
    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    @property
    def read_only(self):
        return not self.writable

    async def execute(self, **arguments):
        self.calls.append(arguments)
        self.started.set()
        try:
            if self.slow:
                await asyncio.Event().wait()
            if self.error:
                raise RuntimeError("private-api-key-in-exception")
            return self.text
        finally:
            self.cleaned = True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_calculator_roundtrip_and_restore_full_chain(tmp_path, stream):
    store = SessionStore(tmp_path)
    provider = ScriptProvider([tools(call()), answer()])
    events = []
    async with AgentRunner(config(), provider, store) as runner:
        reply = await runner.ask("计算 (12+3)*4", on_stream=events.append if stream else None)
        assert reply.content == "结果是 60"
        assert len(provider.requests) == 2
        first, second = provider.requests
        assert first.tools[0].name == "calculator"
        assert first.tool_choice == "auto"
        assert first.parallel_tool_calls is False
        assert second.messages[-1].role == "tool"
        assert second.messages[-1].tool_call_id == "c1"
        assert second.messages[-1].content == "60"
        assert second.messages[-2].tool_calls == (call(),)
        assert [item.role for item in runner.context.history] == ["user", "assistant", "tool", "assistant"]
        assert runner.last_usage.report("chat").calls == 2
        saved = store.load(runner.session_id)
        assert [event.kind for event in saved.events if event.kind != "diagnostic"] == ["tool", "turn"]
        assert len(saved.events[-1].messages) == 4
        assert saved.context_history() == runner.context.history
    async with AgentRunner(config(), ScriptProvider([answer("下一轮")]), store, saved) as restored:
        await restored.ask("继续")
        assert restored.context.history[:4] == saved.context_history()
    if stream:
        assert [event.kind for event in events] == ["start", "start", "delta"]


@pytest.mark.asyncio
async def test_multiple_calls_and_repair_invalid_arguments(tmp_path):
    provider = ScriptProvider([
        tools(call("a", arguments="{bad"), call("b", name="missing", arguments="{}")),
        tools(call("c")), answer(),
    ])
    async with AgentRunner(config(), provider, SessionStore(tmp_path)) as runner:
        await runner.ask("测试")
        returned = provider.requests[1].messages[-2:]
        assert [item.tool_call_id for item in returned] == ["a", "b"]
        assert "JSON" in returned[0].content
        assert "找不到工具" in returned[1].content
        assert runner.context.history[-2].content == "60"
        assert len(runner.context.history) == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ['[]', '{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}', 'null'])
async def test_bad_json_never_executes_tool(tmp_path, raw):
    probe = ProbeTool()
    provider = ScriptProvider([tools(call(name="probe", arguments=raw)), answer("参数错误")])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试")
        assert not probe.calls
        assert provider.requests[1].messages[-1].role == "tool"


@pytest.mark.asyncio
async def test_tool_type_validation_returns_error_to_model(tmp_path):
    provider = ScriptProvider([tools(call(arguments='{"expression":3}')), answer("参数有误")])
    async with AgentRunner(config(), provider, SessionStore(tmp_path)) as runner:
        await runner.ask("测试")
        assert "string" in provider.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_write_tools_are_denied_by_default(tmp_path):
    probe = ProbeTool(writable=True)
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), answer("不可写")])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试")
        assert not probe.calls
        assert "只读" in provider.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_timeout_returns_error_and_cleans_tool(tmp_path):
    probe = ProbeTool(slow=True)
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), answer("工具超时")])
    async with AgentRunner(config(tools={"timeout_seconds": 0.01}), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试")
        assert "超时" in provider.requests[1].messages[-1].content
        assert probe.cleaned


@pytest.mark.asyncio
async def test_cancel_during_tool_preserves_audit_without_committing_turn(tmp_path):
    probe = ProbeTool(slow=True)
    store = SessionStore(tmp_path)
    provider = ScriptProvider([tools(call(name="probe", arguments="{}"))])
    async with AgentRunner(config(), provider, store, tools=ToolRegistry([probe])) as runner:
        task = asyncio.create_task(runner.ask("测试"))
        await asyncio.wait_for(probe.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert probe.cleaned
        assert runner.context.history == ()
        saved = store.load(runner.session_id)
        assert saved.events[0].kind == "tool"
        assert saved.events[0].tool.is_error
        assert saved.context_history() == ()


@pytest.mark.asyncio
async def test_max_rounds_forces_final_answer_without_tools(tmp_path):
    provider = ScriptProvider([tools(call()), answer()])
    async with AgentRunner(config(max_iterations=2), provider, SessionStore(tmp_path)) as runner:
        await runner.ask("测试")
        assert [request.tool_choice for request in provider.requests] == ["auto", None]
        assert "只根据用户问题与已有工具结果总结" in provider.requests[-1].messages[0].content
        assert provider.requests[-1].tools == ()
        assert [message.role for message in provider.requests[-1].messages] == ["system", "user"]
        assert provider.requests[-1].parallel_tool_calls is None
        assert AgentLoop._final_answer_request(provider.requests[-1]) == provider.requests[-1]
        assert "60" in provider.requests[-1].messages[-1].content
        assert "不具备任何工具" not in runner.context.messages[0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_dsml_recovered_once_without_reexecuting_tools_or_saving_markup(tmp_path, stream):
    probe = ProbeTool()
    broken = '<｜｜DSML｜｜ calls><｜｜DSML｜｜ invoke name="probe">'
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), answer(broken), answer("证据不足，无法确认。")])
    events = []
    phases = []
    store = SessionStore(tmp_path)
    async with AgentRunner(config(max_iterations=2), provider, store, tools=ToolRegistry([probe])) as runner:
        result = await runner.ask("查询", on_status=phases.append, on_stream=events.append if stream else None)
        assert result.content == "证据不足，无法确认。"
        assert len(provider.requests) == 3
        assert [request.tool_choice for request in provider.requests] == ["auto", None, None]
        assert len(probe.calls) == 1
        assert "recovering" in phases
        assert "DSML" not in "".join(event.text for event in events)
        saved = store.load(runner.session_id)
        assert [event.kind for event in saved.events if event.kind != "diagnostic"] == ["tool", "turn"]
        assert all("DSML" not in message.content for message in runner.context.history)
        assert runner.last_usage.report().calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("last", ["markup", "tool"])
async def test_failed_dsml_recovery_is_bounded_and_never_committed(tmp_path, last):
    probe = ProbeTool()
    broken = answer('<|DSML|calls>')
    final = broken if last == "markup" else tools(call("second", name="probe", arguments="{}"))
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), broken, final])
    store = SessionStore(tmp_path)
    async with AgentRunner(config(max_iterations=2), provider, store, tools=ToolRegistry([probe])) as runner:
        with pytest.raises(ProviderResponseError):
            await runner.ask("测试", on_stream=lambda event: None)
        assert len(provider.requests) == 3
        assert len(probe.calls) == 1
        assert runner.context.history == ()
        assert [event.kind for event in store.load(runner.session_id).events if event.kind not in {"diagnostic", "pending"}] == ["tool"]


@pytest.mark.asyncio
async def test_character_split_markup_never_reaches_text_callback(tmp_path):
    class Streaming(ScriptProvider):
        supports_streaming = True

        async def stream(self, request):
            response = await self.generate(request)
            for char in response.content:
                yield ModelStreamEvent(kind="delta", text=char)
            yield ModelStreamEvent(kind="done", response=response)

    provider = Streaming([answer('<｜｜DSML｜｜ calls>'), answer("已停止搜索，现有证据不足。")])
    visible = []
    async with AgentRunner(config(), provider, SessionStore(tmp_path)) as runner:
        await runner.ask("测试", on_text=visible.append)
        assert "".join(visible) == "已停止搜索，现有证据不足。"


@pytest.mark.asyncio
async def test_restore_legacy_markup_neutralizes_projection_without_changing_disk(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create().with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="旧问题"),
        ChatMessage(role="assistant", content="<｜｜DSML｜｜ calls>"),
    )))
    store.save(session)
    async with AgentRunner(config(), ScriptProvider([answer()]), store, session) as runner:
        assert "DSML" not in runner.context.history[-1].content
        assert "不能作为事实依据" in runner.context.history[-1].content
        runner.switch_session(str(session.id))
        assert "DSML" not in runner.context.history[-1].content
        assert "DSML" in store.load(session.id).context_history()[-1].content


@pytest.mark.asyncio
async def test_repeated_call_id_and_excessive_calls_are_not_reexecuted(tmp_path):
    probe = ProbeTool()
    repeated = call(name="probe", arguments="{}")
    provider = ScriptProvider([tools(repeated), tools(repeated)])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        with pytest.raises(ProviderResponseError, match="重复"):
            await runner.ask("测试")
        assert len(probe.calls) == 1
        assert runner.context.history == ()
    provider = ScriptProvider([tools(*[call(str(index), name="probe", arguments="{}") for index in range(9)])])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        with pytest.raises(ProviderResponseError, match="超过"):
            await runner.ask("测试")
        assert len(probe.calls) == 1


@pytest.mark.asyncio
async def test_long_result_truncated_and_internal_exception_not_leaked(tmp_path):
    probe = ProbeTool(text="超长结果" * 10000)
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), answer()])
    async with AgentRunner(config(tools={"max_result_chars": 128}), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试")
        content = provider.requests[1].messages[-1].content
        assert len(content) <= 128 and "已截断" in content
        assert estimate_request_tokens(provider.requests[1]) <= config().agent.input_budget
    probe = ProbeTool(error=True)
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), answer("失败")])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试")
        assert "private-api-key" not in provider.requests[1].messages[-1].content


@pytest.mark.asyncio
async def test_length_after_tool_execution_rewrites_without_reexecuting(tmp_path):
    probe = ProbeTool()
    provider = ScriptProvider([tools(call(name="probe", arguments="{}")), answer("长草稿", "length"), answer("精简回答")])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试")
        assert len(probe.calls) == 1
        assert provider.requests[-1].tool_choice == "none"
        assert any(message.role == "tool" and message.content == "done" for message in provider.requests[-1].messages)
        assert [message.role for message in runner.context.history] == ["user", "assistant", "tool", "assistant"]
        assert runner.context.history[-1].content == "精简回答"


@pytest.mark.asyncio
async def test_length_tool_arguments_never_execute(tmp_path):
    probe = ProbeTool()
    provider = ScriptProvider([tools(call(name="probe", arguments="{partial"), reason="length"), answer("精简回答")])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        await runner.ask("测试", on_stream=lambda event: None)
        assert not probe.calls
        assert len(runner.context.history) == 2


@pytest.mark.asyncio
async def test_tool_budget_preflight_prevents_execution(tmp_path):
    probe = ProbeTool()
    provider = ScriptProvider([tools(call(name="probe", arguments=json.dumps({"huge": "x" * 250000})))])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry([probe])) as runner:
        with pytest.raises(ContextError):
            await runner.ask("测试")
        assert not probe.calls


@pytest.mark.asyncio
async def test_compaction_preserves_whole_tool_turn_and_checkpoint_mapping(tmp_path):
    small = config().model_copy(update={"agent": AgentConfig(
        model=ModelConfig(name="fake", max_tokens=512, context_window_tokens=4096),
        context={"reserve_tokens": 1024, "keep_recent_tokens": 500, "summary_max_tokens": 256, "safety_tokens": 128},
    )})
    store = SessionStore(tmp_path)
    session = store.create()
    old = (
        ChatMessage(role="user", content="项目历史资料" * 800),
        ChatMessage(role="assistant", tool_calls=(call(),)),
        ChatMessage(role="tool", content="60", tool_call_id="c1"),
        ChatMessage(role="assistant", content="计算完成，结果是60"),
    )
    recent = (ChatMessage(role="user", content="下一步"), ChatMessage(role="assistant", content="准备继续"))
    session = session.with_event(SessionEvent(kind="turn", messages=old))
    session = session.with_event(SessionEvent(kind="turn", messages=recent))
    store.save(session)
    provider = ScriptProvider([answer("历史摘要：计算结果60，继续开发项目。") for _ in range(20)])
    async with AgentRunner(small, provider, store, session) as runner:
        result = await runner.compact()
        assert result.summarized_messages == 4
        assert result.kept == recent
        saved = store.load(runner.session_id)
        assert saved.events[-1].covered_until == 1
        assert saved.context_history() == recent
        assert saved.history()[:4] == old
    async with AgentRunner(small, ScriptProvider([]), store, saved) as restored:
        assert restored.context.history == recent
        assert "计算结果60" in restored.context.summary


def test_session_rejects_orphan_and_duplicate_tool_results():
    messages = (ChatMessage(role="user", content="测试"),
                ChatMessage(role="tool", content="错误", tool_call_id="orphan"),
                ChatMessage(role="assistant", content="结束"))
    with pytest.raises(ValueError):
        SessionEvent(kind="turn", messages=messages)
    with pytest.raises(ValueError):
        SessionEvent(kind="turn", messages=(ChatMessage(role="user", content="测试"),
                    ChatMessage(role="assistant", tool_calls=(call(),)),
                    ChatMessage(role="assistant", content="缺少工具结果")))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["complete", "cancel", "mismatch"])
async def test_native_stream_waits_for_verified_complete_arguments_before_execution(tmp_path, mode):
    probe = ProbeTool()
    first_chunk = asyncio.Event()
    release = asyncio.Event()

    class NativeProvider(ScriptProvider):
        supports_streaming = True

        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) > 1:
                yield ModelStreamEvent(kind="delta", text="完成")
                yield ModelStreamEvent(kind="done", response=answer("完成"))
                return
            yield ModelStreamEvent(kind="tool_delta", tool_delta=ToolCallDelta(
                index=0, id="c1", name="probe", arguments="{",
            ))
            first_chunk.set()
            await release.wait()
            yield ModelStreamEvent(kind="tool_delta", tool_delta=ToolCallDelta(index=0, arguments="}"))
            yield ModelStreamEvent(kind="done", response=tools(call(name="probe", arguments='{"x":1}' if mode == "mismatch" else "{}")))

    provider = NativeProvider([])
    store = SessionStore(tmp_path)
    async with AgentRunner(config(), provider, store, tools=ToolRegistry([probe])) as runner:
        task = asyncio.create_task(runner.ask("测试", on_stream=lambda event: None))
        await asyncio.wait_for(first_chunk.wait(), timeout=2)
        assert not probe.calls
        if mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            if mode == "mismatch":
                with pytest.raises(ProviderResponseError, match="不一致"):
                    await task
            else:
                assert (await task).content == "完成"
        assert len(probe.calls) == (1 if mode == "complete" else 0)
        if mode != "complete":
            assert runner.context.history == ()
            assert store.load(runner.session_id).context_history() == ()
            assert all(event.kind in {"diagnostic", "pending"} for event in store.load(runner.session_id).events)


@pytest.mark.asyncio
async def test_empty_registry_does_not_implicitly_enable_calculator(tmp_path):
    provider = ScriptProvider([tools(call())])
    async with AgentRunner(config(), provider, SessionStore(tmp_path), tools=ToolRegistry()) as runner:
        with pytest.raises(ProviderResponseError, match="未允许"):
            await runner.ask("测试")
        assert provider.requests[0].tools == ()


@pytest.mark.asyncio
async def test_tool_audit_save_failure_retains_record_for_manual_save(tmp_path, monkeypatch):
    probe = ProbeTool()
    store = SessionStore(tmp_path)
    provider = ScriptProvider([tools(call(name="probe", arguments="{}"))])
    runner = AgentRunner(config(), provider, store, tools=ToolRegistry([probe]))
    save = store.save

    def fail_after_execute(session):
        if session.events:
            raise SessionError("模拟磁盘错误")
        save(session)

    monkeypatch.setattr(store, "save", fail_after_execute)
    with pytest.raises(SessionError):
        await runner.ask("测试")
    assert len(probe.calls) == 1
    assert len(provider.requests) == 1
    assert runner.persistence_error is not None
    assert runner.context.history == ()
    monkeypatch.setattr(store, "save", save)
    await runner.aclose()
    assert store.load(runner.session_id).events[0].tool.content == "done"
