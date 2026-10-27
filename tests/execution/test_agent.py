"""回合内审批、工具执行器及会话恢复的离线联调。"""

import asyncio
import json
import sys
from uuid import uuid4

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.cli.app import LynxTUI
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionError, SessionStore
from lynx_agent.tools import ToolRegistry
from lynx_agent.tools.builtin.filesystem import create_file_tools
from lynx_agent.tools.errors import ToolExecutionError


def config():
    return Config(agent=AgentConfig(model=ModelConfig(name="fake")),
        providers=ProvidersConfig(deepseek=ProviderConfig(api_key="test-secret-value", base_url="https://example.invalid")))


class Provider(ModelProvider):
    def __init__(self, *replies):
        self.replies = list(replies)

    async def generate(self, request):
        return self.replies.pop(0)

    async def aclose(self):
        pass


def call(script="print('ok')"):
    return ModelToolCall(id=uuid4().hex, name="exec_command", arguments=json.dumps({"argv": [sys.executable, "-c", script]}))


def invocation(item):
    return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(item,))


def answer():
    return ModelResponse(content="完成", model="fake", finish_reason="stop")


def runner(tmp_path, provider=None, store=None):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    return AgentRunner(config(), provider or Provider(), store or SessionStore(tmp_path / "sessions"),
        tools=ToolRegistry(create_file_tools(FileAccess(root))))


def operator(agent):
    approvals = []
    def present(item):
        if item:
            approvals.append(item)
            agent.execution.broker.decide(item.identifier, True)
    agent.execution.broker.presenter = present
    agent.shell_command("on confirm")
    return approvals


@pytest.mark.asyncio
async def test_closed_headless_read_and_independent_approvals(tmp_path):
    agent = runner(tmp_path)
    assert "exec_command" not in agent.tools.names
    with pytest.raises(ValueError, match="审批界面"):
        agent.shell_command("on confirm")
    approvals = operator(agent)
    assert "exec_command" in agent.tools.names
    agent.set_file_mode("read")
    assert "exec_command" not in agent.tools.names
    agent.set_file_mode("workspace")
    agent._session = agent._store.create()
    executor = ToolExecutor(agent.tools, config().agent.tools.model_copy(update={"timeout_seconds": 0.01}))
    first, second = await executor.execute(call(), uuid4(), 4000), await executor.execute(call(), uuid4(), 4000)
    assert not first.is_error and not second.is_error and len(approvals) == 2
    assert not executor._writes
    await agent.aclose()


@pytest.mark.asyncio
async def test_real_roundtrip_redacted_session_and_log_recovery(tmp_path):
    item = call("print('test-secret-value')")
    agent = runner(tmp_path, Provider(invocation(item), answer()))
    operator(agent)
    await agent.ask("run once")
    trace = agent.tool_traces[-1]
    assert "test-secret-value" not in trace.model_dump_json()
    saved = agent._store.load(agent.session_id)
    assert "test-secret-value" not in saved.model_dump_json()
    page = json.loads(agent._archive.read(trace.result_id, 0, 1000))
    assert "[已隐藏]" in page["content"]
    assert page["total_chars"] > 0
    old_session = agent.session_id
    agent.switch_session()
    with pytest.raises(ToolExecutionError):
        agent._archive.read(trace.result_id)
    agent.switch_session(old_session)
    assert "[已隐藏]" in json.loads(agent._archive.read(trace.result_id))["content"]
    agent.clear_history()
    with pytest.raises(ToolExecutionError):
        agent._archive.read(trace.result_id)
    await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["off", "permission", "new", "clear"])
async def test_pending_approval_invalidated(tmp_path, boundary):
    agent = runner(tmp_path, Provider(invocation(call()), answer()))
    operator(agent)
    ready = asyncio.Event()
    agent.execution.broker.presenter = lambda item: ready.set() if item else None
    task = asyncio.create_task(agent.ask("wait for operator"))
    await asyncio.wait_for(ready.wait(), 2)
    if boundary == "off":
        agent.shell_command("off")
    else:
        with pytest.raises(ValueError):
            {"permission": lambda: agent.set_file_mode("read"), "new": agent.switch_session, "clear": agent.clear_history}[boundary]()
    await task
    data = json.loads(agent.tool_traces[-1].content)
    assert data["status"] == "rejected" and agent.tool_traces[-1].execution_state == "not_executed"
    assert not list(agent.execution.logs_dir.rglob("*.log"))
    await agent.aclose()


@pytest.mark.asyncio
async def test_save_after_started_failure_retry_never_reexecutes(tmp_path):
    class FaultStore(SessionStore):
        failed = False
        def save(self, session):
            if session.events and session.events[-1].kind == "tool" and not self.failed:
                self.failed = True
                raise SessionError("injected")
            super().save(session)

    store = FaultStore(tmp_path / "sessions")
    agent = runner(tmp_path, Provider(invocation(call()), answer()), store)
    approvals = operator(agent)
    with pytest.raises(SessionError):
        await agent.ask("run once")
    assert agent.can_retry and agent.persistence_error
    agent.save_session()
    session = store.load(agent.session_id)
    await agent.aclose()
    root = tmp_path / "workspace"
    restored = AgentRunner(config(), Provider(answer()), store, session,
        tools=ToolRegistry(create_file_tools(FileAccess(root))))
    assert (await restored.retry()).finish_reason == "stop"
    assert len(approvals) == 1
    assert len(list(restored.execution.logs_dir.rglob("*.log"))) == 1
    await restored.aclose()


@pytest.mark.asyncio
async def test_tui_keys_approve_in_same_turn(tmp_path):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    agent = runner(tmp_path, Provider(invocation(call()), answer()))
    with create_pipe_input() as pipe:
        ui = LynxTUI(agent, config(), agent._store, input_stream=pipe, output_stream=DummyOutput(),
                     preferences_path=tmp_path / "prefs.json")
        task = asyncio.create_task(ui.run())
        try:
            while not ui.application.is_running:
                await asyncio.sleep(0.01)
            assert agent.execution.enabled and ui.state.shell_enabled
            assert "exec_command" in agent.tools.names
            pipe.send_text("run\r")
            async with asyncio.timeout(5):
                while not ui.state.approval_open:
                    await asyncio.sleep(0.01)
            assert "无沙箱" in ui.approval_editor.text
            pipe.send_text("y")
            async with asyncio.timeout(5):
                while ui.state.busy:
                    await asyncio.sleep(0.01)
            assert agent.context.history[-1].content == "完成"
            assert len(agent.tool_traces) == 1
        finally:
            ui.application.exit()
            await task
            await agent.aclose()
