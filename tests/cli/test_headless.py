"""headless 单次运行：事件流、退出码、stdin 读取与参数校验。"""

import io
import json
import sys

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.cli import commands, headless
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import (
    ModelProvider,
    ModelResponse,
    ModelToolCall,
)
from lynx_agent.providers.errors import ProviderConnectionError
from lynx_agent.session import SessionStore
from lynx_agent.tools import Tool, ToolRegistry


class Probe(Tool):
    name = "probe"
    description = "offline probe"
    read_only = True

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **arguments):
        return "probe-result"


class ScriptedProvider(ModelProvider):
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.closed = False

    async def generate(self, request):
        self.requests.append(request)
        value = self.replies.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value

    async def aclose(self):
        self.closed = True


def make_config(*, rewrite=True):
    return Config(
        agent=AgentConfig(
            model=ModelConfig(name="fake"),
            reply_rewrite={"enabled": rewrite},
        ),
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")
        ),
    )


def tool_call(call_id="call-1", name="probe", arguments="{}"):
    return ModelResponse(
        content="", model="fake", finish_reason="tool_calls",
        tool_calls=(ModelToolCall(id=call_id, name=name, arguments=arguments),),
    )


def final(text="完成", reason="stop"):
    return ModelResponse(content=text, model="fake", finish_reason=reason)


def install(monkeypatch, tmp_path, config, provider, *, tools=None):
    monkeypatch.setattr(headless, "load_config", lambda: config)
    monkeypatch.setattr(headless, "SessionStore", lambda: SessionStore(tmp_path / "sessions"))
    registry = tools if tools is not None else [Probe()]

    def fake_from_config(config, session_id=None, store=None, *, tools=None, on_event=None):
        session = store.load(session_id) if session_id is not None else None
        return AgentRunner(
            config, provider, store, session, tools=ToolRegistry(registry), on_event=on_event,
        )

    monkeypatch.setattr(headless.AgentRunner, "from_config", staticmethod(fake_from_config))


def events_of(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_json_mode_emits_required_events_and_exit_zero(monkeypatch, tmp_path):
    provider = ScriptedProvider([tool_call(), final()])
    install(monkeypatch, tmp_path, make_config(), provider)
    out, err = io.StringIO(), io.StringIO()
    code = await headless.run_headless("执行任务", json_mode=True, stdout=out, stderr=err)
    assert code == headless.EXIT_OK
    events = events_of(out)
    kinds = [event["type"] for event in events]
    assert kinds.count("session_id") == 1
    assert kinds[-1] == "done"
    for required in ("text", "tool_call", "usage", "session_id"):
        assert required in kinds, required
    assert events[-1]["exit_code"] == 0
    session_event = next(event for event in events if event["type"] == "session_id")
    assert session_event["session_id"] and session_event["session_id"] == events[-1]["session_id"]
    tool_events = [event for event in events if event["type"] == "tool_call"]
    assert tool_events[0]["name"] == "probe" and tool_events[0]["is_error"] is False
    assert [event["text"] for event in events if event["type"] == "text"] == ["完成"]
    assert err.getvalue() == ""
    assert provider.closed is True


@pytest.mark.asyncio
async def test_plain_mode_prints_answer_and_progress_separately(monkeypatch, tmp_path):
    provider = ScriptedProvider([tool_call(), final("最终回答")])
    install(monkeypatch, tmp_path, make_config(), provider)
    out, err = io.StringIO(), io.StringIO()
    code = await headless.run_headless("任务", json_mode=False, stdout=out, stderr=err)
    assert code == headless.EXIT_OK
    assert out.getvalue() == "最终回答\n"
    assert "工具 probe" in err.getvalue()


@pytest.mark.asyncio
async def test_resume_reuses_session_history(monkeypatch, tmp_path):
    config = make_config()
    store = SessionStore(tmp_path / "sessions")
    session = store.create()
    provider = ScriptedProvider([final("第二次")])
    install(monkeypatch, tmp_path, config, provider)
    out = io.StringIO()
    code = await headless.run_headless(
        "继续", session.id, json_mode=True, stdout=out, stderr=io.StringIO(),
    )
    assert code == 0
    events = events_of(out)
    session_event = next(event for event in events if event["type"] == "session_id")
    assert session_event["resumed"] is True
    assert session_event["session_id"] == str(session.id)
    assert [message.content for message in provider.requests[0].messages][-1] == "继续"


@pytest.mark.asyncio
async def test_failure_returns_one_and_still_emits_done(monkeypatch, tmp_path):
    provider = ScriptedProvider([ProviderConnectionError("断网")])
    install(monkeypatch, tmp_path, make_config(), provider)
    out = io.StringIO()
    code = await headless.run_headless("任务", json_mode=True, stdout=out, stderr=io.StringIO())
    assert code == headless.EXIT_FAILED
    events = events_of(out)
    assert events[-1]["type"] == "done" and events[-1]["exit_code"] == 1
    assert any(event["type"] == "error" for event in events)


@pytest.mark.asyncio
async def test_truncated_answer_is_not_success(monkeypatch, tmp_path):
    provider = ScriptedProvider([final("半截", "length")])
    install(monkeypatch, tmp_path, make_config(rewrite=False), provider)
    out = io.StringIO()
    code = await headless.run_headless("任务", json_mode=True, stdout=out, stderr=io.StringIO())
    assert code == headless.EXIT_FAILED
    events = events_of(out)
    assert any(event["type"] == "error" and event["code"] == "incomplete" for event in events)


@pytest.mark.asyncio
async def test_context_error_is_reported(monkeypatch, tmp_path):
    provider = ScriptedProvider([ContextError("预算不足")])
    install(monkeypatch, tmp_path, make_config(), provider)
    out = io.StringIO()
    code = await headless.run_headless("任务", json_mode=True, stdout=out, stderr=io.StringIO())
    assert code == headless.EXIT_FAILED
    assert "预算不足" in out.getvalue()


@pytest.mark.asyncio
async def test_empty_task_never_calls_model(monkeypatch, tmp_path):
    provider = ScriptedProvider([])
    install(monkeypatch, tmp_path, make_config(), provider)
    out = io.StringIO()
    code = await headless.run_headless("   ", json_mode=True, stdout=out, stderr=io.StringIO())
    assert code == headless.EXIT_FAILED
    assert not provider.requests
    assert events_of(out)[0]["code"] == "empty_task"


def test_read_task_prefers_argument_then_stdin():
    assert headless.read_task("任务", stdin=io.StringIO("忽略")) == "任务"
    assert headless.read_task(None, stdin=io.StringIO("来自管道\n")) == "来自管道\n"
    assert headless.read_task("   ", stdin=io.StringIO("兜底")) == "兜底"

    class Tty(io.StringIO):
        def isatty(self):
            return True

    assert headless.read_task(None, stdin=Tty("不该读取")) == ""


def test_json_requires_headless(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lynx-agent", "--json"])
    with pytest.raises(SystemExit) as error:
        commands.main()
    assert error.value.code == 2


def test_headless_argument_forms(monkeypatch):
    calls = []

    def fake_run_cli(task, session_id, *, json_mode):
        calls.append((task, session_id, json_mode))
        return 0

    monkeypatch.setattr(headless, "run_cli", fake_run_cli)
    monkeypatch.setattr(sys, "argv", ["lynx-agent", "--headless", "--json", "任务文本"])
    with pytest.raises(SystemExit) as error:
        commands.main()
    assert error.value.code == 0
    assert calls == [("任务文本", None, True)]

    monkeypatch.setattr(sys, "argv", ["lynx-agent", "--headless", "--session", "abc", "任务"])
    with pytest.raises(SystemExit):
        commands.main()
    assert calls[-1] == ("任务", "abc", False)

    monkeypatch.setattr(sys, "argv", ["lynx-agent", "--headless"])
    with pytest.raises(SystemExit):
        commands.main()
    assert calls[-1] == (None, None, False)
