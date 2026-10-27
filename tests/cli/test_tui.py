"""真实按键驱动统一 TUI，不访问网络。"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.layout import status_text, transcript_text
from lynx_agent.cli.scrolling import ScrollbarControl
from lynx_agent.cli.state import UIState
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
    ModelStreamEvent,
    ModelToolCall,
    TokenUsage,
)
from lynx_agent.session import SessionEvent, SessionStore
from lynx_agent.web.setup import create_default_tools


class RecordingOutput(DummyOutput):
    def __init__(self, columns=90, rows=28):
        self.columns = columns
        self.rows = rows
        self.text = ""

    def get_size(self):
        return Size(rows=self.rows, columns=self.columns)

    def write(self, data):
        self.text += data

    def write_raw(self, data):
        self.write(data)


class TestProvider(ModelProvider):
    __test__ = False
    supports_streaming = True

    def __init__(self):
        self.requests = []
        self.gate = None
        self.stream_closed = False
        self.closed = False

    async def generate(self, request):
        raise AssertionError("测试应走流式接口")

    async def stream(self, request):
        self.requests.append(request)
        try:
            yield ModelStreamEvent(kind="delta", text="第一段")
            if self.gate is not None:
                await self.gate.wait()
            yield ModelStreamEvent(kind="delta", text="\n第二段")
            yield ModelStreamEvent(kind="done", response=ModelResponse(
                content="第一段\n第二段", model="fake", finish_reason="stop",
                usage=TokenUsage(prompt_tokens=20, completion_tokens=5, total_tokens=25),
            ))
        finally:
            self.stream_closed = True

    async def aclose(self):
        self.closed = True


def config():
    return Config(
        agent=AgentConfig(model=ModelConfig(name="fake-model")),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="fake", base_url="https://example.invalid",
        )),
    )


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout=3)


@pytest.mark.asyncio
async def test_header_help_workspace_and_exit_without_creating_session(tmp_path):
    store = SessionStore(tmp_path)
    provider = TestProvider()
    output = RecordingOutput(columns=48, rows=24)
    project = Path("E:/测试项目/" + "长路径" * 12)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, project=project, input_stream=pipe, output_stream=output)
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            assert "❯ ❯" not in output.text
            pipe.send_text("/help\r")
            await wait_until(lambda: "TUI 导航" in transcript_text(ui.state) and not ui.state.busy)
            assert "API 实际用量（最近一次" not in transcript_text(ui.state)
            pipe.send_text("/workspace\r")
            await wait_until(lambda: "当前项目目录" in transcript_text(ui.state) and not ui.state.busy)
            assert str(project.resolve()) in transcript_text(ui.state)
            assert "fake-model" in output.text
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)
    assert store.list_sessions() == []
    assert not provider.requests
    assert provider.closed


@pytest.mark.asyncio
async def test_header_shows_session_title_without_id_suffix(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create().with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="标题测试"), ChatMessage(role="assistant", content="历史回答"),
    )))
    store.save(session)
    output = RecordingOutput(columns=100)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store, session) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=output)
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: "会话：" in output.text)
            assert ui.state.session_title in output.text
            assert str(session.id)[:8] not in output.text
            assert ui.state.session_id == str(session.id)
            assert store.load(session.id).id == session.id
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_stream_cancel_then_continue_and_query_usage(tmp_path):
    store = SessionStore(tmp_path)
    provider = TestProvider()
    provider.gate = asyncio.Event()
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("第一次\r")
            await wait_until(lambda: "第一段" in transcript_text(ui.state))
            assert ui.state.busy
            assert ui.state.status == "正在生成…"
            assert ui.state.session_id is not None
            assert runner.context.history == ()
            pipe.send_text("\x03\x03")
            await wait_until(lambda: not ui.state.busy)
            assert provider.stream_closed
            assert not provider.closed
            assert store.load(runner.session_id).events == ()
            assert "未保存" in transcript_text(ui.state)
            provider.gate = None
            pipe.send_text("第二次\r")
            await wait_until(lambda: len(store.load(runner.session_id).events) == 1 and not ui.state.busy)
            assert runner.context.history[0].content == "第二次"
            assert "API 实际用量（最近一次" not in transcript_text(ui.state)
            pipe.send_text("/tokens\r")
            await wait_until(lambda: "输入 20，输出 5，合计 25" in transcript_text(ui.state))
            pipe.send_text("\x04")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_session_overlay_navigation_rename_and_new(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create().with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="历史问题"), ChatMessage(role="assistant", content="历史回答"),
    )))
    store.save(session)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("/sessions\r")
            await wait_until(lambda: ui.state.picker_open)
            pipe.send_text("\x1b[B\r")
            await wait_until(lambda: ui.state.session_id == str(session.id) and not ui.state.busy)
            assert "历史回答" in transcript_text(ui.state)
            pipe.send_text("/rename 新名称\r")
            await wait_until(lambda: ui.state.session_title == "新名称")
            assert store.load(session.id).title == "新名称"
            pipe.send_text("/new\r")
            await wait_until(lambda: ui.state.session_id is None and not ui.state.busy)
            assert len(store.list_sessions()) == 1
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_paste_and_scroll_do_not_follow_new_text_while_browsing(tmp_path):
    store = SessionStore(tmp_path)
    provider = TestProvider()
    provider.gate = asyncio.Event()
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(rows=24))
            for index in range(80):
                ui.notice(f"历史显示行 {index}")
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            message = "/clear\n" + "多行资料\n" * 30 + "保留正文"
            pipe.send_text("\x1b[200~" + message + "\x1b[201~\r")
            await wait_until(lambda: len(provider.requests) == 1)
            assert provider.requests[0].messages[-1].content == message
            pipe.send_text("\x1b[5~")
            await wait_until(lambda: not ui.state.follow_tail)
            position = ui.state.scroll_line
            provider.gate.set()
            await wait_until(lambda: not ui.state.busy)
            assert not ui.state.follow_tail
            assert ui.state.unread_activity
            assert ui.state.scroll_line == position
            pipe.send_text("\x1b[1;5F")
            await wait_until(lambda: ui.state.follow_tail)
            assert not ui.state.unread_activity
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.parametrize("arguments, handler", [([], "tui_chat"), (["--plain"], "chat")])
def test_main_uses_tui_by_default_and_keeps_plain_option(monkeypatch, arguments, handler):
    callback = AsyncMock()
    monkeypatch.setattr(commands, handler, callback)
    monkeypatch.setattr("sys.argv", ["lynx-agent", *arguments])
    commands.main()
    callback.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_cancel_before_worker_starts_does_not_leave_busy_or_empty_session(tmp_path):
    store = SessionStore(tmp_path)
    provider = TestProvider()
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("立即取消\r\x03")
            await wait_until(lambda: "操作已取消" in transcript_text(ui.state))
            assert not ui.state.busy
            assert store.list_sessions() == []
            assert provider.requests == []
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_real_mouse_wheel_and_scrollbar_drag_with_wrapped_chinese(tmp_path):
    store = SessionStore(tmp_path)
    output = RecordingOutput(columns=50, rows=24)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=output)
            # 单段长文本也必须能够逐屏滚动，不能仅按原始换行数计算。
            ui.notice("长段落测试" * 500)
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.transcript.render_info is not None)
            info = ui.transcript.render_info
            y = info._y_offset + 2
            pipe.send_text(f"\x1b[<64;5;{y}M")
            await wait_until(lambda: not ui.state.follow_tail)
            position = ui.state.scroll_line
            pipe.send_text(f"\x1b[<64;5;{y}M")
            await wait_until(lambda: ui.state.scroll_line < position)
            assert ui.application.layout.current_control is ui.editor.control
            bar = next(window for window in ui.application.layout.find_all_windows()
                       if isinstance(window.content, ScrollbarControl))
            await wait_until(lambda: bar.content.height == ui.transcript.render_info.window_height)
            assert bar.content.thumb_size < bar.content.height
            x = output.columns
            top = info._y_offset + 1
            bottom = top + info.window_height - 1
            # 点击轨道上端，然后按住左键拖动到中间和底部。
            pipe.send_text(f"\x1b[<0;{x};{top}M")
            await wait_until(lambda: ui.state.scroll_line == 0)
            pipe.send_text(f"\x1b[<32;{x};{top + info.window_height // 2}M")
            await wait_until(lambda: ui.state.scroll_line > 0)
            assert not ui.state.follow_tail
            pipe.send_text(f"\x1b[<32;{x};{bottom}M\x1b[<0;{x};{bottom}m")
            await wait_until(lambda: ui.state.follow_tail)
            assert ui.application.layout.current_control is ui.editor.control
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_display_speed_command_and_validation(tmp_path):
    store = SessionStore(tmp_path)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("/speed 60\r")
            await wait_until(lambda: ui.state.stream_speed == 60 and not ui.state.busy)
            pipe.send_text("/speed 10001\r")
            await wait_until(lambda: "速度范围" in transcript_text(ui.state) and not ui.state.busy)
            assert ui.state.stream_speed == 60
            pipe.send_text("/speed 0\r")
            await wait_until(lambda: ui.state.stream_speed == 0 and not ui.state.busy)
            assert not store.list_sessions()
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_narrow_layout_bounded_editor_and_click_to_latest(tmp_path):
    store = SessionStore(tmp_path)
    provider = TestProvider()
    output = RecordingOutput(columns=40, rows=20)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, project=Path("E:/工作区/" + "长路径/" * 30), input_stream=pipe, output_stream=output)
            for index in range(50):
                ui.notice(f"第 {index} 行测试资料")
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.transcript.render_info is not None)
            assert ui.transcript.render_info.window_height >= 5
            pipe.send_text("\x1b[200~" + "多行编辑\n" * 15 + "\x1b[201~")
            await wait_until(lambda: ui.editor.text.count("\n") == 15)
            await asyncio.sleep(0.15)
            assert ui.editor.window.render_info.window_height <= 6
            assert "Window too small" not in output.text
            pipe.send_text("\x1b[5~")
            await wait_until(lambda: not ui.state.follow_tail)
            ui.notice("新的工具状态")
            assert ui.state.unread_activity
            await asyncio.sleep(0.15)
            # 底栏提示行可点击，恢复跟随但不改变输入内容或焦点。
            pipe.send_text(f"\x1b[<0;4;{output.rows}M\x1b[<0;4;{output.rows}m")
            await wait_until(lambda: ui.state.follow_tail)
            assert not ui.state.unread_activity
            assert ui.editor.text.count("\n") == 15
            assert ui.application.layout.current_control is ui.editor.control
            pipe.send_text("\x03/exit\r")
            await asyncio.wait_for(task, timeout=3)
    assert not provider.requests


@pytest.mark.asyncio
async def test_speed_paces_large_fragments_and_cancel_stops_animation(tmp_path):
    class BurstProvider(TestProvider):
        async def stream(self, request):
            try:
                yield ModelStreamEvent(kind="delta", text="**长文本**" * 100)
                await asyncio.Event().wait()
            finally:
                self.stream_closed = True

    store = SessionStore(tmp_path)
    provider = BurstProvider()
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            ui.state.stream_speed = 30
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("测试速度\r")
            await wait_until(lambda: any(message.role == "Lynx" and message.content for message in ui.state.messages))
            assert 0 < len(ui.state.messages[-1].content) < 700
            pipe.send_text("\x03")
            await wait_until(lambda: not ui.state.busy)
            assert provider.stream_closed
            assert store.load(runner.session_id).events == ()
            snapshot = transcript_text(ui.state)
            await asyncio.sleep(0.08)
            assert transcript_text(ui.state) == snapshot
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("always_length", [False, True])
async def test_tui_rewrites_replace_pending_text_and_restore_without_draft_context(tmp_path, always_length):
    class LengthProvider(TestProvider):
        async def stream(self, request):
            self.requests.append(request)
            if len(self.requests) > 1:
                await self.gate.wait()
            text = "截断草稿" if len(self.requests) == 1 else "**最终回答**"
            reason = "length" if always_length or len(self.requests) == 1 else "stop"
            yield ModelStreamEvent(kind="delta", text=text)
            yield ModelStreamEvent(kind="done", response=ModelResponse(content=text, model="fake", finish_reason=reason))

    store = SessionStore(tmp_path)
    provider = LengthProvider()
    provider.gate = asyncio.Event()
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            ui.state.stream_speed = 1
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("测试超限\r")
            await wait_until(lambda: len(provider.requests) == 2)
            assert ui.state.status == "回复达到上限，正在精简重写…"
            assert ui.state.messages[-1].content == ""
            assert runner.context.history == ()
            provider.gate.set()
            await wait_until(lambda: not ui.state.busy)
            assistant = next(message for message in ui.state.messages if message.role.startswith("Lynx"))
            assert assistant.content == "**最终回答**"
            assert "截断草稿" not in assistant.content
            if always_length:
                assert len(provider.requests) == 3
                assert "截断草稿" in assistant.role
                assert runner.context.history == ()
                assert all(event.kind in {"attempt", "diagnostic", "pending"} for event in store.load(runner.session_id).events)
                assert runner.can_retry
            else:
                assert len(provider.requests) == 2
                assert assistant.role == "Lynx"
                assert runner.context.history[-1].content == "**最终回答**"
            ui._load_transcript()
            assert "尝试" in transcript_text(ui.state)
            assert "历史压缩：约 0" not in transcript_text(ui.state)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_tui_tool_phase_and_tools_command_do_not_create_extra_sessions(tmp_path):
    class ToolProvider(ModelProvider):
        def __init__(self):
            self.called = False

        async def generate(self, request):
            if request.messages[-1].role == "tool":
                assert request.messages[-1].content == "60"
                return ModelResponse(content="结果是 **60**", model="fake", finish_reason="stop")
            self.called = True
            return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(
                ModelToolCall(id="calc", name="calculator", arguments='{"expression":"(12+3)*4"}'),
            ))

        async def aclose(self):
            pass

    provider = ToolProvider()
    store = SessionStore(tmp_path)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput())
            phases = []
            status = ui._status

            def record(phase):
                phases.append(phase)
                status(phase)

            ui._status = record
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("/tools\r")
            await wait_until(lambda: "当前注册的模型工具" in transcript_text(ui.state) and not ui.state.busy)
            assert "calculator" in transcript_text(ui.state)
            assert not store.list_sessions()
            pipe.send_text("计算 (12+3)*4\r")
            await wait_until(lambda: provider.called and not ui.state.busy)
            assert "tooling" in phases
            assert runner.context.history[-1].content == "结果是 **60**"
            card = next(message for message in ui.state.messages if message.role == "工具")
            assert "计算器" in card.content and "本批耗时" in card.content
            assert "expression" not in card.content
            assert ui.state.messages.index(card) < next(index for index, item in enumerate(ui.state.messages) if item.role == "Lynx")
            ui._load_transcript()
            assert "✓ 计算器 · 1 项已返回" in transcript_text(ui.state)
            assert sum(message.role == "工具" for message in ui.state.messages) == 1
            assert "本批耗时" not in transcript_text(ui.state)  # 旧存档无单批计时，不编造恢复耗时。
            assert len(store.list_sessions()) == 1
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


def test_request_timer_in_front_of_status_freezes_and_restarts(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("lynx_agent.cli.state.monotonic", lambda: now[0])
    state = UIState(project=Path.cwd(), model="fake", search_backend="anysearch")
    state.mcp_status = lambda: "2/2 已连接"
    state.memory_status = lambda: "2/2 可写"
    assert status_text(state) == "就绪 · 权限：未启用 · Shell：关闭 · 联网搜索：anysearch"
    state.start_request()
    now[0] = 103.2
    state.status = "正在执行工具…"
    assert status_text(state) == "已用 3.2s · 正在执行工具… · 权限：未启用 · Shell：关闭 · 联网搜索：anysearch"
    now[0] = 108.5
    state.finish_request()
    state.status = "就绪"
    now[0] = 200.0
    assert status_text(state) == "上次耗时 8.5s · 就绪 · 权限：未启用 · Shell：关闭 · 联网搜索：anysearch"
    state.finish_request()
    assert state.last_request_seconds == 8.5
    state.start_request()
    assert state.timing_text() == "已用 0.0s"


@pytest.mark.asyncio
async def test_visible_backend_timer_updates_while_waiting_freezes_after_cancel(tmp_path):
    provider = TestProvider()
    provider.gate = asyncio.Event()
    store = SessionStore(tmp_path)
    settings = config()
    output = RecordingOutput(columns=100)
    with create_pipe_input() as pipe:
        async with AgentRunner(settings, provider, store, tools=create_default_tools(settings)) as runner:
            ui = LynxTUI(runner, settings, store, input_stream=pipe, output_stream=output)
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            # 联网搜索状态已从界面移除，直接看状态对象。
            await wait_until(lambda: ui.state.search_backend == "anysearch")
            assert ui.state.search_backend == "anysearch"
            pipe.send_text("计时测试\r")
            await wait_until(lambda: bool(provider.requests))
            started = ui.state.request_started_at
            await wait_until(lambda: ui.state.timing_text().startswith("已用"))
            first = ui.state.timing_text()
            await wait_until(lambda: ui.state.timing_text() != first)
            assert ui.state.request_started_at == started
            pipe.send_text("\x03")
            await wait_until(lambda: not ui.state.busy)
            assert ui.state.request_started_at is None
            elapsed = ui.state.last_request_seconds
            assert elapsed > 0
            pipe.send_text("/tools\r")
            await wait_until(lambda: "搜索后端：anysearch" in transcript_text(ui.state) and not ui.state.busy)
            assert ui.state.last_request_seconds == elapsed
            provider.gate = None
            pipe.send_text("再次请求\r")
            await wait_until(lambda: len(provider.requests) == 2 and not ui.state.busy)
            assert ui.state.request_started_at is None
            assert ui.state.last_request_seconds is not None
            assert "上次耗时" in status_text(ui.state)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery_fails", [False, True])
async def test_tui_markup_recovery_never_displays_protocol_or_saves_failed_turn(tmp_path, recovery_fails):
    class MarkupProvider(TestProvider):
        async def stream(self, request):
            self.requests.append(request)
            text = '<｜｜DSML｜｜ calls>' if len(self.requests) == 1 or recovery_fails else "现有证据不足，无法确认。"
            try:
                for char in text:
                    yield ModelStreamEvent(kind="delta", text=char)
                yield ModelStreamEvent(kind="done", response=ModelResponse(content=text, model="fake", finish_reason="stop"))
            finally:
                self.stream_closed = True

    settings = config()
    provider = MarkupProvider()
    store = SessionStore(tmp_path)
    output = RecordingOutput(columns=120)
    with create_pipe_input() as pipe:
        async with AgentRunner(settings, provider, store) as runner:
            ui = LynxTUI(runner, settings, store, input_stream=pipe, output_stream=output)
            phases = []
            original_status = ui._status

            def record(phase):
                phases.append(phase)
                original_status(phase)

            ui._status = record
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("工具协议恢复测试\r")
            await wait_until(lambda: len(provider.requests) == 2 and not ui.state.busy)
            assert "recovering" in phases
            assert "DSML" not in transcript_text(ui.state)
            assert "DSML" not in output.text
            assert ui.state.last_request_seconds is not None
            saved = store.load(runner.session_id)
            if recovery_fails:
                assert runner.context.history == ()
                assert saved.context_history() == ()
                assert all(event.kind in {"diagnostic", "pending"} for event in saved.events)
                assert "操作失败" in transcript_text(ui.state)
            else:
                assert runner.context.history[-1].content == "现有证据不足，无法确认。"
                assert [event.kind for event in saved.events if event.kind != "diagnostic"] == ["turn"]
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_tui_hides_legacy_protocol_without_deleting_session(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create().with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="旧问题"), ChatMessage(role="assistant", content="<|DSML|calls>"),
    )))
    store.save(session)
    async with AgentRunner(config(), TestProvider(), store, session) as runner:
        ui = LynxTUI(runner, config(), store, output_stream=RecordingOutput())
        assert "DSML" not in transcript_text(ui.state)
        assert "原始会话记录仍保留" in transcript_text(ui.state)
        assert "DSML" in store.load(session.id).context_history()[-1].content


@pytest.mark.asyncio
async def test_tui_search_on_off_status_help_and_no_network_calls(tmp_path):
    settings = config()
    provider = TestProvider()
    store = SessionStore(tmp_path)
    with create_pipe_input() as pipe:
        async with AgentRunner(settings, provider, store, tools=create_default_tools(settings)) as runner:
            ui = LynxTUI(runner, settings, store, input_stream=pipe, output_stream=RecordingOutput())
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("/search off\r")
            await wait_until(lambda: ui.state.search_backend == "已关闭" and not ui.state.busy)
            assert "联网搜索：已关闭" in status_text(ui.state)
            assert not provider.requests and not store.list_sessions()
            previous_task = ui._task
            pipe.send_text("/new\r")
            # 等待新命令实际被分发，不能把发送前的空闲状态当作命令已完成。
            await wait_until(lambda: ui._task is not previous_task and not ui.state.busy)
            pipe.send_text("/search on\r")
            await wait_until(lambda: ui.state.search_backend == "anysearch" and not ui.state.busy)
            assert "已开启（anysearch）" in transcript_text(ui.state)
            pipe.send_text("/search invalid\r")
            await wait_until(lambda: "用法：/search" in transcript_text(ui.state) and not ui.state.busy)
            assert runner.search_enabled
            pipe.send_text("/help\r")
            await wait_until(lambda: "/search [on/off]" in transcript_text(ui.state) and not ui.state.busy)
            assert not provider.requests and not store.list_sessions()
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_tui_permissions_fixed_workspace_and_file_approval_without_model_calls(tmp_path):
    import json

    access = FileAccess(tmp_path)
    path = tmp_path / "approval.txt"
    path.write_text("old", encoding="utf-8")
    proposed = json.loads(await access.invoke("edit_file", {"path": "approval.txt", "old_text": "old", "new_text": "new"}))
    identifier = proposed["approval_id"]
    settings = config()
    provider = TestProvider()
    store = SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(settings, provider, store, tools=create_default_tools(settings, file_access=access)) as runner:
            ui = LynxTUI(runner, settings, store, project=tmp_path.parent, input_stream=pipe, output_stream=RecordingOutput())
            assert ui.state.project == tmp_path
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            assert ui.state.permissions == "工作区修改"
            pipe.send_text(f"/approve {identifier}\r")
            await wait_until(lambda: "本次仅显示预览" in transcript_text(ui.state) and not ui.state.busy)
            assert path.read_text() == "old"
            pipe.send_text(f"/approve {identifier}\r")
            await wait_until(lambda: "已应用文件修改" in transcript_text(ui.state) and not ui.state.busy)
            assert path.read_text() == "new"
            pipe.send_text("/permissions read\r")
            await wait_until(lambda: ui.state.permissions == "仅查看" and not ui.state.busy)
            assert "write_file" not in runner.tools.names
            pipe.send_text("/permissions full\r")
            await wait_until(lambda: "当前权限尚未改变" in transcript_text(ui.state) and not ui.state.busy)
            assert ui.state.permissions == "仅查看"
            pipe.send_text("/permissions full confirm\r")
            await wait_until(lambda: ui.state.permissions == "完全权限" and not ui.state.busy)
            assert not provider.requests and not store.list_sessions()
            assert "权限：完全权限" in status_text(ui.state)
            pipe.send_text("/new\r")
            # 等待命令确实完成重置，不能在输入尚未被处理时因初始空闲误通过。
            await wait_until(lambda: not ui.state.busy and len(ui.state.messages) == 1)
            pipe.send_text("/permissions\r")
            await wait_until(lambda: "固定工作区" in transcript_text(ui.state) and not ui.state.busy)
            assert runner.permission_label == "完全权限"
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)
