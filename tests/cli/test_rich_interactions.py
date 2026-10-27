"""完整交互与安全边界：剪贴板注入模拟器，不改动用户剪贴板或调用模型 API。"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input.defaults import create_pipe_input
from test_tui import RecordingOutput, TestProvider, config, wait_until

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.clipboard import ClipboardError, write_clipboard
from lynx_agent.cli.completion import AgentCompleter
from lynx_agent.cli.message_actions import copy_source
from lynx_agent.cli.projection import TranscriptProjection
from lynx_agent.cli.selection import Anchor, TextSelection
from lynx_agent.cli.session_browser import filter_choices, session_choices
from lynx_agent.cli.state import DisplayMessage, UIState
from lynx_agent.cli.themes import load_theme, save_theme
from lynx_agent.cli.tool_cards import detail_text
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import ChatMessage, ModelToolCall
from lynx_agent.providers.models import ToolTrace
from lynx_agent.session import SessionEvent, SessionStore
from lynx_agent.tools.errors import ToolExecutionError


def test_copy_preserves_code_and_strips_display_chrome():
    message = DisplayMessage("Lynx", "**你好世界**\n\n```python\nx = '**literal**'\n  y = 2\n```\n")
    assert copy_source(message, "plain").startswith("你好世界")
    assert copy_source(message, "markdown") == message.content
    assert copy_source(message, "code") == "x = '**literal**'\n  y = 2\n"
    with pytest.raises(ValueError):
        copy_source(message, "code", 9)
    with pytest.raises(ClipboardError):
        write_clipboard("bad\0text")


def test_terminal_controls_never_reach_copy_and_combining_marks_are_kept():
    source = DisplayMessage("Lynx", "\x1b[31m**红色**\x1b[0m\x1b]52;c;bad\x07\n")
    assert copy_source(source) == "红色"
    assert "\x1b" not in copy_source(source, "markdown")
    message = DisplayMessage("你", "e\u0301 👩\u200d💻")
    selection = TextSelection()
    selection.begin(Anchor(message.id, 0))
    selection.extend(Anchor(message.id, 1))
    selection.finish([message], lambda item: item.content)
    assert selection.snapshot == "e\u0301"


@pytest.mark.parametrize("success", [True, False])
def test_windows_clipboard_handles_unicode_and_releases_on_failure(monkeypatch, success):
    import ctypes

    from lynx_agent.cli.clipboard import _windows_copy

    memory = ctypes.create_string_buffer(256)
    user = SimpleNamespace(CreateWindowExW=Mock(return_value=123), DestroyWindow=Mock(return_value=1),
                           OpenClipboard=Mock(return_value=1), EmptyClipboard=Mock(return_value=1),
                           SetClipboardData=Mock(return_value=99 if success else 0), CloseClipboard=Mock(return_value=1))
    kernel = SimpleNamespace(GlobalAlloc=Mock(return_value=99), GlobalLock=Mock(return_value=ctypes.addressof(memory)),
                             GlobalUnlock=Mock(return_value=1), GlobalFree=Mock(return_value=None))
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **kwargs: user if name == "user32" else kernel, raising=False)
    if success:
        _windows_copy("你好\nLynx")
        assert memory.raw.startswith("你好\nLynx".encode("utf-16-le") + b"\0\0")
        kernel.GlobalFree.assert_not_called()
    else:
        with pytest.raises(ClipboardError):
            _windows_copy("模拟失败")
        kernel.GlobalFree.assert_called_once_with(99)
    user.CloseClipboard.assert_called_once()
    user.OpenClipboard.assert_called_once_with(123)
    user.DestroyWindow.assert_called_once_with(123)


def test_selection_survives_resize_stream_append_and_cross_message_reverse_drag():
    first, second = DisplayMessage("你", "你好世界" * 8), DisplayMessage("Lynx", "**答案** 然后继续")
    state = UIState(project=Path.cwd(), model="fake", messages=[first, second])
    selection = TextSelection()
    view = TranscriptProjection(state, SimpleNamespace(file_access=None, tool_traces=()), selection)
    view.update(10)
    selection.begin(Anchor(second.id, 2))
    selection.extend(Anchor(first.id, 1))
    view.update(7)
    selection.finish(state.messages, view.plain)
    expected = first.content[1:] + "\n\n答案"
    assert selection.snapshot == expected
    second.content += "新增的文字"
    view.update(30)
    assert selection.snapshot == expected
    assert any("class:selection" in style for i in range(len(view.rows)) for style, _ in view.get_line(i))


def test_projection_is_windowed_cached_and_keeps_all_source_messages():
    state = UIState(project=Path.cwd(), model="fake", messages=[DisplayMessage("你", f"历史 {i}") for i in range(1000)])
    view = TranscriptProjection(state, SimpleNamespace(file_access=None, tool_traces=()), TextSelection())
    view.update(40)
    rows = view.rows
    view.update(40)
    assert view.rows is rows and len(view.cache) <= 96
    assert "加载更早" in "".join(g.char for g in view.rows[0])
    assert "历史 0" not in "\n".join("".join(g.char for g in row) for row in view.rows)
    assert len(state.messages) == 1000 and view.cache_chars <= 160000
    state.history_end_id = view.start_id
    view.update(40)
    assert any(g.action == ("latest",) for row in view.rows for g in row)
    assert len(state.messages) == 1000


def test_file_completion_excludes_sensitive_and_private_roots(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "main.py").write_text("print(1)", encoding="utf-8")
    (root / ".env").write_text("fake", encoding="utf-8")
    private = root / "archive"
    private.mkdir()
    (private / "secret.txt").write_text("not listed", encoding="utf-8")
    access = FileAccess(root)
    access.policy.private_roots.add(private)
    completer = AgentCompleter(SimpleNamespace(file_access=access))
    event = CompleteEvent(completion_requested=True)
    paths = [item.text for item in completer.get_completions(Document("查看 @"), event)]
    assert "@main.py" in paths and not any("env" in item or "archive" in item for item in paths)
    assert not list(completer.get_completions(Document("@../"), event))
    assert not list(completer.get_completions(Document("多行\n/clear"), event))
    assert [item.text for item in completer.get_completions(Document("/them"), event)] == ["/theme"]


def test_tool_details_revalidate_permissions_and_cache(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("OUTSIDE_BODY", encoding="utf-8")
    access = FileAccess(root)
    access.set_mode("full")
    trace = ToolTrace(turn_id=uuid4(), result_id=uuid4(), call=ModelToolCall(
        id="read", name="read_file", arguments=json.dumps({"path": str(outside), "api_key": "fake-key"}),
    ), content="OUTSIDE_BODY", scope_paths=(str(outside),))
    runner = SimpleNamespace(file_access=access, tool_traces=(trace,))
    message = DisplayMessage("工具", "✓ 读取文件", tools=(trace,), expanded=True)
    text, pages = detail_text(message, runner)
    assert "OUTSIDE_BODY" in text and "fake-key" not in text and pages == 1
    state = UIState(project=root, model="fake", messages=[message])
    view = TranscriptProjection(state, runner, TextSelection())
    view.update(60)
    assert "OUTSIDE_BODY" in "\n".join("".join(g.char for g in row) for row in view.rows)
    access.set_mode("read")
    with pytest.raises(ToolExecutionError):
        detail_text(message, runner)
    view.update(60)
    assert "OUTSIDE_BODY" not in "\n".join("".join(g.char for g in row) for row in view.rows)


def test_themes_and_session_search_do_not_change_agent_data(tmp_path):
    path = tmp_path / "ui" / "preferences.json"
    save_theme(path, "light")
    assert load_theme(path) == "light"
    path.write_text("broken", encoding="utf-8")
    assert load_theme(path) == "ansi"
    with pytest.raises(ValueError):
        save_theme(path, "unknown")
    store = SessionStore(tmp_path / "sessions")
    session = store.create().model_copy(update={"title": "查找项目文档"}).with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="问题"), ChatMessage(role="assistant", content="预览关键词 PineOak"),
    )))
    store.save(session)
    choices = session_choices(store.list_sessions())
    filtered = filter_choices(choices, "pineoak")
    assert len(filtered) == 2 and filtered[1][0] == str(session.id)
    assert str(session.id)[:8] not in filtered[1][1]


def coordinates(ui, message_id, offset):
    info = ui.transcript.render_info
    for y, row in enumerate(ui.projection.rows):
        for x, glyph in enumerate(row):
            if glyph.anchor == Anchor(message_id, offset):
                from prompt_toolkit.utils import get_cwidth
                return sum(get_cwidth(g.char) for g in row[:x]) + 1, info._y_offset + y - info.first_visible_line() + 1
    raise AssertionError("未找到文字锚点")


def action_coordinates(ui, action):
    from prompt_toolkit.utils import get_cwidth
    info = ui.transcript.render_info
    for y, row in enumerate(ui.projection.rows):
        for x, glyph in enumerate(row):
            if glyph.action == action:
                return sum(get_cwidth(g.char) for g in row[:x]) + 1, info._y_offset + y - info.first_visible_line() + 1
    raise AssertionError("未找到操作按钮")


@pytest.mark.asyncio
async def test_real_mouse_select_resize_copy_button_and_clipboard_failure(tmp_path):
    provider, clipboard = TestProvider(), AsyncMock()
    store = SessionStore(tmp_path / "sessions")
    output = RecordingOutput(columns=50, rows=26)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=output,
                         clipboard_writer=clipboard, preferences_path=tmp_path / "preferences.json")
            message = DisplayMessage("Lynx", "**你好世界** abc\n\n```python\nx = 1\n```")
            ui.state.messages = [message]
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.transcript.render_info is not None)
            x1, y1 = coordinates(ui, message.id, 0)
            x2, y2 = coordinates(ui, message.id, 3)
            pipe.send_text(f"\x1b[<0;{x1};{y1}M\x1b[<32;{x2};{y2}M\x1b[<0;{x2};{y2}m")
            await wait_until(lambda: ui.selection.snapshot == "你好世界")
            output.columns = 36
            ui.application.invalidate()
            await asyncio.sleep(0.15)
            pipe.send_text("\x1bc")  # Alt+C，复制选区，不取消请求或退出。
            await wait_until(lambda: clipboard.await_count == 1)
            clipboard.assert_awaited_with("你好世界")
            assert not ui.state.selection_available
            await asyncio.sleep(0.15)
            x, y = action_coordinates(ui, ("copy", message.id, "plain"))
            pipe.send_text(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
            await wait_until(lambda: clipboard.await_count == 2)
            clipboard.assert_awaited_with(copy_source(message))
            clipboard.side_effect = ClipboardError("测试占用")
            ui._perform_action(("copy", message.id, "plain"))
            await wait_until(lambda: "复制失败" in ui.state.toast)
            assert not provider.requests and not store.list_sessions()
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_fullscreen_draft_completion_and_session_search_preserve_input(tmp_path):
    provider = TestProvider()
    store = SessionStore(tmp_path / "sessions")
    old = store.create().model_copy(update={"title": "SearchNeedle"})
    store.save(old)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(),
                         clipboard_writer=AsyncMock(), preferences_path=tmp_path / "preferences.json")
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("草稿\x1be")
            await wait_until(lambda: ui.state.draft_expanded)
            pipe.send_text("\r第二行")
            await wait_until(lambda: ui.editor.text == "草稿\n第二行")
            pipe.send_text("\x1be")
            await wait_until(lambda: not ui.state.draft_expanded)
            assert ui.editor.text == "草稿\n第二行" and not provider.requests
            await ui._dispatch("/sessions")
            pipe.send_text("SearchNeedle")
            await wait_until(lambda: ui.state.session_query == "SearchNeedle")
            assert len(ui.state.session_choices) == 2
            pipe.send_text("\x1b")
            await wait_until(lambda: not ui.state.picker_open)
            assert ui.editor.text == "草稿\n第二行"
            pipe.send_text("\x03/them\t")
            await wait_until(lambda: ui.editor.text.startswith("/theme"))
            pipe.send_text("\x03/exit\r")
            await asyncio.wait_for(task, timeout=3)
    assert not provider.requests


@pytest.mark.asyncio
async def test_fullscreen_send_theme_persistence_and_copy_does_not_call_model(tmp_path):
    provider, clipboard = TestProvider(), AsyncMock()
    store = SessionStore(tmp_path / "sessions")
    preferences = tmp_path / "preferences.json"
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(),
                         clipboard_writer=clipboard, preferences_path=preferences)
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("第一行\x1be")
            await wait_until(lambda: ui.state.draft_expanded)
            await asyncio.sleep(0.15)
            assert ui.editor.window.render_info.window_height > 6
            pipe.send_text("\r第二行\x1bs")
            await wait_until(lambda: len(provider.requests) == 1 and not ui.state.busy)
            assert provider.requests[0].messages[-1].content == "第一行\n第二行"
            assert not ui.state.draft_expanded and len(store.list_sessions()) == 1
            pipe.send_text("/theme dark\r")
            await wait_until(lambda: ui.state.theme == "dark" and not ui.state.busy)
            assert load_theme(preferences) == "dark"
            pipe.send_text("/copy\r")
            await wait_until(lambda: clipboard.await_count == 1)
            assert len(provider.requests) == 1
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_copy_while_generating_does_not_cancel_request(tmp_path):
    provider, clipboard = TestProvider(), AsyncMock()
    provider.gate = asyncio.Event()
    store = SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(),
                         clipboard_writer=clipboard, preferences_path=tmp_path / "preferences.json")
            previous = DisplayMessage("Lynx", "**旧回答**")
            ui.state.messages.append(previous)
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("测试等待\r")
            await wait_until(lambda: bool(provider.requests))
            ui._perform_action(("copy", previous.id, "plain"))
            await wait_until(lambda: clipboard.await_count == 1)
            assert ui.state.busy and not provider.stream_closed
            clipboard.assert_awaited_with("旧回答")
            pipe.send_text("\x03")
            await wait_until(lambda: not ui.state.busy)
            assert provider.stream_closed
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_click_session_row_opens_selected_session_without_dropping_draft(tmp_path):
    from prompt_toolkit.formatted_text import fragment_list_to_text

    store = SessionStore(tmp_path / "sessions")
    session = store.create().model_copy(update={"title": "ClickSessionTarget"})
    store.save(session)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(),
                         clipboard_writer=AsyncMock(), preferences_path=tmp_path / "preferences.json")
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            ui.editor.text = "保留的草稿"
            await ui._dispatch("/sessions")
            await asyncio.sleep(0.15)
            target = None
            for window in ui.application.layout.find_all_windows():
                if window.render_info is None:
                    continue
                content = window.content.create_content(80, 20)
                for row in range(min(content.line_count, 20)):
                    if "ClickSessionTarget" in fragment_list_to_text(content.get_line(row)):
                        target = (window.render_info._x_offset + 4, window.render_info._y_offset + row + 1)
            assert target is not None
            x, y = target
            pipe.send_text(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
            await wait_until(lambda: runner.session_id == str(session.id) and not ui.state.busy)
            assert ui.editor.text == "保留的草稿"
            pipe.send_text("\x03/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
async def test_drag_at_edge_continues_scrolling_until_mouse_release(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    output = RecordingOutput(columns=55, rows=24)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=output,
                         clipboard_writer=AsyncMock(), preferences_path=tmp_path / "preferences.json")
            ui.state.messages = [DisplayMessage("你", "\n".join(f"第{i}行 中文边缘拖动" for i in range(100)))]
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.transcript.render_info is not None)
            ui.scroll_to(0)
            await asyncio.sleep(0.15)
            info = ui.transcript.render_info
            top = info._y_offset + 2
            bottom = info._y_offset + info.window_height
            pipe.send_text(f"\x1b[<0;4;{top}M\x1b[<32;4;{bottom}M")
            await wait_until(lambda: ui.state.scroll_line >= 3)
            pipe.send_text(f"\x1b[<0;4;{bottom}m")
            await wait_until(lambda: bool(ui.selection.snapshot))
            stopped = ui.state.scroll_line
            await asyncio.sleep(0.2)
            assert ui.state.scroll_line == stopped
            pipe.send_text("\x1b")
            await wait_until(lambda: not ui.state.selection_available)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)
