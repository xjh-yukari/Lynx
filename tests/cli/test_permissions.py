"""真实鼠标/按键切换权限，检查草稿、文件边界和默认 Shell。"""

import asyncio

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.utils import get_cwidth
from test_effort import EffortProvider
from test_tui import RecordingOutput, config, wait_until

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.filesystem import FileAccess
from lynx_agent.session import SessionStore
from lynx_agent.tools.errors import ToolExecutionError
from lynx_agent.web.setup import create_default_tools


async def click_label(ui, pipe, label):
    # 等待状态变更后的布局完成，使用终端列宽定位中文标签。
    await asyncio.sleep(0.15)
    width = ui.application.output.get_size().columns
    for window in ui.application.layout.find_all_windows():
        info = window.render_info
        if info is None:
            continue
        content = window.content.create_content(width, info.window_height)
        for row in range(min(content.line_count, info.window_height)):
            text = fragment_list_to_text(content.get_line(row))
            if label in text:
                x = info._x_offset + get_cwidth(text[:text.index(label)]) + 1
                y = info._y_offset + row + 1
                pipe.send_text(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
                return
    raise AssertionError(f"未找到可点击标签：{label}")


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [44, 110])
async def test_click_permissions_applies_file_boundary_and_preserves_draft(tmp_path, columns):
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    access = FileAccess(root)
    provider, settings, store = EffortProvider(), config(), SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(settings, provider, store, tools=create_default_tools(settings, file_access=access)) as runner:
            ui = LynxTUI(runner, settings, store, input_stream=pipe,
                         output_stream=RecordingOutput(columns=columns, rows=32),
                         preferences_path=tmp_path / "prefs.json")
            assert runner.execution.enabled and "exec_command" in runner.tools.names
            assert ui.state.shell_enabled
            task = asyncio.create_task(ui.run())
            try:
                await wait_until(lambda: ui.editor.window.render_info is not None)
                pipe.send_text("保留草稿")
                await wait_until(lambda: ui.editor.text == "保留草稿")
                position = ui.editor.buffer.cursor_position
                with pytest.raises(ToolExecutionError, match="超出固定工作区"):
                    access.policy.resolve(str(outside))
                access.policy.sensitive_grants.add("old-grant")
                await access.invoke("write_file", {"path": "inside.txt", "content": "old"})
                await access.invoke("edit_file", {"path": "inside.txt", "old_text": "old", "new_text": "new"})
                assert access.pending

                await click_label(ui, pipe, "工作区修改 ▾")
                await wait_until(lambda: ui.state.permissions_open)
                await click_label(ui, pipe, "完全权限")
                await wait_until(lambda: runner.permission_label == "完全权限")
                assert access.policy.resolve(str(outside)) == outside
                assert not access.pending and not access.policy.sensitive_grants
                assert ui.state.permissions == "完全权限"

                # 选择区期间粘贴和文字不穿透，Enter 只关闭，不提交草稿。
                pipe.send_text("\x1b[200~不应插入\x1b[201~x\r")
                await wait_until(lambda: not ui.state.permissions_open)
                assert ui.editor.text == "保留草稿" and ui.editor.buffer.cursor_position == position
                assert not provider.requests

                await click_label(ui, pipe, "权限：完全权限")
                await wait_until(lambda: ui.state.permissions_open)
                await click_label(ui, pipe, "仅查看")
                await wait_until(lambda: runner.permission_label == "仅查看")
                assert "exec_command" not in runner.tools.names
                pipe.send_text("\x1b[C")
                await wait_until(lambda: runner.permission_label == "工作区修改")
                assert "exec_command" in runner.tools.names
                with pytest.raises(ToolExecutionError, match="超出固定工作区"):
                    access.policy.resolve(str(outside))
                pipe.send_text("\x03")
                await wait_until(lambda: not ui.state.permissions_open)
                assert ui.editor.text == "保留草稿"

                # 两种选择区互斥，关闭 Shell 后切换权限不会重新开启。
                ui.open_effort()
                assert ui.state.effort_open
                ui.open_permissions()
                assert ui.state.permissions_open and not ui.state.effort_open
                runner.shell_command("off")
                ui.pick_permission(2)
                assert not runner.execution.enabled and not ui.state.shell_enabled
                pipe.send_text("\x04")
                await wait_until(lambda: not ui.state.permissions_open)
                assert ui.application.is_running
                pipe.send_text("\x03/exit\r")
                await asyncio.wait_for(task, timeout=3)
            finally:
                if not task.done():
                    ui.application.exit()
                    await task
    assert not provider.requests and not store.list_sessions()


@pytest.mark.asyncio
async def test_permissions_cannot_change_during_request(tmp_path):
    access, settings = FileAccess(tmp_path), config()
    provider, store = EffortProvider(), SessionStore(tmp_path / "sessions")
    provider.gate = asyncio.Event()
    with create_pipe_input() as pipe:
        async with AgentRunner(settings, provider, store, tools=create_default_tools(settings, file_access=access)) as runner:
            ui = LynxTUI(runner, settings, store, input_stream=pipe, output_stream=RecordingOutput(),
                         preferences_path=tmp_path / "prefs.json")
            task = asyncio.create_task(ui.run())
            try:
                await wait_until(lambda: ui.application.is_running)
                pipe.send_text("问题\r")
                await wait_until(lambda: bool(provider.requests))
                ui.open_permissions()
                ui.pick_permission(2)
                assert not ui.state.permissions_open and runner.permission_label == "工作区修改"
                provider.gate.set()
                await wait_until(lambda: not ui.state.busy)
                ui.open_permissions()
                ui.pick_permission(2)
                assert runner.permission_label == "完全权限"
                ui.close_permissions()
                pipe.send_text("/exit\r")
                await asyncio.wait_for(task, timeout=3)
            finally:
                if not task.done():
                    ui.application.exit()
                    await task
