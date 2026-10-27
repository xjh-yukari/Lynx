"""推理选择操作与真实参数发送分别验证，不访问真实模型。"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from test_tui import RecordingOutput, TestProvider, config, wait_until

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.themes import load_effort, load_theme, save_effort, save_theme
from lynx_agent.session import SessionStore


class EffortProvider(TestProvider):
    reasoning_efforts = ("off", "low", "high", "max")
    reasoning_effort = "high"

    def set_reasoning_effort(self, value):
        if value not in self.reasoning_efforts:
            raise ValueError("不支持")
        self.reasoning_effort = value


def test_theme_effort_preferences_preserve_each_other_and_scope_model(tmp_path):
    path = tmp_path / "ui.json"
    save_theme(path, "dark")
    save_effort(path, "deepseek-flash", "max")
    assert load_theme(path) == "dark" and load_effort(path, "deepseek-flash") == "max"
    assert load_effort(path, "another-model") is None
    save_theme(path, "light")
    assert load_effort(path, "deepseek-flash") == "max"
    path.write_text('{"theme": [], "effort": {}, "effort_model": "deepseek-flash"}', encoding="utf-8")
    assert load_theme(path) == "ansi" and load_effort(path, "deepseek-flash") is None


@pytest.mark.asyncio
async def test_click_model_and_adjust_effort_preserves_draft_and_single_row_input(tmp_path):
    provider, store = EffortProvider(), SessionStore(tmp_path / "sessions")
    preferences = tmp_path / "ui.json"
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(columns=100),
                         preferences_path=preferences, clipboard_writer=AsyncMock())
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.editor.window.render_info is not None)
            assert ui.editor.window.render_info.window_height == 1
            pipe.send_text("保留草稿")
            await wait_until(lambda: ui.editor.text == "保留草稿")
            target = None
            for window in ui.application.layout.find_all_windows():
                if window.render_info:
                    text = fragment_list_to_text(window.content.create_content(100, 1).get_line(0))
                    if "high ▾" in text:
                        target = (window.render_info._x_offset + 2, window.render_info._y_offset + 1)
            assert target is not None
            x, y = target
            pipe.send_text(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
            await wait_until(lambda: ui.state.effort_open)
            pipe.send_text("\x1b[C")
            await wait_until(lambda: runner.reasoning_effort == "max")
            assert ui.editor.text == "保留草稿"
            pipe.send_text("\r")
            await wait_until(lambda: not ui.state.effort_open)
            assert ui.editor.text == "保留草稿" and not provider.requests
            assert load_effort(preferences, "fake-model") == "max"
            pipe.send_text("\x1b\r第二行")
            await wait_until(lambda: ui.editor.text == "保留草稿\n第二行")
            await asyncio.sleep(0.15)
            assert ui.editor.window.render_info.window_height == 2
            pipe.send_text("\x03/effort\r")
            await wait_until(lambda: ui.state.effort_open and not ui.state.busy)
            pipe.send_text("\x1b[D\x1b[D\x1b[D")
            await wait_until(lambda: runner.reasoning_effort == "off")
            await asyncio.sleep(0.15)
            choice = next(window for window in ui.application.layout.find_all_windows()
                          if window.render_info and "推理强度" in fragment_list_to_text(window.content.create_content(100, 4).get_line(0)))
            row = fragment_list_to_text(choice.content.create_content(100, 4).get_line(1))
            x, y = row.index("Max") + 1, choice.render_info._y_offset + 2
            pipe.send_text(f"\x1b[<0;{x};{y}M\x1b[<0;{x};{y}m")
            await wait_until(lambda: runner.reasoning_effort == "max")
            pipe.send_text("\x1b")
            await wait_until(lambda: not ui.state.effort_open)
            pipe.send_text("/effort high\r")
            await wait_until(lambda: runner.reasoning_effort == "high" and not ui.state.busy)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)
    assert store.list_sessions() == []


@pytest.mark.asyncio
async def test_unsupported_model_does_not_offer_fake_tiers(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(config(), TestProvider(), store) as runner:
        ui = LynxTUI(runner, config(), store, input_stream=None, output_stream=RecordingOutput(),
                     preferences_path=tmp_path / "ui.json")
        ui.open_effort()
        assert not ui.state.effort_open
        with pytest.raises(ValueError, match="未声明"):
            runner.set_reasoning_effort("max")


@pytest.mark.asyncio
async def test_busy_guard_and_saved_effort_restoration(tmp_path):
    preferences = tmp_path / "ui.json"
    save_effort(preferences, "fake-model", "low")
    provider = EffortProvider()
    provider.gate = asyncio.Event()
    store = SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), provider, store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=RecordingOutput(),
                         preferences_path=preferences)
            assert runner.reasoning_effort == "low"
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: ui.application.is_running)
            pipe.send_text("问题\r")
            await wait_until(lambda: bool(provider.requests))
            ui.open_effort()
            assert not ui.state.effort_open
            with pytest.raises(ValueError, match="等待"):
                runner.set_reasoning_effort("max")
            provider.gate.set()
            await wait_until(lambda: not ui.state.busy)
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)
