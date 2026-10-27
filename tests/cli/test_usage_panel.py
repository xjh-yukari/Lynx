"""Token/缓存/费用入口使用真实鼠标和键盘；不请求付费接口。"""

import asyncio

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.utils import get_cwidth
from test_permissions import click_label
from test_tui import RecordingOutput, TestProvider, config, wait_until

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.usage import UsageTracker
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.statusbar import usage_metrics
from lynx_agent.cli.usage_details import breakdown_text, usage_details
from lynx_agent.config import Config
from lynx_agent.providers import TokenUsage
from lynx_agent.session import SessionStore


def official_config():
    data = config().model_dump()
    data["agent"]["model"]["name"] = "deepseek-flash"
    data["providers"]["deepseek"]["base_url"] = "https://api.deepseek.com"
    return Config.model_validate(data)


def record(tracker, prompt=1000, cached=800):
    tracker.finish(tracker.start("chat"), TokenUsage(
        prompt_tokens=prompt, completion_tokens=100, total_tokens=prompt + 100,
        cached_prompt_tokens=cached,
    ))


@pytest.mark.parametrize("width", [5, 10, 20, 40, 80, 110, 160])
def test_stats_remain_clickable_in_narrow_windows(width):
    tracker = UsageTracker()
    record(tracker, 1234567, 1000000)
    fragments = usage_metrics(tracker.breakdown(), "≈¥9999.9999* · 谷", width=width)
    assert get_cwidth("".join(item[1] for item in fragments)) <= width
    assert {item[2] for item in fragments} >= {"tokens", "cost"}


def test_unknown_usage_and_zero_cache_are_distinct():
    tracker = UsageTracker()
    tracker.finish(tracker.start("chat"), None)
    text = breakdown_text("测试", tracker.breakdown())
    assert "未知" in text and "合计 0" not in text
    assert "Token —*" in "".join(item[1] for item in usage_metrics(tracker.breakdown(), "费用—", width=100))
    tracker = UsageTracker()
    record(tracker, cached=0)
    text = breakdown_text("测试", tracker.breakdown())
    assert "命中 0" in text and "未命中 1,000" in text and "0.00%" in text


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [40, 110])
async def test_mouse_keyboard_details_preserve_draft_and_do_not_call_model(tmp_path, columns):
    cfg, provider, store = official_config(), TestProvider(), SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(cfg, provider, store) as runner:
            record(runner.last_usage)
            ui = LynxTUI(runner, cfg, store, input_stream=pipe,
                         output_stream=RecordingOutput(columns=columns, rows=32),
                         preferences_path=tmp_path / "prefs.json")
            original_messages = list(ui.state.messages)
            task = asyncio.create_task(ui.run())
            try:
                await wait_until(lambda: ui.editor.window.render_info is not None)
                pipe.send_text("保留草稿")
                await wait_until(lambda: ui.editor.text == "保留草稿")
                position = ui.editor.buffer.cursor_position
                await click_label(ui, pipe, "Token")
                await wait_until(lambda: ui.state.usage_panel == "tokens")
                assert "输入 1,000 · 输出 100 · 合计 1,100" in ui.usage_editor.text
                assert "命中 800 · 未命中 200" in ui.usage_editor.text
                assert "命中率 80.00%" in ui.usage_editor.text
                assert "元/百万 Tokens" in ui.usage_editor.text
                await click_label(ui, pipe, "费用" if columns == 40 else "会话 ≈¥")
                await wait_until(lambda: ui.state.usage_panel == "cost")
                await click_label(ui, pipe, "Token")
                await wait_until(lambda: ui.state.usage_panel == "tokens")
                pipe.send_text("\x1b[6~")  # PgDn 只滚动详情。
                await wait_until(lambda: ui.usage_editor.buffer.cursor_position > 0)
                assert ui.state.follow_tail
                pipe.send_text("\x1b[200~不应插入\x1b[201~x\r")
                await wait_until(lambda: not ui.state.usage_panel)
                assert ui.editor.text == "保留草稿" and ui.editor.buffer.cursor_position == position
                assert not provider.requests

                await click_label(ui, pipe, "费用" if columns == 40 else "会话 ≈¥")
                await wait_until(lambda: ui.state.usage_panel == "cost")
                assert ui.usage_editor.text.startswith("模型费用估算")
                assert "缓存命中输入 ¥" in ui.usage_editor.text
                assert "未命中输入 ¥" in ui.usage_editor.text
                # 运行中获得新 usage，面板自动更新且不重复记账。
                record(runner.last_usage, 500, 100)
                await wait_until(lambda: "合计 1,700" in ui.usage_editor.text)
                assert runner.costs.usage(runner.session_id).total_tokens == 1700
                ui.application.output.columns = 70
                ui.application.invalidate()
                await asyncio.sleep(0.15)
                pipe.send_text("\x1b")
                await wait_until(lambda: not ui.state.usage_panel)
                assert ui.editor.text == "保留草稿"
                assert ui.state.messages == original_messages
                assert not provider.requests and not store.list_sessions()
            finally:
                ui.application.exit()
                await task


@pytest.mark.asyncio
async def test_statistics_work_during_request_and_yield_to_approval(tmp_path):
    cfg, provider, store = config(), TestProvider(), SessionStore(tmp_path / "sessions")
    provider.gate = asyncio.Event()
    with create_pipe_input() as pipe:
        async with AgentRunner(cfg, provider, store) as runner:
            ui = LynxTUI(runner, cfg, store, input_stream=pipe, output_stream=RecordingOutput(),
                         preferences_path=tmp_path / "prefs.json")
            task = asyncio.create_task(ui.run())
            try:
                await wait_until(lambda: ui.application.is_running)
                pipe.send_text("问题\r")
                await wait_until(lambda: bool(provider.requests))
                await click_label(ui, pipe, "Token")
                await wait_until(lambda: ui.state.usage_panel == "tokens")
                assert "未知" in ui.usage_editor.text
                assert "未计价" in ui.usage_editor.text
                provider.gate.set()
                await wait_until(lambda: not ui.state.busy and "合计 25" in ui.usage_editor.text)
                assert ui.state.usage_panel == "tokens"
                assert len(provider.requests) == 1
                ui.state.approval_open = True
                ui.state.usage_panel = ""
                ui.toggle_usage("cost")
                assert not ui.state.usage_panel
                ui.state.approval_open = False
                with set_app(ui.application):
                    control = next(item for item in ui.application.layout.find_all_controls()
                                   if callable(getattr(item, "text", None)) and item.text.__name__ == "metrics_line")
                    assert "25" in fragment_list_to_text(control.text())
                assert "第三方接口" in usage_details(runner, cfg)
            finally:
                ui.application.exit()
                await task
