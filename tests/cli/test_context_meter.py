"""上下文条和缓存统计分离；所有请求/剪贴板均离线模拟。"""

import asyncio
from types import SimpleNamespace

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.utils import get_cwidth
from test_tui import RecordingOutput, TestProvider, config, wait_until

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.context import AgentContext
from lynx_agent.agent.tokens import (
    estimate_context_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
)
from lynx_agent.agent.usage import UsageTracker
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.metrics import (
    ContextMeter,
    ContextReading,
    context_bar,
    measure_context,
)
from lynx_agent.cli.themes import theme_style
from lynx_agent.providers import ChatMessage, TokenUsage
from lynx_agent.session import SessionStore
from lynx_agent.tools import ToolRegistry


def fake_runner():
    context = AgentContext("系统提示")
    context.memory_profiles = "用户偏好"
    context.memory_recall = "召回资料"
    context.add_user_message("问题")
    return SimpleNamespace(context_meter_messages=context.messages, runtime_context="安全提示", tools=ToolRegistry(),
                           session_id="a", last_usage=UsageTracker())


def test_segments_total_matches_shared_token_estimate_and_shrinks():
    runner, cfg = fake_runner(), config()
    runner.context_meter_messages += (ChatMessage(role="tool", content="结果正文" * 200, tool_call_id="fake"),)
    reading = measure_context(runner, cfg, "草稿")
    expected = (estimate_context_tokens(runner.context_meter_messages, "草稿").total
                + estimate_tool_tokens(runner.tools.definitions()) + estimate_text_tokens(runner.runtime_context))
    assert reading.total == expected and reading.tools > 0 and reading.draft > 0
    runner.context_meter_messages = runner.context_meter_messages[:1]
    assert measure_context(runner, cfg).total < reading.total


@pytest.mark.parametrize("width", [0, 1, 5, 24, 40, 80, 120])
@pytest.mark.parametrize("used", [0, 20, 80, 120])
def test_progress_geometry_and_overflow_numbers(width, used):
    reading = ContextReading(system=used // 2, messages=used - used // 2, window=100, trigger=90)
    fragments = context_bar(reading, width)
    text = fragment_list_to_text(fragments)
    assert get_cwidth(text) == width
    assert sum("context.free" not in style for style, _ in fragments) == min(width, round(width * used / 100))
    if width >= 24:
        assert f"{used:.1f}%" in text
    if used >= 90 and width:
        assert all("context.danger" in style for style, _ in fragments)


def test_cache_is_last_chat_not_summary_and_resets_on_session_switch(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr("lynx_agent.cli.metrics.monotonic", lambda: clock[0])
    runner = fake_runner()
    meter = ContextMeter(runner, config())
    meter.sample("")
    assert meter.cache_percent is None
    usage = TokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110, cached_prompt_tokens=75)
    runner.last_usage.finish(runner.last_usage.start("chat"), usage)
    runner.last_usage.finish(runner.last_usage.start("memory"), usage.model_copy(update={"cached_prompt_tokens": 0}))
    clock[0] += 1
    meter.sample("")
    assert meter.cache_percent == 75
    runner.last_usage.finish(runner.last_usage.start("chat"), usage.model_copy(update={"cached_prompt_tokens": None}))
    meter.sample("")
    assert meter.cache_percent is None
    runner.last_usage.finish(runner.last_usage.start("chat"), usage.model_copy(update={"cached_prompt_tokens": 0}))
    meter.sample("")
    assert meter.cache_percent == 0
    runner.session_id, runner.last_usage = "b", UsageTracker()
    meter.sample("")
    assert meter.cache_percent is None


def test_meter_throttles_draft_changes_and_does_not_rescan_idle_history(monkeypatch):
    from lynx_agent.cli import metrics
    original, calls = metrics.measure_context, []

    def measure(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(metrics, "measure_context", measure)
    clock = [1.0]
    monkeypatch.setattr(metrics, "monotonic", lambda: clock[0])
    meter = ContextMeter(fake_runner(), config())
    meter.sample("")
    for _ in range(20):
        clock[0] += 1
        meter.sample("")
    assert len(calls) == 1
    meter.sample("新草稿")
    meter.sample("新草稿继续")
    assert len(calls) == 2
    clock[0] += 0.3
    meter.sample("新草稿继续")
    assert len(calls) == 3


def test_usage_aggregate_does_not_invent_missing_cache_data():
    tracker = UsageTracker()
    tracker.finish(tracker.start("chat"), TokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11, cached_prompt_tokens=5))
    tracker.finish(tracker.start("chat"), TokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    assert tracker.report().usage.cached_prompt_tokens is None


@pytest.mark.asyncio
async def test_live_rounded_input_context_click_and_new_session(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    output = RecordingOutput(columns=80, rows=28)
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), TestProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=output,
                         preferences_path=tmp_path / "ui.json")
            task = asyncio.create_task(ui.run())
            await wait_until(lambda: "╭" in output.text and "◆" in output.text)
            assert "╯" in output.text and "class:input" in ui.editor.window.style
            pipe.send_text("测试\r")
            await wait_until(lambda: not ui.state.busy and bool(runner.context.history))
            ui.context_meter.invalidate()
            await asyncio.sleep(0.3)
            reading = ui._sample_context()
            assert not runner.context_meter_inflight
            assert reading == measure_context(runner, config())
            # 进度条点击只显示本地统计，不发模型请求。
            window = next(w for w in ui.application.layout.find_all_windows()
                          if w.render_info and "context.free" in str(w.content.create_content(80, 1).get_line(0)))
            y = window.render_info._y_offset + 1
            calls = len(ui.runner._provider.requests)
            # 底栏已合并成一行，上下文短条在最右侧，点在右边缘才是它。
            pipe.send_text(f"\x1b[<0;{output.columns - 3};{y}M\x1b[<0;{output.columns - 3};{y}m")
            await wait_until(lambda: any("上下文本地估算" in item.content for item in ui.state.messages))
            assert len(ui.runner._provider.requests) == calls
            pipe.send_text("/new\r")
            await wait_until(lambda: runner.session_id is None and not ui.state.busy)
            await asyncio.sleep(0.3)
            assert ui._sample_context().total < reading.total
            assert ui.context_meter.cache_percent is None
            pipe.send_text("/exit\r")
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.parametrize("theme", ["ansi", "dark", "light"])
def test_user_orange_and_code_selection_style_are_independent(theme):
    style = theme_style(theme)
    assert style.get_attrs_for_style_str("class:input").color == style.get_attrs_for_style_str("class:message.user").color
    assert style.get_attrs_for_style_str("class:message.assistant").color != style.get_attrs_for_style_str("class:message.user").color
