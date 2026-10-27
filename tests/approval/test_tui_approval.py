"""TUI：独立审批选择、批量审批卡片、允许一次/本批次/拒绝和停止按钮。"""

import asyncio

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import ModelProvider
from lynx_agent.session import SessionStore
from lynx_agent.tools.builtin.batch import BatchReplaceTextTool
from lynx_agent.web.setup import create_default_tools


class RecordingOutput(DummyOutput):
    def __init__(self, columns=110, rows=34):
        self.columns = columns
        self.rows = rows
        self.text = ""

    def get_size(self):
        return Size(rows=self.rows, columns=self.columns)

    def write(self, data):
        self.text += data

    def write_raw(self, data):
        self.write(data)


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


async def click_label(ui, pipe, label):
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


class QuietProvider(ModelProvider):
    __test__ = False

    async def generate(self, request):
        raise AssertionError("审批测试不应请求模型")

    async def aclose(self):
        pass


async def build_ui(tmp_path, *, columns=110):
    root = tmp_path / "workspace"
    root.mkdir()
    access = FileAccess(root)
    settings = config()
    store = SessionStore(tmp_path / "sessions")
    runner = AgentRunner(settings, QuietProvider(), store,
                         tools=create_default_tools(settings, file_access=access))
    pipe_context = create_pipe_input()
    pipe = pipe_context.__enter__()
    ui = LynxTUI(runner, settings, store, input_stream=pipe,
                 output_stream=RecordingOutput(columns=columns, rows=34),
                 preferences_path=tmp_path / "prefs.json")
    task = asyncio.create_task(ui.run())
    await wait_until(lambda: ui.application.is_running)
    return runner, ui, pipe, task, pipe_context, root


@pytest.mark.asyncio
async def test_approval_selector_is_independent_from_file_permissions(tmp_path):
    runner, ui, pipe, task, pipe_context, _ = await build_ui(tmp_path)
    try:
        assert runner.approval_label == "常用操作自动"
        assert runner.permission_label == "工作区修改"
        await click_label(ui, pipe, "审批：自动 ▾")
        await wait_until(lambda: ui.state.approval_selector_open)
        await click_label(ui, pipe, "暂停执行")
        await wait_until(lambda: runner.approval_label == "暂停执行")
        assert runner.permission_label == "工作区修改"
        assert "batch_replace_text" not in runner.tools.names
        assert not ui.state.shell_enabled
        await click_label(ui, pipe, "审批：暂停 ▾")
        await wait_until(lambda: ui.state.approval_selector_open)
        await click_label(ui, pipe, "常用操作自动")
        await wait_until(lambda: runner.approval_label == "常用操作自动")
        assert runner.permission_label == "工作区修改"
        assert ui.state.shell_enabled
        pipe.send_text("\r")
        await wait_until(lambda: not ui.state.approval_selector_open)
        pipe.send_text("/exit\r")
        await asyncio.wait_for(task, timeout=3)
    finally:
        if not task.done():
            ui.application.exit()
            await task
        pipe_context.__exit__(None, None, None)


@pytest.mark.asyncio
async def test_batch_card_supports_batch_decision_and_paging(tmp_path):
    runner, ui, pipe, task, pipe_context, root = await build_ui(tmp_path)
    try:
        paths = []
        for index in range(15):
            path = root / f"{index:02d}.txt"
            path.write_text("hello", encoding="utf-8")
            paths.append(str(path))
        tool = BatchReplaceTextTool(runner.file_access, runner.approval)
        parsed = tool._parse({"paths": paths, "old_text": "hello", "new_text": "hi", "purpose": "分页审批"})
        _, plan = tool.build_plan(parsed)
        request = asyncio.create_task(runner.approval.authorize_batch(plan))
        await wait_until(lambda: ui.state.approval_open)
        assert ui.state.approval_supports_batch and "第 1/2 页" in ui.approval_editor.text
        pipe.send_text("\x1b[6~")
        await wait_until(lambda: runner.approval.pending is not None and runner.approval.pending.page == 1)
        pipe.send_text("b")
        decision = await asyncio.wait_for(request, timeout=3)
        assert decision.allowed and not decision.reused
        assert runner.approval.grants
        await wait_until(lambda: not ui.state.approval_open)
        assert (root / "00.txt").read_text(encoding="utf-8") == "hello"  # 只授权，未执行

        second = root / "second.txt"
        second.write_text("hello", encoding="utf-8")
        parsed = tool._parse({"paths": [str(second)], "old_text": "hello", "new_text": "hi", "purpose": "一次"})
        _, plan = tool.build_plan(parsed)
        request = asyncio.create_task(runner.approval.authorize_batch(plan))
        await wait_until(lambda: ui.state.approval_open)
        pipe.send_text("y")
        decision = await asyncio.wait_for(request, timeout=3)
        assert decision.allowed and decision.grant is not None
        assert decision.grant.identifier not in runner.approval.grants

        third = root / "third.txt"
        third.write_text("hello", encoding="utf-8")
        parsed = tool._parse({"paths": [str(third)], "old_text": "hello", "new_text": "hi", "purpose": "拒绝"})
        _, plan = tool.build_plan(parsed)
        request = asyncio.create_task(runner.approval.authorize_batch(plan))
        await wait_until(lambda: ui.state.approval_open)
        pipe.send_text("n")
        decision = await asyncio.wait_for(request, timeout=3)
        assert not decision.allowed
        assert str(third).casefold() in runner.approval.denied_paths

        ui.state.batch_running = True
        ui.application.invalidate()
        await asyncio.sleep(0.1)
        await click_label(ui, pipe, "[停止]")
        await wait_until(lambda: runner.approval.stop_event.is_set())
        pipe.send_text("/exit\r")
        await asyncio.wait_for(task, timeout=3)
    finally:
        if not task.done():
            ui.application.exit()
            await task
        pipe_context.__exit__(None, None, None)
