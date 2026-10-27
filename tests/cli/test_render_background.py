"""模拟终端默认黑底，检查真实 renderer 重绘后是否残留未绘制色块。"""

import asyncio

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.styles import DEFAULT_ATTRS
from prompt_toolkit.utils import get_cwidth
from test_effort import EffortProvider
from test_tui import RecordingOutput, config, wait_until

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.state import DisplayMessage
from lynx_agent.session import SessionStore


class RasterOutput(RecordingOutput):
    """按 Output 操作维护背景栅格；擦除使用终端当前颜色，而非应用主题。"""

    def __init__(self, columns=100, rows=32):
        super().__init__(columns, rows)
        self.x = self.y = 0
        self.attrs = DEFAULT_ATTRS
        self.pixels = {}

    def set_attributes(self, attrs, color_depth):
        self.attrs = attrs

    def reset_attributes(self):
        self.attrs = DEFAULT_ATTRS

    def write(self, data):
        self.text += data
        for char in data:
            if char == "\r":
                self.x = 0
            elif char == "\n":
                self.y = min(self.rows - 1, self.y + 1)
            else:
                width = get_cwidth(char)
                for offset in range(width):
                    if 0 <= self.x + offset < self.columns:
                        self.pixels[self.y, self.x + offset] = self.attrs.bgcolor or "terminal-default"
                self.x += width

    def write_raw(self, data):
        # Renderer 的游标、颜色与清屏操作通过对应 Output 方法记录。
        pass

    def cursor_forward(self, amount):
        self.x += amount

    def cursor_backward(self, amount):
        self.x = max(0, self.x - amount)

    def cursor_up(self, amount):
        self.y = max(0, self.y - amount)

    def cursor_goto(self, row=0, column=0):
        self.y, self.x = row, column

    def erase_down(self):
        for y in range(self.y, self.rows):
            for x in range(self.x if y == self.y else 0, self.columns):
                self.pixels[y, x] = self.attrs.bgcolor or "terminal-default"

    def erase_end_of_line(self):
        for x in range(self.x, self.columns):
            self.pixels[self.y, x] = self.attrs.bgcolor or "terminal-default"

    def assert_surface_painted(self, extra_backgrounds=()):
        allowed = {"20242a", *extra_backgrounds}
        holes = [(y, x) for y in range(self.rows) for x in range(self.columns)
                 if self.pixels.get((y, x), "terminal-default") not in allowed]
        assert not holes, f"{len(holes)} 个背景单元格未绘制，示例：{holes[:12]}"


@pytest.mark.asyncio
async def test_background_survives_blank_rows_shorter_reply_and_resize(tmp_path):
    output, store = RasterOutput(), SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(config(), EffortProvider(), store) as runner:
            ui = LynxTUI(runner, config(), store, input_stream=pipe, output_stream=output,
                         preferences_path=tmp_path / "prefs.json")
            ui.state.messages[:] = [DisplayMessage("你", "天气如何？"), DisplayMessage("Lynx", (
                "北京今天阴转小雨。\n\n补充信息：\n\n"
                "- 白天微风。\n- 明天气温下降。\n\n"
                "来源：https://www.example.com/weather/" + "long-path/" * 12
            ))]
            task = asyncio.create_task(ui.run())
            try:
                await wait_until(lambda: ui.application.renderer._last_screen is not None)
                output.assert_surface_painted()
                ui.state.messages[-1].content = "\n\n".join(f"第 {index} 行测试中文滚动。" for index in range(40))
                ui.application.invalidate()
                await asyncio.sleep(0.2)
                ui.scroll_lines(-6)
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                ui.jump_to_latest()
                ui.state.messages[-1].content = "简短回复。"
                ui.application.invalidate()
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                pipe.send_text("保留草稿")
                await wait_until(lambda: ui.editor.text == "保留草稿")
                ui.open_effort()
                await asyncio.sleep(0.2)
                ui.close_effort()
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                ui.toggle_usage("tokens")
                await asyncio.sleep(0.2)
                # TextArea 的滚动条有明确的轨道/滑块背景，不是透明色块。
                output.assert_surface_painted(extra_backgrounds=("aaaaaa", "444444"))
                ui.close_usage()
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                ui.state.picker_open = True
                ui.application.layout.focus(ui.search_editor)
                ui.application.invalidate()
                await asyncio.sleep(0.2)
                ui.close_picker()
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                output.columns, output.rows = 72, 26
                ui.application.invalidate()
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                output.columns, output.rows = 110, 36
                ui.application.invalidate()
                await asyncio.sleep(0.2)
                output.assert_surface_painted()
                assert ui.editor.text == "保留草稿"
            finally:
                ui.application.exit()
                await task
