"""金额底栏与只读查询命令。"""

from pathlib import Path

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from test_tui import RecordingOutput, TestProvider, config

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.config import Config
from lynx_agent.session import SessionStore


@pytest.mark.asyncio
async def test_footer_and_cost_command(tmp_path):
    data = config().model_dump()
    data["agent"]["model"]["name"] = "deepseek-flash"
    data["providers"]["deepseek"]["base_url"] = "https://api.deepseek.com"
    cfg = Config.model_validate(data)
    provider = TestProvider()
    store = SessionStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        async with AgentRunner(cfg, provider, store) as runner:
            ui = LynxTUI(runner, cfg, store, project=Path(tmp_path), input_stream=pipe, output_stream=RecordingOutput(160))
            with set_app(ui.application):
                fragments = [control.text() for control in ui.application.layout.find_all_controls()
                             if callable(getattr(control, "text", None)) and getattr(control.text, "__name__", "") == "metrics_line"]
                assert len(fragments) == 1
                assert "≈¥0.0000" in fragment_list_to_text(fragments[0])
                cost_fragment = next(item for item in fragments[0] if len(item) > 2 and "≈¥" in item[1])
                cost_fragment[2](MouseEvent(Point(0, 0), MouseEventType.MOUSE_UP, MouseButton.LEFT, frozenset()))
                assert ui.state.usage_panel == "cost"
                assert "非账单/账户余额" in ui.usage_editor.text
                ui.close_usage()
                for width in (20, 40, 80):
                    from prompt_toolkit.utils import get_cwidth
                    ui.application.output.columns = width
                    control = next(control for control in ui.application.layout.find_all_controls()
                                   if callable(getattr(control, "text", None)) and getattr(control.text, "__name__", "") == "model_line")
                    assert get_cwidth(fragment_list_to_text(control.text())) <= width
                    metrics = next(control for control in ui.application.layout.find_all_controls()
                                   if callable(getattr(control, "text", None)) and getattr(control.text, "__name__", "") == "metrics_line")
                    assert get_cwidth(fragment_list_to_text(metrics.text())) <= width
                    assert "Token" in fragment_list_to_text(metrics.text())
                    assert "费用" in fragment_list_to_text(metrics.text()) or "¥" in fragment_list_to_text(metrics.text())
            await ui._dispatch("/cost")
            assert "非账单/账户余额" in ui.state.messages[-1].content
            assert not provider.requests
            assert runner.session_id is None
