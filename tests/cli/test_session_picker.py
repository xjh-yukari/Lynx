"""通过真实输入管道验证选择器按键。"""

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lynx_agent.cli.session_picker import NEW_SESSION, build_picker
from lynx_agent.session import Session


@pytest.mark.asyncio
@pytest.mark.parametrize("keys, expected", [
    ("\r", "new"),
    ("\x1b[B\r", "first"),
    ("\x1b[B\x1b[B\x1b[A\r", "first"),
    ("\x1b", "cancel"),
])
async def test_selection_keys(keys, expected):
    session = Session(title="中文历史会话")
    with create_pipe_input() as pipe:
        application = build_picker([session], input_stream=pipe, output_stream=DummyOutput())
        pipe.send_text(keys)
        result = await application.run_async()
    assert result == {"new": NEW_SESSION, "first": str(session.id), "cancel": None}[expected]
