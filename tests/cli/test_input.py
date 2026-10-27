"""使用终端按键与粘贴事件验证输入行为，不请求 API。"""

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.cli.input import ChatInput
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import ModelProvider, ModelResponse


@pytest.mark.asyncio
@pytest.mark.parametrize("keys, expected", [
    ("你好\r", "你好"),
    ("第一行\x1b\r第二行\r", "第一行\n第二行"),
    ("  保留空白  \r", "  保留空白  "),
])
async def test_enter_and_alt_enter(keys, expected):
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        pipe.send_text(keys)
        assert await asyncio.wait_for(reader.read(), timeout=3) == expected


@pytest.mark.asyncio
async def test_long_paste_waits_for_enter_and_preserves_text():
    text = "第一行\n" + "长文本、中文🙂与代码 print('hello')\n" * 1200 + "\n最后一行  "
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        inserted = asyncio.Event()

        def on_change(buffer):
            if buffer.text == text:
                inserted.set()

        reader.session.default_buffer.on_text_changed += on_change
        task = asyncio.create_task(reader.read())
        try:
            pipe.send_text("\x1b[200~" + text + "\x1b[201~")
            await asyncio.wait_for(inserted.wait(), timeout=3)
            assert not task.done()
            pipe.send_text("\r")
            assert await asyncio.wait_for(task, timeout=3) == text
        finally:
            if not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


@pytest.mark.asyncio
async def test_paste_normalizes_windows_newlines():
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        pipe.send_text("\x1b[200~甲\r\n乙\r丙\x1b[201~\r")
        assert await asyncio.wait_for(reader.read(), timeout=3) == "甲\n乙\n丙"


@pytest.mark.asyncio
async def test_ctrl_c_discards_draft_and_allows_next_input():
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        pipe.send_text("未发送的草稿\x03")
        with pytest.raises(KeyboardInterrupt):
            await reader.read()
        pipe.send_text("新消息\r")
        assert await asyncio.wait_for(reader.read(), timeout=3) == "新消息"
        assert "未发送的草稿" not in reader.session.history.get_strings()


@pytest.mark.asyncio
async def test_ctrl_d_exits_with_draft():
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        pipe.send_text("不发送的草稿\x04")
        with pytest.raises(EOFError):
            await reader.read()


@pytest.mark.asyncio
async def test_multiline_paste_goes_to_model_once_and_is_not_a_command(monkeypatch):
    config = Config(
        agent=AgentConfig(model=ModelConfig(name="fake")),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="unused", base_url="https://example.invalid",
        )),
    )
    provider = AsyncMock(spec=ModelProvider)
    provider.generate.return_value = ModelResponse(content="已收到", model="fake", finish_reason="stop")
    runner = AgentRunner(config, provider)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    text = "/clear\n" + "测试资料\n" * 1000 + "exit\n/compact\n  结束  "
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        monkeypatch.setattr(commands, "ChatInput", lambda: reader)
        original_read = reader.read
        read_count = 0

        async def read():
            nonlocal read_count
            read_count += 1
            if read_count == 2:
                pipe.send_text("/exit\r")
            return await original_read()

        monkeypatch.setattr(reader, "read", read)
        pipe.send_text("\x1b[200~" + text + "\x1b[201~\r")
        await asyncio.wait_for(commands.chat(), timeout=5)
    provider.generate.assert_awaited_once()
    assert provider.generate.call_args.args[0].messages[-1].content == text
    assert runner.context.history[0].content == text
    provider.aclose.assert_awaited_once()
