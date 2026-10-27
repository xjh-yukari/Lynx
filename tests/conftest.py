"""CLI 测试使用异步脚本输入，不访问真实终端。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def chat_inputs(monkeypatch):
    def install(values):
        async def run_operation(operation):
            return await operation(lambda phase: None)

        async def run_stream_operation(operation):
            return await operation(lambda phase: None, lambda text: None)

        reader = SimpleNamespace(
            read=AsyncMock(side_effect=values), run_operation=run_operation,
            run_stream_operation=run_stream_operation,
        )
        monkeypatch.setattr("lynx_agent.cli.commands.ChatInput", lambda: reader)
        return reader
    return install
