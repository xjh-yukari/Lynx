"""MCP 配置、Runner 生命周期、模型调用链和 CLI/TUI 集成。"""

import sys
from pathlib import Path

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.cli.app import LynxTUI
from lynx_agent.cli.layout import status_text
from lynx_agent.cli.state import UIState
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
    load_config,
)
from lynx_agent.mcp.config import MCPConfig, MCPServerConfig
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionStore

SERVER = Path(__file__).with_name("fixture_server.py").resolve()


class Provider(ModelProvider):
    def __init__(self):
        self.requests = []
        self.closed = False

    async def generate(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                content="",
                model="fake",
                finish_reason="tool_calls",
                tool_calls=[
                    ModelToolCall(
                        id="mcp1", name="mcp__demo__add", arguments='{"a":2,"b":3}'
                    ),
                ],
            )
        return ModelResponse(content="结果是5", model="fake", finish_reason="stop")

    async def aclose(self):
        self.closed = True


def config(tmp_path):
    return Config(
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")
        ),
        agent=AgentConfig(model=ModelConfig(name="fake")),
        mcp=MCPConfig(
            servers={
                "demo": MCPServerConfig(
                    enabled=True,
                    command=sys.executable,
                    args=[str(SERVER)],
                    cwd=str(tmp_path),
                    allowed_tools=["add"],
                    read_only_tools=["add"],
                )
            }
        ),
    )


@pytest.mark.asyncio
async def test_runner_calls_mcp_and_preserves_tool_trace(tmp_path):
    provider = Provider()
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(config(tmp_path), provider, store) as runner:
        assert runner.mcp.connections["demo"].state == "connected"
        reply = await runner.ask("计算2加3")
        assert reply.content == "结果是5"
        assert any(tool.name == "mcp__demo__add" for tool in provider.requests[0].tools)
        assert provider.requests[1].messages[-1].role == "tool"
        assert "5" in provider.requests[1].messages[-1].content
        session = store.load(runner.session_id)
        assert any(
            event.kind == "tool" and event.tool.call.name == "mcp__demo__add"
            for event in session.events
        )
        assert "只读" in await runner.mcp.command("tools")
    assert provider.closed
    assert not any(name.startswith("mcp__") for name in runner.tools.names)


@pytest.mark.asyncio
async def test_plain_commands_do_not_invoke_provider(
    tmp_path, monkeypatch, chat_inputs, capsys
):
    cfg = config(tmp_path)
    provider = Provider()
    runner = AgentRunner(cfg, provider)
    monkeypatch.setattr(commands, "load_config", lambda: cfg)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(
        ["/mcp", "/mcp tools", "/mcp disconnect demo", "/mcp connect demo", "/exit"]
    )
    await commands.chat()
    output = capsys.readouterr().out
    assert "connected" in output and "disconnected" in output
    assert "mcp__demo__add" in output
    assert not provider.requests


@pytest.mark.asyncio
async def test_tui_dispatch_commands_use_same_manager(tmp_path):
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    cfg = config(tmp_path)
    provider = Provider()
    async with AgentRunner(cfg, provider) as runner:
        app = LynxTUI(
            runner,
            cfg,
            SessionStore(tmp_path / "sessions"),
            input_stream=DummyInput(),
            output_stream=DummyOutput(),
        )
        await app._dispatch("/mcp tools")
        assert "mcp__demo__add" in app.state.messages[-1].content
        await app._dispatch("/mcp disconnect demo")
        assert "0/1" in runner.mcp.label
        assert "0/1" not in status_text(app.state)
        await app._dispatch("/mcp connect demo")
        assert "1/1" in runner.mcp.label
        assert "1/1" not in status_text(app.state)
    assert not provider.requests


def test_toml_loader_accepts_mcp(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n", encoding="utf-8"
    )
    (tmp_path / "config.toml").write_text(
        "[mcp.servers.demo]\nenabled = false\n"
        f"command = '{sys.executable}'\ncwd = '{tmp_path}'\n"
        "allowed_tools = ['add']\nread_only_tools = ['add']\n",
        encoding="utf-8",
    )
    cfg = load_config(env_path)
    assert not cfg.mcp.servers["demo"].enabled
    assert cfg.mcp.servers["demo"].allowed_tools == ("add",)


def test_status_without_mcp_remains_usable(tmp_path):
    state = UIState(project=tmp_path, model="fake")
    assert "MCP" not in status_text(state)
    assert "记忆" not in status_text(state)
    assert "权限" in status_text(state)
