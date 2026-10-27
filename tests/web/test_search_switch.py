import asyncio

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.formatting import format_tools
from lynx_agent.cli.search import handle_search_command
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import ModelProvider, ModelResponse
from lynx_agent.session import SessionStore
from lynx_agent.tools import ToolCall
from lynx_agent.web.setup import create_default_tools


def settings(**web):
    return Config(agent=AgentConfig(model=ModelConfig(name="fake")),
                  providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")),
                  web=web)


class Provider(ModelProvider):
    def __init__(self):
        self.requests = []
        self.gate = None

    async def generate(self, request):
        self.requests.append(request)
        if self.gate:
            await self.gate.wait()
        return ModelResponse(content="离线测试回答", model="fake", finish_reason="stop")

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_toggle_changes_definitions_and_execution_preserves_other_tools_and_sessions(tmp_path):
    config = settings()
    provider = Provider()
    store = SessionStore(tmp_path)
    registry = create_default_tools(config)
    search_tool = registry.get("web_search")
    async with AgentRunner(config, provider, store, tools=registry) as runner:
        assert "已开启（anysearch）" in handle_search_command(runner)
        assert "已关闭" in handle_search_command(runner, "off")
        assert runner.search_backend == "已关闭"
        assert "web_search" not in registry.names
        assert "calculator" in registry.names and "web_fetch" in registry.names
        assert "通过 /search off 关闭" in format_tools(runner, config)
        assert "关键词搜索已关闭" in runner.runtime_context
        assert not store.list_sessions()
        result = await registry.execute(ToolCall(id="disabled", name="web_search", arguments={"query": "test"}))
        assert result.is_error
        await runner.ask("不联网回答")
        assert "web_search" not in [tool.name for tool in provider.requests[-1].tools]
        runner.switch_session(None)
        assert not runner.search_enabled
        handle_search_command(runner, "on")
        assert registry.get("web_search") is search_tool
        handle_search_command(runner, "on")
        assert {"calculator", "web_fetch", "web_search"}.issubset(registry.names)
        assert "关键词搜索已关闭" not in runner.runtime_context
    fresh = AgentRunner(config, Provider(), tools=create_default_tools(config))
    assert fresh.search_enabled
    await fresh.aclose()


@pytest.mark.asyncio
async def test_enable_from_config_off_and_missing_key_does_not_change_state():
    config = settings(search={"enabled": False})
    async with AgentRunner(config, Provider(), tools=create_default_tools(config)) as runner:
        assert not runner.search_enabled
        handle_search_command(runner, "on")
        assert runner.search_enabled
        assert not config.web.search.enabled
        with pytest.raises(ValueError, match="用法"):
            handle_search_command(runner, "invalid")
        assert runner.search_enabled
    config = settings(search={"provider": "brave"})
    async with AgentRunner(config, Provider(), tools=create_default_tools(config)) as runner:
        with pytest.raises(ValueError, match="BRAVE_SEARCH_API_KEY"):
            handle_search_command(runner, "on")
        assert not runner.search_enabled


@pytest.mark.asyncio
async def test_toggle_during_request_rejected_and_after_cancel_allowed(tmp_path):
    config = settings()
    provider = Provider()
    provider.gate = asyncio.Event()
    async with AgentRunner(config, provider, SessionStore(tmp_path), tools=create_default_tools(config)) as runner:
        task = asyncio.create_task(runner.ask("等待测试"))
        try:
            async with asyncio.timeout(2):
                while not provider.requests:
                    await asyncio.sleep(0)
            with pytest.raises(ValueError, match="先等待或取消"):
                runner.set_search_enabled(False)
            assert runner.search_enabled
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        runner.set_search_enabled(False)
        assert not runner.search_enabled
