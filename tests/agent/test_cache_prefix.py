"""离线验证请求前缀、动态快照和真实预算，不请求服务端缓存。"""

from datetime import UTC, datetime

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.cli.projection import session_messages
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.providers.deepseek import DeepSeekProvider
from lynx_agent.session import SessionStore
from lynx_agent.tools import Tool, ToolRegistry
from lynx_agent.tools.builtin.filesystem import create_file_tools


class SearchTool(Tool):
    name = "web_search"
    description = "离线模拟工具"
    read_only = True

    @property
    def parameters(self):
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **arguments):
        return "离线结果"


class Provider(ModelProvider):
    def __init__(self, replies=()):
        self.replies = iter(replies)
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return next(self.replies, ModelResponse(content="答复", model="fake", finish_reason="stop"))

    async def aclose(self):
        pass


def settings():
    return Config(agent=AgentConfig(model=ModelConfig(name="fake")), providers=ProvidersConfig(
        deepseek=ProviderConfig(api_key="unused", base_url="https://example.invalid")))


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            return datetime(2026, 10, 19, 12, 0, cls.calls, tzinfo=UTC)
    monkeypatch.setattr("lynx_agent.agent.loop.datetime", Clock)
    return Clock


@pytest.mark.asyncio
async def test_tool_turn_prefix_memory_changes_and_restore(tmp_path, clock):
    provider = Provider((ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(
        ModelToolCall(id="search-1", name="web_search", arguments="{}"),)),))
    store, cfg = SessionStore(tmp_path / "sessions"), settings()
    async with AgentRunner(cfg, provider, store, tools=ToolRegistry([SearchTool()])) as runner:
        runner.context.memory_recall = "[相关长期记忆]召回 A"
        await runner.ask("问题一")
        assert clock.calls == 1 and len(provider.requests) == 2
        first, followup = [DeepSeekProvider._messages(request) for request in provider.requests]
        assert first == followup[:len(first)]
        assert "当前本机时间" not in first[0]["content"] and "召回 A" not in first[0]["content"]
        assert "当前本机时间" in first[1]["content"] and "召回 A" in first[1]["content"]
        for _ in range(20):
            _ = runner.runtime_context
        assert clock.calls == 1
        runner.context.memory_recall = "[相关长期记忆]召回 B"
        await runner.ask("问题二")
        assert clock.calls == 2
        second = DeepSeekProvider._messages(provider.requests[-1])
        assert followup == second[:len(followup)]
        assert "召回 B" in second[-1]["content"] and "召回 B" not in second[1]["content"]
        saved = store.load(runner.session_id)
        assert saved.context_history()[0].content == "问题一"
        assert next(message.content for message in session_messages(saved) if message.role == "你") == "问题一"
    restored_provider = Provider()
    async with AgentRunner(cfg, restored_provider, store, saved, tools=ToolRegistry([SearchTool()])) as runner:
        runner.context.memory_recall = "[相关长期记忆]召回 C"
        await runner.ask("第三问")
        third = DeepSeekProvider._messages(restored_provider.requests[-1])
        assert second == third[:len(second)] and clock.calls == 3


@pytest.mark.asyncio
async def test_rewrite_reuses_snapshot(tmp_path, clock):
    provider = Provider((ModelResponse(content="截断草稿", model="fake", finish_reason="length"),))
    async with AgentRunner(settings(), provider, SessionStore(tmp_path), tools=ToolRegistry([SearchTool()])) as runner:
        await runner.ask("问题")
        assert len(provider.requests) == 2 and clock.calls == 1
        first, rewritten = [DeepSeekProvider._messages(request) for request in provider.requests]
        assert first == rewritten[:len(first)]
        assert runner.context.history[0].content == "问题"


@pytest.mark.asyncio
async def test_snapshot_is_checked_before_api_call(tmp_path):
    provider = Provider()
    async with AgentRunner(settings(), provider, SessionStore(tmp_path)) as runner:
        runner.context.memory_recall = "x" * 300000
        with pytest.raises(ContextError):
            await runner.ask("短问题")
        assert not provider.requests and not runner.context.history


@pytest.mark.asyncio
async def test_permissions_remain_live(tmp_path, clock):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    access, provider = FileAccess(workspace), Provider()
    tools = ToolRegistry([SearchTool(), *create_file_tools(access)])
    async with AgentRunner(settings(), provider, SessionStore(tmp_path / "sessions"), tools=tools) as runner:
        await runner.ask("第一轮")
        snapshot = runner.context.history[0].runtime_context
        assert "工作区修改" in snapshot
        runner.set_file_mode("read")
        await runner.ask("第二轮")
        assert "仅查看" in provider.requests[-1].messages[-1].runtime_context
        assert runner.context.history[0].runtime_context == snapshot
        assert not access.policy.writable and clock.calls == 2


@pytest.mark.asyncio
async def test_retry_refreshes_runtime_without_executing_tools(tmp_path, clock):
    provider = Provider((ModelResponse(content="", model="fake", finish_reason="length"),))
    async with AgentRunner(settings(), provider, SessionStore(tmp_path), tools=ToolRegistry([SearchTool()])) as runner:
        await runner.ask("未完成问题")
        old = provider.requests[0].messages[-1].runtime_context
        assert runner.can_retry and clock.calls == 1
        await runner.retry()
        assert clock.calls == 2 and len(provider.requests) == 2
        assert not provider.requests[-1].tools
        assert runner.context.history[0].content == "未完成问题"
        assert runner.context.history[0].runtime_context != old
