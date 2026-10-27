"""端到端：模型连续调用批量工具，批准一次后 Agent 循环内复用授权。"""

import json

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionStore
from lynx_agent.web.setup import create_default_tools


class BatchProvider(ModelProvider):
    __test__ = False

    def __init__(self):
        self.requests = []
        self.calls = 0

    async def generate(self, request):
        self.requests.append(request)
        if self.calls == 0:
            self.calls += 1
            return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(
                ModelToolCall(id="first", name="batch_replace_text", arguments=json.dumps({
                    "paths": [str(self.first), str(self.second)],
                    "old_text": "hello", "new_text": "hi", "purpose": "整批问候",
                })),
            ))
        if self.calls == 1:
            self.calls += 1
            return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(
                ModelToolCall(id="second", name="batch_replace_text", arguments=json.dumps({
                    "paths": [str(self.first), str(self.second)],
                    "old_text": "hi", "new_text": "yo", "purpose": "继续整批",
                })),
            ))
        return ModelResponse(content="批量替换完成。", model="fake", finish_reason="stop")

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_agent_batch_calls_use_one_authorization_and_complete(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    first = root / "a.txt"
    second = root / "b.txt"
    first.write_text("hello world hello", encoding="utf-8")
    second.write_text("hello", encoding="utf-8")
    settings = Config(agent=AgentConfig(model=ModelConfig(name="fake")), providers=ProvidersConfig(
        deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))
    provider = BatchProvider()
    provider.first, provider.second = first, second
    access = FileAccess(root)
    store = SessionStore(tmp_path / "sessions")
    runner = AgentRunner(settings, provider, store,
                         tools=create_default_tools(settings, file_access=access))
    cards = []

    def present(pending):
        if pending is not None:
            cards.append(pending)
            runner.approval.decide(pending.identifier, "batch")

    runner.approval.presenter = present
    response = await runner.ask("批量替换问候")
    assert response.content == "批量替换完成。"
    assert first.read_text(encoding="utf-8") == "yo world yo"
    assert second.read_text(encoding="utf-8") == "yo"
    assert len(cards) == 1
    traces = [item for item in runner.tool_traces if item.call.name == "batch_replace_text"]
    assert len(traces) == 2 and not any(item.is_error for item in traces)
