"""重新启用缓存优化后，存量快照原样读取、发送和计数。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lynx_agent.agent.compaction import ContextCompactor
from lynx_agent.agent.context import AgentContext
from lynx_agent.agent.tokens import estimate_message_tokens, estimate_text_tokens
from lynx_agent.config.schema import AgentConfig, ContextConfig, ModelConfig
from lynx_agent.providers import ChatMessage, ModelRequest, ModelResponse
from lynx_agent.providers.deepseek import DeepSeekProvider
from lynx_agent.session import Session, SessionEvent, SessionStore


def test_saved_snapshot_survives_loading_and_is_sent_and_counted(tmp_path):
    user = ChatMessage(role="user", content="用户原文", runtime_context="优化期间保存的资料")
    session = Session(events=(SessionEvent(kind="turn", messages=(
        user, ChatMessage(role="assistant", content="原回答"),
    )),))
    store = SessionStore(tmp_path)
    store.save(session)
    restored = store.load(session.id)
    message = restored.context_history()[0]
    assert message.runtime_context == user.runtime_context
    assert message.content == "用户原文"
    assert DeepSeekProvider._messages(ModelRequest(messages=(message,))) == [{"role": "user", "content": message.request_content}]
    assert estimate_message_tokens(message) == estimate_text_tokens(message.request_content)


@pytest.mark.asyncio
async def test_compaction_includes_active_snapshot_as_history():
    provider = SimpleNamespace(generate=AsyncMock(return_value=ModelResponse(content="精简摘要", model="fake", finish_reason="stop")))
    config = AgentConfig(model=ModelConfig(name="fake"), context=ContextConfig(keep_recent_tokens=1))
    context = AgentContext("系统规则")
    context.commit_turn((ChatMessage(role="user", content="用户原文" * 500, runtime_context="优化期间保存的资料"),
                         ChatMessage(role="assistant", content="答复" * 500)))
    result = await ContextCompactor(provider, config).prepare(context, force=True)
    assert result is not None and provider.generate.await_count > 0
    assert any("优化期间保存的资料" in message.content
               for call in provider.generate.call_args_list for message in call.args[0].messages)
