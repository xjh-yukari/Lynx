"""AgentContext 测试。"""

import pytest

from lynx_agent.agent import AgentContext


def test_context_preserves_message_order() -> None:
    context = AgentContext("You are Lynx.")

    context.add_user_message("你好")
    context.add_assistant_message("你好，我是 Lynx。")

    assert [message.role for message in context.messages] == [
        "system",
        "user",
        "assistant",
    ]
    assert context.messages[1].content == "你好"


def test_messages_returns_immutable_snapshot() -> None:
    context = AgentContext("You are Lynx.")
    snapshot = context.messages

    context.add_user_message("新消息")

    assert len(snapshot) == 1
    assert len(context.messages) == 2


def test_clear_keeps_system_prompt() -> None:
    context = AgentContext("You are Lynx.")
    context.add_user_message("临时消息")

    context.clear()

    assert len(context) == 1
    assert context.messages[0].role == "system"
    assert context.messages[0].content == "You are Lynx."


@pytest.mark.parametrize("content", ["", "   ", "\n"])
def test_context_rejects_empty_messages(content: str) -> None:
    context = AgentContext("You are Lynx.")

    with pytest.raises(ValueError, match="消息不能为空"):
        context.add_user_message(content)


def test_context_rejects_empty_system_prompt() -> None:
    with pytest.raises(ValueError, match="system_prompt 不能为空"):
        AgentContext("  ")
