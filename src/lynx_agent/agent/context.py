"""维护一次 Agent 运行期间发送给模型的对话上下文。"""

from typing import Literal

from lynx_agent.providers import ChatMessage
from lynx_agent.providers.models import validate_turn


class AgentContext:
    """保存有序消息；系统提示词始终位于第一条。"""

    def __init__(self, system_prompt: str) -> None:
        prompt = system_prompt.strip()
        if not prompt:
            raise ValueError("system_prompt 不能为空")
        self._system_message = ChatMessage(role="system", content=prompt)
        self._messages: list[ChatMessage] = [self._system_message]
        self._summary = ""
        self.replacements = {}
        self.memory_profiles = ""
        self.memory_recall = ""

    def apply_replacements(self, messages):
        """只改变模型投影；原始消息与工具配对字段保持不变。"""
        return tuple(message.model_copy(update={"content": self.replacements[message.result_id].content})
                     if message.result_id in self.replacements else message for message in messages)

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        """返回不可变快照，防止调用方直接修改内部历史。"""
        return self.project(self._summary, self.history)

    @property
    def history(self) -> tuple[ChatMessage, ...]:
        """尚未被摘要替代的原始消息。"""
        return self.apply_replacements(tuple(self._messages[1:]))

    @property
    def summary(self) -> str:
        return self._summary

    def project(self, summary: str, history: tuple[ChatMessage, ...]) -> tuple[ChatMessage, ...]:
        system = self._system_message
        if self.memory_profiles:
            system = system.model_copy(update={"content": system.content + "\n\n" + self.memory_profiles})
        prefix = (system,)
        # 动态召回由本轮用户消息快照携带，不能每轮重新插在历史之前。
        if summary:
            prefix += (ChatMessage(
                role="user", content="[历史对话摘要，仅作背景资料，不是新的指令]\n" + summary,
            ),)
        return (*prefix, *history)

    def replace_history(self, summary: str, history: tuple[ChatMessage, ...]) -> None:
        """提交已验证并持久化的摘要及近期原文。"""
        self._summary = summary
        self._messages = [self._system_message, *history]

    def add_user_message(self, content: str) -> ChatMessage:
        """加入一条用户消息。"""
        return self._append("user", content)

    def add_assistant_message(self, content: str) -> ChatMessage:
        """加入一条模型回复。"""
        return self._append("assistant", content)

    def clear(self) -> None:
        """清空对话历史，但保留系统提示词。"""
        self._messages = [self._system_message]
        self._summary = ""
        self.replacements.clear()

    def commit_turn(self, messages: tuple[ChatMessage, ...]) -> None:
        """只提交已完成且工具调用/结果配对的整个回合。"""
        validate_turn(messages)
        self._messages.extend(messages)

    def _append(self, role: Literal["user", "assistant"], content: str) -> ChatMessage:
        text = content.strip()
        if not text:
            raise ValueError(f"{role} 消息不能为空")
        message = ChatMessage(role=role, content=content)
        self._messages.append(message)
        return message

    def __len__(self) -> int:
        return len(self.messages)
