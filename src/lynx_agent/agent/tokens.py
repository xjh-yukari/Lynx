"""本地启发式 Token 估算，不等同于供应商 tokenizer。"""

import json
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil

from lynx_agent.providers import ChatMessage, ModelRequest

_MESSAGE_CACHE = OrderedDict()
_MESSAGE_CACHE_LIMIT = 4096


def _discard_message(identifier, reference):
    cached = _MESSAGE_CACHE.get(identifier)
    if cached is not None and cached[0] is reference:
        _MESSAGE_CACHE.pop(identifier, None)


def estimate_text_tokens(text: str) -> int:
    """粗估：每 4 个 ASCII 字符约 1 Token，其他字符每个约 1.5 Token。

    这是用于观察上下文增长的经验权重，非 DeepSeek 官方计数；
    中文、代码、标点、emoji 和不同模型均可能产生明显误差。
    """
    # ASCII 快路径避免逐字符 Python 循环；混合文本计数同样保持原估算权重。
    ascii_count = len(text) if text.isascii() else len(text.encode("ascii", errors="ignore"))
    return ceil(ascii_count / 4 + (len(text) - ascii_count) * 1.5)


def estimate_message_tokens(message: ChatMessage) -> int:
    # 消息不可变；按对象身份弱引用缓存，不保留正文，也不改变消息相等性/序列化。
    identifier = id(message)
    cached = _MESSAGE_CACHE.get(identifier)
    if cached is not None and cached[0]() is message:
        _MESSAGE_CACHE.move_to_end(identifier)
        return cached[1]
    extra = ""
    if message.tool_calls:
        extra += json.dumps([call.model_dump() for call in message.tool_calls], ensure_ascii=False)
    if message.tool_call_id:
        extra += message.tool_call_id
    if message.reasoning_content:
        extra += message.reasoning_content
    value = estimate_text_tokens(message.request_content + extra)
    reference = weakref.ref(message, lambda ref: _discard_message(identifier, ref))
    _MESSAGE_CACHE[identifier] = (reference, value)
    if len(_MESSAGE_CACHE) > _MESSAGE_CACHE_LIMIT:
        _MESSAGE_CACHE.popitem(last=False)
    return value


def estimate_tool_tokens(tools) -> int:
    return estimate_text_tokens(json.dumps([tool.to_openai_schema() for tool in tools], ensure_ascii=False)) + 16 if tools else 0


def estimate_request_tokens(request: ModelRequest) -> int:
    return estimate_context_tokens(request.messages).total + estimate_tool_tokens(request.tools)


class TokenBudget:
    """每个会话内只向上校正的预算估算；不替代或改写供应商实际 usage。"""

    def __init__(self, enabled=False):
        self.enabled = enabled
        self.factor = 1.0

    def reset(self):
        self.factor = 1.0

    def scale(self, value):
        return ceil(value * self.factor)

    def text(self, text):
        return self.scale(estimate_text_tokens(text))

    def context(self, messages, pending=None):
        return self.scale(estimate_context_tokens(messages, pending).total)

    def tools(self, tools):
        return self.scale(estimate_tool_tokens(tools))

    def request(self, request):
        return self.scale(estimate_request_tokens(request))

    def observe(self, request, usage):
        if not self.enabled or usage is None or usage.prompt_tokens <= 0:
            return False
        local = estimate_request_tokens(request)
        if local <= 0 or usage.prompt_tokens <= self.scale(local):
            return False
        self.factor = max(self.factor, usage.prompt_tokens / local * 1.05)
        return True


@dataclass(frozen=True)
class TokenEstimate:
    """各部分文本及消息封装的估算量。"""

    system: int
    history: int
    pending: int
    framing: int
    message_count: int

    @property
    def total(self) -> int:
        return self.system + self.history + self.pending + self.framing


def estimate_context_tokens(
    messages: Sequence[ChatMessage], pending_input: str | None = None,
) -> TokenEstimate:
    """估算消息快照；待发送输入只参与计算，不写入上下文。"""
    system = sum(estimate_message_tokens(item) for item in messages if item.role == "system")
    history = sum(estimate_message_tokens(item) for item in messages if item.role != "system")
    count = len(messages) + int(pending_input is not None)
    return TokenEstimate(
        system=system,
        history=history,
        pending=estimate_text_tokens(pending_input) if pending_input is not None else 0,
        framing=count * 4 + 3,
        message_count=count,
    )
