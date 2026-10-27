"""检查泄漏到正文中的内部工具标记，并跨流式片段保留可疑前缀。"""

import re

from lynx_agent.providers.errors import ProviderToolMarkupError

_MARKER = re.compile(r"<[|｜\s]*[|｜][|｜\s]*DSML", re.IGNORECASE)
_PREFIX = re.compile(r"<[|｜\s]*(?:D(?:S(?:M(?:L)?)?)?)?", re.IGNORECASE)


def contains_tool_markup(text: str) -> bool:
    return _MARKER.search(text) is not None


def reject_tool_markup(text: str) -> None:
    if contains_tool_markup(text):
        raise ProviderToolMarkupError("模型将内部工具标记写入正文；已拦截，未执行或保存异常调用")


class TextOutputGuard:
    def __init__(self):
        self.pending = ""

    def feed(self, fragment: str) -> str:
        self.pending += fragment
        reject_tool_markup(self.pending)
        start = self.pending.rfind("<")
        if start >= 0 and _PREFIX.fullmatch(self.pending[start:]):
            safe, self.pending = self.pending[:start], self.pending[start:]
        else:
            safe, self.pending = self.pending, ""
        return safe

    def finish(self) -> str:
        reject_tool_markup(self.pending)
        if re.search(r"[|｜]\s*D", self.pending, re.IGNORECASE):
            raise ProviderToolMarkupError("模型返回了不完整的内部工具标记；已拦截，本轮未保存")
        safe, self.pending = self.pending, ""
        return safe
