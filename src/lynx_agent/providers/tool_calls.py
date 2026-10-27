"""拼装完整流式工具调用；中途片段绝不执行。"""

from lynx_agent.providers.errors import ProviderResponseError
from lynx_agent.providers.models import ModelToolCall, ToolCallDelta


class ToolCallAccumulator:
    def __init__(self):
        self.parts: dict[int, dict[str, str]] = {}
        self.size = 0

    def add(self, delta: ToolCallDelta) -> None:
        self.size += len(delta.id) + len(delta.name) + len(delta.arguments)
        if self.size > 262144:
            raise ProviderResponseError("工具调用参数增量超过安全上限")
        part = self.parts.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
        for key in part:
            part[key] += getattr(delta, key)

    def finish(self) -> tuple[ModelToolCall, ...]:
        calls = []
        if sorted(self.parts) != list(range(len(self.parts))):
            raise ProviderResponseError("工具调用增量索引不连续")
        for index in sorted(self.parts):
            part = self.parts[index]
            if not part["id"] or not part["name"]:
                raise ProviderResponseError("流式工具调用缺少 ID 或名称")
            calls.append(ModelToolCall(**part))
        if len({call.id for call in calls}) != len(calls):
            raise ProviderResponseError("工具调用 ID 重复")
        return tuple(calls)
