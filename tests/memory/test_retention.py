"""提取安全边界：不连接网络或用户记忆库。"""

import json

import pytest
from pydantic import ValidationError

from lynx_agent.agent.usage import UsageTracker
from lynx_agent.memory.errors import MemoryError
from lynx_agent.memory.retention import eligible, extract
from lynx_agent.providers import ModelResponse


class Provider:
    def __init__(self, content, finish="stop"):
        self.content = content
        self.finish = finish
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return ModelResponse(content=self.content, model="fake", finish_reason=self.finish)


@pytest.mark.parametrize("question,answer", [
    ("不要保存这次结论", "名称是霜降前哨。"),
    ("Don't remember this", "Named Dawn."),
    ("我的偏好是中文", "今后使用中文。"),
    ("更新 SOUL", "好的。"),
    ("a" * 4001, "done"),
    ("问题", "a" * 16001),
])
def test_skip_opt_out_profiles_and_oversized(question, answer):
    assert not eligible(question, answer)


@pytest.mark.parametrize("secret", ["DEEPSEEK_API_KEY=sk-abcdefghijklmnop", "test@example.com", "13812345678"])
def test_sensitive_inputs_rejected(secret):
    with pytest.raises(MemoryError):
        eligible("任务结论", secret)


@pytest.mark.asyncio
async def test_extract_is_bounded_tool_free_and_tracks_usage():
    body = "北京本次降温命名为霜降前哨。"
    provider = Provider(json.dumps({"conclusions": [{"title": "北京降温命名", "body": body}]}))
    tracker = UsageTracker()
    items = await extract(provider, tracker, "给北京本次降温起名", body)
    assert items[0].body == body
    request = provider.requests[0]
    assert not request.tools and request.max_tokens == 1200
    assert len(request.messages) == 2
    assert tracker.report("memory").calls == 1


@pytest.mark.parametrize("content,finish", [
    ('{"conclusions":[{"title":"伪造","body":"不是原文"}]}', "stop"),
    ('{"conclusions":[{"title":"画像","body":"原文","kind":"user"}]}', "stop"),
    ('{"conclusions":[]}', "length"),
    ('```json\n{"conclusions":[]}\n```', "stop"),
])
@pytest.mark.asyncio
async def test_invalid_extraction_rejected(content, finish):
    with pytest.raises((MemoryError, ValidationError)):
        await extract(Provider(content, finish), UsageTracker(), "任务", "原文")
