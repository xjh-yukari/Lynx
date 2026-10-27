"""录制代理的 Anthropic 分支：dsh 走 Messages 协议，用量字段与 OpenAI 不同。"""

import json

import pytest
import pytest_asyncio
from agent_bench.config import ModelSettings, ProxySettings, RunSettings
from agent_bench.proxy.recorder import ProxyRecorder
from agent_bench.proxy.server import RecordingProxy, detect_protocol, make_scope
from aiohttp import ClientSession

from tests.fake_anthropic import FakeAnthropic


def test_protocol_detection():
    assert detect_protocol("v1/messages") == "anthropic"
    assert detect_protocol("/v1/messages/") == "anthropic"
    assert detect_protocol("chat/completions") == "openai"
    assert detect_protocol("") == "openai"


@pytest_asyncio.fixture
async def stack():
    upstream = await FakeAnthropic().start()
    recorder = ProxyRecorder("run-a")
    proxy = RecordingProxy(
        run_id="run-a", proxy=ProxySettings(),
        run=RunSettings(retries=0, retry_backoff_seconds=0),
        model=ModelSettings(base_url=upstream.base_url,
                            anthropic_base_url=upstream.base_url),
        api_key="sk-test", recorder=recorder,
    )
    await proxy.start()
    try:
        yield proxy, recorder, upstream
    finally:
        await proxy.stop()
        await upstream.stop()


@pytest.mark.asyncio
async def test_anthropic_stream_usage_is_recorded(stack):
    proxy, recorder, upstream = stack
    scope = make_scope("dsh", "case-1")
    await proxy.set_turn(scope, 1)
    lines = []
    async with ClientSession() as session, session.post(
        f"{proxy.scope_url(scope)}/v1/messages",
        json={"model": "deepseek-flash", "stream": True, "max_tokens": 64,
              "tools": [{"name": "mcp__bench_env__notes_set", "description": "存笔记",
                         "input_schema": {"type": "object", "properties": {}}}],
              "messages": [{"role": "user", "content": "把结论存进笔记库：key 用 k1，内容是 AB-12"}]},
    ) as response:
        assert response.status == 200
        async for raw in response.content:
            lines.append(raw.decode())
    text = "".join(lines)
    assert "tool_use" in text and "input_json_delta" in text
    (record,) = recorder.records
    assert record.agent == "dsh" and record.case_id == "case-1" and record.turn == 1
    # Anthropic 的 input_tokens 不含缓存：总输入 = input + cache_read + cache_creation。
    assert record.cached_prompt_tokens == 300
    assert record.prompt_tokens is not None and record.prompt_tokens > 300
    assert record.cached_prompt_tokens <= record.prompt_tokens
    assert record.completion_tokens == 25
    assert record.usage_injected is False  # Anthropic 不补注 stream_options
    assert record.ttft_ms is not None
    # 上游收到的请求没有被塞进 OpenAI 专有字段。
    assert "stream_options" not in upstream.requests[0]


@pytest.mark.asyncio
async def test_anthropic_non_stream_usage(stack):
    proxy, recorder, upstream = stack
    scope = make_scope("dsh", "case-2")
    async with ClientSession() as session, session.post(
        f"{proxy.scope_url(scope)}/v1/messages",
        json={"model": "deepseek-flash", "max_tokens": 64,
              "messages": [{"role": "user", "content": "你好"}]},
    ) as response:
        payload = await response.json()
    assert payload["content"][0]["type"] == "text"
    (record,) = recorder.records
    assert record.prompt_tokens == (upstream.input_tokens
                                    + len(json.dumps(upstream.requests[0]["messages"],
                                                     ensure_ascii=False)) // 4
                                    + upstream.cache_read_tokens)
    assert record.completion_tokens == 25


@pytest.mark.asyncio
async def test_anthropic_uses_its_own_upstream_root(stack):
    proxy, _recorder, upstream = stack
    scope = make_scope("dsh", "case-3")
    async with ClientSession() as session, session.post(
        f"{proxy.scope_url(scope)}/v1/messages",
        json={"model": "deepseek-flash", "max_tokens": 8,
              "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        await response.read()
    # 上游是假服务，能收到请求说明 /p/<scope>/v1/messages 被正确转发。
    assert upstream.requests


@pytest.mark.asyncio
async def test_anthropic_tool_result_round_trip(stack):
    """模拟 dsh 的第二轮：把工具结果放回最后一条 user 消息。"""
    proxy, recorder, _upstream = stack
    scope = make_scope("dsh", "case-4")
    body = {
        "model": "deepseek-flash", "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "查一下 release-summary"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "mcp__bench_env__notes_get",
                 "input": {"key": "release-summary"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "{\"value\": \"AB-12\"}"}]},
        ],
    }
    async with ClientSession() as session, session.post(
        f"{proxy.scope_url(scope)}/v1/messages", json=body) as response:
        payload = await response.json()
    assert "AB-12" in payload["content"][0]["text"]
    (record,) = recorder.records
    assert record.messages == 3
    assert record.tool_definitions == 0
