"""录制代理：转发、usage 补注与剥离、重试与密钥不外泄（假上游，不联网）。"""

import json

import pytest
import pytest_asyncio
from agent_bench.config import ModelSettings, ProxySettings, RunSettings
from agent_bench.models import ModelCallRecord
from agent_bench.proxy.recorder import ProxyRecorder
from agent_bench.proxy.server import RecordingProxy, make_scope
from aiohttp import ClientSession, web

API_KEY = "sk-test-secret-value-123456"


class Upstream:
    """最简假的 DeepSeek 上游：支持普通与流式响应。"""

    def __init__(self):
        self.requests = []
        self.fail_times = 0
        self.seen_auth = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        self.seen_auth.append(request.headers.get("Authorization", ""))
        if self.fail_times > 0:
            self.fail_times -= 1
            return web.json_response({"error": {"message": "rate limited"}}, status=429)
        if body.get("stream"):
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await response.prepare(request)
            chunks = [
                {"choices": [{"index": 0, "delta": {"content": "你"}}]},
                {"choices": [{"index": 0, "delta": {"content": "好"}}]},
            ]
            if body.get("stream_options", {}).get("include_usage"):
                chunks.append({"choices": [], "usage": {
                    "prompt_tokens": 11, "completion_tokens": 2,
                    "prompt_cache_hit_tokens": 4}})
            for chunk in chunks:
                await response.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response
        return web.json_response({
            "choices": [{"message": {"content": "回答"}}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 3,
                      "prompt_cache_hit_tokens": 5},
        })


@pytest_asyncio.fixture
async def stack():
    upstream = Upstream()
    app = web.Application()
    app.router.add_post("/chat/completions", upstream.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    recorder = ProxyRecorder("run-1")
    proxy = RecordingProxy(
        run_id="run-1", proxy=ProxySettings(), run=RunSettings(retries=2, retry_backoff_seconds=0),
        model=ModelSettings(base_url=f"http://127.0.0.1:{port}"), api_key=API_KEY,
        recorder=recorder,
    )
    await proxy.start()
    try:
        yield proxy, recorder, upstream
    finally:
        await proxy.stop()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_non_stream_usage_and_key_injection(stack):
    proxy, recorder, upstream = stack
    scope = make_scope("fake", "case-1")
    await proxy.set_turn(scope, 2)
    async with ClientSession() as session, session.post(f"{proxy.scope_url(scope)}/chat/completions",
                                json={"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                      "tools": [{"type": "function"}]}) as response:
            assert response.status == 200
            payload = await response.json()
    assert payload["choices"][0]["message"]["content"] == "回答"
    (record,) = recorder.records
    assert record.agent == "fake" and record.case_id == "case-1" and record.turn == 2
    assert record.prompt_tokens == 21 and record.completion_tokens == 3
    assert record.cached_prompt_tokens == 5
    assert record.tool_definitions == 1
    assert record.usage_injected is False
    assert API_KEY in upstream.seen_auth[0]
    assert API_KEY not in json.dumps(record.model_dump(mode="json"))


@pytest.mark.asyncio
async def test_stream_injects_usage_then_strips_the_extra_chunk(stack):
    proxy, recorder, upstream = stack
    scope = make_scope("fake", "case-2")
    lines = []
    async with ClientSession() as session, session.post(f"{proxy.scope_url(scope)}/chat/completions",
                                json={"model": "m", "stream": True,
                                      "messages": [{"role": "user", "content": "hi"}]}) as response:
            async for raw in response.content:
                lines.append(raw.decode())
    text = "".join(lines)
    assert "你" in text and "好" in text
    assert "usage" not in text  # 补注换来的 usage 分片被剥离
    assert upstream.requests[0]["stream_options"]["include_usage"] is True
    (record,) = recorder.records
    assert record.usage_injected is True
    assert record.prompt_tokens == 11
    assert record.ttft_ms is not None


@pytest.mark.asyncio
async def test_client_requested_usage_is_preserved(stack):
    proxy, recorder, upstream = stack
    scope = make_scope("fake", "case-3")
    lines = []
    async with ClientSession() as session, session.post(
        f"{proxy.scope_url(scope)}/chat/completions",
        json={"model": "m", "stream": True,
              "stream_options": {"include_usage": True},
              "messages": [{"role": "user", "content": "hi"}]},
    ) as response:
        async for raw in response.content:
            lines.append(raw.decode())
    text = "".join(lines)
    assert "usage" in text  # 客户端本来就要 usage，保留
    assert upstream.requests[0]["stream_options"]["include_usage"] is True
    (record,) = recorder.records
    assert record.usage_injected is False
    assert record.prompt_tokens == 11


@pytest.mark.asyncio
async def test_retries_are_counted(stack):
    proxy, recorder, upstream = stack
    upstream.fail_times = 2
    scope = make_scope("fake", "case-4")
    async with ClientSession() as session, session.post(f"{proxy.scope_url(scope)}/chat/completions",
                                json={"model": "m", "messages": []}) as response:
            assert response.status == 200
    (record,) = recorder.records
    assert record.retries == 2
    assert record.status == 200


@pytest.mark.asyncio
async def test_retries_exhausted_returns_429(stack):
    proxy, recorder, upstream = stack
    upstream.fail_times = 5
    scope = make_scope("fake", "case-5")
    async with ClientSession() as session, session.post(f"{proxy.scope_url(scope)}/chat/completions",
                                json={"model": "m", "messages": []}) as response:
            assert response.status == 429
    (record,) = recorder.records
    assert record.status == 429
    assert record.retries == 2


@pytest.mark.asyncio
async def test_turn_control_endpoint(stack):
    proxy, recorder, _upstream = stack
    scope = make_scope("fake", "case-6")
    async with ClientSession() as session:
        async with session.put(f"{proxy.base_url}/__bench/turn",
                               json={"scope": scope, "turn": 7}) as response:
            assert response.status == 200
        async with session.get(f"{proxy.base_url}/__bench/health") as response:
            assert (await response.json())["ok"] is True
        async with session.post(f"{proxy.scope_url(scope)}/chat/completions",
                                json={"model": "m", "messages": []}) as response:
            await response.read()
    assert recorder.records[0].turn == 7


def test_recorder_cost_and_totals():
    from agent_bench.config import PriceSettings
    from agent_bench.proxy.recorder import cost_of

    records = [ModelCallRecord(run_id="r", agent="a", case_id="c", prompt_tokens=1000,
                               completion_tokens=100, cached_prompt_tokens=400)]
    price = PriceSettings(input_per_million=2.0, cached_input_per_million=0.5,
                          output_per_million=8.0)
    expected = (600 * 2.0 + 400 * 0.5 + 100 * 8.0) / 1_000_000
    assert abs(cost_of(records, price) - expected) < 1e-12
