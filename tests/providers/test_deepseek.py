"""通过模拟 HTTP 测试真实 SDK 的请求和响应转换，不访问网络。"""

import asyncio
import json

import httpx2
import pytest
from openai import AsyncOpenAI

from lynx_agent.agent import AgentRunner
from lynx_agent.config.schema import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import (
    ChatMessage,
    ModelRequest,
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    create_provider,
)
from lynx_agent.session import SessionStore


def make_config(provider: str = "deepseek") -> Config:
    return Config(
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake-secret", base_url="https://example.invalid"),
        ),
        agent=AgentConfig(model=ModelConfig(provider=provider, name="test-model")),
    )


def completion() -> dict:
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("extra,expected", [
    ({"prompt_cache_hit_tokens": 3}, 3),
    ({"prompt_tokens_details": {"cached_tokens": 2}}, 2),
    ({"prompt_cache_hit_tokens": 1, "prompt_tokens_details": {"cached_tokens": 2}}, 1),
    ({"prompt_cache_hit_tokens": 0}, 0),
    ({}, None),
    ({"prompt_cache_hit_tokens": 1000}, None),
    ({"prompt_cache_hit_tokens": -1}, None),
])
async def test_cache_usage_from_real_sdk_transport(mock_transport, streaming, extra, expected):
    payload = completion()
    payload["usage"].update(extra)
    if streaming:
        body = SSEBody([sse_chunk("Hello"), sse_chunk(reason="stop"),
                        sse_chunk(usage=payload["usage"], empty_choices=True), b"data: [DONE]\n\n"])
        mock_transport(lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    else:
        mock_transport(lambda request: httpx2.Response(200, json=payload))
    async with create_provider(make_config()) as provider:
        request = ModelRequest(messages=[ChatMessage(role="user", content="Hi")])
        if streaming:
            events = [event async for event in provider.stream(request)]
            result = events[-1].response
        else:
            result = await provider.generate(request)
        assert result.usage.cached_prompt_tokens == expected
        assert result.usage.prompt_tokens == 4


@pytest.fixture
def mock_transport(monkeypatch):
    """为 Provider 创建的 SDK 客户端安装离线 transport。"""
    def install(handler):
        def client_factory(**kwargs):
            return AsyncOpenAI(
                **kwargs,
                http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
            )
        monkeypatch.setattr("lynx_agent.providers.deepseek.AsyncOpenAI", client_factory)
    return install


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("effort", ["off", "low", "high", "max"])
async def test_reasoning_effort_reaches_api_and_can_change(mock_transport, streaming, effort):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if streaming:
            stream = SSEBody([sse_chunk("Hello"), sse_chunk(reason="stop"), b"data: [DONE]\n\n"])
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)
        return httpx2.Response(200, json=completion())
    mock_transport(handler)
    cfg = make_config()
    cfg = cfg.model_copy(update={"agent": cfg.agent.model_copy(update={
        "model": cfg.agent.model.model_copy(update={"name": "deepseek-flash", "reasoning_effort": effort}),
    })})
    async with create_provider(cfg) as provider:
        assert provider.reasoning_effort == effort
        request = ModelRequest(messages=[ChatMessage(role="user", content="Hi")])
        if streaming:
            _ = [event async for event in provider.stream(request)]
        else:
            await provider.generate(request)
        assert bodies[-1]["thinking"]["type"] == ("disabled" if effort == "off" else "enabled")
        assert bodies[-1].get("reasoning_effort") == (None if effort == "off" else effort)
        provider.set_reasoning_effort("low")
        if streaming:
            _ = [event async for event in provider.stream(request)]
        else:
            await provider.generate(request)
        assert bodies[-1]["reasoning_effort"] == "low"
        assert bodies[-1]["thinking"]["type"] == "enabled"


@pytest.mark.asyncio
async def test_generate_converts_request_response_and_closes(mock_transport):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=completion())

    mock_transport(handler)
    async with create_provider(make_config()) as provider:
        result = await provider.generate(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))
        await provider.generate(ModelRequest(
            messages=[ChatMessage(role="user", content="Hi")], temperature=0, max_tokens=10,
        ))
    assert result.content == "Hello"
    assert result.usage.total_tokens == 6
    assert result.finish_reason == "stop"
    assert bodies[0]["messages"] == [{"role": "user", "content": "Hi"}]
    assert "max_tokens" not in bodies[0]
    assert bodies[1]["temperature"] == 0
    assert bodies[1]["max_tokens"] == 10
    assert provider._client.is_closed()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,error", [
    (401, ProviderAuthenticationError),
    (429, ProviderRateLimitError),
    (500, ProviderResponseError),
])
async def test_http_errors_are_translated_without_body(mock_transport, status, error):
    def handler(request):
        return httpx2.Response(status, json={"error": {"message": "private-server-detail"}})
    mock_transport(handler)
    async with create_provider(make_config()) as provider:
        with pytest.raises(error) as caught:
            await provider.generate(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))
    assert "private-server-detail" not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.asyncio
@pytest.mark.parametrize("network_error,error", [
    (httpx2.ReadTimeout, ProviderTimeoutError),
    (httpx2.ConnectError, ProviderConnectionError),
])
async def test_network_errors_are_translated(mock_transport, network_error, error):
    def handler(request):
        raise network_error("offline", request=request)
    mock_transport(handler)
    async with create_provider(make_config()) as provider:
        with pytest.raises(error):
            await provider.generate(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))


@pytest.mark.asyncio
async def test_empty_choices_are_rejected(mock_transport):
    payload = completion()
    payload["choices"] = []
    mock_transport(lambda request: httpx2.Response(200, json=payload))
    async with create_provider(make_config()) as provider:
        with pytest.raises(ProviderResponseError, match="choices"):
            await provider.generate(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))


def test_factory_rejects_unknown_provider():
    with pytest.raises(ProviderConfigurationError):
        create_provider(make_config("unknown"))


@pytest.mark.asyncio
async def test_nonstream_tool_markup_is_not_a_successful_response(mock_transport):
    payload = completion()
    payload["choices"][0]["message"]["content"] = '<｜｜DSML｜｜ calls>'
    mock_transport(lambda request: httpx2.Response(200, json=payload))
    async with create_provider(make_config()) as provider:
        with pytest.raises(ProviderResponseError, match="已拦截"):
            await provider.generate(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))


def sse_chunk(text=None, *, reason=None, usage=None, empty_choices=False, delta=None):
    payload = {
        "id": "stream-test", "object": "chat.completion.chunk", "created": 0,
        "model": "test-model",
        "choices": [] if empty_choices else [{
            "index": 0, "delta": delta if delta is not None else {"content": text},
            "finish_reason": reason,
        }],
    }
    if usage is not None:
        payload["usage"] = usage
    return ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode()


class SSEBody(httpx2.AsyncByteStream):
    def __init__(self, events, *, gate=None, failure=None):
        self.events = events
        self.gate = gate
        self.failure = failure
        self.closed = False

    async def __aiter__(self):
        for index, event in enumerate(self.events):
            if index == 1 and self.gate is not None:
                await self.gate.wait()
            yield event
        if self.failure is not None:
            raise self.failure("private-network-detail")

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_sdk_stream_markup_is_blocked_before_display_and_connection_closed(mock_transport):
    marker = '<｜｜DSML｜｜ calls>'
    body = SSEBody([*[sse_chunk(char) for char in marker], sse_chunk(reason="stop"), b"data: [DONE]\n\n"])
    mock_transport(lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    visible = []
    async with create_provider(make_config()) as provider:
        with pytest.raises(ProviderResponseError, match="已拦截"):
            async for event in provider.stream(ModelRequest(messages=[ChatMessage(role="user", content="Hi")])):
                if event.kind == "delta":
                    visible.append(event.text)
        assert visible == []
        assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("separate_usage", [False, True])
async def test_stream_yields_before_completion_and_collects_usage(mock_transport, separate_usage):
    gate = asyncio.Event()
    usage = {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    events = [sse_chunk("你"), sse_chunk("好🙂")]
    if separate_usage:
        events.extend([sse_chunk(reason="stop"), sse_chunk(usage=usage, empty_choices=True)])
    else:
        events.append(sse_chunk(reason="stop", usage=usage))
    events.append(b"data: [DONE]\n\n")
    body = SSEBody(events, gate=gate)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)

    mock_transport(handler)
    async with create_provider(make_config()) as provider:
        iterator = provider.stream(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))
        first = await anext(iterator)
        assert first.kind == "delta"
        assert first.text == "你"
        assert not gate.is_set()
        gate.set()
        remaining = [event async for event in iterator]
        assert remaining[0].text == "好🙂"
        assert remaining[-1].response.content == "你好🙂"
        assert remaining[-1].response.usage.total_tokens == 13
        assert remaining[-1].response.finish_reason == "stop"
        assert body.closed
        assert not provider._client.is_closed()
    assert requests[0]["stream"] is True
    assert requests[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_cancel_closes_response_but_keeps_provider_usable(mock_transport):
    gate = asyncio.Event()
    body = SSEBody([sse_chunk("部分"), sse_chunk("后续")], gate=gate)

    def handler(request):
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        return httpx2.Response(200, json=completion())

    mock_transport(handler)
    async with create_provider(make_config()) as provider:
        request = ModelRequest(messages=[ChatMessage(role="user", content="Hi")])
        iterator = provider.stream(request)
        assert (await anext(iterator)).text == "部分"
        task = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await iterator.aclose()
        assert body.closed
        assert (await provider.generate(request)).content == "Hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure, expected", [
    (httpx2.ReadTimeout, ProviderTimeoutError),
    (httpx2.ReadError, ProviderConnectionError),
    (None, ProviderResponseError),
])
async def test_broken_stream_is_not_a_complete_response(mock_transport, failure, expected):
    body = SSEBody([sse_chunk("部分")], failure=failure)
    mock_transport(lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body,
    ))
    async with create_provider(make_config()) as provider:
        events = []
        with pytest.raises(expected) as caught:
            async for event in provider.stream(ModelRequest(messages=[ChatMessage(role="user", content="Hi")])):
                events.append(event)
        assert [event.kind for event in events] == ["delta"]
        assert body.closed
        assert "private-network-detail" not in str(caught.value)


@pytest.mark.asyncio
async def test_stream_authentication_error_is_safe(mock_transport):
    mock_transport(lambda request: httpx2.Response(401, json={"error": {"message": "private"}}))
    async with create_provider(make_config()) as provider:
        request = ModelRequest(messages=[ChatMessage(role="user", content="Hi")])
        with pytest.raises(ProviderAuthenticationError):
            await anext(provider.stream(request))


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "length"])
async def test_stream_without_usage_is_unknown_not_zero(mock_transport, reason):
    body = SSEBody([sse_chunk("回答"), sse_chunk(reason=reason), b"data: [DONE]\n\n"])
    mock_transport(lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body,
    ))
    async with create_provider(make_config()) as provider:
        events = [event async for event in provider.stream(
            ModelRequest(messages=[ChatMessage(role="user", content="Hi")]),
        )]
        assert events[-1].response.usage is None
        assert events[-1].response.finish_reason == reason


@pytest.mark.asyncio
async def test_stream_yields_tool_arguments_but_rejects_missing_end(mock_transport):
    body = SSEBody([sse_chunk(delta={"tool_calls": [{
        "index": 0, "id": "call-1", "type": "function",
        "function": {"name": "calculator", "arguments": "{}"},
    }]})])
    mock_transport(lambda request: httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body,
    ))
    async with create_provider(make_config()) as provider:
        iterator = provider.stream(ModelRequest(messages=[ChatMessage(role="user", content="Hi")]))
        first = await anext(iterator)
        assert first.kind == "tool_delta"
        assert first.tool_delta.arguments == "{}"
        with pytest.raises(ProviderResponseError):
            await anext(iterator)
        assert body.closed


@pytest.mark.asyncio
async def test_empty_length_is_returned_for_agent_recovery(mock_transport):
    payload = completion()
    payload["choices"][0]["message"]["content"] = None
    payload["choices"][0]["finish_reason"] = "length"

    def handler(request):
        if json.loads(request.content).get("stream"):
            body = SSEBody([sse_chunk(reason="length"), b"data: [DONE]\n\n"])
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        return httpx2.Response(200, json=payload)

    mock_transport(handler)
    async with create_provider(make_config()) as provider:
        request = ModelRequest(messages=[ChatMessage(role="user", content="问题")])
        reply = await provider.generate(request)
        assert reply.content == "" and reply.finish_reason == "length"
        events = [event async for event in provider.stream(request)]
        assert len(events) == 1
        assert events[0].response.content == ""
        assert events[0].response.finish_reason == "length"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_sdk_tool_call_roundtrip_serializes_definitions_arguments_and_results(mock_transport, tmp_path, streaming):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("stream"):
            if len(bodies) == 1:
                events = [
                    sse_chunk(delta={"reasoning_content": "内部推理元数据"}),
                    sse_chunk(delta={"tool_calls": [{"index": 0, "id": "calc-1", "type": "function", "function": {
                        "name": "calculator", "arguments": '{"expression":"(12+',
                    }}]}),
                    sse_chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": '3)*4"}'}}]}),
                    sse_chunk(reason="tool_calls"), b"data: [DONE]\n\n",
                ]
            else:
                events = [sse_chunk("结果是 60"), sse_chunk(reason="stop"), b"data: [DONE]\n\n"]
            return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=SSEBody(events))
        payload = completion()
        if len(bodies) == 1:
            payload["choices"][0]["message"] = {
                "role": "assistant", "content": None, "reasoning_content": "内部推理元数据",
                "tool_calls": [{"id": "calc-1", "type": "function", "function": {
                    "name": "calculator", "arguments": '{"expression":"(12+3)*4"}',
                }}],
            }
            payload["choices"][0]["finish_reason"] = "tool_calls"
        else:
            payload["choices"][0]["message"]["content"] = "结果是 60"
        return httpx2.Response(200, json=payload)

    mock_transport(handler)
    provider = create_provider(make_config())
    async with AgentRunner(make_config(), provider, SessionStore(tmp_path)) as runner:
        reply = await runner.ask("计算 (12+3)*4", on_stream=(lambda event: None) if streaming else None)
        assert reply.content == "结果是 60"
        assert len(bodies) == 2
        assert bodies[0]["tools"][0]["function"]["name"] == "calculator"
        assert bodies[0]["tool_choice"] == "auto"
        assert bodies[0]["parallel_tool_calls"] is False
        assert bodies[0]["max_tokens"] == 8192
        assert bodies[1]["messages"][-1] == {"role": "tool", "tool_call_id": "calc-1", "content": "60"}
        assistant = bodies[1]["messages"][-2]
        assert assistant["content"] is None
        assert assistant["reasoning_content"] == "内部推理元数据"
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"expression":"(12+3)*4"}'
        assert runner.context.history[1].reasoning_content == "内部推理元数据"


@pytest.mark.asyncio
async def test_interleaved_stream_tool_parameters_assemble_by_index(mock_transport):
    from lynx_agent.tools import CalculatorTool

    body = SSEBody([
        sse_chunk(delta={"tool_calls": [{"index": 0, "id": "a", "type": "function", "function": {
            "name": "calculator", "arguments": '{"expression":"1',
        }}, {"index": 1, "id": "b", "type": "function", "function": {
            "name": "calculator", "arguments": '{"expression":"2',
        }}]}),
        sse_chunk(delta={"tool_calls": [{"index": 1, "function": {"arguments": '+3"}'}},
                                      {"index": 0, "function": {"arguments": '+4"}'}}]}),
        sse_chunk(reason="tool_calls"), b"data: [DONE]\n\n",
    ])
    mock_transport(lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    async with create_provider(make_config()) as provider:
        events = [event async for event in provider.stream(ModelRequest(
            messages=[ChatMessage(role="user", content="测试")], tools=(CalculatorTool().definition,),
        ))]
        calls = events[-1].response.tool_calls
        assert [item.id for item in calls] == ["a", "b"]
        assert [json.loads(item.arguments)["expression"] for item in calls] == ["1+4", "2+3"]
        assert body.closed


@pytest.mark.asyncio
async def test_incomplete_tool_metadata_is_not_a_completed_call(mock_transport):
    body = SSEBody([sse_chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}),
                    sse_chunk(reason="tool_calls"), b"data: [DONE]\n\n"])
    mock_transport(lambda request: httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    async with create_provider(make_config()) as provider:
        with pytest.raises(ProviderResponseError, match="ID"):
            _ = [event async for event in provider.stream(ModelRequest(messages=[ChatMessage(role="user", content="测试")]))]
        assert body.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_confirmed_context_overflow_uses_typed_safe_error(mock_transport, streaming):
    from lynx_agent.providers import ProviderContextOverflowError

    mock_transport(lambda request: httpx2.Response(400, json={
        "error": {"code": "context_length_exceeded", "message": "private-server-data"},
    }))
    async with create_provider(make_config()) as provider:
        request = ModelRequest(messages=(ChatMessage(role="user", content="test"),))
        with pytest.raises(ProviderContextOverflowError) as caught:
            if streaming:
                _ = [event async for event in provider.stream(request)]
            else:
                await provider.generate(request)
        assert "private-server-data" not in str(caught.value)
