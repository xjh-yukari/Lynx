"""通过 OpenAI 兼容接口调用 DeepSeek，支持完整响应和流式文本。"""

from collections.abc import AsyncIterator

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    RateLimitError,
    omit,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from pydantic import ValidationError

from lynx_agent.config.schema import ModelConfig, ProviderConfig
from lynx_agent.providers.base import ModelProvider
from lynx_agent.providers.errors import (
    ProviderAuthenticationError,
    ProviderConfigurationError,
    ProviderConnectionError,
    ProviderContextOverflowError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from lynx_agent.providers.models import (
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelToolCall,
    TokenUsage,
    ToolCallDelta,
)
from lynx_agent.providers.output_guard import TextOutputGuard, reject_tool_markup
from lynx_agent.providers.tool_calls import ToolCallAccumulator


def _token_usage(usage):
    cached = getattr(usage, "prompt_cache_hit_tokens", None)
    if cached is None:
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None)
    # 可选统计异常不能让已经生成的答复失败，也不能伪造零命中。
    if type(cached) is not int or not 0 <= cached <= usage.prompt_tokens:
        cached = None
    return TokenUsage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                      total_tokens=usage.total_tokens, cached_prompt_tokens=cached)


def _status_error(exc):
    body = exc.body
    error = body.get("error", body) if isinstance(body, dict) else {}
    code = error.get("code") if isinstance(error, dict) else None
    if exc.status_code in {400, 413} and isinstance(code, str) and code in {"context_length_exceeded", "context_window_exceeded", "max_context_length_exceeded"}:
        return ProviderContextOverflowError("DeepSeek 确认请求超过上下文窗口")
    return ProviderResponseError(f"DeepSeek 返回 HTTP {exc.status_code}")


class DeepSeekProvider(ModelProvider):
    """配置由调用者传入；本模块不读取 .env。"""

    supports_streaming = True

    def __init__(self, provider_config: ProviderConfig, model_config: ModelConfig) -> None:
        if model_config.provider != "deepseek":
            raise ProviderConfigurationError("DeepSeekProvider 只接受 deepseek 配置")
        if not provider_config.api_key.get_secret_value().strip():
            raise ProviderConfigurationError("DeepSeek API Key 不能为空")
        if not model_config.name.strip():
            raise ProviderConfigurationError("模型名称不能为空")
        if model_config.reasoning_effort is not None and model_config.name not in {"deepseek-flash", "deepseek-v4-pro"}:
            raise ProviderConfigurationError("该模型尚未声明推理强度能力；请移除 reasoning_effort 或使用已支持的模型")
        self.model_config = model_config
        self._client = AsyncOpenAI(
            api_key=provider_config.api_key.get_secret_value(),
            base_url=str(provider_config.base_url),
            timeout=provider_config.timeout_seconds,
            max_retries=0,
        )

    @staticmethod
    def _messages(request: ModelRequest) -> list[ChatCompletionMessageParam]:
        messages: list[ChatCompletionMessageParam] = []
        for message in request.messages:
            if message.role == "system":
                messages.append({"role": "system", "content": message.content})
            elif message.role == "user":
                messages.append({"role": "user", "content": message.request_content})
            else:
                if message.role == "tool":
                    messages.append({"role": "tool", "content": message.content, "tool_call_id": message.tool_call_id})
                    continue
                entry = {"role": "assistant", "content": message.content or None}
                if message.tool_calls:
                    entry["tool_calls"] = [{"id": call.id, "type": "function", "function": {
                        "name": call.name, "arguments": call.arguments,
                    }} for call in message.tool_calls]
                if message.reasoning_content is not None:
                    entry["reasoning_content"] = message.reasoning_content
                messages.append(entry)
        return messages

    @property
    def reasoning_efforts(self):
        return ("off", "low", "high", "max") if self.model_config.name in {"deepseek-flash", "deepseek-v4-pro"} else ()

    @property
    def reasoning_effort(self):
        return self.model_config.reasoning_effort or ("high" if self.reasoning_efforts else None)

    def set_reasoning_effort(self, effort):
        if effort not in self.reasoning_efforts:
            raise ValueError("当前模型不支持此推理档位")
        self.model_config = self.model_config.model_copy(update={"reasoning_effort": effort})

    def _reasoning_parameters(self):
        effort = self.model_config.reasoning_effort
        if effort is None:
            return {}  # 未配置时保持原有服务端默认行为。
        if effort == "off":
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {"reasoning_effort": effort, "extra_body": {"thinking": {"type": "enabled"}}}

    @staticmethod
    def _tool_parameters(request: ModelRequest) -> dict:
        if not request.tools:
            return {}
        parameters = {"tools": [tool.to_openai_schema() for tool in request.tools]}
        if request.tool_choice is not None:
            parameters["tool_choice"] = request.tool_choice
        if request.parallel_tool_calls is not None:
            parameters["parallel_tool_calls"] = request.parallel_tool_calls
        return parameters

    async def generate(self, request: ModelRequest) -> ModelResponse:
        """调用模型；不自动重试，避免隐式重复计费。"""
        temperature = request.temperature
        if temperature is None:
            temperature = self.model_config.temperature
        max_tokens = request.max_tokens
        if max_tokens is None:
            max_tokens = self.model_config.max_tokens
        try:
            response = await self._client.chat.completions.create(
                model=self.model_config.name,
                messages=self._messages(request),
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else omit,
                stream=False,
                **self._reasoning_parameters(),
                **self._tool_parameters(request),
            )
        except AuthenticationError:
            raise ProviderAuthenticationError("DeepSeek 认证失败，请检查 API Key") from None
        except RateLimitError:
            raise ProviderRateLimitError("DeepSeek 请求频率或额度受限") from None
        except APITimeoutError:
            raise ProviderTimeoutError("DeepSeek 请求超时") from None
        except APIConnectionError:
            raise ProviderConnectionError("无法连接 DeepSeek") from None
        except APIStatusError as exc:
            raise _status_error(exc) from None
        except (APIError, ValueError):
            raise ProviderResponseError("DeepSeek 返回了无法解析的响应") from None
        return self._parse_response(response)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """转发文本增量，流结束并关闭连接后才返回最终事件。"""
        parts: list[str] = []
        model = self.model_config.name
        finish_reason = None
        usage = None
        calls = ToolCallAccumulator()
        reasoning_parts: list[str] = []
        output_guard = TextOutputGuard()
        temperature = request.temperature if request.temperature is not None else self.model_config.temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self.model_config.max_tokens
        try:
            stream = await self._client.chat.completions.create(
                model=self.model_config.name,
                messages=self._messages(request),
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else omit,
                stream=True,
                stream_options={"include_usage": True},
                **self._reasoning_parameters(),
                **self._tool_parameters(request),
            )
            async with stream:
                async for chunk in stream:
                    if chunk.model:
                        model = chunk.model
                    if chunk.usage is not None:
                        usage = _token_usage(chunk.usage)
                    # 同时兼容 usage 位于结束片段或单独的空 choices 片段。
                    if not chunk.choices:
                        continue
                    if len(chunk.choices) != 1 or chunk.choices[0].index != 0:
                        raise ProviderResponseError("DeepSeek 流返回了不支持的多个候选")
                    choice = chunk.choices[0]
                    delta = choice.delta
                    if delta is None:
                        raise ProviderResponseError("DeepSeek 流缺少 delta")
                    if delta.function_call:
                        raise ProviderResponseError("不支持旧版 function_call，请使用 tools 协议")
                    if delta.tool_calls:
                        if finish_reason is not None:
                            raise ProviderResponseError("工具参数在流结束后仍返回")
                        for part in delta.tool_calls:
                            if part.type not in {None, "function"}:
                                raise ProviderResponseError("不支持的工具类型")
                            tool_delta = ToolCallDelta(
                                index=part.index, id=part.id or "",
                                name=part.function.name or "" if part.function else "",
                                arguments=part.function.arguments or "" if part.function else "",
                            )
                            calls.add(tool_delta)
                            yield ModelStreamEvent(kind="tool_delta", tool_delta=tool_delta)
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not isinstance(reasoning, str) or finish_reason is not None:
                            raise ProviderResponseError("推理元数据格式或顺序不合法")
                        reasoning_parts.append(reasoning)
                    if delta.refusal or choice.finish_reason == "content_filter":
                        raise ProviderResponseError("DeepSeek 拒绝了本次请求")
                    if delta.content:
                        if finish_reason is not None:
                            raise ProviderResponseError("DeepSeek 流在结束后仍返回文本")
                        parts.append(delta.content)
                        safe = output_guard.feed(delta.content)
                        if safe:
                            yield ModelStreamEvent(kind="delta", text=safe)
                    if choice.finish_reason is not None:
                        if choice.finish_reason not in {"stop", "length", "tool_calls"}:
                            raise ProviderResponseError("DeepSeek 流返回了不支持的结束原因")
                        finish_reason = choice.finish_reason
        except AuthenticationError:
            raise ProviderAuthenticationError("DeepSeek 认证失败，请检查 API Key") from None
        except RateLimitError:
            raise ProviderRateLimitError("DeepSeek 请求频率或额度受限") from None
        except APITimeoutError:
            raise ProviderTimeoutError("DeepSeek 请求超时") from None
        except APIConnectionError:
            raise ProviderConnectionError("无法连接 DeepSeek 或流式连接中断") from None
        except APIStatusError as exc:
            raise _status_error(exc) from None
        except (APIError, ValueError, TypeError, AttributeError):
            raise ProviderResponseError("DeepSeek 返回了无法解析的流式响应") from None
        tail = output_guard.finish()
        if tail:
            yield ModelStreamEvent(kind="delta", text=tail)
        content = "".join(parts)
        if finish_reason is None or (not content.strip() and finish_reason not in {"length", "tool_calls"}):
            raise ProviderResponseError("DeepSeek 流未完整结束或未返回文本，本轮未保存")
        try:
            yield ModelStreamEvent(kind="done", response=ModelResponse(
                content=content, model=model, finish_reason=finish_reason, usage=usage,
                tool_calls=calls.finish() if finish_reason != "length" else (),
                reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            ))
        except ValidationError:
            raise ProviderResponseError("工具调用最终响应字段不合法") from None

    @staticmethod
    def _parse_response(response: ChatCompletion) -> ModelResponse:
        if not response.choices:
            raise ProviderResponseError("DeepSeek 响应缺少 choices")
        if len(response.choices) != 1:
            raise ProviderResponseError("DeepSeek 返回了不支持的多个候选")
        choice = response.choices[0]
        if choice.message is None:
            raise ProviderResponseError("DeepSeek 响应缺少 message")
        reject_tool_markup(choice.message.content or "")
        if choice.message.function_call:
            raise ProviderResponseError("不支持旧版 function_call，请使用 tools 协议")
        if any(call.type != "function" for call in choice.message.tool_calls or ()):
            raise ProviderResponseError("不支持的工具类型")
        if choice.message.refusal or choice.finish_reason == "content_filter":
            raise ProviderResponseError("DeepSeek 拒绝了本次请求")
        if not choice.message.content and choice.finish_reason not in {"length", "tool_calls"}:
            raise ProviderResponseError("DeepSeek 未返回文本内容")
        try:
            usage = None
            if response.usage is not None:
                usage = _token_usage(response.usage)
            return ModelResponse(
                content=choice.message.content or "",
                model=response.model,
                finish_reason=choice.finish_reason,
                usage=usage,
                tool_calls=tuple(ModelToolCall(
                    id=call.id, name=call.function.name, arguments=call.function.arguments,
                ) for call in choice.message.tool_calls or () if call.type == "function") if choice.finish_reason != "length" else (),
                reasoning_content=getattr(choice.message, "reasoning_content", None),
            )
        except ValidationError:
            raise ProviderResponseError("DeepSeek 响应字段不合法") from None

    async def aclose(self) -> None:
        await self._client.close()
