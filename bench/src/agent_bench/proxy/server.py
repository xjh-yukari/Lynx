"""OpenAI 兼容录制代理：转发到真实上游，并记录每次调用的用量与延迟。

设计要点：
- 只绑定 127.0.0.1；每个 agent/用例有独立的 URL 前缀，避免互相串账。
- 请求未带 `stream_options.include_usage` 时由代理补注，拿到 usage 后
  再把多出来的 usage-only 分片从返回流里剥掉，避免改变 agent 行为。
- 429/5xx 在尚未向 agent 发送任何字节前按退避重试，重试次数计入结果。
- 任何密钥都不写入记录、日志或报告。
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
from contextlib import suppress
from time import monotonic
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout, web
from aiohttp.client_exceptions import ClientConnectionResetError

from agent_bench.config import ModelSettings, ProxySettings, RunSettings
from agent_bench.models import ModelCallRecord, utc_now
from agent_bench.proxy.recorder import ProxyRecorder

RETRY_STATUS = {429, 500, 502, 503, 504}
SAFE_SCOPE = re.compile(r"[^A-Za-z0-9_.~-]")

PROTOCOL_OPENAI = "openai"
PROTOCOL_ANTHROPIC = "anthropic"


def make_scope(agent: str, case_id: str) -> str:
    return SAFE_SCOPE.sub("_", f"{agent}~{case_id}")


def detect_protocol(tail: str) -> str:
    """按路径判断上游协议：dsh 走 Anthropic Messages，Lynx 走 OpenAI Chat。"""
    path = "/" + tail.strip("/").lower()
    if path.endswith("/messages"):
        return PROTOCOL_ANTHROPIC
    return PROTOCOL_OPENAI


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _usage_from_payload(payload: dict[str, Any],
                        protocol: str = PROTOCOL_OPENAI) -> dict[str, int | None]:
    """从一次响应（或响应分片）里取用量，两种协议字段名不同。"""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    if protocol == PROTOCOL_ANTHROPIC:
        # Anthropic 的 input_tokens **不含**缓存读写，总输入要把它们加回来，
        # 否则缓存命中率会大于 1，费用也会少算。
        cached = _int_or_none(usage.get("cache_read_input_tokens"))
        written = _int_or_none(usage.get("cache_creation_input_tokens"))
        fresh = _int_or_none(usage.get("input_tokens"))
        total_input = None
        if fresh is not None or cached is not None or written is not None:
            total_input = (fresh or 0) + (cached or 0) + (written or 0)
        return {
            "prompt_tokens": total_input,
            "completion_tokens": _int_or_none(usage.get("output_tokens")),
            "cached_prompt_tokens": cached,
        }
    cached = _int_or_none(usage.get("prompt_cache_hit_tokens"))
    if cached is None:
        details = usage.get("prompt_tokens_details")
        cached = _int_or_none(details.get("cached_tokens")) if isinstance(details, dict) else None
    return {
        "prompt_tokens": _int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": _int_or_none(usage.get("completion_tokens")),
        "cached_prompt_tokens": cached,
    }


class UsageCollector:
    """累积一次调用的用量；Anthropic 把它拆在 message_start 与 message_delta 里。"""

    def __init__(self, protocol: str) -> None:
        self.protocol = protocol
        self.prompt: int | None = None
        self.completion: int | None = None
        self.cached: int | None = None

    def feed(self, payload: dict[str, Any]) -> bool:
        """喂一个 JSON 对象；返回是否读到了新的用量。"""
        if self.protocol == PROTOCOL_ANTHROPIC:
            found = False
            if payload.get("type") == "message_start":
                message = payload.get("message")
                if isinstance(message, dict):
                    self._merge(_usage_from_payload(message, self.protocol))
                    found = True
            elif payload.get("type") in {"message_delta", "message_stop"} or payload.get("usage"):
                self._merge(_usage_from_payload(payload, self.protocol))
                found = True
            return found
        if payload.get("usage"):
            self._merge(_usage_from_payload(payload, self.protocol))
            return True
        return False

    def _merge(self, values: dict[str, int | None]) -> None:
        if values.get("prompt_tokens") is not None:
            self.prompt = values["prompt_tokens"]
        if values.get("completion_tokens") is not None:
            self.completion = values["completion_tokens"]
        if values.get("cached_prompt_tokens") is not None:
            self.cached = values["cached_prompt_tokens"]

    def result(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt,
            "completion_tokens": self.completion,
            "cached_prompt_tokens": self.cached,
        }


def _request_shape(body: dict[str, Any]) -> tuple[int, int, int]:
    messages = body.get("messages")
    if not isinstance(messages, list):
        messages = []
    chars = 0
    system = body.get("system")
    if isinstance(system, str):
        chars += len(system)
    elif system is not None:
        chars += len(json.dumps(system, ensure_ascii=False))
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            chars += len(json.dumps(content, ensure_ascii=False))
        for call in message.get("tool_calls") or ():
            chars += len(json.dumps(call, ensure_ascii=False))
    tools = body.get("tools")
    return len(messages), chars, len(tools) if isinstance(tools, list) else 0


class RecordingProxy:
    """一个 run 一个代理实例；上游地址、密钥只存在于内存。"""

    def __init__(
        self, *, run_id: str, proxy: ProxySettings, model: ModelSettings,
        run: RunSettings, api_key: str, recorder: ProxyRecorder,
    ) -> None:
        self.run_id = run_id
        self.settings = proxy
        self.model = model
        self.run = run
        self._api_key = api_key
        self.recorder = recorder
        self._session: ClientSession | None = None
        self._runner: web.AppRunner | None = None
        self._turns: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self.port = proxy.port

    # ---- 生命周期 ----
    async def start(self) -> str:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_put("/__bench/turn", self._handle_turn)
        app.router.add_get("/__bench/health", self._handle_health)
        app.router.add_route("*", "/p/{scope}", self._handle_forward)
        app.router.add_route("*", "/p/{scope}/{tail:.*}", self._handle_forward)
        self._session = ClientSession(timeout=ClientTimeout(total=None, sock_connect=30, sock_read=600))
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.settings.host, self.settings.port))
        sock.listen(128)
        self.port = sock.getsockname()[1]
        await web.SockSite(self._runner, sock).start()
        return self.base_url

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    @property
    def base_url(self) -> str:
        return f"http://{self.settings.host}:{self.port}"

    def scope_url(self, scope: str) -> str:
        """被测 agent 的 base_url；后面追加的 /chat/completions 会被原样转发。"""
        return f"{self.base_url}/p/{scope}"

    async def set_turn(self, scope: str, turn: int) -> None:
        async with self._lock:
            self._turns[scope] = turn

    def current_turn(self, scope: str) -> int:
        return self._turns.get(scope, 0)

    # ---- 管理接口（仅本机）----
    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True, "run_id": self.run_id})

    async def _handle_turn(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
            scope = str(payload["scope"])
            turn = int(payload["turn"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return web.json_response({"ok": False, "error": "需要 scope 与 turn"}, status=400)
        await self.set_turn(scope, turn)
        return web.json_response({"ok": True, "scope": scope, "turn": turn})

    # ---- 转发 ----
    async def _handle_forward(self, request: web.Request) -> web.StreamResponse:
        scope = request.match_info["scope"]
        tail = request.match_info.get("tail", "")
        agent, _, case_id = scope.partition("~")
        turn = self.current_turn(scope)
        raw = await request.read()
        body: dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                body = parsed if isinstance(parsed, dict) else {}
            except (ValueError, UnicodeError):
                body = {}
        stream = bool(body.get("stream"))
        protocol = detect_protocol(tail)
        injected = False
        # 只有 OpenAI 协议需要补注 include_usage；Anthropic 的用量本来就在
        # message_start / message_delta 里，补注反而会给上游塞未知字段。
        if stream and protocol == PROTOCOL_OPENAI:
            options = body.get("stream_options")
            if not isinstance(options, dict):
                options = {}
            if not options.get("include_usage"):
                body["stream_options"] = {**options, "include_usage": True}
                raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
                injected = True

        messages, chars, tools = _request_shape(body)
        url = self._upstream_url(tail, request.query_string, protocol)
        if protocol == PROTOCOL_ANTHROPIC:
            # Anthropic Messages 用 x-api-key 鉴权，不发送 Bearer。
            headers = {
                "Content-Type": request.headers.get("Content-Type", "application/json"),
                "x-api-key": self._api_key,
                "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
                "Accept": request.headers.get("Accept", "application/json"),
            }
            if "anthropic-beta" in request.headers:
                headers["anthropic-beta"] = request.headers["anthropic-beta"]
        else:
            headers = {
                "Content-Type": request.headers.get("Content-Type", "application/json"),
                "Authorization": f"Bearer {self._api_key}",
                "Accept": request.headers.get("Accept", "application/json"),
            }
        started = monotonic()
        attempt = 0
        response = None
        while True:
            try:
                response = await self._session.post(url, data=raw, headers=headers)  # type: ignore[union-attr]
            except ClientError as error:
                if attempt < self.run.retries:
                    attempt += 1
                    await asyncio.sleep(self.run.retry_backoff_seconds * (2 ** (attempt - 1)))
                    continue
                return await self._fail(
                    agent=agent, case_id=case_id, turn=turn, messages=messages, chars=chars,
                    tools=tools, injected=injected, started=started, retries=attempt,
                    message=f"{type(error).__name__}: 上游连接失败",
                )
            if response.status in RETRY_STATUS and attempt < self.run.retries:
                await response.release()
                attempt += 1
                await asyncio.sleep(self.run.retry_backoff_seconds * (2 ** (attempt - 1)))
                continue
            break

        common = {
            "run_id": self.run_id, "agent": agent, "case_id": case_id, "turn": turn,
            "messages": messages, "request_chars": chars, "tool_definitions": tools,
            "status": response.status, "usage_injected": injected, "retries": attempt,
        }
        if not stream:
            return await self._finish_json(response, common, started, protocol)
        return await self._finish_stream(request, response, common, started, injected, protocol)

    def _upstream_url(self, tail: str, query: str, protocol: str = PROTOCOL_OPENAI) -> str:
        """按协议选上游根地址：OpenAI 用 base_url，Anthropic 用 anthropic_base_url。"""
        if protocol == PROTOCOL_ANTHROPIC:
            base = (self.model.anthropic_base_url
                    or f"{self.model.base_url.rstrip('/')}/anthropic").rstrip("/")
        else:
            base = self.model.base_url.rstrip("/")
        path = tail.strip("/")
        url = f"{base}/{path}" if path else base
        return f"{url}?{query}" if query else url

    async def _record(self, payload: dict[str, Any]) -> None:
        self.recorder.add(ModelCallRecord(**payload, at=utc_now()))

    async def _fail(self, *, agent: str, case_id: str, turn: int, messages: int, chars: int,
                    tools: int, injected: bool, started: float, retries: int,
                    message: str) -> web.Response:
        await self._record({
            "run_id": self.run_id, "agent": agent, "case_id": case_id, "turn": turn,
            "messages": messages, "request_chars": chars, "tool_definitions": tools,
            "status": 502, "error": message, "usage_injected": injected,
            "retries": retries, "latency_ms": (monotonic() - started) * 1000,
        })
        return web.json_response({"error": {"message": message, "type": "proxy_error"}}, status=502)

    async def _finish_json(self, response, common: dict[str, Any], started: float,
                           protocol: str) -> web.Response:
        try:
            data = await response.read()
        finally:
            response.release()
        usage = {"prompt_tokens": None, "completion_tokens": None, "cached_prompt_tokens": None}
        error = None
        if response.status < 400:
            try:
                payload = json.loads(data)
                collector = UsageCollector(protocol)
                if isinstance(payload, dict):
                    collector.feed(payload)
                usage = collector.result()
            except (ValueError, UnicodeError):
                error = "上游返回了无法解析的 JSON"
        else:
            error = f"上游返回 HTTP {response.status}"
        await self._record({**common, **usage, "error": error,
                            "latency_ms": (monotonic() - started) * 1000})
        return web.Response(body=data, status=response.status,
                            content_type=response.content_type)

    async def _finish_stream(self, request: web.Request, response, common: dict[str, Any],
                             started: float, injected: bool, protocol: str) -> web.StreamResponse:
        out = web.StreamResponse(
            status=response.status,
            headers={"Content-Type": response.headers.get("Content-Type", "text/event-stream")},
        )
        await out.prepare(request)
        buffer = b""
        first_byte: float | None = None
        collector = UsageCollector(protocol)
        error = None if response.status < 400 else f"上游返回 HTTP {response.status}"
        try:
            async for chunk in response.content.iter_any():
                buffer += chunk
                while True:
                    index = buffer.find(b"\n")
                    if index < 0:
                        break
                    line = buffer[: index + 1]
                    buffer = buffer[index + 1:]
                    keep = self._inspect_line(line, injected, collector)
                    if not keep:
                        continue
                    if first_byte is None:
                        first_byte = monotonic()
                    await out.write(line)
            if buffer and self._inspect_line(buffer, injected, collector):
                await out.write(buffer)
        except (ConnectionResetError, ClientConnectionResetError):
            # agent 提前断开（例如只取到 final 就退出）：记账已完成，不当成代理故障。
            error = "agent 提前断开了连接"
        finally:
            response.release()
            with suppress(Exception):
                await out.write_eof()
            await self._record({
                **common, **collector.result(), "error": error,
                "ttft_ms": None if first_byte is None else (first_byte - started) * 1000,
                "latency_ms": (monotonic() - started) * 1000,
            })
        return out

    @staticmethod
    def _inspect_line(line: bytes, injected: bool, collector: UsageCollector) -> bool:
        """把这一行喂给用量收集器；返回是否应当转发给 agent。"""
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            return True
        payload = stripped[5:].strip()
        if payload == b"[DONE]":
            return True
        try:
            data = json.loads(payload)
        except (ValueError, UnicodeError):
            return True
        if not isinstance(data, dict):
            return True
        collector.feed(data)
        # 补注换来的 usage-only 分片不能进入 agent 的流。
        return not (injected and data.get("usage") and not data.get("choices"))
