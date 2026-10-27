"""本地假的 Anthropic Messages 上游，只用于框架自检，不访问真实 API。

dsh 走的是 Anthropic Messages 协议，这里用最小实现驱动它完成两轮工具调用，
既不给真实模型付费，也不依赖网络。

规则（按优先级）：
1. 最后一条消息是 tool_result → 返回文本，并把工具结果原文带进回答（便于判分）。
2. 用户要求“存进笔记库 / notes_set” → 发起 notes_set 调用，key 与内容从用户文本里取。
3. 用户要求“查一下 / 存的是什么” → 发起 notes_get 调用。
4. 其他 → 直接回答。
"""

from __future__ import annotations

import json
import re
from typing import Any

from aiohttp import web

TOOL_KEY = re.compile(r"key\s*(?:必须)?\s*(?:用|是|为)?\s*[:：]?\s*([A-Za-z0-9_-]{2,})")
TOKEN = re.compile(r"[A-Z0-9]{2,}-[A-Z0-9]{2,}")
WORD = re.compile(r"[A-Za-z0-9_-]{3,}")


def _last_message(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    return messages[-1] if messages and isinstance(messages[-1], dict) else {}


def _last_user_text(body: dict[str, Any]) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [block.get("text", "") for block in content
                     if isinstance(block, dict) and block.get("type") == "text"]
            if parts:
                return "\n".join(parts)
    return ""


def _tool_results(body: dict[str, Any]) -> list[str]:
    """只看最后一条消息：只有它才是本轮的工具结果。"""
    content = _last_message(body).get("content")
    if not isinstance(content, list):
        return []
    found = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            continue
        inner = block.get("content")
        if isinstance(inner, str):
            found.append(inner)
        elif isinstance(inner, list):
            found.extend(item.get("text", "") for item in inner if isinstance(item, dict))
    return found


def _pick_tool(body: dict[str, Any], suffix: str) -> str | None:
    for tool in body.get("tools") or []:
        name = tool.get("name", "")
        if isinstance(name, str) and name.endswith(suffix):
            return name
    return None


def _plan(body: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
    """返回 (类型, 工具参数, 文本)。"""
    results = _tool_results(body)
    if results:
        return "text", None, f"已经拿到工具结果：{results[-1][:400]}"
    text = _last_user_text(body)
    store_tool = _pick_tool(body, "notes_set")
    if store_tool and ("notes_set" in text or "存进笔记库" in text):
        key = TOOL_KEY.search(text)
        value = TOKEN.search(text)
        return "tool", {"name": store_tool,
                        "input": {"key": key.group(1) if key else "note",
                                  "value": value.group(0) if value else "ok"}}, ""
    read_tool = _pick_tool(body, "notes_get")
    if read_tool and ("查一下" in text or "存的是什么" in text):
        matched = TOOL_KEY.search(text) or WORD.search(text)
        key = matched.group(1) if matched and matched.lastindex else (
            matched.group(0) if matched else "note")
        return "tool", {"name": read_tool, "input": {"key": key}}, ""
    return "text", None, "好的，已按要求回答。"


class FakeAnthropic:
    """可嵌入测试的假上游；用量字段写成固定值，便于断言。"""

    def __init__(self, *, input_tokens: int = 1200, output_tokens: int = 25,
                 cache_read_tokens: int = 300) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.requests: list[dict[str, Any]] = []
        self.port = 0
        self._runner: web.AppRunner | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/{tail:.*}", self.handle)
        return app

    async def start(self) -> FakeAnthropic:
        self._runner = web.AppRunner(self.app(), access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append(body)
        kind, tool, text = _plan(body)
        prompt_tokens = self.input_tokens + len(
            json.dumps(body.get("messages") or [], ensure_ascii=False)) // 4
        if not body.get("stream"):
            content = ([{"type": "tool_use", "id": "toolu_1", "name": tool["name"],
                         "input": tool["input"]}] if kind == "tool"
                       else [{"type": "text", "text": text}])
            return web.json_response({
                "id": "msg_mock", "type": "message", "role": "assistant",
                "model": body.get("model"), "content": content,
                "stop_reason": "tool_use" if kind == "tool" else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": prompt_tokens,
                          "output_tokens": self.output_tokens,
                          "cache_read_input_tokens": self.cache_read_tokens},
            })

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def send(event: str, payload: dict[str, Any]) -> None:
            await response.write(
                f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode())

        await send("message_start", {"type": "message_start", "message": {
            "id": "msg_mock", "type": "message", "role": "assistant",
            "model": body.get("model"), "content": [], "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": 1,
                      "cache_read_input_tokens": self.cache_read_tokens}}})
        if kind == "tool":
            await send("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1",
                                  "name": tool["name"], "input": {}}})
            await send("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "input_json_delta",
                          "partial_json": json.dumps(tool["input"], ensure_ascii=False)}})
            stop_reason = "tool_use"
        else:
            await send("content_block_start", {
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}})
            await send("content_block_delta", {
                "type": "content_block_delta", "index": 0,
                "delta": {"type": "text_delta", "text": text}})
            stop_reason = "end_turn"
        await send("content_block_stop", {"type": "content_block_stop", "index": 0})
        await send("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": self.output_tokens}})
        await send("message_stop", {"type": "message_stop"})
        await response.write_eof()
        return response
