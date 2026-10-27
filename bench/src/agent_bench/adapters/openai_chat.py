"""通用 HTTP agent 服务适配器。

协议（由被测服务实现，全部是普通 JSON）：
- POST {url}/sessions  → {"session_id": "..."}     可选；不支持时由本适配器自增 ID
- POST {url}/messages  → {"text": "..."}，请求体 {"session_id": "...", "message": "..."}
- POST {url}/reset     → 清空该服务的长期记忆，可选

模型调用由服务自己发起；如果服务不把 base_url 指向录制代理，本适配器的
Token 与费用列会是空的（报告中会标注“未记录”）。
"""

from __future__ import annotations

from time import monotonic
from typing import Any

from aiohttp import ClientError, ClientSession, ClientTimeout

from agent_bench.adapters.base import AdapterError, BaseAdapter, BenchEnv
from agent_bench.models import TurnResult


class OpenAIChatAdapter(BaseAdapter):
    kind = "openai_chat"

    def __init__(self, name: str, settings, context) -> None:
        super().__init__(name, settings, context)
        self._session = ""
        self._counter = 0
        self._client: ClientSession | None = None

    async def setup(self, workspace, env: BenchEnv) -> None:
        await super().setup(workspace, env)
        if not self.settings.url:
            raise AdapterError("openai_chat 适配器缺少 url")
        self._client = ClientSession(timeout=ClientTimeout(total=self.settings.request_timeout_seconds))
        await self._probe()

    async def teardown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def _probe(self) -> None:
        assert self._client is not None
        url = f"{self.settings.url.rstrip('/')}{self.settings.health_path}"
        try:
            async with self._client.get(url) as response:
                if response.status >= 400:
                    raise AdapterError(f"被测服务健康检查失败：HTTP {response.status}")
        except ClientError as error:
            raise AdapterError(f"无法连接被测服务：{type(error).__name__}") from error

    async def new_session(self) -> str:
        self._counter += 1
        self._session = f"bench-{self.name}-{self._counter}"
        assert self._client is not None
        url = f"{self.settings.url.rstrip('/')}/sessions"
        try:
            async with self._client.post(url, json={}) as response:
                if response.status < 400:
                    payload = await response.json()
                    if isinstance(payload, dict) and payload.get("session_id"):
                        self._session = str(payload["session_id"])
        except (ClientError, ValueError):
            pass  # 可选接口：不支持时沿用本地自增 ID
        return self._session

    async def reset_memory(self) -> None:
        if self._client is None or not self.settings.url:
            return
        url = f"{self.settings.url.rstrip('/')}/reset"
        try:
            async with self._client.post(url, json={"session_id": self._session}) as response:
                await response.read()
        except ClientError:
            pass  # 可选接口
        await self.new_session()

    async def send(self, message: str) -> TurnResult:
        if self._client is None or not self.settings.url:
            return self._denied()
        url = f"{self.settings.url.rstrip('/')}/messages"
        started = monotonic()
        payload: dict[str, Any] = {"session_id": self._session, "message": message}
        try:
            async with self._client.post(url, json=payload) as response:
                body = await response.json(content_type=None)
                status = response.status
        except (ClientError, ValueError) as error:
            return TurnResult(text="", elapsed_seconds=monotonic() - started,
                              error=f"调用被测服务失败：{type(error).__name__}",
                              finish_reason="error")
        elapsed = monotonic() - started
        if status >= 400:
            return TurnResult(text="", elapsed_seconds=elapsed,
                              error=f"被测服务返回 HTTP {status}", finish_reason="error")
        if not isinstance(body, dict):
            return TurnResult(text="", elapsed_seconds=elapsed,
                              error="被测服务返回的不是 JSON 对象", finish_reason="error")
        session = body.get("session_id")
        if isinstance(session, str) and session:
            self._session = session
        return TurnResult(text=str(body.get("text", "")), elapsed_seconds=elapsed,
                          finish_reason=str(body.get("finish_reason", "stop")),
                          diagnostics={"session": self._session})
