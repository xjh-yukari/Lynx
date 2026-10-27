"""一个长期任务拥有一个 stdio 会话；进入和退出始终发生在同一任务。"""

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from mcp import Client, StdioServerParameters

from lynx_agent.mcp.errors import MCPError
from lynx_agent.mcp.results import result_text
from lynx_agent.mcp.tool import MCPTool


class MCPConnection:
    def __init__(self, name, config, registry, *, expose_tools=True):
        self.name = name
        self.config = config
        self.registry = registry
        self.expose_tools = expose_tools
        self.state = "disabled" if not config.enabled else "disconnected"
        self.error = ""
        self.rejected = 0
        self.registered = []
        self.adapters = {}
        self.remote_tools = {}
        self._task = None
        self._queue = asyncio.Queue(maxsize=1)
        self._ready = asyncio.Event()
        self._busy = False
        self._phase = "启动"
        self.knowledge = None
        self._heartbeat_interval = 15 if config.transport == "streamable_http" else 2

    @asynccontextmanager
    async def _client(self):
        import os

        if self.config.transport == "streamable_http":
            import httpx2
            from mcp.client.streamable_http import streamable_http_client

            if self.config.token_file:
                try:
                    key_path = Path(self.config.token_file)
                    if key_path.stat().st_size > 4096:
                        raise ValueError
                    token = key_path.read_text(encoding="ascii").strip()
                except (OSError, UnicodeError, ValueError):
                    raise MCPError("HTTP MCP认证文件无法读取或格式无效；未输出文件内容") from None
            else:
                token = os.environ.get(self.config.token_env, "")
            if not 32 <= len(token) <= 1024 or not token.isascii() or any(char.isspace() for char in token):
                raise MCPError("HTTP MCP认证密钥缺失或无效，需要32–1024位无空白ASCII字符")
            async with (
                httpx2.AsyncClient(headers={"Authorization": "Bearer " + token}, follow_redirects=False,
                                   trust_env=False, timeout=self.config.call_timeout) as http,
                Client(streamable_http_client(self.config.url, http_client=http),
                       read_timeout_seconds=self.config.call_timeout + 1) as client,
            ):
                yield client
            return
        command = Path(self.config.command)
        if not command.is_file() or not Path(self.config.cwd).is_dir():
            raise MCPError("服务程序或工作目录不存在")
        if command.suffix.lower() in {".bat", ".cmd", ".ps1"}:
            raise MCPError("只允许直接运行程序，不执行 shell 脚本")
        params = StdioServerParameters(command=str(command), args=list(self.config.args), cwd=self.config.cwd,
                                       env={key: value.get_secret_value() for key, value in self.config.env.items()})
        from mcp.client.stdio import stdio_client

        with open(os.devnull, "w", encoding="utf-8") as errlog:  # noqa: ASYNC230
            async with Client(stdio_client(params, errlog=errlog), read_timeout_seconds=self.config.call_timeout + 1) as client:
                yield client

    async def connect(self):
        if not self.config.enabled:
            raise MCPError("服务配置未启用；请先由用户修改配置并重启")
        if self.state == "connected":
            return
        await self.close()
        self.state, self.error, self.rejected = "connecting", "", 0
        self._ready.clear()
        self._task = asyncio.create_task(self._serve(), name=f"mcp:{self.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), self.config.startup_timeout)
        except TimeoutError:
            await self.close()
            self.state, self.error = "error", "启动或工具发现超时"
        except asyncio.CancelledError:
            await self.close()
            raise

    async def _discover(self, client):
        tools = []
        cursor = None
        cursors = set()
        names = set()
        for _ in range(20):
            page = await client.list_tools(cursor=cursor)
            for definition in page.tools:
                if definition.name not in self.config.allowed_tools:
                    continue
                if definition.name in names:
                    raise MCPError("服务返回重复工具定义")
                names.add(definition.name)
                try:
                    tool = MCPTool(self, definition)
                    if tool.name in self.registry.names:
                        raise ValueError("工具名称冲突")
                    tools.append(tool)
                except (ValueError, TypeError):
                    self.rejected += 1
            cursor = page.next_cursor
            if not cursor:
                return tools
            if cursor in cursors:
                raise MCPError("工具分页游标重复")
            cursors.add(cursor)
        raise MCPError("工具发现超过分页限制")

    async def _serve(self):
        pending = None
        self._phase = "启动或协议协商"
        try:
            async with self._client() as client:
                self._phase = "工具发现"
                tools = await self._discover(client)
                for tool in tools:
                    if self.expose_tools:
                        self.registry.register(tool)
                        self.registered.append(tool.name)
                    self.adapters[tool.name] = tool
                    self.remote_tools[tool.remote_name] = tool
                self.state = "connected"
                self._ready.set()
                while True:
                    try:
                        pending = await asyncio.wait_for(self._queue.get(), self._heartbeat_interval)
                    except TimeoutError:
                        self._phase = "连接保活"
                        async with asyncio.timeout(self.config.call_timeout):
                            if self.config.transport == "streamable_http":
                                # 当前HTTP SDK服务不支持旧ping；使用只读发现方法检查连接。
                                await client.list_tools()
                            else:
                                await client.session.send_ping()
                        continue
                    name, arguments, future, raw = pending
                    if future.cancelled():
                        pending = None
                        continue
                    try:
                        self._phase = "工具调用"
                        async with asyncio.timeout(self.config.call_timeout):
                            response = await client.call_tool(name, arguments)
                    except TimeoutError:
                        self.state, self.error = "error", "MCP 请求超时；已断开，未自动重试"
                        if not future.done():
                            future.set_exception(TimeoutError("MCP 请求超时"))
                        raise
                    try:
                        text = response if raw else result_text(response, self.config.max_result_chars)
                    except MCPError as error:
                        future.set_exception(error)
                    else:
                        future.set_result(text)
                    pending = None
        except TimeoutError:
            self.state, self.error = "error", "MCP 请求超时；已断开，未自动重试"
            if pending is not None and not pending[2].done():
                pending[2].set_exception(TimeoutError("MCP 请求超时"))
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - 第三方错误不可回显参数和密钥
            self.state = "error"
            if not self.error:
                reason = f"：{error}" if isinstance(error, MCPError) else ""
                self.error = f"{self._phase}失败（{type(error).__name__}）{reason}；未自动重试"
        finally:
            for name in self.registered:
                self.registry.unregister(name)
            self.registered.clear()
            self.adapters.clear()
            self.remote_tools.clear()
            if self.state != "error":
                self.state = "disconnected"
            self._ready.set()
            if pending is not None and not pending[2].done():
                pending[2].set_exception(
                    MCPError(
                        "MCP 连接中断；外部操作是否生效未知，未重试",
                        execution_state="unknown",
                    )
                )
            while not self._queue.empty():
                _, _, future, _ = self._queue.get_nowait()
                if not future.done():
                    future.set_exception(MCPError("MCP 连接已关闭，未执行排队调用"))

    async def call(self, name, arguments, *, raw=False):
        if self.state != "connected" or name not in self.config.allowed_tools:
            raise MCPError("服务不可用或工具不在白名单")
        if name not in self.config.read_only_tools and not self.config.allow_write:
            raise MCPError("MCP 写入未获服务级授权")
        if self._busy:
            raise MCPError("此 MCP 服务正在执行调用，请稍后再试")
        self._busy = True
        future = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait((name, arguments, future, raw))
            return await future
        except asyncio.CancelledError:
            # 保守策略：取消本次调用同时断开服务，不重放任何操作。
            await self.close()
            raise
        finally:
            self._busy = False

    async def close(self):
        if self._task is not None:
            self._task.cancel()
            done, _ = await asyncio.wait(
                {self._task}, timeout=self.config.shutdown_timeout
            )
            if not done:
                self.state, self.error = "error", "服务清理尚未完成；禁止重新连接"
                raise MCPError(self.error)
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self.state != "disabled":
            self.state = "disconnected"
