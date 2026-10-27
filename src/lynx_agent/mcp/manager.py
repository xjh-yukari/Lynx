"""管理配置内的服务，不接受来自模型的启动配置。"""

from lynx_agent.mcp.client import MCPConnection
from lynx_agent.mcp.errors import MCPError


class MCPManager:
    def __init__(self, config, registry):
        self.connections = {
            name: MCPConnection(name, server, registry)
            for name, server in config.servers.items()
        }

    async def start(self):
        for connection in self.connections.values():
            if connection.config.enabled:
                try:
                    await connection.connect()
                except MCPError:
                    continue

    async def close(self):
        errors = []
        for connection in self.connections.values():
            try:
                await connection.close()
            except MCPError as error:
                errors.append(error)
        if errors:
            raise MCPError("部分 MCP 服务未完成关闭，请检查本地子进程")

    @property
    def label(self):
        if not self.connections:
            return "未配置"
        count = sum(item.state == "connected" for item in self.connections.values())
        return f"{count}/{len(self.connections)} 已连接"

    def status(self):
        if not self.connections:
            return "MCP：未配置服务。"
        return "\n".join(
            f"{name}：{item.state}；已注册 {len(item.registered)}；拒绝定义 {item.rejected}"
            + (f"；{item.error}" if item.error else "")
            for name, item in self.connections.items()
        )

    async def command(self, argument):
        parts = argument.split()
        if not parts:
            return self.status()
        if parts == ["tools"]:
            lines = []
            for connection in self.connections.values():
                for name in connection.registered:
                    tool = connection.adapters.get(name)
                    if tool is not None:
                        lines.append(
                            f"{name}：{'只读' if tool.read_only else '写入'}；{'可用' if tool.available else '已禁用'}"
                        )
            return "\n".join(lines) or "MCP：没有已注册工具。"
        if len(parts) != 2 or parts[0] not in {"connect", "disconnect"}:
            raise MCPError("用法：/mcp [tools | connect 服务名 | disconnect 服务名]")
        connection = self.connections.get(parts[1])
        if connection is None:
            raise MCPError("MCP 服务未配置")
        if parts[0] == "connect":
            await connection.connect()
        else:
            await connection.close()
        return self.status()
