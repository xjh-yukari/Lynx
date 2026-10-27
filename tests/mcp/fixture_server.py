"""仅供离线测试的本地服务，不注册到用户配置。"""

import asyncio
import os

from mcp.server import MCPServer

server = MCPServer("Lynx test fixture")


@server.tool()
def add(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


@server.tool()
def identity() -> dict:
    """仅用于测试进程及显式环境变量。"""
    return {
        "pid": os.getpid(),
        "secret": os.environ.get("DEEPSEEK_API_KEY"),
        "explicit": os.environ.get("LYNX_MCP_TEST"),
    }


@server.tool()
async def slow() -> str:
    """等待以测试超时和取消。"""
    await asyncio.sleep(30)
    return "done"


@server.tool()
def crash() -> str:
    """模拟服务进程异常退出。"""
    os._exit(7)


@server.tool()
def fail() -> str:
    """返回服务工具错误。"""
    raise ValueError("fixture failure")


@server.tool()
def large() -> str:
    """产生长文本。"""
    return "x" * 20000


@server.tool()
def change() -> str:
    """模拟有副作用工具，故意没有真实副作用。"""
    return "changed"


if __name__ == "__main__":
    server.run()
