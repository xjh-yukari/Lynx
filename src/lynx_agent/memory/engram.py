"""固定 Engram 3.4.0 的 MCP 返回协议，不直接读取或编辑 SQLite。"""

import json
import re

from lynx_agent.mcp.client import MCPConnection
from lynx_agent.mcp.config import MCPServerConfig
from lynx_agent.memory.cache import atomic_json, read_json
from lynx_agent.memory.errors import MemoryError, MemoryNotFound
from lynx_agent.memory.models import MARKER, MemoryRecord
from lynx_agent.tools import ToolRegistry

TOOLS = ("mem_save", "mem_get_observation", "mem_search", "mem_update", "mem_delete")


class EngramBackend:
    def __init__(self, config, directory, owner, name):
        self.directory = directory
        self.owner = owner
        server = MCPServerConfig(
            enabled=True,
            command=config.executable,
            args=("mcp", "--project", owner),
            cwd=str(directory),
            env={"ENGRAM_DATA_DIR": str(directory), "ENGRAM_PROJECT": owner},
            allowed_tools=TOOLS,
            read_only_tools=("mem_get_observation", "mem_search"),
            allow_write=True,
            call_timeout=config.timeout_seconds,
            max_result_chars=100000,
        )
        self.connection = MCPConnection(
            name, server, ToolRegistry(), expose_tools=False
        )

    async def start(self):
        project_config = self.directory / ".engram" / "config.json"
        existing = read_json(project_config, None)
        if existing is not None and existing != {"project_name": self.owner}:
            raise MemoryError("记忆库项目绑定与预期不同，未覆盖")
        if existing is None:
            atomic_json(project_config, {"project_name": self.owner})
        await self.connection.connect()
        if self.connection.state != "connected":
            raise MemoryError("Engram 未连接，记忆功能已降级")
        if not set(TOOLS).issubset(self.connection.remote_tools):
            raise MemoryError("Engram 工具不完整或版本不兼容")

    async def close(self):
        await self.connection.close()

    async def call(self, name, arguments):
        tool = self.connection.remote_tools.get(name)
        if tool is None:
            raise MemoryError("Engram 服务未就绪")
        tool.validate_arguments(arguments)
        try:
            result = await self.connection.call(name, arguments, raw=True)
            if any(item.type != "text" for item in result.content):
                raise MemoryError("Engram 返回格式不兼容")
            text = "".join(item.text for item in result.content)
            if result.is_error:
                if name == "mem_get_observation" and re.fullmatch(
                    r"Observation #\d+ not found", text
                ):
                    raise MemoryNotFound("记录不存在或已删除")
                raise MemoryError("Engram 拒绝操作，未确认成功；请检查服务状态")
            if result.result_type != "complete":
                raise MemoryError("Engram 请求未完整结束")
            if len(text) > 200000:
                raise MemoryError("Engram 返回超过边界")
            data = json.loads(text) if text.startswith("{") else {"result": text}
            if data.get("truncated") or data.get("truncation", {}).get("truncated"):
                raise MemoryError("Engram 内容发生截断，不能启用")
            return data
        except MemoryError:
            raise
        except Exception as error:
            raise MemoryError("Engram 操作未确认；写入不得盲目重试") from error

    async def save(self, record):
        result = await self.call(
            "mem_save",
            {
                "title": record.title,
                "content": record.encode(),
                "type": "config"
                if record.kind in {"soul", "user", "agents"}
                else "decision",
                "project": self.owner,
                "scope": "project",
                "topic_key": record.topic,
                "capture_prompt": False,
            },
        )
        if result.get("project") != self.owner or not isinstance(result.get("id"), int):
            raise MemoryError("Engram 保存归属或返回编号异常")
        return result["id"]

    def decode(self, content):
        # 只接受整行的应用记录，不从正文内任意匹配片段。
        for line in content.splitlines():
            if line.startswith(MARKER):
                record = MemoryRecord.model_validate_json(line[len(MARKER) :])
                if record.owner != self.owner:
                    raise MemoryError("记忆归属不匹配")
                return record
        raise MemoryError("此记录不是受管理的 Lynx 记忆")

    async def get(self, identifier, *, history=False):
        data = await self.call(
            "mem_get_observation",
            {"id": identifier, "project": self.owner, "include_history": history},
        )
        text = data.get("result", "")
        if f"\nProject: {self.owner}\n" not in text:
            raise MemoryError("实际记录归属不符，拒绝返回内容")
        record = self.decode(text)
        versions = []
        if history:
            for item in data.get("history", []):
                versions.append(self.decode(item["content"]))
        return record, versions

    async def search(self, query, limit):
        data = await self.call(
            "mem_search",
            {
                "query": query,
                "project": self.owner,
                "scope": "project",
                "limit": limit,
                "match_mode": "any",
                "response_format": "compact",
            },
        )
        return [
            item["id"]
            for item in data.get("results", [])
            if item.get("project") == self.owner
        ]

    async def delete(self, identifier):
        await self.call(
            "mem_delete", {"id": identifier, "expected_project": self.owner}
        )
