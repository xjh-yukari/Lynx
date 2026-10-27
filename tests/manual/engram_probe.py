"""验证已下载 Engram 的真实 MCP 协议，所有数据写入临时目录。"""

import asyncio
import json
import sys
import tempfile

from mcp import Client, StdioServerParameters


async def main():
    with tempfile.TemporaryDirectory(prefix="lynx-engram-probe-") as directory:
        params = StdioServerParameters(
            command=sys.argv[1],
            args=["mcp", "--project", "lynx-probe"],
            cwd=directory,
            env={"ENGRAM_DATA_DIR": directory},
        )
        async with Client(params) as client:
            definitions = await client.list_tools()
            wanted = {
                "mem_save",
                "mem_search",
                "mem_get_observation",
                "mem_update",
                "mem_delete",
                "mem_session_start",
            }
            for tool in definitions.tools:
                if tool.name in wanted:
                    print(tool.name, json.dumps(tool.input_schema, ensure_ascii=False))
            for name, args in [
                (
                    "mem_session_start",
                    {"id": "probe", "project": "lynx-probe", "directory": directory},
                ),
                (
                    "mem_save",
                    {
                        "title": "Lynx probe",
                        "type": "config",
                        "content": '{"schema_version":1,"body":"test"}',
                        "project": "lynx-probe",
                        "session_id": "probe",
                        "topic_key": "profile/user/default",
                        "capture_prompt": False,
                    },
                ),
                ("mem_search", {"query": "Lynx", "project": "lynx-probe"}),
                (
                    "mem_get_observation",
                    {"id": 1, "project": "lynx-probe", "include_history": True},
                ),
                (
                    "mem_save",
                    {
                        "title": "Lynx probe",
                        "type": "config",
                        "content": 'LYNX_MEMORY_V1:{"body":"second"}',
                        "project": "lynx-probe",
                        "topic_key": "profile/user/default",
                        "capture_prompt": False,
                    },
                ),
                (
                    "mem_get_observation",
                    {"id": 1, "project": "lynx-probe", "include_history": True},
                ),
                ("mem_delete", {"id": 1, "expected_project": "lynx-probe"}),
                ("mem_get_observation", {"id": 1, "project": "lynx-probe"}),
            ]:
                if name == "mem_session_start" and "--no-session" in sys.argv:
                    continue
                result = await client.call_tool(name, args)
                print(name, result.model_dump_json(by_alias=True, exclude_none=True))


if __name__ == "__main__":
    asyncio.run(main())
