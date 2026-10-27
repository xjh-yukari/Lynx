"""仅测试联网工具，不调用大模型、不创建会话。"""

import argparse
import asyncio
import json
import sys

from lynx_agent.config import ConfigError, load_config
from lynx_agent.tools import ToolCall
from lynx_agent.web.setup import create_default_tools, search_unavailable_reason


async def main() -> int:
    parser = argparse.ArgumentParser(description="测试网页工具：不调用模型；搜索消耗所选服务额度，有 Key 时可能计费")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--search", metavar="QUERY")
    group.add_argument("--fetch", metavar="URL")
    parser.add_argument("--freshness", choices=["day", "week", "month", "year"])
    args = parser.parse_args()
    try:
        config = load_config()
    except ConfigError as error:
        print(error)
        return 1
    registry = create_default_tools(config)
    name = "web_search" if args.search else "web_fetch"
    if name not in registry.names:
        print(search_unavailable_reason(config.web.search) if args.search else "网页读取已关闭")
        return 1
    arguments = {"query": args.search} if args.search else {"url": args.fetch}
    if args.search and args.freshness:
        arguments["freshness"] = args.freshness
    result = await registry.execute(ToolCall(id="manual-web-check", name=name, arguments=arguments))
    if result.is_error:
        print("联网工具失败：" + result.content)
        return 1
    try:
        print(json.dumps(json.loads(result.content), ensure_ascii=False, indent=2))
    except ValueError:
        print(result.content)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    raise SystemExit(asyncio.run(main()))
