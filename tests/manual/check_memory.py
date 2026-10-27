"""检查实际配置和本地记忆连接；不调用模型，不保存用户记忆。"""

import asyncio

from lynx_agent.config import load_config
from lynx_agent.config.paths import get_home_path
from lynx_agent.filesystem.policy import WORKSPACE
from lynx_agent.memory.service import MemoryService


async def main():
    config = load_config()
    memory = MemoryService(
        config.memory, get_home_path() / "memory", WORKSPACE, lambda: "read"
    )
    try:
        await memory.start()
        print(memory.status())
        print("有效档案：", ", ".join(memory.profiles) or "未创建")
        print(
            "受管理记录数：",
            sum(
                not loc.deleted
                for index in memory.indexes.values()
                for loc in index.values()
            ),
        )
        print(
            "本检查未调用模型 API，未写入用户事实或画像。首次连接会建立数据库和运行元数据。"
        )
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
