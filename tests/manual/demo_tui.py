"""离线预览统一 TUI；不读取密钥，不连接网络，退出即清理演示会话。"""

import argparse
import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import (
    ModelProvider,
    ModelResponse,
    ModelStreamEvent,
    ModelToolCall,
    ToolCallDelta,
)
from lynx_agent.session import SessionStore
from lynx_agent.tools.builtin.calculator import CalculatorTool
from lynx_agent.tools.builtin.filesystem import create_file_tools
from lynx_agent.tools.registry import ToolRegistry


class DemoProvider(ModelProvider):
    supports_streaming = True

    async def generate(self, request):
        return ModelResponse(
            content="离线摘要：正在测试 Lynx 的终端交互，保留会话原文。",
            model="offline-demo", finish_reason="stop",
        )

    async def stream(self, request):
        text = (
            "## 离线交互演示\n\n这是模拟回复，**没有调用真实 API**。\n\n"
            "- 拖动正文后按 Alt+C，或点击下方按钮复制。\n"
            "- Alt+E 展开草稿，Enter 换行，Alt+S 发送。\n"
            "- /sessions 搜索会话；/theme light 切换主题；输入 @ 选择临时示例文件。\n\n"
            "```python\nmessage = '你好，Lynx'\nprint(message)\n```\n"
            "点击回答下方的 [复制] 即可复制完整正文。工具演示可用 --tools 启动。"
        )
        for index in range(0, len(text), 4):
            await asyncio.sleep(0.08)
            yield ModelStreamEvent(kind="delta", text=text[index:index + 4])
        yield ModelStreamEvent(kind="done", response=ModelResponse(
            content=text, model="offline-demo", finish_reason="stop",
        ))

    async def aclose(self):
        pass


class ToolDemoProvider(DemoProvider):
    """模拟模型提出固定算式，实际执行本地计算器并根据真实结果回复。"""

    async def stream(self, request):
        if request.messages[-1].role == "tool":
            text = f"本地计算器返回：**{request.messages[-1].content}**。\n这是离线工具演示，未调用 API；演示固定计算 `(12 + 3) * 4`。"
            for index in range(0, len(text), 4):
                await asyncio.sleep(0.04)
                yield ModelStreamEvent(kind="delta", text=text[index:index + 4])
            yield ModelStreamEvent(kind="done", response=ModelResponse(content=text, model="offline-tools", finish_reason="stop"))
        elif request.tools and request.tool_choice != "none":
            yield ModelStreamEvent(kind="tool_delta", tool_delta=ToolCallDelta(
                index=0, id="demo-calc", name="calculator", arguments='{"expression":"(12+',
            ))
            await asyncio.sleep(0.3)
            yield ModelStreamEvent(kind="tool_delta", tool_delta=ToolCallDelta(index=0, arguments='3)*4"}'))
            yield ModelStreamEvent(kind="done", response=ModelResponse(
                content="", model="offline-tools", finish_reason="tool_calls", tool_calls=(ModelToolCall(
                    id="demo-calc", name="calculator", arguments='{"expression":"(12+3)*4"}',
                ),),
            ))
        else:
            async for event in super().stream(request):
                yield event


async def main():
    parser = argparse.ArgumentParser(description="离线 TUI 演示，不读取密钥或联网")
    parser.add_argument("--tools", action="store_true", help="演示模型工具调用与计算器执行")
    args = parser.parse_args()
    config = Config(
        agent=AgentConfig(model=ModelConfig(name="offline-tools" if args.tools else "offline-demo")),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="unused", base_url="https://example.invalid",
        )),
    )
    with TemporaryDirectory(prefix="lynx-tui-demo-") as temporary:
        workspace = Path(temporary) / "workspace"
        workspace.mkdir()
        (workspace / "demo.py").write_text("print('hello Lynx')\n", encoding="utf-8")
        (workspace / "README.md").write_text("# 离线示例文件\n", encoding="utf-8")
        store = SessionStore(Path(temporary) / "sessions")
        tools = ToolRegistry([CalculatorTool(), *create_file_tools(FileAccess(workspace))])
        async with AgentRunner(config, ToolDemoProvider() if args.tools else DemoProvider(), store, tools=tools) as runner:
            await LynxTUI(runner, config, store, preferences_path=Path(temporary) / "preferences.json").run()


if __name__ == "__main__":
    asyncio.run(main())
