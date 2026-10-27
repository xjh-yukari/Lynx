"""离线演示真实压缩流程；固定摘要仅验证机制，不代表摘要质量。"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.tokens import estimate_context_tokens
from lynx_agent.config import (
    AgentConfig,
    Config,
    ContextConfig,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import ChatMessage, ModelProvider, ModelRequest, ModelResponse
from lynx_agent.session import SessionEvent, SessionStore


class DemoProvider(ModelProvider):
    """不会连接网络的演示 Provider。"""

    async def generate(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            content="目标：开发 Lynx。约束：运行数据存 E 盘；保留会话原文。下一步：完成 CLI。",
            model="offline-demo", finish_reason="stop",
        )

    async def aclose(self) -> None:
        pass


async def main() -> None:
    config = Config(
        agent=AgentConfig(
            model=ModelConfig(name="offline-demo", max_tokens=512, context_window_tokens=4096),
            context=ContextConfig(
                reserve_tokens=1024, keep_recent_tokens=500,
                summary_max_tokens=256, safety_tokens=128,
            ),
        ),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="unused", base_url="https://example.invalid",
        )),
    )
    with TemporaryDirectory(prefix="lynx-compaction-") as temporary:
        store = SessionStore(Path(temporary))
        session = store.create()
        for index in range(8):
            session = session.with_event(SessionEvent(kind="turn", messages=(
                ChatMessage(role="user", content=f"第 {index} 轮：" + "运行数据存 E 盘，保留原文。" * 80),
                ChatMessage(role="assistant", content="记录项目要求。" * 20),
            )))
        session = session.with_event(SessionEvent(kind="turn", messages=(
            ChatMessage(role="user", content="当前任务：完成 CLI。"),
            ChatMessage(role="assistant", content="继续实现 CLI。"),
        )))
        store.save(session)
        async with AgentRunner(config, DemoProvider(), store, session) as agent:
            # 正常 ask 路径会自动触发，演示无需手动调用 compact。
            await agent.ask("继续开发")
            result = agent.last_compaction
            assert result is not None
            snapshot = agent.context.messages
            print("离线模拟：不调用真实 API；固定摘要不用于证明模型摘要质量。")
            print(f"自动触发压缩：是；输入预算：{config.agent.input_budget}")
            print(f"压缩前：{result.before_tokens}；压缩后：{result.after_tokens} Tokens（估算）")
            print(f"分批摘要请求：{result.summary_calls} 次")
        loaded = store.load(session.id)
        assert loaded.events[:len(session.events)] == session.events
        async with AgentRunner(config, DemoProvider(), store, loaded) as restored:
            assert restored.context.messages == snapshot
            print(f"恢复后上下文估算：{estimate_context_tokens(restored.context.messages).total}")
            print("检查通过：近期原文保留、Session 原文完整、检查点重启恢复一致。")


if __name__ == "__main__":
    asyncio.run(main())
