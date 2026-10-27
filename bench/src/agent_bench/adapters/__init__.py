"""适配器工厂：新增 agent 一般只需改 bench.toml，不必改这里。"""

from agent_bench.adapters.base import AdapterError, BaseAdapter, BenchEnv, RunContext
from agent_bench.config import AgentSettings


def create_adapter(name: str, settings: AgentSettings, context: RunContext) -> BaseAdapter:
    if settings.kind == "lynx":
        from agent_bench.adapters.lynx import LynxAdapter

        return LynxAdapter(name, settings, context)
    if settings.kind == "cli":
        from agent_bench.adapters.cli import CliAdapter

        return CliAdapter(name, settings, context)
    if settings.kind == "openai_chat":
        from agent_bench.adapters.openai_chat import OpenAIChatAdapter

        return OpenAIChatAdapter(name, settings, context)
    raise AdapterError(f"未知的适配器类型：{settings.kind}")


__all__ = ["AdapterError", "BaseAdapter", "BenchEnv", "RunContext", "create_adapter"]
