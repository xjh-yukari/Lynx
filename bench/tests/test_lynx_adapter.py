"""Lynx 适配器的配置改写：窗口对齐、记忆隔离与测试环境 MCP。

只做配置构造，不启动 agent、不调用模型，也不写任何会话或记忆数据。
"""

import sys

import pytest
from agent_bench.adapters.base import AdapterError, BenchEnv, RunContext
from agent_bench.config import AgentSettings, ModelSettings
from agent_bench.env_mcp.server import TOOL_NAMES


class StubProxy:
    def scope_url(self, scope: str) -> str:
        return f"http://127.0.0.1:9/p/{scope}"


def context_for() -> RunContext:
    return RunContext(run_id="run", model=ModelSettings(), proxy=StubProxy(),  # type: ignore[arg-type]
                      recorder=None,  # type: ignore[arg-type]
                      api_key="test-key", turn_timeout_seconds=60)


def make_adapter(tmp_path, **settings):
    from agent_bench.adapters.lynx import LynxAdapter

    payload = {"kind": "lynx", "memory_enabled": True}
    payload.update(settings)
    return LynxAdapter("lynx", AgentSettings.model_validate(payload), context_for())


def test_small_benchmark_window_normalizes_million_profile(tmp_path, monkeypatch):
    from agent_bench.adapters.lynx import SAFETY_TOKENS

    from lynx_agent.config import Config

    base = Config.model_validate({
        "providers": {"deepseek": {"api_key": "fake", "base_url": "https://example.invalid"}},
        "agent": {"model": {"name": "fake", "context_window_tokens": 1000000},
                  "context": {"reserve_tokens": 65536, "safety_tokens": 32768,
                              "keep_recent_tokens": 131072, "summary_max_tokens": 8192,
                              "summary_generation_max_tokens": 32768, "summary_input_max_tokens": 131072}},
    })
    monkeypatch.setattr("lynx_agent.config.load_config", lambda: base)
    adapter = make_adapter(tmp_path, memory_enabled=False)
    adapter.env = BenchEnv(workspace=tmp_path / "workspace", state_dir=tmp_path / "state",
                           scope="test", align_window=32768)
    adapter.workspace = adapter.env.workspace
    config = adapter._build_config()
    assert config.agent.model.context_window_tokens == 32768
    assert config.agent.context.safety_tokens == SAFETY_TOKENS
    assert config.agent.output_limit + config.agent.context.safety_tokens <= config.agent.context.reserve_tokens
    assert config.agent.summary_input_limit <= config.agent.input_budget
    assert base.agent.context.safety_tokens == 32768  # 不反向改写正式配置。


@pytest.mark.asyncio
async def test_build_config_alignment_memory_and_mcp(tmp_path):
    from lynx_agent.config import load_config

    try:
        load_config()
    except Exception as error:  # noqa: BLE001 - 没有 .env 时跳过，不伪造密钥
        pytest.skip(f"本机没有可用的 Lynx 配置：{type(error).__name__}")

    adapter = make_adapter(tmp_path)
    env = BenchEnv(workspace=tmp_path / "ws", state_dir=tmp_path / "ws" / "env",
                   scope="lynx~case-r0", align_window=32768)
    env.reset()
    adapter.workspace = env.workspace
    adapter.env = env
    config = adapter._build_config()

    assert config.agent.model.context_window_tokens == 32768
    assert config.agent.model.name == ModelSettings().name
    assert config.agent.model.temperature == ModelSettings().temperature
    assert config.agent.input_budget == 32768 - config.agent.context.reserve_tokens
    assert config.agent.context.keep_recent_tokens < config.agent.input_budget
    assert config.agent.context.summary_max_tokens < config.agent.input_budget
    assert config.agent.tools.allow_write_tools is True
    assert str(config.providers.deepseek.base_url).startswith("http://127.0.0.1:9/p/lynx~case-r0")
    assert config.memory.data_dir == str(env.workspace / "memory")
    assert config.memory.auto_approve_for_bench is True
    assert config.memory.auto_save is False
    assert config.knowledge.enabled is False
    server = config.mcp.servers["bench_env"]
    assert server.command == sys.executable
    assert server.args == ("-m", "agent_bench.env_mcp.server")
    assert server.allowed_tools == TOOL_NAMES
    assert set(server.read_only_tools) < set(server.allowed_tools)
    assert server.env["BENCH_ENV_DIR"].get_secret_value() == str(env.state_dir)


@pytest.mark.asyncio
async def test_too_small_alignment_window_is_rejected(tmp_path):
    from lynx_agent.config import load_config

    try:
        load_config()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"本机没有可用的 Lynx 配置：{type(error).__name__}")

    adapter = make_adapter(tmp_path)
    env = BenchEnv(workspace=tmp_path / "ws", state_dir=tmp_path / "ws" / "env",
                   scope="lynx~case-r0", align_window=12000)
    env.reset()
    adapter.workspace = env.workspace
    adapter.env = env
    with pytest.raises(AdapterError):
        adapter._build_config()


@pytest.mark.asyncio
async def test_memory_capability_requires_executable(tmp_path):
    from lynx_agent.config import load_config

    try:
        load_config()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"本机没有可用的 Lynx 配置：{type(error).__name__}")

    adapter = make_adapter(tmp_path, memory_enabled=False)
    env = BenchEnv(workspace=tmp_path / "ws", state_dir=tmp_path / "ws" / "env",
                   scope="lynx~case-r0", align_window=32768)
    env.reset()
    adapter.workspace = env.workspace
    adapter.env = env
    adapter.capabilities.discard("cross_session_memory")
    config = adapter._build_config()
    assert config.memory.enabled is False
