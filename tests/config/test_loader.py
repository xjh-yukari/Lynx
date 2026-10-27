import pytest

from lynx_agent.config import (
    ConfigValidationError,
    clear_config_cache,
    load_config,
)
from lynx_agent.config.schema import DEFAULT_SYSTEM_PROMPT, AgentConfig, ModelConfig

_ENVIRONMENT_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TIMEOUT_SECONDS",
    "MODEL_TEMPERATURE",
    "MODEL_MAX_TOKENS",
    "AGENT_SYSTEM_PROMPT",
    "AGENT_MAX_ITERATIONS",
    "MODEL_CONTEXT_WINDOW_TOKENS",
    "CONTEXT_RESERVE_TOKENS",
    "CONTEXT_KEEP_RECENT_TOKENS",
    "CONTEXT_SUMMARY_MAX_TOKENS",
    "CONTEXT_SUMMARY_GENERATION_MAX_TOKENS",
    "CONTEXT_SUMMARY_INPUT_MAX_TOKENS",
    "CONTEXT_COMPACTION_TRIGGER_RATIO",
    "CONTEXT_CALIBRATE_TOKENS",
    "CONTEXT_SAFETY_TOKENS",
    "CONTEXT_AUTO_COMPACT",
    "MODEL_PRESET",
    "MODEL_MAX_OUTPUT_TOKENS",
    "REPLY_REWRITE_ENABLED",
    "REPLY_REWRITE_MAX_ATTEMPTS",
    "TOOLS_TIMEOUT_SECONDS",
    "TOOLS_MAX_CALLS_PER_RESPONSE",
    "TOOLS_MAX_ARGUMENT_CHARS",
    "TOOLS_MAX_RESULT_CHARS",
    "TOOLS_ALLOW_WRITE",
)


def test_default_response_style_is_shared_by_schema_and_loader(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY=test-key\nDEEPSEEK_MODEL=test-model\n", encoding="utf-8")
    assert load_config(path).agent.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert AgentConfig(model=ModelConfig(name="test-model")).system_prompt == DEFAULT_SYSTEM_PROMPT
    assert "simple greetings" in DEFAULT_SYSTEM_PROMPT
    assert "workspace" in DEFAULT_SYSTEM_PROMPT
    assert "safety notices" in DEFAULT_SYSTEM_PROMPT


def test_explicit_response_prompt_is_not_overwritten(tmp_path):
    path = tmp_path / ".env"
    path.write_text("DEEPSEEK_API_KEY=test-key\nDEEPSEEK_MODEL=test-model\nAGENT_SYSTEM_PROMPT=Custom rules\n", encoding="utf-8")
    assert load_config(path).agent.system_prompt == "Custom rules"


@pytest.fixture(autouse=True)
def isolate_config_environment(
    monkeypatch: pytest.MonkeyPatch,
):
    """确保每个测试使用独立的环境变量和配置缓存。"""

    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)

    clear_config_cache()
    yield
    clear_config_cache()


def test_load_config_from_env_file(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        (
            "DEEPSEEK_API_KEY=test-key\n"
            "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
            "DEEPSEEK_MODEL=test-model\n"
        ),
        encoding="utf-8",
    )

    config = load_config(env_path)

    assert config.providers.deepseek.api_key.get_secret_value() == "test-key"
    assert str(config.providers.deepseek.base_url) == "https://api.deepseek.com/"
    assert config.providers.deepseek.timeout_seconds == 60.0
    assert config.agent.model.name == "test-model"
    assert config.agent.model.temperature == 0.2
    assert config.agent.max_iterations == 8


def test_million_context_settings_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\nMODEL_CONTEXT_WINDOW_TOKENS=1000000\n"
        "MODEL_MAX_TOKENS=8192\nCONTEXT_RESERVE_TOKENS=65536\n"
        "CONTEXT_KEEP_RECENT_TOKENS=131072\nCONTEXT_SUMMARY_MAX_TOKENS=8192\n"
        "CONTEXT_SUMMARY_GENERATION_MAX_TOKENS=32768\nCONTEXT_SUMMARY_INPUT_MAX_TOKENS=131072\n"
        "CONTEXT_SAFETY_TOKENS=32768\nCONTEXT_COMPACTION_TRIGGER_RATIO=0.9\n"
        "CONTEXT_CALIBRATE_TOKENS=true\nTOOL_PRUNING_PRESSURE_RATIO=0.85\n"
        "DEEPSEEK_TIMEOUT_SECONDS=300\nRECOVERY_MAX_SECONDS=600\n", encoding="utf-8",
    )
    monkeypatch.setenv("CONTEXT_SUMMARY_INPUT_MAX_TOKENS", "65536")
    config = load_config(path)
    assert config.agent.input_budget == 934464
    assert config.agent.compaction_trigger == 841017
    assert config.agent.summary_input_limit == 65536
    assert config.agent.context.calibrate_tokens
    assert config.agent.summary_generation_limit == 32768
    assert config.providers.deepseek.timeout_seconds == 300
    assert config.agent.recovery.max_seconds == 600


def test_load_config_rejects_missing_api_key(tmp_path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEEPSEEK_MODEL=test-model\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ConfigValidationError,
        match="DEEPSEEK_API_KEY",
    ):
        load_config(env_path)


def test_system_environment_overrides_env_file(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        (
            "DEEPSEEK_API_KEY=file-key\n"
            "DEEPSEEK_MODEL=file-model\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "environment-model")

    config = load_config(env_path)

    assert (
        config.providers.deepseek.api_key.get_secret_value()
        == "environment-key"
    )
    assert config.agent.model.name == "environment-model"


def test_context_budget_environment_overrides_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n"
        "MODEL_CONTEXT_WINDOW_TOKENS=4096\nMODEL_MAX_TOKENS=512\n"
        "CONTEXT_RESERVE_TOKENS=1024\nCONTEXT_KEEP_RECENT_TOKENS=500\n"
        "CONTEXT_SUMMARY_MAX_TOKENS=256\nCONTEXT_SAFETY_TOKENS=128\n"
        "CONTEXT_AUTO_COMPACT=true\n", encoding="utf-8",
    )
    monkeypatch.setenv("CONTEXT_AUTO_COMPACT", "false")
    loaded = load_config(path)
    assert loaded.agent.input_budget == 3072
    assert loaded.agent.context.auto_compact is False
