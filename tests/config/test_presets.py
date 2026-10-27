"""预设与环境配置优先级、默认输出及能力校验。"""

import pytest
from pydantic import ValidationError

from lynx_agent.config import (
    AgentConfig,
    ConfigLoadError,
    ConfigValidationError,
    ModelConfig,
    clear_config_cache,
    load_config,
)
from lynx_agent.config.loader import _ENVIRONMENT_KEYS


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    clear_config_cache()
    yield
    clear_config_cache()


def test_default_limit_and_configured_limit():
    assert AgentConfig(model=ModelConfig(name="fake")).output_limit == 8192
    assert AgentConfig(model=ModelConfig(name="fake", max_tokens=10000)).output_limit == 10000


def test_explicit_model_output_capability_is_checked():
    with pytest.raises(ValidationError):
        AgentConfig(model=ModelConfig(name="fake", max_tokens=8192, max_output_tokens=4096))


def test_presets_load_without_legacy_model_variable(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        'model_preset = "daily"\n[model_presets.daily]\nname = "fake"\nmax_tokens = 9000\n'
        '[model_presets.small]\nname = "small"\nmax_tokens = 1024\n', encoding="utf-8",
    )
    loaded = load_config(env)
    assert loaded.agent.model.name == "fake"
    assert loaded.agent.output_limit == 9000
    assert loaded.model_presets["small"].max_tokens == 1024


def test_system_then_dotenv_override_selected_preset(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nMODEL_MAX_TOKENS=6000\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        'model_preset = "first"\n[model_presets.first]\nname = "first"\nmax_tokens = 8192\n'
        '[model_presets.second]\nname = "second"\nmax_tokens = 10000\n', encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PRESET", "second")
    monkeypatch.setenv("MODEL_MAX_TOKENS", "7000")
    loaded = load_config(env)
    assert loaded.model_preset == "second"
    assert loaded.agent.model.name == "second"
    assert loaded.agent.output_limit == 7000
    assert loaded.model_presets["second"].max_tokens == 10000


def test_dotenv_overrides_preset_and_rewrite_settings(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=override\nMODEL_MAX_TOKENS=1024\n"
                   "REPLY_REWRITE_ENABLED=false\nREPLY_REWRITE_MAX_ATTEMPTS=1\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        'model_preset = "daily"\n[model_presets.daily]\nname = "base"\nmax_tokens = 8192\n', encoding="utf-8",
    )
    config = load_config(env)
    assert config.agent.output_limit == 1024
    assert config.agent.model.name == "override"
    assert config.agent.reply_rewrite.enabled is False
    assert config.agent.reply_rewrite.max_attempts == 1


@pytest.mark.parametrize("content", ["not toml @", 'unexpected = "field"',
                                     '[model_presets.bad]\nname = "x"\nmax_tokens = -1'])
def test_invalid_preset_file_is_rejected(tmp_path, content):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(content, encoding="utf-8")
    with pytest.raises(ConfigLoadError):
        load_config(env)


def test_missing_selected_preset_is_explicit_error(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nMODEL_PRESET=missing\n", encoding="utf-8")
    with pytest.raises(ConfigValidationError):
        load_config(env)


def test_explicit_missing_preset_path_is_error(tmp_path):
    with pytest.raises(ConfigLoadError):
        load_config(tmp_path / ".env", preset_path=tmp_path / "missing.toml")
