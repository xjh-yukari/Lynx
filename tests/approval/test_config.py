"""审批配置默认值、preset 与 APPROVAL_MODE 覆盖。"""

import pytest

from lynx_agent.approval.config import ApprovalConfig
from lynx_agent.config import clear_config_cache, load_config


@pytest.fixture(autouse=True)
def clean_config():
    clear_config_cache()
    yield
    clear_config_cache()


def test_approval_defaults_and_toml_section(tmp_path):
    assert ApprovalConfig().mode == "auto"
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text(
        '[approval]\nmode = "paused"\nmax_batch_files = 7\nbackup_enabled = false\n', encoding="utf-8")
    loaded = load_config(env)
    assert loaded.approval.mode == "paused"
    assert loaded.approval.max_batch_files == 7
    assert loaded.approval.backup_enabled is False


def test_environment_overrides_approval_mode(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n", encoding="utf-8")
    (tmp_path / "config.toml").write_text('[approval]\nmode = "auto"\n', encoding="utf-8")
    monkeypatch.setenv("APPROVAL_MODE", "ask")
    clear_config_cache()
    loaded = load_config(env)
    assert loaded.approval.mode == "ask"


def test_invalid_approval_mode_is_rejected(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n", encoding="utf-8")
    monkeypatch.setenv("APPROVAL_MODE", "forever")
    clear_config_cache()
    with pytest.raises(Exception, match="配置验证失败"):
        load_config(env)
