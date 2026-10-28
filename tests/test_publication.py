"""公开发布文件与只读扫描器回归；密钥样本均在临时仓库中构造。"""

import importlib.util
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from lynx_agent.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("publication_check", ROOT / "scripts" / "check_publication.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, check=True).stdout


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "user.name", "publication-test")
    git(tmp_path, "config", "user.email", "publication-test@example.invalid")
    return tmp_path


def test_readme_and_supporting_documents_have_valid_local_links():
    assert (ROOT / "README.md").read_text(encoding="utf-8").startswith("# Lynx Agent ｜轻量化个人智能助手")
    for relative in ("README.md", "CONTRIBUTING.md", "SECURITY.md", "THIRD_PARTY_NOTICES.md", "docs/open-source-release.md"):
        path = ROOT / relative
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if "://" not in target and not target.startswith("#"):
                assert (path.parent / target.split("#", 1)[0]).exists()


def test_apache_license_metadata_and_notice_agree():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    assert project["description"] != "Add your description here"
    assert set(project["license-files"]) == {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Version 2.0, January 2004" in license_text and "END OF TERMS AND CONDITIONS" in license_text
    assert len(license_text) > 9000
    assert "Copyright 2026 xjh-yukari and contributors" in (ROOT / "NOTICE").read_text(encoding="utf-8")


def test_fresh_example_loads_without_private_config(tmp_path, monkeypatch):
    monkeypatch.setattr("lynx_agent.config.loader._collect_environment", lambda values: values)
    env = tmp_path / ".env"
    env.write_text((ROOT / ".env.example").read_text(encoding="utf-8"), encoding="utf-8")
    config = load_config(env_path=env)
    assert config.providers.deepseek.api_key.get_secret_value().startswith("replace-")
    assert config.agent.model.max_tokens == 8192
    assert not config.memory.enabled and not config.knowledge.enabled and not config.training.enabled
    assert not config.web.search.enabled and not config.web.fetch.enabled


@pytest.mark.parametrize("path", ["private.env", "credentials.json", "secrets.json", "sessions/one.jsonl",
                                   ".venv-training/pyvenv.cfg", "registry.sqlite3-wal", "memory/data.json",
                                   "services/document_rag/grants.json"])
def test_new_private_paths_ignored_but_examples_kept(path):
    assert subprocess.run(["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "--quiet", path],
                          cwd=ROOT, capture_output=True, check=False).returncode == 0
    assert subprocess.run(["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "--quiet", ".env.example"],
                          cwd=ROOT, capture_output=True, check=False).returncode == 1


def test_clean_repo_and_known_secret_scan_never_return_value(repository):
    (repository / "README.md").write_text("safe", encoding="utf-8")
    git(repository, "add", "README.md")
    git(repository, "commit", "--quiet", "-m", "safe")
    assert CHECK.scan(repository, history=True)["status"] == "passed"
    secret = b"synthetic-sensitive-value-for-test"
    (repository / "README.md").write_bytes(secret)
    result = CHECK.scan(repository, history=True, secrets=[secret])
    assert result["status"] == "blocked"
    assert secret.decode() not in str(result)


def test_index_and_deleted_history_secrets_still_block_publication(repository):
    secret = "sk-" + "Ab9KlmPq02Zyx" * 3
    (repository / "sample.txt").write_text(secret, encoding="utf-8")
    git(repository, "add", "sample.txt")
    (repository / "sample.txt").write_text("safe working tree", encoding="utf-8")
    assert CHECK.scan(repository)["status"] == "blocked"
    git(repository, "commit", "--quiet", "-m", "test history only")
    git(repository, "add", "sample.txt")
    git(repository, "commit", "--quiet", "-m", "remove sample")
    assert CHECK.scan(repository)["status"] == "passed"
    assert CHECK.scan(repository, history=True)["status"] == "blocked"


def test_private_filename_blocks_even_without_recognizable_key(repository):
    (repository / ".env").write_text("X=harmless", encoding="utf-8")
    git(repository, "add", ".env")
    assert CHECK.scan(repository)["status"] == "blocked"


def test_private_key_shape_and_extra_secret_files_are_never_printed(repository, monkeypatch):
    monkeypatch.delenv("SYNTHETIC_SECRET", raising=False)
    value = b"-----BEGIN " + b"PRIVATE KEY-----"
    assert CHECK.inspect_bytes(value, []) == ["private_key"]
    (repository / ".env").write_text('API_KEY="some-private-fixture-value" # local only\n', encoding="utf-8")
    values = CHECK.local_secrets(repository)
    assert b"some-private-fixture-value" in values


def test_oversized_working_file_blocks_instead_of_silent_skip(repository):
    path = repository / "large.txt"
    path.write_text("small", encoding="utf-8")
    git(repository, "add", "large.txt")
    with path.open("wb") as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    assert any(f["reason"] == "working_file_unscanned_size_limit" for f in CHECK.scan(repository)["findings"])
