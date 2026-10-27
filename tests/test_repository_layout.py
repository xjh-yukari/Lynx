"""目录整理回归：入口、人工脚本与测试语料不能因迁移断链。"""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "document_rag"


def test_runtime_launcher_stays_in_scripts():
    assert (ROOT / "scripts" / "rag_runtime.py").is_file()
    assert (ROOT / "docs" / "product-plan.md").is_file()


@pytest.mark.parametrize("name", [
    "check_memory", "check_web", "demo_compaction", "demo_tui", "engram_probe",
    "tui_preview", "tui_mockups", "validate_rag_real", "check_rag_persistence",
])
def test_manual_program_compiles_without_execution(name):
    path = ROOT / "tests" / "manual" / f"{name}.py"
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_long_context_materials_exist():
    for number in range(1, 7):
        path = ROOT / "tests" / "fixtures" / "long_context" / f"part-{number}.txt"
        assert path.stat().st_size > 0


def test_rag_sample_plans_reference_existing_fixtures():
    plans = [FIXTURES / "plan.json", ROOT / "services" / "document_rag" / "import.example.json"]
    for plan in plans:
        for entry in json.loads(plan.read_text(encoding="utf-8")):
            # 示例使用当前部署绝对路径；只验证迁移后的仓库内映射，兼容其他检出位置。
            prefix = "tests/fixtures/document_rag/"
            normalized = entry["path"].replace("\\", "/")
            assert prefix in normalized
            assert (FIXTURES / normalized.split(prefix, 1)[1]).is_file()


@pytest.mark.parametrize("relative", [
    "README.md", "docs/product-plan.md", "tests/README.md", "services/document_rag/README.md",
])
def test_maintenance_document_links_exist(relative):
    path = ROOT / relative
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        if "://" in target or target.startswith("#"):
            continue
        assert (path.parent / target.split("#", 1)[0]).exists(), (relative, target)


@pytest.mark.parametrize("name", [
    ".env", ".env.local", "config.toml", "credentials.key", "private.pem",
    ".idea/workspace.xml", ".venv/pyvenv.cfg", ".pytest_cache/cache", "results/run.json",
])
def test_git_ignores_local_secrets_and_artifacts(name):
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "--quiet", name],
        cwd=ROOT, capture_output=True, check=False,
    )
    assert result.returncode == 0, name


@pytest.mark.parametrize("name", [
    "AGENTS.md", "config.example.toml", "web.example.env",
    "services/document_rag/.env.example", "uv.lock", "tests/test_repository_layout.py",
])
def test_git_keeps_rules_examples_and_tests(name):
    result = subprocess.run(
        ["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "--quiet", name],
        cwd=ROOT, capture_output=True, check=False,
    )
    assert result.returncode == 1, name


def test_project_rules_require_tests_verified_commits_and_handoff():
    rules = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    for required in ("新增或更新相关测试", "uv run --locked pytest -q", "ruff check src tests",
                     "必须创建对应的 Git commit", "commit 短哈希", "不能宣称已通过全部验证"):
        assert required in rules
