"""主包、锁文件、安装元数据和终端显示的版本必须一致。"""

import tomllib
from importlib.metadata import version
from pathlib import Path

from lynx_agent.cli.layout import package_version

ROOT = Path(__file__).resolve().parents[1]


def test_release_metadata_and_ui_match_locked_project_version():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = next(package for package in lock["package"] if package["name"] == project["name"])
    assert locked["version"] == project["version"]
    assert version(project["name"]) == project["version"]
    assert package_version() == project["version"]
    release = f"V{project['version']}"
    for relative in ("README.md", "docs/product-plan.md"):
        assert release in (ROOT / relative).read_text(encoding="utf-8")
