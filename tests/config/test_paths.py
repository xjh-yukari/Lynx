from pathlib import Path

import pytest

from lynx_agent.config import paths


def test_project_root_contains_pyproject() -> None:
    project_root = paths.get_project_root()

    assert project_root == Path("E:/github/nanobot")
    assert (project_root / "pyproject.toml").is_file()


def test_home_path_is_next_to_project() -> None:
    expected = paths.get_project_root().parent / "lynx_agent_data"

    assert paths.get_home_path() == expected.resolve()


def test_env_path_is_inside_project() -> None:
    expected = paths.get_project_root() / ".env"

    assert paths.get_env_path() == expected.resolve()


def test_runtime_subdir_is_created_under_data_dir(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        paths,
        "get_data_dir",
        lambda: tmp_path,
    )

    result = paths.get_runtime_subdir("sessions")

    assert result == (tmp_path / "sessions").resolve()
    assert result.is_dir()


@pytest.mark.parametrize(
    "name",
    [
        "",
        ".",
        "..",
        "nested/path",
    ],
)
def test_runtime_subdir_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError):
        paths.get_runtime_subdir(name)