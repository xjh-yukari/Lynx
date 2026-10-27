"""应用运行路径管理。"""

from __future__ import annotations

from pathlib import Path

_RUNTIME_DIR_NAME = "lynx_agent_data"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回规范化路径。"""

    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """返回源码项目根目录。"""

    return Path(__file__).resolve().parents[3]


def get_env_path() -> Path:
    """返回项目本地 .env 文件路径。"""

    return get_project_root() / ".env"


def get_home_path() -> Path:
    """返回与源码仓库同级的 Lynx Agent 运行数据目录。"""

    return (get_project_root().parent / _RUNTIME_DIR_NAME).resolve()


def get_data_dir() -> Path:
    """返回应用运行数据目录。"""

    return ensure_dir(get_home_path())


def get_runtime_subdir(name: str) -> Path:
    """返回运行数据目录下的指定子目录。"""

    if not name or name in {".", ".."}:
        raise ValueError("运行目录名称不能为空或使用相对路径标记")

    if Path(name).name != name:
        raise ValueError("运行目录名称不能包含路径分隔符")

    return ensure_dir(get_data_dir() / name)


def get_workspace_dir() -> Path:
    """返回 Agent 工作区目录。"""

    return get_runtime_subdir("workspace")


def get_sessions_dir() -> Path:
    """返回对话会话存储目录。"""

    return get_runtime_subdir("sessions")


def get_memory_dir() -> Path:
    """返回长期记忆存储目录。"""

    return get_runtime_subdir("memory")


def get_logs_dir() -> Path:
    """返回日志目录。"""

    return get_runtime_subdir("logs")
