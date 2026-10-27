"""状态目录的读写：原子 JSON + 进程间文件锁 + 追加式调用记录。"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"
CONTROL_FILE = "control.json"
RECORDS_FILE = "records.jsonl"
LOCK_FILE = ".lock"

DEFAULT_STATE: dict[str, Any] = {
    "kv": {},
    "orders": {},
    "files": {},
    "documents": {},
    "pages": {},
    "counter": {},
    "seq": 0,
}

DEFAULT_CONTROL: dict[str, Any] = {
    "faults": {},      # {tool: {"mode": "error"|"timeout", "times": N, "message": str, "seconds": float}}
    "notes": "",       # 给管理侧观察的备注，不暴露给 agent
}


@contextmanager
def file_lock(path: Path, timeout: float = 10.0) -> Iterator[None]:
    """跨进程互斥；Windows 用 msvcrt，其他平台用 flock。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    stream.seek(0)
    if not stream.read(1):
        stream.write(b"0")
        stream.flush()
    stream.seek(0)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("测试环境状态目录被占用，未继续写入") from None
                time.sleep(0.01)
        yield
    finally:
        with suppress(Exception):
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = dict(target)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class EnvStore:
    """测试环境的唯一入口；bench 主进程与 stdio MCP 服务器共享同一个目录。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    # ---- 路径 ----
    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    @property
    def control_path(self) -> Path:
        return self.root / CONTROL_FILE

    @property
    def records_path(self) -> Path:
        return self.root / RECORDS_FILE

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            write_json(self.state_path, DEFAULT_STATE)
        if not self.control_path.exists():
            write_json(self.control_path, DEFAULT_CONTROL)

    # ---- 管理通道（仅 bench 主进程使用）----
    def reset(self, state: dict[str, Any] | None = None,
              control: dict[str, Any] | None = None) -> None:
        """用深合并写入初始业务状态与故障控制，并清空调用记录。"""
        self.ensure()
        with self._lock, file_lock(self.root / LOCK_FILE):
            current = deep_merge(DEFAULT_STATE, state or {})
            current["seq"] = 0
            write_json(self.state_path, current)
            write_json(self.control_path, deep_merge(DEFAULT_CONTROL, control or {}))
            self.records_path.write_text("", encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        self.ensure()
        return read_json(self.state_path, DEFAULT_STATE)

    def control(self) -> dict[str, Any]:
        self.ensure()
        return read_json(self.control_path, DEFAULT_CONTROL)

    def set_control(self, patch: dict[str, Any]) -> None:
        self.ensure()
        with self._lock, file_lock(self.root / LOCK_FILE):
            write_json(self.control_path, deep_merge(read_json(self.control_path, DEFAULT_CONTROL), patch))

    def records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        entries = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except ValueError:
                continue
        return entries

    # ---- 供工具实现调用 ----
    def mutate(self, operation: Callable[[dict[str, Any], dict[str, Any]], Any]) -> Any:
        """在同一把锁内读改写业务状态与故障控制；operation 必须原地修改。

        返回 operation 的返回值，便于调用方取回一次性结果（如消耗的故障）。
        """
        self.ensure()
        with self._lock, file_lock(self.root / LOCK_FILE):
            state = read_json(self.state_path, DEFAULT_STATE)
            control = read_json(self.control_path, DEFAULT_CONTROL)
            result = operation(state, control)
            write_json(self.state_path, state)
            write_json(self.control_path, control)
            return result

    def read(self, operation: Callable[[dict[str, Any], dict[str, Any]], Any]) -> Any:
        self.ensure()
        with self._lock, file_lock(self.root / LOCK_FILE):
            return operation(read_json(self.state_path, DEFAULT_STATE),
                             read_json(self.control_path, DEFAULT_CONTROL))

    def consume_fault(self, tool: str) -> dict[str, Any] | None:
        """按剩余次数消耗一次故障注入；没有剩余时返回 None。"""
        def operation(state, control):
            faults = control.get("faults", {})
            fault = faults.get(tool)
            if not isinstance(fault, dict):
                return None
            remaining = int(fault.get("times", 0))
            if remaining <= 0:
                return None
            fault["times"] = remaining - 1
            return dict(fault)

        return self.mutate(operation)

    def next_seq(self) -> int:
        def operation(state, control):
            state["seq"] = int(state.get("seq", 0)) + 1
            return state["seq"]

        return self.mutate(operation)

    def append_record(self, record: dict[str, Any]) -> None:
        self.ensure()
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock, file_lock(self.root / LOCK_FILE), \
                self.records_path.open("a", encoding="utf-8") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
