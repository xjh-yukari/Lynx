"""原子定位索引和只读派生缓存；正文的有效来源仍是 Engram。"""

import json
import os
import tempfile
from pathlib import Path

from lynx_agent.memory.errors import MemoryError


def read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise MemoryError("记忆索引或缓存损坏，未静默重新初始化") from error


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".lynx-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if Path(name).exists():
            Path(name).unlink()
