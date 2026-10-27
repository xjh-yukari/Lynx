"""审批与批量操作审计：只记录元数据，不写文件正文或凭据。"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock


def sanitize(text):
    """去掉控制字符，避免审计或界面被工具输出伪造换行和终端转义。"""
    return "".join(char for char in str(text) if char in "\n\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159))


class AuditLog:
    """追加式 JSONL；写入失败不影响主流程，但会在返回值中暴露。"""

    def __init__(self, directory=None):
        self.directory = Path(directory).absolute() if directory is not None else None
        self.path = self.directory / "audit.jsonl" if self.directory is not None else None
        self._lock = Lock()
        self.error = None

    def record(self, event, **fields):
        if self.path is None:
            return
        item = {"time": datetime.now(UTC).isoformat(), "event": sanitize(event)}
        item.update({key: sanitize(value) if isinstance(value, (str, Path)) else value
                     for key, value in fields.items()})
        try:
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                os.chmod(self.path, 0o600)
        except OSError:
            self.error = "审批审计写入失败；操作记录只在内存中"

    def entries(self, limit=20):
        if self.path is None or not self.path.is_file():
            return ()
        try:
            with self.path.open(encoding="utf-8") as stream:
                return tuple(stream.readlines()[-limit:])
        except OSError:
            return ()
