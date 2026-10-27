"""以 UTF-8 JSONL 快照保存完整会话，原子替换防止半写覆盖。"""

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from uuid import UUID

from pydantic import ValidationError

from lynx_agent.config import get_sessions_dir
from lynx_agent.session.errors import SessionError
from lynx_agent.session.models import Session, SessionEvent


class SessionStore:
    """第一版用于单进程顺序读写同一个会话。"""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory if directory is not None else get_sessions_dir()

    def _path(self, session_id: str | UUID) -> Path:
        try:
            identifier = UUID(str(session_id))
        except ValueError as error:
            raise SessionError("会话 ID 必须是有效的 UUID") from error
        return self.directory / f"{identifier}.jsonl"

    def create(self) -> Session:
        session = Session()
        self.save(session)
        return session

    def save(self, session: Session) -> None:
        target = self._path(session.id)
        temporary: Path | None = None
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.directory, delete=False,
            ) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(
                    session.model_dump(mode="json", exclude={"events"}), ensure_ascii=False,
                ) + "\n")
                for event in session.events:
                    stream.write(event.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        except OSError as error:
            raise SessionError(f"无法保存会话 {session.id}，请检查磁盘和目录权限") from error
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def load(self, session_id: str | UUID) -> Session:
        path = self._path(session_id)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            metadata = json.loads(lines[0])
            if not isinstance(metadata, dict) or "events" in metadata:
                raise ValueError("元信息格式无效")
            events = tuple(SessionEvent.model_validate_json(line) for line in lines[1:])
            session = Session.model_validate({**metadata, "events": events})
            session.context_history()  # 回收引用和目标哈希也必须在加载边界验证。
            if str(session.id) != path.stem:
                raise ValueError("会话 ID 与文件名不一致")
            return session
        except FileNotFoundError as error:
            raise SessionError(f"会话不存在：{path.stem}") from error
        except (OSError, UnicodeError, ValueError, IndexError, ValidationError) as error:
            raise SessionError(f"会话无法读取或格式损坏：{path.name}") from error

    def list_sessions(self) -> list[Session]:
        try:
            sessions = [self.load(path.stem) for path in self.directory.glob("*.jsonl")]
        except OSError as error:
            raise SessionError("无法列出会话目录") from error
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)
