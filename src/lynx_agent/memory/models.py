"""Lynx 自有记录协议，不把 Engram 的 type 字段当作授权。"""

import hashlib
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from lynx_agent.knowledge.models import Evidence

Kind = Literal["soul", "user", "agents", "fact", "decision", "experience", "handoff"]
PROFILES = {"soul", "user", "agents"}
MARKER = "LYNX_MEMORY_V1:"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    kind: Kind
    owner: str
    topic: str
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=6000)
    revision: int = Field(default=1, ge=1)
    confirmed: Literal[True] = True
    source_session: str | None = None
    operation_id: str = Field(default_factory=lambda: uuid4().hex)
    index_terms: str = ""
    updated_at: str = Field(default_factory=now)
    sources: tuple[Evidence, ...] = Field(default=(), max_length=32)

    def encode(self):
        # 保持旧记录逐字序列化兼容，否则已存哈希会失效。
        if not self.sources:
            return MARKER + self.model_dump_json(exclude={"sources"})
        return MARKER + self.model_dump_json()


class Locator(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store: Literal["profile", "project"]
    id: int
    kind: Kind
    topic: str
    content_hash: str
    revision: int
    deleted: bool = False


class Change(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    action: Literal["save", "delete"]
    key: str
    record: MemoryRecord | None = None
    expected_hash: str | None = None
    mode: str
    created: float
    viewed: bool = False
