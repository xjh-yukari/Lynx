import hashlib
import json
import re
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    uri: str = "http://127.0.0.1:19540"
    policy_file: Path
    state_dir: Path
    project_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,128}$")
    # 固定 Hugging Face commit；本地路径也必须提供同一版本标识。
    model: str = "BAAI/bge-m3"
    model_revision: str
    device: str = "cpu"
    batch_size: int = Field(default=4, ge=1, le=64)
    chunk_tokens: int = Field(default=600, ge=64, le=2000)
    overlap_tokens: int = Field(default=80, ge=0, le=256)
    model_max_tokens: int = Field(default=8192, ge=128, le=8192)
    candidates: int = Field(default=20, ge=10, le=100)
    mode: str = "hybrid"
    output_tokens: int = Field(default=2400, ge=100, le=4000)
    queue_size: int = Field(default=4, ge=1, le=32)
    timeout: float = Field(default=60, gt=0, le=240)
    port: int = Field(default=8766, ge=1024, le=65535)
    max_file_bytes: int = Field(default=16777216, ge=1024, le=67108864)

    @model_validator(mode="after")
    def validate_settings(self):
        if self.overlap_tokens >= self.chunk_tokens or self.chunk_tokens + 2 > self.model_max_tokens:
            raise ValueError("切块/重叠/模型上限不匹配")
        if not self.policy_file.is_absolute() or not self.state_dir.is_absolute():
            raise ValueError("授权和运行目录必须为绝对路径")
        if self.mode not in {"dense", "hybrid"}:
            raise ValueError("mode 必须为 dense/hybrid")
        if not re.fullmatch(r"[a-f0-9]{40}", self.model_revision):
            raise ValueError("必须固定模型 commit，不能使用浮动 main")
        return self

    def manifest(self):
        return {key: str(value) for key, value in self.model_dump().items()
                if key in {"model", "model_revision", "chunk_tokens", "overlap_tokens", "model_max_tokens"}} | {
                    "schema": 2, "dimension": 1024, "normalized": True, "metric": "IP",
                    "dense_index": "AUTOINDEX", "sparse_index": "SPARSE_INVERTED_INDEX", "preprocessing": "utf8-offset-structure-v2"}


def load(path):
    return Settings.model_validate(tomllib.loads(Path(path).read_text(encoding="utf-8")))


def grant(settings, kb_id):
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", kb_id):
        raise ValueError("invalid_argument")
    try:
        if settings.policy_file.stat().st_size > 65536:
            raise ValueError
        data = json.loads(settings.policy_file.read_text(encoding="utf-8"))
        entry = data["grants"][kb_id]
        if data["project_id"] != settings.project_id or entry["enabled"] is not True:
            raise ValueError
        if not re.fullmatch(r"lynx_kb_[a-z0-9_]{1,50}", entry["alias"]):
            raise ValueError
        return entry
    except (OSError, KeyError, TypeError, ValueError):
        raise PermissionError("permission_denied") from None
