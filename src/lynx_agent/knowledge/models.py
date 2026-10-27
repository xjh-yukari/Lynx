from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    project_id: str = Field(min_length=1, max_length=128)
    kb_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    index_version: str = Field(min_length=1, max_length=128)
    chunk_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    source: str = Field(min_length=1, max_length=2048)
    page: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=512)


class KnowledgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    enabled: bool = False
    policy_file: str = ""
    server_name: str = "documents"

    @model_validator(mode="after")
    def validate_policy(self):
        if self.enabled and not Path(self.policy_file).is_absolute():
            raise ValueError("启用文档知识库必须指定授权文件绝对路径")
        return self
