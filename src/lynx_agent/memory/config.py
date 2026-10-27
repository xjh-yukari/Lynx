"""长期记忆的显式配置。"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    executable: str = ""
    data_dir: str = ""
    agent_id: str = Field(default="lynx", pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    user_id: str = Field(default="default", pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    project_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,39}$")
    allow_write: bool = True
    auto_recall: bool = True
    auto_save: bool = False
    auto_save_timeout_seconds: float = Field(default=20, gt=0, le=60)
    # 仅评测框架使用：还需要进程环境 LYNX_BENCH_MODE=1 同时成立才生效。
    auto_approve_for_bench: bool = False
    profile_tokens: int = Field(default=2000, ge=128, le=10000)
    recall_tokens: int = Field(default=1500, ge=64, le=10000)
    max_results: int = Field(default=5, ge=1, le=10)
    timeout_seconds: float = Field(default=15, gt=0, le=60)

    @model_validator(mode="after")
    def validate_paths(self):
        if self.enabled and (
            not self.executable or not Path(self.executable).is_absolute()
        ):
            raise ValueError("启用记忆必须指定 Engram 程序绝对路径")
        if self.data_dir and not Path(self.data_dir).is_absolute():
            raise ValueError("记忆数据目录必须是绝对路径")
        return self
