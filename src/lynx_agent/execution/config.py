"""执行限额不包含开关或自动审批选项。"""

from pydantic import BaseModel, ConfigDict, Field


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    default_timeout_seconds: int = Field(default=60, ge=1, le=300)
    approval_timeout_seconds: float = Field(default=600, gt=0, le=600)
    extra_env_keys: tuple[str, ...] = ()
