"""训练模块默认关闭，配置不提供自动审批或任意执行入口。"""

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    enabled: bool = False
    distribution: str = Field(default="Ubuntu-24.04", pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
    root: str = Field(default="~/lynx-training", min_length=3, max_length=1024)
    rpc_timeout_seconds: int = Field(default=20, ge=1, le=25)
