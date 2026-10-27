"""不可变参数与审批快照。"""

from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

RISK = ("本机执行，无沙箱；命令可读写当前系统用户可访问的文件，包括 .env。"
        "工作目录不是访问边界；脱敏只覆盖已知密钥，不能防止其他敏感内容外泄。")


class CommandArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    argv: list[str] = Field(min_length=1, max_length=128)
    working_dir: str = "."
    timeout_seconds: int | None = Field(default=None, ge=1, le=300)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value):
        if any(not arg or len(arg) > 16384 or "\0" in arg for arg in value):
            raise ValueError("程序参数为空或过长")
        if any(arg in {"|", "||", "&&", ">", ">>", "<", ";"} for arg in value):
            raise ValueError("不支持管道、重定向和命令拼接")
        return value


@dataclass(frozen=True)
class ApprovalRequest:
    argv: tuple[str, ...]
    working_dir: str
    timeout_seconds: int
    permission_mode: str
    session_id: str
    generation: int
    identifier: str
    created: float
    contents: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(cls, argv, directory, timeout, mode, session, generation, contents=()):
        return cls(tuple(argv), str(directory), timeout, mode, session, generation, uuid4().hex,
                   monotonic(), tuple(contents))
