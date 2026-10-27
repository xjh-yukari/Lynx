"""可信的本地服务配置，不允许模型修改。"""

from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator


class MCPServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    transport: Literal["stdio", "streamable_http"] = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    cwd: str = ""
    url: str | None = None
    token_env: str | None = None
    token_file: str | None = None
    env: dict[str, SecretStr] = Field(default_factory=dict)
    allowed_tools: tuple[str, ...] = ()
    read_only_tools: tuple[str, ...] = ()
    allow_write: bool = False
    startup_timeout: float = Field(default=15, gt=0, le=120)
    call_timeout: float = Field(default=20, gt=0, le=300)
    shutdown_timeout: float = Field(default=10, gt=0, le=60)
    max_result_chars: int = Field(default=16000, ge=128, le=1000000)

    @model_validator(mode="after")
    def check_local_config(self):
        if self.transport == "stdio" and (not Path(self.command).is_absolute() or not Path(self.cwd).is_absolute()):
            raise ValueError("MCP command 与 cwd 必须使用绝对路径")
        if self.transport == "streamable_http":
            target = urlsplit(self.url or "")
            if (target.scheme != "http" or target.hostname not in {"127.0.0.1", "::1"}
                    or target.username or target.password or target.query or target.fragment or target.path != "/mcp"):
                raise ValueError("HTTP MCP 第一版仅接受本机 IP 的 http://地址:端口/mcp")
            if bool(self.token_env) == bool(self.token_file):
                raise ValueError("HTTP MCP 必须且只能指定 token_env 或 token_file")
            if self.token_env and not self.token_env.isidentifier():
                raise ValueError("HTTP MCP认证环境变量名无效")
            if self.token_file and not Path(self.token_file).is_absolute():
                raise ValueError("HTTP MCP认证文件必须使用绝对路径")
        if not set(self.read_only_tools).issubset(self.allowed_tools):
            raise ValueError("只读工具必须属于白名单")
        if any(
            not name or len(name) > 128 or not name.isprintable()
            for name in self.allowed_tools
        ):
            raise ValueError("工具白名单名称不合法")
        return self


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    servers: dict[str, MCPServerConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_names(self):
        import re

        if any(
            re.fullmatch(r"[a-z][a-z0-9_]{0,23}", name) is None for name in self.servers
        ):
            raise ValueError("服务名称只能使用小写字母、数字和下划线，最长 24 字符")
        return self
