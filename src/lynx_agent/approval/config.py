"""审批策略配置；这里只有限额和默认档位，不提供“永久放行任意程序”。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ApprovalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    # 启动默认：工作区修改 + 常用操作自动。运行期的切换不写回配置文件。
    mode: Literal["ask", "auto", "paused"] = "auto"
    approval_timeout_seconds: float = Field(default=600, gt=0, le=600)
    max_batch_files: int = Field(default=200, ge=1, le=5000)
    max_batch_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    max_batch_operations: int = Field(default=1000, ge=1, le=100000)
    preview_page_size: int = Field(default=10, ge=1, le=100)
    # 批量覆盖前的同目录外备份；备份目录属于内部保护范围。
    backup_enabled: bool = True
    download_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    download_timeout_seconds: float = Field(default=60, gt=0, le=300)
    download_max_redirects: int = Field(default=5, ge=0, le=10)
    extract_max_files: int = Field(default=200, ge=1, le=10000)
    extract_max_bytes: int = Field(default=50 * 1024 * 1024, ge=1024)
    extract_max_ratio: int = Field(default=200, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_batch_limits(self):
        if self.max_batch_bytes < 1024:
            raise ValueError("批量任务字节上限过小")
        return self
