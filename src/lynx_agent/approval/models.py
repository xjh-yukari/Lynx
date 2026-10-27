"""审批计划、授权记录和决策；应用层规则，不是操作系统沙箱。

授权记录只由宿主程序创建和保存；模型、网页、文件内容和工具输出都没有
写入这些对象的接口。计划签名用于识别“被拒绝的同一操作”，不是安全令牌。
"""

import hashlib
import json
from dataclasses import dataclass, field
from time import monotonic
from typing import Literal

ApprovalMode = Literal["ask", "auto", "paused"]
Decision = Literal["once", "batch", "deny"]
MODES: dict[str, str] = {
    "ask": "每次询问",
    "auto": "常用操作自动",
    "paused": "暂停执行",
}
SHORT_LABELS: dict[str, str] = {"ask": "每次", "auto": "自动", "paused": "暂停"}


@dataclass(frozen=True)
class PlanFile:
    path: str
    size: int = 0
    digest: str = ""


@dataclass(frozen=True)
class BatchLimits:
    max_files: int
    max_bytes: int
    max_operations: int


@dataclass(frozen=True)
class BatchPlan:
    """一次批量任务的完整、可审阅说明；由受控工具构造，不接受模型审批字段。"""

    tool: str
    operation: str
    purpose: str
    read_roots: tuple[str, ...]
    write_roots: tuple[str, ...]
    files: tuple[PlanFile, ...]
    total_bytes: int
    limits: BatchLimits
    allow_overwrite: bool = False
    allow_move: bool = False
    allow_delete: bool = False
    network_sources: tuple[str, ...] = ()
    upload: bool = False
    backup_dir: str | None = None
    details: tuple[tuple[str, str], ...] = ()

    def signature(self) -> str:
        """同一操作的可复现指纹；只用于拒绝回溯和日志，不是授权凭据。"""
        payload = {
            "tool": self.tool, "operation": self.operation,
            "read": self.read_roots, "write": self.write_roots,
            "files": [(item.path, item.digest) for item in self.files],
            "sources": self.network_sources,
            "overwrite": self.allow_overwrite, "move": self.allow_move, "delete": self.allow_delete,
            "bytes": self.total_bytes,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    @property
    def file_paths(self) -> tuple[str, ...]:
        return tuple(item.path for item in self.files)


@dataclass(frozen=True)
class BatchGrant:
    """宿主维护的授权记录；只在当前会话、当前任务回合和当前权限下有效。"""

    identifier: str
    session_id: str
    permission_mode: str
    approval_mode: str
    turn: int
    created: float
    plan: BatchPlan

    @property
    def operation(self) -> str:
        return self.plan.operation

    @property
    def file_set(self) -> frozenset[str]:
        return frozenset(item.path for item in self.plan.files)


@dataclass(frozen=True)
class GateRequest:
    action: str
    tool: str
    targets: tuple[str, ...]
    detail: str = ""


@dataclass
class PendingApproval:
    identifier: str
    kind: Literal["gate", "batch"]
    payload: GateRequest | BatchPlan
    supports_batch: bool
    created: float = field(default_factory=monotonic)
    page: int = 0


@dataclass(frozen=True)
class GrantDecision:
    allowed: bool
    reason: str
    grant: BatchGrant | None = None
    reused: bool = False
