"""确定性审批规则与宿主维护的批量授权；不是沙箱，也不依赖 AI 审核模型。"""

import asyncio
import threading
from pathlib import Path
from time import monotonic
from uuid import uuid4

from lynx_agent.approval.audit import AuditLog
from lynx_agent.approval.cards import page_count, pending_text
from lynx_agent.approval.config import ApprovalConfig
from lynx_agent.approval.models import (
    MODES,
    BatchGrant,
    BatchPlan,
    Decision,
    GateRequest,
    GrantDecision,
    PendingApproval,
)
from lynx_agent.tools.errors import ToolExecutionError

READ_ACTIONS = {"read"}
WRITE_ACTIONS = {"create", "replace", "rename", "copy", "download", "extract"}


class ApprovalManager:
    """一个 Runner 内的审批状态；所有授权只存在于内存，不写回配置。"""

    def __init__(self, config: ApprovalConfig, *, session_id, permission_mode, data_dir=None,
                 backup_dir=None, secrets=()):
        self.config = config
        self.mode = config.mode
        self._session_id = session_id
        self._permission_mode = permission_mode
        self.presenter = None
        self.on_progress = None
        self.progress = None
        self.pending = None
        self.generation = 0
        self.turn = 0
        self.grants = {}
        self.denied_paths = set()
        self._denied_signatures = set()
        self._future = None
        self._lock = asyncio.Lock()
        self.stop_event = threading.Event()
        self.secrets = tuple(secrets)
        self.audit = AuditLog(data_dir)
        self.backup_dir = Path(backup_dir).absolute() if backup_dir is not None else None

    @property
    def label(self):
        return MODES[self.mode]

    @property
    def writable(self):
        return self.mode != "paused"

    @property
    def has_presenter(self):
        return self.presenter is not None

    def clean(self, text):
        value = str(text)
        for secret in self.secrets:
            if secret:
                value = value.replace(secret, "[已隐藏]")
        return value

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError("审批档位必须为 ask、auto 或 paused")
        if mode == self.mode:
            return
        self.mode = mode
        self.invalidate("approval_mode_changed")
        self.audit.record("approval_mode", mode=mode, label=MODES[mode])

    def begin_turn(self):
        self.turn += 1
        self.grants.clear()
        self.denied_paths.clear()
        self._denied_signatures.clear()
        self.stop_event.clear()
        self.progress = None
        self.audit.record("turn_begin", turn=self.turn, mode=self.mode)

    def end_turn(self):
        self.grants.clear()
        self.pending = None
        self.stop_event.clear()
        self.progress = None
        self._denied_signatures.clear()
        self.denied_paths.clear()
        self.audit.record("turn_end", turn=self.turn, mode=self.mode)

    def invalidate(self, reason="invalidate"):
        """切换会话、权限、审批档位或撤销时作废授权与待审批卡片。"""
        self.generation += 1
        self.grants.clear()
        self.denied_paths.clear()
        self._denied_signatures.clear()
        self._deny_pending()
        self.audit.record("invalidate", reason=reason, turn=self.turn)

    def revoke(self):
        self.invalidate("user_revoked")
        self.audit.record("approval_revoked", mode=self.mode)

    def _deny_pending(self):
        if self._future is not None and not self._future.done():
            self._future.set_result("deny")
        self.pending = None

    def request_stop(self):
        self.stop_event.set()
        self.audit.record("stop_requested", turn=self.turn)
        return "已请求停止当前批量操作；已发生的修改会如实报告，不会自动回滚。"

    def should_stop(self):
        return self.stop_event.is_set()

    def report_progress(self, *, phase, done, total, succeeded, failed, skipped, message=""):
        self.progress = {"phase": phase, "done": done, "total": total,
                         "succeeded": succeeded, "failed": failed, "skipped": skipped,
                         "message": self.clean(message)}
        if self.on_progress is not None:
            try:
                self.on_progress(self.progress)
            except Exception:  # noqa: BLE001 - 宿主进度回调失败不能中断批量任务
                self.audit.record("progress_callback_failed")

    def progress_text(self):
        if not self.progress:
            return ""
        item = self.progress
        text = (f"批量任务：{item['phase']} {item['done']}/{item['total']} · "
                f"成功 {item['succeeded']} · 失败 {item['failed']} · 跳过 {item['skipped']}")
        return text + (f" · {item['message']}" if item["message"] else "")

    def record_operation(self, *, tool, authorization, target, result, detail="", recovery=""):
        self.audit.record("operation", tool=self.clean(tool), authorization=self.clean(authorization),
                          target=self.clean(target), result=self.clean(result),
                          detail=self.clean(detail), recovery=self.clean(recovery), turn=self.turn)

    def _grant(self, plan, session):
        return BatchGrant(identifier=uuid4().hex, session_id=session or "",
                          permission_mode=self._permission_mode(), approval_mode=self.mode,
                          turn=self.turn, created=monotonic(), plan=plan)

    def _matches(self, grant, plan):
        if grant.turn != self.turn or grant.approval_mode != self.mode:
            return False
        if grant.session_id != (self._session_id() or ""):
            return False
        if grant.permission_mode != self._permission_mode():
            return False
        if grant.operation != plan.operation:
            return False
        if not set(plan.read_roots) <= set(grant.plan.read_roots):
            return False
        if not set(plan.write_roots) <= set(grant.plan.write_roots):
            return False
        if not set(plan.network_sources) <= set(grant.plan.network_sources):
            return False
        if not set(plan.file_paths) <= grant.file_set:
            return False
        if plan.allow_overwrite and not grant.plan.allow_overwrite:
            return False
        if plan.allow_move and not grant.plan.allow_move:
            return False
        return not (plan.allow_delete and not grant.plan.allow_delete)

    def matching_grant(self, plan):
        if self.mode != "auto":
            return None
        return next((grant for grant in self.grants.values() if self._matches(grant, plan)), None)

    async def _request(self, pending: PendingApproval):
        if self.presenter is None:
            self.audit.record("approval_unavailable", kind=pending.kind, identifier=pending.identifier)
            return "deny"
        async with self._lock:
            self.pending = pending
            self._future = asyncio.get_running_loop().create_future()
            try:
                self.presenter(pending)
                try:
                    async with asyncio.timeout(self.config.approval_timeout_seconds):
                        decision = await self._future
                except TimeoutError:
                    self.audit.record("approval_timeout", kind=pending.kind, identifier=pending.identifier)
                    decision = "deny"
            except Exception:  # noqa: BLE001 - 审批界面异常时只能拒绝
                decision = "deny"
            finally:
                self.pending = None
                self._future = None
                try:
                    self.presenter(None)
                except Exception:  # noqa: BLE001 - 关闭卡片失败不影响拒绝结果
                    self.audit.record("presenter_close_failed")
            return decision

    def decide(self, identifier, decision: bool | Decision):
        if isinstance(decision, bool):
            decision = "once" if decision else "deny"
        if decision not in {"once", "batch", "deny"}:
            return False
        if self.pending is None or self.pending.identifier != identifier or self._future is None or self._future.done():
            return False
        if decision == "batch" and not self.pending.supports_batch:
            return False
        self._future.set_result(decision)
        self.audit.record("approval_decision", kind=self.pending.kind, identifier=identifier, decision=decision)
        return True

    def set_page(self, delta):
        if self.pending is None or self.pending.kind != "batch":
            return None
        total = page_count(self.pending.payload, self.config.preview_page_size)
        self.pending.page = max(0, min(total - 1, self.pending.page + delta))
        return self.pending_text()

    def pending_text(self):
        if self.pending is None:
            return "没有待审批的批量任务。"
        return pending_text(self.pending, self.config.preview_page_size)

    def approvals(self):
        pending = self.pending_text()
        if self.pending is None:
            return ("没有待审批的批量任务。\n"
                    f"审批档位：{self.label}；批量授权只由终端界面创建，模型和外部内容不能批准。\n"
                    "文件覆盖差异仍使用 /approvals、/approve、/reject 流程。")
        hint = "允许一次：/approve <编号>；拒绝：/reject <编号>"
        if self.pending.supports_batch:
            hint += "；允许本批次请在 TUI 按 B（命令行为 /approve <编号> batch）"
        return f"审批 {self.pending.identifier}\n{pending}\n{hint}"

    def _check_denied(self, plan):
        if plan.signature() in self._denied_signatures:
            raise ToolExecutionError("用户已拒绝该操作；本回合不能换工具或包装方式重试同一操作")
        denied = [path for path in plan.file_paths if path.casefold() in self.denied_paths]
        if denied:
            raise ToolExecutionError("用户已拒绝写入这些目标；本回合不能换工具重试：" + ", ".join(denied[:3]))

    def _record_denial(self, plan):
        self._denied_signatures.add(plan.signature())
        self.denied_paths.update(path.casefold() for path in plan.file_paths)
        self.audit.record("operation_denied", tool=plan.tool, operation=plan.operation,
                          files=len(plan.files), turn=self.turn)

    def path_denied(self, path):
        return str(path).casefold() in self.denied_paths

    def ensure_commit_allowed(self):
        """文件覆盖审批最终提交前再次确认审批档位。"""
        if self.mode == "paused":
            raise ValueError("审批档位为暂停执行：已保留预览，但不能提交修改；请先切换审批档位")

    async def allow_gate(self, action, tool, targets, detail=""):
        """只读/新建等单步操作的审批；auto 下只读与新建自动放行。"""
        normalized = tuple(str(target) for target in targets)
        if action in WRITE_ACTIONS and self._permission_mode() == "read":
            raise ToolExecutionError("当前文件权限为仅查看，禁止写入；审批不能突破权限限制")
        if action in WRITE_ACTIONS and any(self.path_denied(path) for path in normalized):
            raise ToolExecutionError("用户已拒绝该目标；本回合不能换工具或包装方式重试")
        if self.mode == "paused":
            if action in READ_ACTIONS:
                return True
            raise ToolExecutionError("审批档位为暂停执行：可以分析和生成预览，但不能修改、下载或执行命令")
        if self.mode == "auto":
            self.audit.record("gate_auto", action=action, tool=tool, targets=list(normalized))
            return True
        if self.presenter is None:
            raise ToolExecutionError("当前入口没有人工审批界面，未执行；请改用交互界面或调整审批档位")
        pending = PendingApproval(uuid4().hex[:12], "gate", GateRequest(action, tool, normalized, detail),
                                  supports_batch=False)
        decision = await self._request(pending)
        if decision == "deny":
            self.denied_paths.update(path.casefold() for path in normalized if action in WRITE_ACTIONS)
            self.audit.record("gate_denied", action=action, tool=tool, targets=list(normalized))
            raise ToolExecutionError("用户拒绝或审批过期；操作没有执行。不要换工具重试同一操作，先询问用户")
        self.audit.record("gate_approved", action=action, tool=tool, targets=list(normalized), decision=decision)
        return True

    async def authorize_batch(self, plan: BatchPlan) -> GrantDecision:
        """请求一次可审阅的批量授权；auto 档位复用当前任务内已批准范围。"""
        if self.mode == "paused":
            raise ToolExecutionError("审批档位为暂停执行：已生成批量计划，但不会修改、下载或执行")
        if self._permission_mode() == "read":
            raise ToolExecutionError("当前文件权限为仅查看，禁止批量写入、下载或解压；审批不能突破权限限制")
        if plan.upload:
            raise ToolExecutionError("本版本不提供上传工具，不能批准上传授权")
        if plan.allow_delete:
            raise ToolExecutionError("本版本不提供永久删除工具，不能批准删除授权")
        if not plan.files:
            raise ToolExecutionError("批量任务没有可执行文件，未申请授权")
        if len(plan.files) > self.config.max_batch_files or plan.total_bytes > self.config.max_batch_bytes:
            raise ToolExecutionError(
                f"批量任务超过配置上限（{self.config.max_batch_files} 个文件 / "
                f"{self.config.max_batch_bytes} 字节）；未执行，请拆分任务")
        if len(plan.files) > plan.limits.max_operations:
            raise ToolExecutionError("批量任务操作次数超过执行上限；未执行，请拆分任务")
        self._check_denied(plan)
        grant = self.matching_grant(plan)
        if grant is not None:
            self.audit.record("grant_reused", grant=grant.identifier, operation=plan.operation,
                              files=len(plan.files), turn=self.turn)
            return GrantDecision(True, "复用当前任务已批准的授权范围", grant, True)
        session = self._session_id() or ""
        supports_batch = self.mode == "auto"
        pending = PendingApproval(uuid4().hex[:12], "batch", plan, supports_batch=supports_batch)
        decision = await self._request(pending)
        if decision == "deny":
            self._record_denial(plan)
            return GrantDecision(False, "用户拒绝、审批过期或没有审批界面；批量任务没有执行")
        if decision == "batch" and not supports_batch:
            decision = "once"
        created = self._grant(plan, session)
        if decision == "batch":
            self.grants[created.identifier] = created
        self.audit.record("grant_created", grant=created.identifier, operation=plan.operation,
                          decision=decision, files=len(plan.files), bytes=plan.total_bytes,
                          writable=list(plan.write_roots), readable=list(plan.read_roots),
                          sources=list(plan.network_sources), turn=self.turn)
        return GrantDecision(True, "用户允许一次" if decision == "once" else "用户允许本批次",
                             created, False)

    def status_text(self):
        return (f"审批档位：{self.label}\n"
                f"批量任务上限：{self.config.max_batch_files} 个文件 / "
                f"{self.config.max_batch_bytes} 字节；下载上限 {self.config.download_max_bytes} 字节\n"
                "自动范围仅限代码审查过的内置工具；任意程序、脚本、测试和构建仍需逐次审批。\n"
                "这是应用层权限与审批机制，不是操作系统沙箱，不能限制获批程序的实际访问范围。\n"
                "授权仅限当前任务与当前会话；切换会话、重启、权限或审批档位变化后失效。")
