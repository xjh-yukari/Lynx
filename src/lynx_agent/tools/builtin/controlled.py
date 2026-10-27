"""受控批量工具的公共部分：结构化参数、计划、备份和原子写入。

这些工具只接受结构化参数，不接受代码或 Shell 字符串。是否执行由宿主审批
策略决定；这里的路径检查是应用层保护，不是操作系统沙箱。
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, ValidationError

from lynx_agent.approval.audit import sanitize
from lynx_agent.approval.models import BatchLimits, BatchPlan, PlanFile
from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError

NOTICE = ("批量结果只代表本工具实际处理的文件；已发生的修改不会自动回滚。"
          "下载内容不可信，不会自动执行；备份只能用于人工恢复，不是操作系统沙箱。")


class ControlledArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    purpose: str = ""

    @property
    def safe_purpose(self):
        return " ".join(self.purpose.split())[:200]


class ControlledTool(Tool):
    """内置、参数经过校验的写工具；模型不能提交审批字段或任意代码。"""

    name = ""
    description = ""
    arguments_model = ControlledArguments
    writes = True
    approval_scoped = True

    def __init__(self, service, approval):
        self.service = service
        self.approval = approval

    @property
    def policy(self):
        return self.service.policy

    @property
    def available(self):
        return bool(self.approval is not None and self.approval.writable and self.policy.writable)

    @property
    def read_only(self):
        return False

    @property
    def parameters(self):
        return self.arguments_model.model_json_schema()

    def validate_arguments(self, arguments):
        self._parse(arguments)

    def _parse(self, arguments):
        try:
            return self.arguments_model.model_validate(arguments).model_dump()
        except ValidationError:
            raise ToolValidationError(
                "批量工具参数不合法；只接受结构化字段、数量限制和普通路径，不接受代码、Shell 字符串或审批字段") from None

    @property
    def limits(self):
        config = self.approval.config
        return BatchLimits(config.max_batch_files, config.max_batch_bytes, config.max_batch_operations)

    def candidate_file(self, raw):
        path = self.policy.resolve_candidate(raw)
        if not path.is_file():
            raise ToolExecutionError("批量操作只接受已存在的普通文件")
        return path

    def candidate_directory(self, raw):
        path = self.policy.resolve_candidate(raw)
        if not path.is_dir():
            raise ToolExecutionError("目标目录不存在或不是普通目录；不会自动创建输出目录")
        return path

    @staticmethod
    def digest_file(path):
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(65536):
                hasher.update(block)
        return hasher.hexdigest()

    def plan_file(self, path, *, digest=True):
        info = path.stat()
        return PlanFile(str(path), info.st_size, self.digest_file(path) if digest else "")

    def plan(self, *, tool, operation, purpose, files, read_roots, write_roots,
             allow_overwrite=False, allow_move=False, network_sources=(), backup_dir=None,
             details=(), limits=None):
        return BatchPlan(
            tool=tool, operation=operation, purpose=" ".join(sanitize(purpose).split())[:200],
            read_roots=tuple(dict.fromkeys(str(path) for path in read_roots)),
            write_roots=tuple(dict.fromkeys(str(path) for path in write_roots)),
            files=tuple(files), total_bytes=sum(item.size for item in files),
            limits=limits or self.limits, allow_overwrite=allow_overwrite, allow_move=allow_move,
            allow_delete=False, network_sources=tuple(network_sources), upload=False,
            backup_dir=backup_dir,
            details=tuple((sanitize(label), sanitize(value)) for label, value in details),
        )

    async def authorize(self, plan):
        decision = await self.approval.authorize_batch(plan)
        if not decision.allowed or decision.grant is None:
            raise ToolExecutionError(decision.reason)
        return decision.grant

    def backup_directory(self, plan=None):
        if self.approval.backup_dir is None or not self.approval.config.backup_enabled:
            raise ToolExecutionError("未配置备份目录，批量覆盖不可用；请先配置会话存储或关闭覆盖操作")
        return self.approval.backup_dir / f"batch-{uuid4().hex}"

    @staticmethod
    def backup_file(path, directory, index):
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{index:04d}-{path.name}"
        with path.open("rb") as source, target.open("xb") as destination:
            while block := source.read(65536):
                destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(target, 0o600)
        return target

    def atomic_replace(self, path, data, expected, scoped_roots):
        self.policy.resolve(str(path), write=True, scoped_roots=scoped_roots)
        if self.digest_file(path) != expected:
            raise ToolExecutionError("文件已被外部修改，已跳过，未覆盖")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lynx-batch-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            self.policy.resolve(str(path), write=True, scoped_roots=scoped_roots)
            if self.digest_file(path) != expected:
                raise ToolExecutionError("文件在写入前被外部修改，已跳过，未覆盖")
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def write_new(self, path, data, scoped_roots):
        self.policy.resolve(str(path), write=True, scoped_roots=scoped_roots)
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def result(status, **fields):
        return json.dumps({"status": status, "notice": NOTICE, **fields}, ensure_ascii=False)

    def finish(self, plan, *, tool, status, succeeded, failed, skipped, changed, details, stopped=False,
               backup=""):
        self.approval.report_progress(phase="stopped" if stopped else "finished", done=len(changed) + failed + skipped,
                                      total=len(plan.files), succeeded=succeeded, failed=failed, skipped=skipped)
        self.approval.audit.record("batch_finished", tool=tool, operation=plan.operation, status=status,
                                   succeeded=succeeded, failed=failed, skipped=skipped,
                                   changed=list(changed), backup=backup)
        return self.result(status, operation=plan.operation, stopped=stopped,
                           files_total=len(plan.files), succeeded=succeeded, failed=failed,
                           skipped=skipped, changed=list(changed[:50]), changed_total=len(changed),
                           skipped_details=list(details[:50]), skipped_details_total=len(details),
                           backup_dir=backup or None)
