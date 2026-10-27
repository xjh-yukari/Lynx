"""安全解压：固定实现，拒绝越界、链接、重名、设备文件和超限归档。"""

import asyncio
import os
import stat
import tarfile
import zipfile
from contextlib import contextmanager

from pydantic import Field

from lynx_agent.approval.models import BatchLimits, PlanFile
from lynx_agent.tools.builtin.controlled import ControlledArguments, ControlledTool
from lynx_agent.tools.errors import ToolExecutionError


class ExtractArguments(ControlledArguments):
    archive: str = Field(min_length=1, max_length=4096)
    destination: str = Field(min_length=1, max_length=4096)


class _Stopped(Exception):
    pass


def _safe_parts(name):
    if not name or len(name) > 1024 or "\0" in name or any(ord(char) < 32 for char in name):
        raise ToolExecutionError("归档成员名称不合法")
    if "\\" in name:
        raise ToolExecutionError("归档成员名称包含反斜杠，拒绝解压")
    normalized = name.rstrip("/")
    if not normalized:
        return ()
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ToolExecutionError("归档成员包含上级路径或空路径，拒绝解压")
    if normalized.startswith("/") or ":" in parts[0]:
        raise ToolExecutionError("归档成员使用绝对路径或盘符，拒绝解压")
    return tuple(parts)


class ExtractArchiveTool(ControlledTool):
    name = "extract_archive"
    description = ("把已下载或已存在的 ZIP/TAR 归档解压到授权输出目录。拒绝路径越界、符号链接、硬链接、"
                   "设备文件、重名成员和超限总大小；不覆盖同名文件，不执行归档内容。需要用户审批。不是沙箱。")
    arguments_model = ExtractArguments

    @property
    def extract_limits(self):
        config = self.approval.config
        return BatchLimits(min(config.max_batch_files, config.extract_max_files),
                           config.extract_max_bytes, config.max_batch_operations)

    def _members(self, archive):
        suffix = archive.suffix.casefold()
        if suffix == ".zip":
            with zipfile.ZipFile(archive) as bundle:
                result = []
                for info in bundle.infolist():
                    if info.flag_bits & 0x1:
                        raise ToolExecutionError("归档包含加密成员，本版本不解压")
                    mode = (info.external_attr >> 16) & 0o170000
                    if stat.S_ISLNK(mode):
                        raise ToolExecutionError("归档包含符号链接，拒绝解压")
                    parts = _safe_parts(info.filename)
                    if not parts:
                        continue
                    result.append((parts, info.is_dir(), info.file_size, info.filename))
                return "zip", result
        if suffix in {".tar", ".tgz", ".gz", ".bz2", ".xz"}:
            with tarfile.open(archive, "r:*") as bundle:
                result = []
                for member in bundle.getmembers():
                    if member.islnk() or member.issym() or member.ischr() or member.isblk() or member.isfifo() or member.isdev():
                        raise ToolExecutionError("归档包含链接或特殊文件，拒绝解压")
                    parts = _safe_parts(member.name)
                    if not parts:
                        continue
                    result.append((parts, member.isdir(), member.size, member.name))
                return "tar", result
        raise ToolExecutionError("只支持 ZIP、TAR、TAR.GZ 归档")

    def build_plan(self, arguments):
        archive = self.candidate_file(arguments["archive"])
        destination = self.candidate_directory(arguments["destination"])
        archive_digest = self.digest_file(archive)
        kind, members = self._members(archive)
        if len(members) > self.extract_limits.max_files:
            raise ToolExecutionError("归档成员数超过执行上限；未解压，请拆分归档")
        total = sum(size for _, is_dir, size, _ in members if not is_dir)
        if total > self.extract_limits.max_bytes:
            raise ToolExecutionError("归档解压后总大小超过配置上限；未解压")
        if total and archive.stat().st_size and total / archive.stat().st_size > self.approval.config.extract_max_ratio:
            raise ToolExecutionError("归档压缩比异常，可能为压缩炸弹；未解压")
        entries, files, seen, skipped = [], [], set(), 0
        for parts, is_dir, size, raw_name in members:
            target = destination.joinpath(*parts)
            key = str(target).casefold()
            if key in seen:
                raise ToolExecutionError(f"归档包含重名成员：{raw_name}")
            seen.add(key)
            self.policy.resolve(str(target), write=True, scoped_roots=(destination,))
            if is_dir:
                entries.append(("dir", parts, target, 0))
                continue
            if target.exists():
                skipped += 1
                entries.append(("skip", parts, target, size))
                continue
            entries.append(("file", parts, target, size))
            files.append(PlanFile(str(target), size, ""))
        if not files:
            raise ToolExecutionError("归档没有可写出的新文件（可能都已存在）；未解压")
        plan = self.plan(
            tool=self.name, operation="extract", purpose=arguments.get("purpose", ""),
            files=files, read_roots={archive.parent}, write_roots={destination},
            details=(("归档格式", kind), ("归档文件", str(archive)),
                     ("归档 SHA-256", archive_digest),
                     ("成员总数", str(len(members))), ("解压总大小", f"{total} 字节"),
                     ("已存在跳过", str(skipped)), ("覆盖", "禁止"),
                     ("链接/设备文件", "拒绝"), ("执行", "不自动运行任何解压内容")),
            limits=self.extract_limits,
        )
        return plan, entries, archive, archive_digest

    @contextmanager
    def _open_archive(self, archive):
        try:
            if archive.suffix.casefold() == ".zip":
                with zipfile.ZipFile(archive) as bundle:
                    yield bundle
            else:
                with tarfile.open(archive, "r:*") as bundle:
                    yield bundle
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as error:
            raise ToolExecutionError("归档无法安全打开或成员不可读，未完成解压") from error

    async def execute(self, **arguments):
        parsed = self._parse(arguments)
        plan, entries, archive, archive_digest = self.build_plan(parsed)
        grant = await self.authorize(plan)
        if self.digest_file(archive) != archive_digest:
            raise ToolExecutionError("归档在审批后已被外部修改，授权作废；未解压")
        write_roots = tuple(grant.plan.write_roots)
        destination = self.policy.resolve(grant.plan.write_roots[0], write=True, scoped_roots=write_roots)
        changed, skipped, failed, errors = [], [], 0, []
        processed = 0
        with self._open_archive(archive) as bundle:
            for kind, parts, target, size in entries:
                if self.approval.should_stop():
                    return self.finish(plan, tool=self.name, status="stopped", succeeded=len(changed),
                                       failed=failed, skipped=len(skipped), changed=changed,
                                       details=skipped + errors, stopped=True)
                if kind == "skip":
                    skipped.append({"path": str(target), "reason": "目标已存在，不覆盖"})
                    continue
                try:
                    target = self.policy.resolve(str(target), write=True, scoped_roots=write_roots)
                    if not target.is_relative_to(destination):
                        raise ToolExecutionError("成员路径越界，已跳过")
                    if kind == "dir":
                        self._mkdir(target, destination)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self.policy.resolve(str(target), write=True, scoped_roots=write_roots)
                    if target.exists():
                        skipped.append({"path": str(target), "reason": "目标已存在，不覆盖"})
                        continue
                    source = self._open_member(bundle, parts)
                    if source is None:
                        raise ToolExecutionError("归档成员不是普通文件")
                    with source, target.open("xb") as writer:
                        written = 0
                        try:
                            while block := source.read(65536):
                                if self.approval.should_stop():
                                    raise _Stopped
                                written += len(block)
                                if written > size:
                                    raise ToolExecutionError("解压成员超过声明大小，已删除")
                                writer.write(block)
                            writer.flush()
                            os.fsync(writer.fileno())
                        except BaseException:
                            writer.close()
                            target.unlink(missing_ok=True)
                            raise
                    changed.append(str(target))
                    self.approval.record_operation(tool=self.name, authorization=grant.identifier,
                                                   target=str(target), result="extracted",
                                                   detail=f"{written} 字节")
                except _Stopped:
                    target.unlink(missing_ok=True)
                    return self.finish(plan, tool=self.name, status="stopped", succeeded=len(changed),
                                       failed=failed, skipped=len(skipped), changed=changed,
                                       details=skipped + errors, stopped=True)
                except (ToolExecutionError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
                    target.unlink(missing_ok=True)
                    failed += 1
                    errors.append({"path": str(target), "reason": str(error)})
                finally:
                    processed += 1
                    self.approval.report_progress(phase="running", done=processed, total=len(entries),
                                                  succeeded=len(changed), failed=failed, skipped=len(skipped))
                    await asyncio.sleep(0)
        status = "completed" if not failed else "partial"
        return self.finish(plan, tool=self.name, status=status, succeeded=len(changed), failed=failed,
                           skipped=len(skipped), changed=changed, details=skipped + errors)

    def _open_member(self, bundle, parts):
        name = "/".join(parts)
        if isinstance(bundle, zipfile.ZipFile):
            info = next((item for item in bundle.infolist() if item.filename.rstrip("/") == name), None)
            return bundle.open(info) if info is not None else None
        member = next((item for item in bundle.getmembers() if item.name.rstrip("/") == name), None)
        return bundle.extractfile(member) if member is not None else None

    def _mkdir(self, target, destination):
        if not target.is_relative_to(destination):
            raise ToolExecutionError("目录成员路径越界")
        target.mkdir(parents=True, exist_ok=True)
