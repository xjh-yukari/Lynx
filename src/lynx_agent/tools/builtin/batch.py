"""批量文本替换、复制和改名：固定实现，模型只提交结构化参数。"""

import asyncio
import codecs
import hashlib
import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from lynx_agent.approval.models import PlanFile
from lynx_agent.tools.builtin.controlled import ControlledArguments, ControlledTool
from lynx_agent.tools.errors import ToolExecutionError

MAX_TEXT_BYTES = 1024 * 1024


class ReplaceArguments(ControlledArguments):
    paths: list[str] = Field(min_length=1, max_length=200)
    old_text: str = Field(min_length=1, max_length=4096)
    new_text: str = Field(max_length=4096)


class CopyArguments(ControlledArguments):
    sources: list[str] = Field(min_length=1, max_length=200)
    destination: str = Field(min_length=1, max_length=4096)


class RenameItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: str = Field(min_length=1, max_length=4096)
    new_name: str = Field(min_length=1, max_length=255)


class RenameArguments(ControlledArguments):
    renames: list[RenameItem] = Field(min_length=1, max_length=200)


class BatchReplaceTextTool(ControlledTool):
    name = "batch_replace_text"
    description = ("在已授权批量任务中，对明确列出的 UTF-8 文本文件做固定字符串全部替换。"
                   "只接受结构化路径和文本，不接受正则、代码或 Shell；覆盖前生成备份并校验原文件 SHA-256，"
                   "文件被外部修改时跳过。需要用户在审批卡片中允许一次或本批次。不是沙箱。")
    arguments_model = ReplaceArguments

    def build_plan(self, arguments):
        if arguments["old_text"] == arguments["new_text"]:
            raise ToolExecutionError("新旧文本相同，没有可执行修改")
        files, roots = [], set()
        for raw in arguments["paths"]:
            path = self.candidate_file(raw)
            self.policy.text_file(path)
            self.policy.resolve(str(path), write=True, scoped_roots=(path.parent,))
            if path.stat().st_size > MAX_TEXT_BYTES:
                raise ToolExecutionError(f"文件超过 1 MiB，批量文本替换跳过：{path}")
            files.append(self.plan_file(path))
            roots.add(path.parent)
        backup = self.backup_directory()
        chosen = tuple(files)
        return chosen, self.plan(
            tool=self.name, operation="replace_text", purpose=arguments.get("purpose", ""),
            files=files, read_roots=roots, write_roots=roots, allow_overwrite=True,
            backup_dir=str(backup),
            details=(("替换预览", f"{arguments['old_text'][:80]!r} → {arguments['new_text'][:80]!r}"),
                     ("替换", "固定字符串全部替换"), ("编码", "UTF-8（保留 BOM）")),
        )

    async def execute(self, **arguments):
        parsed = self._parse(arguments)
        files, plan = self.build_plan(parsed)
        grant = await self.authorize(plan)
        write_roots = tuple(grant.plan.write_roots)
        old, new = parsed["old_text"], parsed["new_text"]
        backup_root = (Path(grant.plan.backup_dir) / f"run-{uuid4().hex}"
                       if grant.plan.backup_dir else None)
        changed, skipped, failed, errors = [], [], 0, []
        for index, item in enumerate(files, 1):
            if self.approval.should_stop():
                return self.finish(plan, tool=self.name, status="stopped", succeeded=len(changed),
                                   failed=failed, skipped=len(skipped), changed=changed, details=skipped,
                                   stopped=True, backup=str(backup_root or ""))
            try:
                path = self.policy.resolve(item.path, write=True, scoped_roots=write_roots)
                if not path.is_file() or self.digest_file(path) != item.digest:
                    skipped.append({"path": item.path, "reason": "文件已被外部修改或不再是普通文件"})
                    continue
                data = path.read_bytes()
                bom = data.startswith(codecs.BOM_UTF8)
                try:
                    text = data.decode("utf-8-sig" if bom else "utf-8")
                except UnicodeDecodeError:
                    skipped.append({"path": item.path, "reason": "不是 UTF-8 文本，已跳过"})
                    continue
                if old not in text:
                    skipped.append({"path": item.path, "reason": "未找到匹配文本，已跳过"})
                    continue
                replacement = text.replace(old, new)
                encoded = replacement.encode("utf-8")
                if bom:
                    encoded = codecs.BOM_UTF8 + encoded
                if len(encoded) > MAX_TEXT_BYTES:
                    skipped.append({"path": item.path, "reason": "替换后超过 1 MiB，已跳过"})
                    continue
                backup_path = None
                if backup_root is not None:
                    backup_path = self.backup_file(path, backup_root, index)
                self.atomic_replace(path, encoded, item.digest, write_roots)
                changed.append(item.path)
                self.approval.record_operation(
                    tool=self.name, authorization=grant.identifier, target=item.path, result="replaced",
                    detail="固定字符串替换", recovery=str(backup_path) if backup_path else "")
            except (ToolExecutionError, OSError) as error:
                failed += 1
                errors.append({"path": item.path, "reason": str(error)})
            finally:
                self.approval.report_progress(phase="running", done=index, total=len(files),
                                              succeeded=len(changed), failed=failed, skipped=len(skipped))
                await asyncio.sleep(0)
        status = "completed" if not failed else "partial"
        return self.finish(plan, tool=self.name, status=status, succeeded=len(changed), failed=failed,
                           skipped=len(skipped), changed=changed, details=skipped + errors,
                           backup=str(backup_root or ""))


class BatchCopyFilesTool(ControlledTool):
    name = "batch_copy_files"
    description = ("把明确列出的文件复制到已授权输出目录，只新建文件，绝不覆盖同名文件。"
                   "需要用户在审批卡片中允许一次或本批次；不执行、不读取正文给模型。不是沙箱。")
    arguments_model = CopyArguments

    def build_plan(self, arguments):
        destination = self.candidate_directory(arguments["destination"])
        sources, targets, mapping, read_roots = [], [], {}, set()
        seen = set()
        for raw in arguments["sources"]:
            source = self.candidate_file(raw)
            target = destination / source.name
            key = str(target).casefold()
            if key in seen:
                raise ToolExecutionError(f"多个来源会写到同一目标：{target}")
            self.policy.resolve(str(target), write=True, scoped_roots=(destination,))
            seen.add(key)
            sources.append(source)
            read_roots.add(source.parent)
            mapping[str(target)] = source
            targets.append(self.plan_file(source))
        # 目标文件尚不存在，计划里记录目标路径与来源大小/校验值。
        files = tuple(
            PlanFile(str(destination / source.name), info.size, info.digest)
            for source, info in zip(sources, targets, strict=True)
        )
        plan = self.plan(
            tool=self.name, operation="copy_files", purpose=arguments.get("purpose", ""),
            files=files, read_roots=read_roots, write_roots={destination},
            details=(("覆盖", "禁止；同名目标会跳过"),
                     ("已存在目标（将跳过）", str(sum(1 for item in files if Path(item.path).exists()))),
                     ("来源文件数", str(len(sources)))),
        )
        return plan, mapping

    async def execute(self, **arguments):
        parsed = self._parse(arguments)
        plan, mapping = self.build_plan(parsed)
        grant = await self.authorize(plan)
        write_roots = tuple(grant.plan.write_roots)
        read_roots = tuple(grant.plan.read_roots)
        destination = self.policy.resolve(grant.plan.write_roots[0], write=True, scoped_roots=write_roots)
        changed, skipped, failed, errors = [], [], 0, []
        for index, item in enumerate(plan.files, 1):
            if self.approval.should_stop():
                return self.finish(plan, tool=self.name, status="stopped", succeeded=len(changed),
                                   failed=failed, skipped=len(skipped), changed=changed,
                                   details=skipped + errors, stopped=True)
            source = mapping.get(item.path)
            try:
                if source is None:
                    skipped.append({"path": item.path, "reason": "计划缺少来源，已跳过"})
                    continue
                source = self.policy.resolve(str(source), scoped_roots=read_roots)
                target = self.policy.resolve(item.path, write=True, scoped_roots=write_roots)
                if target.parent != destination:
                    raise ToolExecutionError("目标目录已变化，拒绝复制")
                if target.exists():
                    skipped.append({"path": item.path, "reason": "目标已存在，不覆盖"})
                    continue
                if self.digest_file(source) != item.digest:
                    skipped.append({"path": item.path, "reason": "来源文件已被外部修改，已跳过"})
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                self._copy_file(source, target, item.size)
                changed.append(item.path)
                self.approval.record_operation(tool=self.name, authorization=grant.identifier,
                                               target=item.path, result="copied",
                                               detail=f"来源 {source}")
            except (ToolExecutionError, OSError) as error:
                failed += 1
                errors.append({"path": item.path, "reason": str(error)})
            finally:
                self.approval.report_progress(phase="running", done=index, total=len(plan.files),
                                              succeeded=len(changed), failed=failed, skipped=len(skipped))
                await asyncio.sleep(0)
        status = "completed" if not failed else "partial"
        return self.finish(plan, tool=self.name, status=status, succeeded=len(changed), failed=failed,
                           skipped=len(skipped), changed=changed, details=skipped + errors)

    def _copy_file(self, source, target, max_bytes):
        hasher = hashlib.sha256()
        written = 0
        with source.open("rb") as reader, target.open("xb") as writer:
            try:
                while block := reader.read(65536):
                    written += len(block)
                    if written > max_bytes:
                        raise ToolExecutionError("来源文件在审批后增大，已删除不完整目标")
                    hasher.update(block)
                    writer.write(block)
                writer.flush()
                os.fsync(writer.fileno())
            except BaseException:
                writer.close()
                target.unlink(missing_ok=True)
                raise
        if hasher.hexdigest() != self.digest_file(source):
            target.unlink(missing_ok=True)
            raise ToolExecutionError("复制过程中来源文件发生变化，已删除不完整目标")


class BatchRenameFilesTool(ControlledTool):
    name = "batch_rename_files"
    description = ("在已授权目录内批量改名，只接受新文件名，不移动目录、不覆盖同名文件、不删除内容。"
                   "需要用户在审批卡片中允许一次或本批次；审计记录新旧名称用于人工恢复。不是沙箱。")
    arguments_model = RenameArguments

    def build_plan(self, arguments):
        pairs, files, roots = [], [], set()
        seen = set()
        for item in arguments["renames"]:
            source = self.candidate_file(item["source"])
            name = item["new_name"]
            if any(char in name for char in '/\\:*?"<>|') or name in {".", ".."}:
                raise ToolExecutionError("新文件名只能是单个文件名，不能包含路径或非法字符")
            target = source.parent / name
            key = str(target).casefold()
            if key in seen:
                raise ToolExecutionError(f"多个改名会写到同一目标：{target}")
            self.policy.resolve(str(target), write=True, scoped_roots=(source.parent,))
            seen.add(key)
            pairs.append((source, target))
            roots.add(source.parent)
            files.append(self.plan_file(source))
        targets = tuple(
            PlanFile(str(target), info.size, info.digest)
            for (_, target), info in zip(pairs, files, strict=True)
        )
        plan = self.plan(
            tool=self.name, operation="rename_files", purpose=arguments.get("purpose", ""),
            files=targets, read_roots=roots, write_roots=roots, allow_move=True,
            details=(("移动", "仅同一目录内改名"), ("覆盖", "禁止；同名目标会跳过"),
                     ("已存在目标（将跳过）", str(sum(1 for item in targets if Path(item.path).exists())))),
        )
        return plan, pairs

    async def execute(self, **arguments):
        parsed = self._parse(arguments)
        plan, pairs = self.build_plan(parsed)
        grant = await self.authorize(plan)
        write_roots = tuple(grant.plan.write_roots)
        changed, skipped, failed, errors = [], [], 0, []
        for index, ((source, target), item) in enumerate(zip(pairs, plan.files, strict=True), 1):
            if self.approval.should_stop():
                return self.finish(plan, tool=self.name, status="stopped", succeeded=len(changed),
                                   failed=failed, skipped=len(skipped), changed=changed,
                                   details=skipped + errors, stopped=True)
            try:
                source = self.policy.resolve(str(source), write=True, scoped_roots=write_roots)
                target = self.policy.resolve(str(target), write=True, scoped_roots=write_roots)
                if source.parent != target.parent:
                    raise ToolExecutionError("改名不能跨目录")
                if target.exists():
                    skipped.append({"path": str(target), "reason": "目标已存在，不覆盖"})
                    continue
                if self.digest_file(source) != item.digest:
                    skipped.append({"path": str(source), "reason": "文件已被外部修改，已跳过"})
                    continue
                if os.name == "nt":
                    os.rename(source, target)
                else:
                    # POSIX rename 会覆盖目标；先建硬链接保证不覆盖，再删除原名。
                    os.link(source, target)
                    try:
                        os.unlink(source)
                    except OSError:
                        target.unlink(missing_ok=True)
                        raise
                changed.append(str(target))
                self.approval.record_operation(tool=self.name, authorization=grant.identifier,
                                               target=str(target), result="renamed",
                                               detail="原名 " + str(source), recovery=f"{target} <- {source}")
            except (ToolExecutionError, OSError) as error:
                failed += 1
                errors.append({"path": str(item.path), "reason": str(error)})
            finally:
                self.approval.report_progress(phase="running", done=index, total=len(plan.files),
                                              succeeded=len(changed), failed=failed, skipped=len(skipped))
                await asyncio.sleep(0)
        status = "completed" if not failed else "partial"
        return self.finish(plan, tool=self.name, status=status, succeeded=len(changed), failed=failed,
                           skipped=len(skipped), changed=changed, details=skipped + errors)


def create_batch_tools(service, approval):
    return [tool(service, approval) for tool in (BatchReplaceTextTool, BatchCopyFilesTool, BatchRenameFilesTool)]
