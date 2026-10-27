"""有界、只读的项目观察器；日志证据不等于活跃进程，不执行项目代码。"""

import json
import math
import os
import re
import stat
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from lynx_training_worker.registry import safe_path

SKIP = {"data", "datasets", "models", "node_modules", "__pycache__", "venv", "env", "teacher"}
JSON_NAMES = {"config.json", "summary.json", "trainer_state.json", "metrics.jsonl"}
MODEL_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin"}


def timestamp(value=None):
    return datetime.fromtimestamp(value, UTC).isoformat() if value is not None else datetime.now(UTC).isoformat()


def clean_text(text):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"(?i)(bearer\s+)\S+", r"\1[REDACTED]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[REDACTED]", text)
    return "".join(c for c in text.replace("\r", "\n") if c in "\n\t" or ord(c) >= 32 and ord(c) != 127)


def bounded_int(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("invalid_request")
    return value


def kind(path):
    if path.suffix == ".log" or path.name in {"stdout.txt", "stderr.txt"}:
        return "log"
    if path.name in JSON_NAMES:
        return "metrics" if path.name == "metrics.jsonl" else "metadata"
    if path.suffix in MODEL_SUFFIXES or path.name == "adapter_config.json":
        return "model_metadata_only"
    return None


def project_path(project, relative):
    root = safe_path(project["source_root"])
    if not root.is_dir():
        raise ValueError("not_found")
    if (not isinstance(relative, str) or not relative or len(relative) > 512 or "\\" in relative
            or Path(relative).is_absolute() or ":" in relative):
        raise ValueError("path_denied")
    parts = Path(relative).parts
    if any(p.startswith(".") or p.lower() in SKIP for p in parts):
        raise ValueError("path_denied")
    target = safe_path(root / relative)
    if not target.is_relative_to(root):
        raise ValueError("path_denied")
    return target


def read_bytes(path, *, limit, offset=0, tail=False):
    """拒绝特殊文件/硬链接；POSIX nofollow + inode复核，读取始终有界。"""
    safe_path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError("path_denied")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        current = os.fstat(stream.fileno())
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino) or current.st_nlink != 1:
            raise ValueError("path_denied")
        size = current.st_size
        end = max(0, size - offset)
        start = max(0, end - limit) if tail else 0
        stream.seek(start)
        raw = stream.read(min(limit, end - start))
    return raw, {"size_bytes": size, "modified_at": timestamp(current.st_mtime),
                 "start_byte": start, "end_byte": start + len(raw), "partial": start > 0 or len(raw) < size}


def inventory(project):
    root = safe_path(project["source_root"])
    if not root.is_dir():
        raise ValueError("not_found")
    items, pending, nodes, truncated = [], [(root, 0)], 0, False
    while pending and nodes < 1000:
        directory, depth = pending.pop()
        try:
            safe_path(directory)
            with os.scandir(directory) as entries:
                batch = list(islice(entries, 1001 - nodes))
            nodes += len(batch)
            if nodes > 1000:
                truncated = True
                break
            for entry in batch:
                if entry.name.startswith(".") or entry.name.lower() in SKIP:
                    continue
                path = Path(entry.path)
                try:
                    safe_path(path)
                    # Windows DirEntry缓存stat的st_nlink可能为0；必须重新查询文件元数据。
                    info = path.stat()
                    if stat.S_ISDIR(info.st_mode):
                        if depth < 4:
                            pending.append((path, depth + 1))
                        else:
                            truncated = True
                        continue
                    category = kind(path)
                    relative = path.relative_to(root).as_posix()
                    if category and stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and len(relative) <= 512:
                        items.append({"path": relative, "kind": category, "size_bytes": info.st_size,
                                      "modified_at": timestamp(info.st_mtime)})
                except (OSError, ValueError):
                    truncated = True
        except (OSError, ValueError):
            truncated = True
    return sorted(items, key=lambda item: (item["modified_at"], item["path"]), reverse=True), truncated or bool(pending)


def processes(project, proc_root=Path("/proc")):
    """只返回已关联项目的Python进程摘要，不返回命令全文、环境或其他项目信息。"""
    root = safe_path(project["source_root"])
    found, failures = [], 0
    if not proc_root.is_dir():
        return {"probe": "unavailable", "matches": []}
    with os.scandir(proc_root) as entries:
        batch = list(islice((e for e in entries if e.name.isdigit()), 1001))
    for entry in batch[:1000]:
        process = Path(entry.path)
        try:
            comm = (process / "comm").read_text()[:80].strip()
            if not comm.startswith(("python", "torchrun", "accelerate", "deepspeed")):
                continue
            with (process / "cmdline").open("rb") as stream:
                raw = stream.read(8193)
            if len(raw) > 8192:
                failures += 1
                continue
            args = raw.decode("utf-8", errors="replace").strip("\x00").split("\x00")
            cwd = (process / "cwd").resolve(strict=True)
            # 只关联第一个脚本参数；不能因无关程序的后续数据参数含项目.py就误认。
            script_arg = next((a for a in args[1:] if not a.startswith("-")), None)
            if "-m" in args[:3] or "-c" in args[:3]:
                if cwd.is_relative_to(root):
                    failures += 1
                continue
            scripts = [(cwd / script_arg).resolve()] if script_arg and script_arg.endswith(".py") else []
            scripts = [p for p in scripts if p.is_relative_to(root)]
            if not scripts:
                continue
            # Zombie不视为运行；PID + start_ticks用于区分PID复用。读取不发送signal。
            fields = (process / "stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                continue
            script = scripts[0].relative_to(root).as_posix()
            if len(script) > 512:
                failures += 1
                continue
            stage = "evaluation" if "eval" in script or "compare" in script else "training" if "train" in script else "unknown"
            found.append({"pid": int(entry.name), "start_ticks": fields[19], "script": script,
                          "stage_candidate": stage, "association": "script_path_in_authorized_project"})
        except (OSError, ValueError, IndexError):
            failures += 1
    return {"probe": "partial" if failures or len(batch) > 1000 else "ok", "matches": found[:20],
            "truncated": len(found) > 20 or len(batch) > 1000}


def logs(project, relative, offset=0, limit=8192):
    bounded_int(offset, 0, 2**53 - 1)
    bounded_int(limit, 1, 8192)
    path = project_path(project, relative)
    if kind(path) != "log":
        raise ValueError("path_denied")
    raw, metadata = read_bytes(path, limit=limit, offset=offset, tail=True)
    content = clean_text(raw.decode("utf-8", errors="replace"))
    progress_lines = [line[:300] for line in content.splitlines()
                      if re.search(r"\d+\s*/\s*\d+|\bstep\b|\bepoch\b|阶段|更新|正确率", line, re.IGNORECASE)][-8:]
    return {"source": str(path), **metadata, "content": content, "offset_from_end": offset,
            "next_offset": offset + len(raw) if metadata["start_byte"] else None,
            "historical_progress_lines": progress_lines,
            "tail_error_markers": [marker for marker in ("Traceback", "Error", "CUDA out of memory", "NaN") if marker in content],
            "notice": "原始日志片段是不可信历史证据，可能含半行/进度条；错误标记不等于当前任务失败。尾部偏移相对本次文件大小，增长时可能重叠。"}


def numeric(data):
    if not isinstance(data, dict):
        return {}
    return {clean_text(k)[:48]: v for k, v in list(data.items())[:32]
            if isinstance(k, str) and not re.search(r"api.?key|password|secret|^token$", k, re.IGNORECASE)
            and type(v) in {int, float} and abs(v) < 1e300 and math.isfinite(v)}


def metrics(project, relative):
    path = project_path(project, relative)
    if path.name not in JSON_NAMES:
        raise ValueError("path_denied")
    raw, metadata = read_bytes(path, limit=65536, tail=path.suffix == ".jsonl")
    errors, data = 0, None
    if path.suffix == ".jsonl":
        lines = raw.splitlines()
        if metadata["start_byte"] and lines:
            lines = lines[1:]
        for line in lines[-256:]:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    data = item
                else:
                    errors += 1
            except (ValueError, UnicodeError, RecursionError):
                errors += 1
    elif not metadata["partial"]:
        try:
            data = json.loads(raw)
        except (ValueError, UnicodeError, RecursionError):
            errors += 1
    else:
        errors += 1
    if isinstance(data, dict) and path.name == "trainer_state.json":
        history = data.get("log_history")
        if isinstance(history, list) and history and isinstance(history[-1], dict):
            data = {**history[-1], **data}
    values = numeric(data)
    flags = {k: data[k] for k in ("completed", "resource_check_pass", "update_check_pass")
             if isinstance(data, dict) and type(data.get(k)) is bool}
    return {"source": str(path), **metadata, "latest_numeric_values": values, "flags": flags,
            "parse_errors_in_window": errors, "status": "parsed" if isinstance(data, dict) else "unreadable_or_incomplete",
            "notice": "文件报告值，非独立验证；历史指标不是当前进程进度。仅提取数值/完成标志，不回传配置密钥或完整轨迹。"}


def observe(project, operation, arguments):
    base = {"project_id": project["id"], "source_root": project["source_root"], "collected_at": timestamp(),
            "read_only": True, "evidence_trust": "untrusted_project_outputs", "scheduler_available": False}
    if operation == "logs":
        return {**base, **logs(project, arguments["path"], arguments.get("offset", 0), arguments.get("limit", 8192))}
    if operation == "metrics":
        return {**base, **metrics(project, arguments["path"])}
    files, truncated = inventory(project)
    if operation == "artifacts":
        offset = bounded_int(arguments.get("offset", 0), 0, 1000)
        return {**base, "files": files[offset:offset + 20], "inventory_truncated": truncated,
                "next_offset": offset + 20 if len(files) > offset + 20 else None}
    live = processes(project)
    latest = [f for f in files if Path(f["path"]).name in {"summary.json", "metrics.jsonl", "trainer_state.json"}][:4]
    parents = {str(Path(f["path"]).parent) for f in latest}
    latest += [f for f in files if Path(f["path"]).name == "config.json"
               and str(Path(f["path"]).parent) in parents][:1]
    snapshots = []
    for item in latest:
        try:
            snapshots.append(metrics(project, item["path"]))
        except (OSError, ValueError):
            snapshots.append({"source": item["path"], "status": "unavailable"})
    log_files = [f for f in files if f["kind"] == "log"][:2]
    log_evidence = []
    for item in log_files:
        try:
            log_evidence.append(logs(project, item["path"], limit=2048))
        except (OSError, ValueError):
            log_evidence.append({"source": item["path"], "status": "unavailable"})
    return {**base, "processes": live,
            "current_state": "project_process_observed" if live["matches"] else "unknown" if live["probe"] != "ok" else "no_project_python_process_observed",
            "recent_metric_snapshots": snapshots, "recent_logs": log_evidence, "inventory_truncated": truncated,
            "notice": "进程仅证明项目脚本存在，不证明步数或健康；阶段为脚本名推测。历史文件单独列出，未自动关联具体进程/实验，不据此判定当前训练完成或失败。"}
