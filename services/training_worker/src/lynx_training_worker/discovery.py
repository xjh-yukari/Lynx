"""有界静态发现，不import训练项目、不执行入口、不读取数据集和密钥文件。"""

import ast
import csv
import hashlib
import os
import platform
import shutil
import subprocess
from itertools import islice
from pathlib import Path

from lynx_training_worker.registry import safe_path

SKIP = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "data", "datasets", "models", "checkpoints", "outputs", "runs", "results"}
SYMBOLS = {"SFTTrainer": "sft", "GRPOTrainer": "grpo", "DPOTrainer": "dpo", "Trainer": "trainer"}


def environment():
    release = platform.freedesktop_os_release().get("PRETTY_NAME", "unknown") if platform.system() == "Linux" else platform.system()
    memory = None
    if Path("/proc/meminfo").is_file():
        line = Path("/proc/meminfo").read_text().splitlines()[0]
        memory = int(line.split()[1]) * 1024
    gpu = []
    nvidia = Path("/usr/lib/wsl/lib/nvidia-smi")
    if nvidia.is_file():
        try:
            result = subprocess.run([str(nvidia), "--query-gpu=name,memory.total,memory.free,driver_version",
                                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                    timeout=5, check=True)
            for row in csv.reader(result.stdout[:16000].splitlines()):
                if len(row) == 4:
                    gpu.append({"name": row[0].strip(), "memory_total_mib": int(row[1]),
                                "memory_free_mib": int(row[2]), "driver": row[3].strip()})
        except (OSError, ValueError, subprocess.SubprocessError):
            gpu = []
    return {"platform": release, "kernel": platform.release(), "cpu_count": os.cpu_count(),
            "memory_total_bytes": memory, "home": str(Path.home()),
            "disk_free_bytes": shutil.disk_usage(Path.home()).free,
            "gpu_probe": "ok" if gpu else "unavailable", "gpus": gpu,
            "framework": "unverified", "training_started": False}


def inspect(project):
    root = safe_path(project["source_root"])
    if not root.is_dir():
        raise ValueError("not_found")
    evidence, candidates, frameworks = [], [], set()
    nodes = 0
    budget = 1024 * 1024
    truncated = False
    directories = [(root, 0)]
    while directories:
        directory, depth = directories.pop()
        safe_path(directory)
        with os.scandir(directory) as iterator:
            entries = sorted(islice(iterator, max(1, 1001 - nodes)), key=lambda entry: entry.name)
        nodes += len(entries)
        if nodes > 1000:
            truncated = True
            break
        for entry in entries:
            name = entry.name
            if entry.is_dir(follow_symlinks=False):
                if (depth < 3 and not name.startswith(".") and name not in SKIP and not entry.is_symlink()
                        and not getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400):
                    directories.append((Path(entry.path), depth + 1))
                continue
            if len(evidence) >= 64 or budget <= 0:
                truncated = True
                break
            if name.startswith(".") or not (name.endswith(".py") or name in {"pyproject.toml", "requirements.txt"}):
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                continue
            try:
                safe_path(path)
            except ValueError:
                truncated = True
                continue
            metadata = path.stat()
            if metadata.st_nlink != 1 or getattr(metadata, "st_file_attributes", 0) & 0x400:
                truncated = True
                continue
            size = metadata.st_size
            if size > min(131072, budget):
                truncated = True
                continue
            with path.open("rb") as stream:
                raw = stream.read(131073)
            budget -= len(raw)
            if len(raw) > 131072:
                truncated = True
                continue
            text = raw.decode("utf-8", errors="replace")
            item = {"file": str(path.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest(), "symbols": []}
            if name.endswith(".py"):
                try:
                    tree = ast.parse(text)
                except (SyntaxError, ValueError, RecursionError):
                    item["parse_error"] = True
                else:
                    symbols = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
                    symbols.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
                    item["symbols"] = sorted(symbols.intersection(SYMBOLS))
                    methods = sorted({SYMBOLS[symbol] for symbol in item["symbols"]})
                    imports = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
                    imports.update(alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
                    frameworks.update(imports.intersection({"trl", "transformers", "unsloth", "openrlhf", "torch"}))
                    if methods:
                        candidates.append({"file": item["file"], "method_candidates": methods, "verified": False})
            evidence.append(item)
    return {"project_id": project["id"], "source_root": str(root), "framework_candidates": sorted(frameworks),
            "entrypoint_candidates": candidates, "evidence": evidence, "truncated": truncated,
            "discovery": "static_only", "resume_support": "unverified", "training_started": False}


def plan(project, discovered):
    return {"project_id": project["id"], "status": "planning_only", "executable": False,
            "framework_candidates": discovered["framework_candidates"],
            "entrypoint_candidates": discovered["entrypoint_candidates"],
            "stages": ["需求与共同评测协议确认", "隔离代码修改、测试、commit", "框架与checkpoint适配验证",
                       "批准的微型SFT与GRPO训练", "checkpoint暂停及续训验收", "共同评测、历史对比与SQLite记录"],
            "blockers": ["M1尚未实现持久调度与训练审批", "解释器、数据、模型、指标协议待接入验证", "暂停续训能力未实测"],
            "resource_policy": {"max_gpu_jobs": 1, "runtime_limit_seconds": None},
            "notice": "本计划未启动训练，不保证本机显存足以运行任意模型。"}
