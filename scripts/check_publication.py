"""发布前只读检查：索引/工作树及可选Git历史；仅报告位置，不输出凭据。"""

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

PATTERNS = {
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "api_key_shape": re.compile(rb"\bsk-[A-Za-z0-9_-]{24,}\b"),
    "github_token_shape": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    "aws_key_shape": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
}
SECRET_NAME = re.compile(r"(?:api[_-]?key|token|password|secret)", re.IGNORECASE)
# bench/tests/test_proxy.py中已核对的合成占位符；仅豁免格式规则，精确密钥匹配仍拦截。
SYNTHETIC = {"5bfe2fcb866dba4b0eaea7173dfdfbaea58874001c6f909745fd729a830365f0"}


def git(root, *args, data=None):
    return subprocess.run(["git", *args], cwd=root, input=data, capture_output=True, check=True).stdout


def forbidden(path):
    name = Path(path).name.lower()
    example = name.endswith((".example", ".template", ".example.env"))
    return (not example and (name == ".env" or name.startswith(".env.")
                            or name in {"config.toml", "credentials.json", "secrets.json"}
                            or name.endswith((".env", ".key", ".pem", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db"))))


def local_secrets(root, extra_files=()):
    values = set()

    def collect(value, name=""):
        if isinstance(value, dict):
            for key, child in value.items():
                collect(child, key)
        elif (isinstance(value, str) and SECRET_NAME.search(name) and not name.lower().endswith(("_file", "_env"))
              and len(value.strip()) >= 8):
            values.add(value.strip().encode())

    collect(dict(os.environ))
    config = root / "config.toml"
    if config.is_file():
        collect(tomllib.loads(config.read_text(encoding="utf-8")))
    env = root / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.strip().removeprefix("export ").partition("=")
            if separator and not name.startswith("#"):
                tokens = shlex.split(value, comments=True)
                if tokens:
                    collect(tokens[0], name.strip())
    for filename in extra_files:
        value = Path(filename).read_bytes().strip()
        if not 8 <= len(value) <= 4096:
            raise ValueError("invalid_secret_file")
        values.add(value)
    return values


def inspect_bytes(data, secrets):
    reasons = [name for name, pattern in PATTERNS.items()
               if any(hashlib.sha256(match.group()).hexdigest() not in SYNTHETIC for match in pattern.finditer(data))]
    if any(secret in data for secret in secrets):
        reasons.append("known_local_secret")
    return reasons


def scan(root, *, history=False, secrets=()):
    root = Path(root).resolve()
    findings, blobs = [], {}
    entries = git(root, "ls-files", "--stage", "-z").split(b"\x00")
    tracked = []
    for entry in filter(None, entries):
        metadata, raw_path = entry.split(b"\t", 1)
        mode, oid, _ = metadata.decode().split()
        path = raw_path.decode("utf-8", errors="replace")
        tracked.append(path)
        blobs[oid] = path
        if forbidden(path) or mode in {"120000", "160000"}:
            findings.append({"path": path, "reason": "private_path_or_external_link"})
        file = root / path
        if file.is_file() and not file.is_symlink():
            if file.stat().st_size > 8 * 1024 * 1024:
                findings.append({"path": path, "reason": "working_file_unscanned_size_limit"})
            else:
                for reason in inspect_bytes(file.read_bytes(), secrets):
                    findings.append({"path": path, "reason": reason, "scope": "working_tree"})
    if history:
        for line in git(root, "rev-list", "--objects", "--all").decode("utf-8", errors="replace").splitlines():
            oid, _, path = line.partition(" ")
            blobs.setdefault(oid, path)
            if path and forbidden(path):
                findings.append({"path": path, "reason": "private_path_in_history", "object": oid})
    selected, budget = [], 64 * 1024 * 1024
    metadata = git(root, "cat-file", "--batch-check", data="".join(oid + "\n" for oid in blobs).encode())
    for line in metadata.decode().splitlines():
        oid, kind, size = line.split()
        if kind != "blob":
            continue
        size = int(size)
        if size > min(8 * 1024 * 1024, budget):
            findings.append({"path": blobs[oid], "object": oid, "reason": "blob_unscanned_size_limit"})
            continue
        budget -= size
        selected.append(oid)
    stream = io.BytesIO(git(root, "cat-file", "--batch", data="".join(oid + "\n" for oid in selected).encode()))
    for oid in selected:
        _, _, size = stream.readline().split()
        data = stream.read(int(size))
        stream.read(1)
        for reason in inspect_bytes(data, secrets):
            findings.append({"path": blobs[oid], "object": oid, "reason": reason, "scope": "git_blob"})
    return {"status": "blocked" if findings else "passed", "tracked_files": len(tracked),
            "scanned_git_blobs": len(selected), "history_checked": history, "findings": findings,
            "notice": "有界启发式检查，不保证发现所有密钥、个人信息或图像内容；不扫描未跟踪文件，不改写历史。"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--secret-file", action="append", default=[], help="额外已知密钥文件，只读取以匹配，不输出内容")
    args = parser.parse_args()
    try:
        result = scan(args.repo, history=args.history, secrets=local_secrets(args.repo, args.secret_file))
    except (OSError, ValueError, subprocess.SubprocessError):
        print(json.dumps({"status": "blocked", "error": "检查未完成；未输出原始异常或凭据"}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
