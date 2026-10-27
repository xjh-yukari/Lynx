"""M1本地SQLite台账：短事务串行写，查询不自动建库。"""

import contextlib
import math
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

SCHEMA = """
CREATE TABLE registry_meta(version INTEGER PRIMARY KEY CHECK(version=1));
INSERT INTO registry_meta VALUES(1);
CREATE TABLE training_projects(
 id TEXT PRIMARY KEY, distribution TEXT NOT NULL, source_root TEXT NOT NULL,
 created_at TEXT NOT NULL, UNIQUE(distribution,source_root));
CREATE TABLE imported_experiments(
 id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES training_projects(id),
 method TEXT NOT NULL CHECK(method IN ('sft','grpo','other')),
 protocol TEXT NOT NULL, evidence TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE imported_metrics(
 experiment_id TEXT NOT NULL REFERENCES imported_experiments(id),
 name TEXT NOT NULL, value REAL NOT NULL CHECK(value BETWEEN -1.7976931348623157e308 AND 1.7976931348623157e308),
 PRIMARY KEY(experiment_id,name));
"""


def identifier(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("invalid_request")
    return value


def safe_text(value, maximum=200):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 for c in value):
        raise ValueError("invalid_request")
    return value


def safe_path(value):
    safe_text(str(value), 4096)
    path = Path(value).expanduser().absolute()
    if ".." in path.parts or any(parent.is_symlink() or (
            parent.exists() and getattr(parent.stat(), "st_file_attributes", 0) & 0x400)
            for parent in (path, *path.parents)):
        raise ValueError("path_denied")
    if path.is_file() and path.stat().st_nlink != 1:
        raise ValueError("path_denied")
    return path


class Registry:
    def __init__(self, root):
        self.root = safe_path(root)
        if (self.root == Path(self.root.anchor) or self.root in Path.home().parents or self.root == Path.home()
                or str(self.root) in {"/tmp", "/var", "/usr", "/opt", "/etc", "/proc", "/sys", "/dev", "/run", "/mnt", "/home"}):
            raise ValueError("path_denied")
        self.database = self.root / "registry" / "experiments.sqlite3"

    @contextlib.contextmanager
    def connect(self, *, write=False):
        safe_path(self.database)
        if not write and not self.database.exists():
            yield None
            return
        if write:
            if self.root.is_dir() and any(self.root.iterdir()) and not self.database.is_file():
                raise ValueError("path_denied")
            self.database.parent.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                self.root.chmod(0o700)
                self.database.parent.chmod(0o700)
        connection = sqlite3.connect(str(self.database) if write else self.database.as_uri() + "?mode=ro",
                                     uri=not write, timeout=3)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not tables:
                    # 不使用executescript隐式提交；建表和版本在同一事务中发布。
                    for statement in SCHEMA.split(";"):
                        if statement.strip():
                            connection.execute(statement)
            versions = connection.execute("SELECT version FROM registry_meta").fetchall()
            if len(versions) != 1 or versions[0][0] != 1:
                raise ValueError("registry_unavailable")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()
            if write and os.name != "nt" and self.database.exists():
                self.database.chmod(0o600)

    def register(self, source, distribution):
        path = safe_path(source)
        safe_text(distribution, 80)
        if not path.is_dir() or path == Path(path.anchor) or path in self.root.parents or self.root in path.parents or path == self.root:
            raise ValueError("path_denied")
        if str(path).startswith(("/proc/", "/sys/", "/dev/", "/etc/")) or str(path) in {"/proc", "/sys", "/dev", "/etc"}:
            raise ValueError("path_denied")
        with self.connect(write=True) as connection:
            connection.execute("INSERT OR IGNORE INTO training_projects VALUES(?,?,?,?)",
                               (str(uuid4()), distribution, str(path), datetime.now(UTC).isoformat()))
            return dict(connection.execute("SELECT * FROM training_projects WHERE distribution=? AND source_root=?",
                                           (distribution, str(path))).fetchone())

    def projects(self):
        with self.connect() as connection:
            return [] if connection is None else [dict(row) for row in connection.execute(
                "SELECT * FROM training_projects ORDER BY created_at,id LIMIT 100")]

    def project(self, project_id):
        identifier(project_id)
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM training_projects WHERE id=?", (project_id,)).fetchone() if connection else None
            if row is None:
                raise ValueError("not_found")
            project = dict(row)
            safe_path(project["source_root"])
            return project

    def record(self, project_id, method, protocol, evidence, metrics, experiment_id):
        self.project(project_id)
        identifier(experiment_id)
        if method not in {"sft", "grpo", "other"} or not isinstance(metrics, dict) or not 1 <= len(metrics) <= 30:
            raise ValueError("invalid_request")
        safe_text(protocol, 80)
        safe_text(evidence, 200)
        for name, value in metrics.items():
            safe_text(name, 80)
            try:
                finite = type(value) in {int, float} and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("invalid_request")
        with self.connect(write=True) as connection:
            existing = connection.execute("SELECT * FROM imported_experiments WHERE id=?", (experiment_id,)).fetchone()
            if existing:
                old_metrics = dict(connection.execute("SELECT name,value FROM imported_metrics WHERE experiment_id=?", (experiment_id,)))
                if (existing["project_id"], existing["method"], existing["protocol"], existing["evidence"], old_metrics) != (project_id, method, protocol, evidence, metrics):
                    raise ValueError("invalid_request")
                return {"experiment_id": experiment_id, "verification": "user_reported_unverified", "reused": True}
            connection.execute("INSERT INTO imported_experiments VALUES(?,?,?,?,?,?)",
                               (experiment_id, project_id, method, protocol, evidence, datetime.now(UTC).isoformat()))
            connection.executemany("INSERT INTO imported_metrics VALUES(?,?,?)",
                                   [(experiment_id, name, value) for name, value in metrics.items()])
        return {"experiment_id": experiment_id, "verification": "user_reported_unverified", "reused": False}

    def experiments(self, project_id):
        self.project(project_id)
        with self.connect() as connection:
            items = [dict(row) for row in connection.execute(
                "SELECT * FROM imported_experiments WHERE project_id=? ORDER BY created_at,id LIMIT 50", (project_id,))]
            for item in items:
                item["metrics"] = dict(connection.execute("SELECT name,value FROM imported_metrics WHERE experiment_id=?", (item["id"],)))
                item["verification"] = "user_reported_unverified"
            return items

    def compare(self, project_id, experiment_ids):
        self.project(project_id)
        if not isinstance(experiment_ids, list) or len(experiment_ids) != 2 or len(set(experiment_ids)) != 2:
            raise ValueError("invalid_request")
        records = []
        with self.connect() as connection:
            for experiment_id in experiment_ids:
                identifier(experiment_id)
                row = connection.execute("SELECT * FROM imported_experiments WHERE id=? AND project_id=?",
                                         (experiment_id, project_id)).fetchone()
                if row is None:
                    raise ValueError("not_found")
                record = dict(row)
                record["metrics"] = dict(connection.execute("SELECT name,value FROM imported_metrics WHERE experiment_id=?", (experiment_id,)))
                records.append(record)
        baseline, candidate = records
        same_protocol = baseline["protocol"] == candidate["protocol"]
        differences = {}
        if same_protocol:
            for name in baseline["metrics"].keys() & candidate["metrics"].keys():
                difference = candidate["metrics"][name] - baseline["metrics"][name]
                differences[name] = difference if math.isfinite(difference) else None
        return {"project_id": project_id, "baseline": baseline, "candidate": candidate,
                "same_protocol_label": same_protocol, "candidate_minus_baseline": differences,
                "verification": "user_reported_unverified", "comparability": "not_verified",
                "notice": "仅人工登记值的数值差；数据、模型、预算和协议内容未核验，不能证明训练提升。协议名不一致时不计算差值。"}
