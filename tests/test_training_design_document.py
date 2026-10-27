"""技术稿的离线契约校验，不代表训练worker功能验收。"""

import json
import re
import sqlite3
from pathlib import Path

import pytest

DOCUMENT = Path(__file__).resolve().parents[1] / "docs" / "training-assistant-technical-design.md"


def document():
    return DOCUMENT.read_text(encoding="utf-8")


def design_database():
    blocks = re.findall(r"```sql\n(.*?)\n```", document(), re.DOTALL)
    assert len(blocks) == 1
    connection = sqlite3.connect(":memory:")
    connection.executescript(blocks[0])
    return connection


def test_design_sql_creates_all_required_entities():
    connection = design_database()
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {
            "projects", "requirements", "code_versions", "test_results", "environments",
            "plans", "approvals", "experiment_groups", "experiments", "jobs", "attempts",
            "requests", "dispatch_outbox", "job_events", "metrics", "artifacts", "checkpoints",
            "reports", "schema_versions",
        } <= tables
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_design_sql_rejects_orphan_records_and_duplicate_registration():
    connection = design_database()
    try:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO requirements VALUES ('r', 'missing', 1, '{}', 'hash', 'now')")
        connection.execute("INSERT INTO projects VALUES ('p', 'Ubuntu-24.04', '/tmp/project', 1, 1, 'now')")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("INSERT INTO projects VALUES ('p2', 'Ubuntu-24.04', '/tmp/project', 1, 1, 'now')")
    finally:
        connection.close()


def test_document_json_examples_have_matching_request_and_response():
    examples = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", document(), re.DOTALL)]
    assert len(examples) == 2
    request, response = examples
    assert request["api_version"] == response["api_version"] == 1
    assert request["request_id"] == response["request_id"]
    assert request["operation"] == "submit_job"
    assert response["status"] == "accepted"
    assert response["execution_state"] == "queued"
    assert response["error"] is None


def test_design_traces_all_first_release_requirements():
    text = document()
    for number in range(1, 13):
        assert re.search(rf"^\| F{number:02d} \|", text, re.MULTILINE)
    assert "技术设计稿，尚未实现" in text
    assert "本轮尝试联网重取失败" in text
    assert "不创建worker包、训练目录、数据库或unit" in text
