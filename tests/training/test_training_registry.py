import sqlite3
from uuid import uuid4

import pytest
from lynx_training_worker.registry import Registry


def setup_project(tmp_path):
    source = tmp_path / "project with space 中文"
    source.mkdir()
    registry = Registry(tmp_path / "state")
    return registry, registry.register(str(source), "Ubuntu-24.04")


def test_read_does_not_create_registry(tmp_path):
    root = tmp_path / "not-created"
    assert Registry(root).projects() == []
    assert not root.exists()


def test_registration_is_persistent_and_idempotent(tmp_path):
    registry, project = setup_project(tmp_path)
    again = registry.register(project["source_root"], project["distribution"])
    assert again == project
    assert Registry(registry.root).projects() == [project]
    with registry.connect() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_cannot_register_internal_root_or_parent(tmp_path):
    registry = Registry(tmp_path / "state")
    registry.root.mkdir()
    for path in (tmp_path, registry.root):
        with pytest.raises(ValueError, match="path_denied"):
            registry.register(str(path), "Ubuntu-24.04")


def test_no_read_of_unknown_project(tmp_path):
    registry, _ = setup_project(tmp_path)
    with pytest.raises(ValueError, match="not_found"):
        registry.project(str(uuid4()))
    with pytest.raises(ValueError):
        registry.project("../../outside")


def test_imported_records_are_unverified_and_request_is_idempotent(tmp_path):
    registry, project = setup_project(tmp_path)
    experiment = str(uuid4())
    arguments = (project["id"], "sft", "eval-v1", "user report", {"accuracy": 0.7}, experiment)
    assert not registry.record(*arguments)["reused"]
    assert registry.record(*arguments)["reused"]
    with pytest.raises(ValueError, match="invalid_request"):
        registry.record(project["id"], "sft", "eval-v1", "user report", {"accuracy": 0.9}, experiment)
    records = Registry(registry.root).experiments(project["id"])
    assert len(records) == 1
    assert records[0]["verification"] == "user_reported_unverified"
    assert records[0]["metrics"] == {"accuracy": 0.7}


@pytest.mark.parametrize("value", [True, "1.0", float("nan"), float("inf"), 10**400])
def test_invalid_metrics_leave_no_partial_record(tmp_path, value):
    registry, project = setup_project(tmp_path)
    with pytest.raises(ValueError, match="invalid_request"):
        registry.record(project["id"], "grpo", "eval-v1", "manual", {"score": value}, str(uuid4()))
    assert registry.experiments(project["id"]) == []


def test_foreign_keys_and_rollback(tmp_path):
    registry, _ = setup_project(tmp_path)
    with pytest.raises(sqlite3.IntegrityError), registry.connect(write=True) as connection:
        connection.execute("INSERT INTO imported_experiments VALUES(?,?,?,?,?,?)",
                           (str(uuid4()), str(uuid4()), "sft", "eval", "manual", "now"))
    with registry.connect() as connection:
        assert connection.execute("SELECT count(*) FROM imported_experiments").fetchone()[0] == 0


def test_unknown_database_is_not_overwritten(tmp_path):
    registry = Registry(tmp_path / "state")
    registry.database.parent.mkdir(parents=True)
    with sqlite3.connect(registry.database) as connection:
        connection.execute("CREATE TABLE unrelated(value TEXT)")
        connection.execute("INSERT INTO unrelated VALUES('keep')")
    with pytest.raises(sqlite3.Error):
        registry.projects()
    with sqlite3.connect(registry.database) as connection:
        assert connection.execute("SELECT value FROM unrelated").fetchone()[0] == "keep"


def test_comparison_is_scoped_and_never_claims_verified_training(tmp_path):
    registry, project = setup_project(tmp_path)
    first, second, different = (str(uuid4()) for _ in range(3))
    for experiment, method, protocol, score in ((first, "sft", "eval-v1", 0.5),
                                              (second, "grpo", "eval-v1", 0.6),
                                              (different, "grpo", "eval-v2", 0.7)):
        registry.record(project["id"], method, protocol, "manual", {"score": score}, experiment)
    result = registry.compare(project["id"], [first, second])
    assert result["candidate_minus_baseline"]["score"] == pytest.approx(0.1)
    assert result["comparability"] == "not_verified"
    assert registry.compare(project["id"], [first, different])["candidate_minus_baseline"] == {}
    with pytest.raises(ValueError, match="not_found"):
        registry.compare(project["id"], [first, str(uuid4())])


def test_shared_nonempty_directory_is_not_adopted_as_registry(tmp_path):
    root = tmp_path / "existing-data"
    root.mkdir()
    original = root / "keep.txt"
    original.write_text("keep", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="path_denied"):
        Registry(root).register(str(source), "test")
    assert original.read_text(encoding="utf-8") == "keep"
    assert not (root / "registry").exists()
