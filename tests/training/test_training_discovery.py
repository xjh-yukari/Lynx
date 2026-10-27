from pathlib import Path

import pytest
from lynx_training_worker.__main__ import dispatch
from lynx_training_worker.discovery import inspect, plan
from lynx_training_worker.registry import Registry


def test_static_discovery_does_not_execute_code_or_read_secrets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    marker = source / "executed.txt"
    (source / "train.py").write_text(
        f"from trl import SFTTrainer, GRPOTrainer\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "trainer = SFTTrainer()\nother = GRPOTrainer()\n", encoding="utf-8")
    (source / ".env").write_text("API_KEY=must-not-read-this", encoding="utf-8")
    (source / "data").mkdir()
    (source / "data" / "secret.py").write_text("secret='hidden'", encoding="utf-8")
    registry = Registry(tmp_path / "state")
    project = registry.register(str(source), "Ubuntu-24.04")
    result = inspect(project)
    assert not marker.exists()
    assert result["framework_candidates"] == ["trl"]
    assert result["entrypoint_candidates"][0]["method_candidates"] == ["grpo", "sft"]
    assert [item["file"] for item in result["evidence"]] == ["train.py"]
    assert "must-not-read-this" not in repr(result)
    assert result["resume_support"] == "unverified"
    proposed = plan(project, result)
    assert proposed["executable"] is False
    assert proposed["resource_policy"]["runtime_limit_seconds"] is None
    assert proposed["blockers"]


def test_discovery_bounds_file_size_and_depth(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "huge.py").write_bytes(b"x" * 140000)
    deep = source / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    (deep / "train.py").write_text("raise RuntimeError('not read')", encoding="utf-8")
    result = inspect({"source_root": str(source), "id": "test"})
    assert result["truncated"]
    assert result["evidence"] == []


def test_symlink_source_is_denied_when_supported(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(actual, target_is_directory=True)
    except OSError:
        pytest.skip("symlink permission unavailable")
    with pytest.raises(ValueError, match="path_denied"):
        inspect({"source_root": str(link), "id": "test"})


@pytest.mark.parametrize("operation", ["submit_job", "pause", "resume", "cancel", "exec_command"])
def test_backend_rejects_execution_operations(tmp_path, operation):
    packet = {"api_version": 1, "root": str(tmp_path / "state"), "operation": operation, "arguments": {}}
    with pytest.raises(ValueError, match="unsupported"):
        dispatch(packet)
    assert not (tmp_path / "state").exists()


def test_inspection_requires_registration_not_arbitrary_path(tmp_path):
    packet = {"api_version": 1, "root": str(tmp_path / "state"), "operation": "inspect",
              "arguments": {"project_id": "../../private", "path": str(Path.home())}}
    with pytest.raises(ValueError, match="invalid_request"):
        dispatch(packet)


def test_discovery_does_not_follow_hard_links(tmp_path):
    import os

    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "outside.py"
    secret.write_text("from trl import GRPOTrainer\nx=GRPOTrainer()", encoding="utf-8")
    try:
        os.link(secret, source / "linked.py")
    except OSError:
        pytest.skip("hard link not available")
    result = inspect({"source_root": str(source), "id": "test"})
    assert result["evidence"] == []
    assert result["truncated"]


def test_discovery_stops_large_directory_enumeration(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for number in range(1005):
        (source / f"{number}.txt").touch()
    result = inspect({"source_root": str(source), "id": "test"})
    assert result["truncated"]
    assert result["evidence"] == []
