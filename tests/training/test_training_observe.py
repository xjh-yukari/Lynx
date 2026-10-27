import json
import os
from uuid import uuid4

import pytest
from lynx_training_worker import observe as observation
from lynx_training_worker.__main__ import dispatch
from lynx_training_worker.registry import Registry

from lynx_agent.tools import ToolCall, ToolRegistry
from lynx_agent.tools.errors import ToolValidationError
from lynx_agent.training.config import TrainingConfig
from lynx_agent.training.service import TrainingService
from lynx_agent.training.tools import training_tools


@pytest.fixture
def registered(tmp_path):
    source = tmp_path / "training-source"
    source.mkdir()
    registry = Registry(tmp_path / "registry-root")
    project = registry.register(str(source), "test")
    return source, registry, project


def request(registry, project, operation, **arguments):
    return dispatch({"api_version": 1, "root": str(registry.root), "operation": operation,
                     "arguments": {"project_id": project["id"], **arguments}})


def test_historical_completion_is_not_live_progress(registered, monkeypatch):
    source, registry, project = registered
    run = source / "outputs" / "run1"
    run.mkdir(parents=True)
    (run / "summary.json").write_text('{"completed":true,"optimizer_updates":40}', encoding="utf-8")
    (source / "train.log").write_text("[4/5] eval\nTraceback\nJSONDecodeError\n", encoding="utf-8")
    monkeypatch.setattr(observation, "processes", lambda project: {"probe": "ok", "matches": []})
    result = request(registry, project, "progress")
    assert result["current_state"] == "no_project_python_process_observed"
    assert result["recent_metric_snapshots"][0]["flags"] == {"completed": True}
    assert result["recent_logs"][0]["tail_error_markers"] == ["Traceback", "Error"]
    assert result["collected_at"] and result["read_only"]
    assert result["recent_metric_snapshots"][0]["source"].endswith("summary.json")


@pytest.mark.parametrize("probe,matches,state", [
    ("unavailable", [], "unknown"), ("partial", [], "unknown"),
    ("ok", [{"pid": 123, "stage_candidate": "evaluation"}], "project_process_observed"),
])
def test_process_probe_does_not_infer_running_from_files(registered, monkeypatch, probe, matches, state):
    _, registry, project = registered
    monkeypatch.setattr(observation, "processes", lambda project: {"probe": probe, "matches": matches})
    assert request(registry, project, "progress")["current_state"] == state


def test_logs_bounded_tail_pagination_and_redaction(registered):
    source, registry, project = registered
    text = "abcdef0123456789"
    (source / "run.log").write_text(text, encoding="utf-8")
    result = request(registry, project, "logs", path="run.log", limit=5)
    assert result["content"] == text[-5:]
    assert result["next_offset"] == 5
    previous = request(registry, project, "logs", path="run.log", limit=5, offset=5)
    assert previous["content"] == text[-10:-5]
    assert request(registry, project, "logs", path="run.log", offset=100)["next_offset"] is None
    (source / "run.log").write_text("\x1b[31mstep 4/40\rAPI_KEY=private-value\nBearer abc123\x00", encoding="utf-8")
    result = request(registry, project, "logs", path="run.log")
    assert "private-value" not in result["content"] and "abc123" not in result["content"]
    assert "\x1b" not in result["content"] and "\x00" not in result["content"]
    assert result["historical_progress_lines"] == ["step 4/40"]


@pytest.mark.parametrize("arguments", [
    {"path": "run.log", "offset": -1}, {"path": "run.log", "offset": True},
    {"path": "run.log", "limit": 8193}, {"path": "run.log", "limit": 0},
    {"path": "../outside.log"}, {"path": "/etc/run.log"}, {"path": "C:\\run.log"},
    {"path": ".env"}, {"path": "data/run.log"}, {"path": "summary.json"},
    {"path": "run.log", "arbitrary": True},
])
def test_log_permission_and_protocol_boundaries(registered, arguments):
    _, registry, project = registered
    with pytest.raises(ValueError):
        request(registry, project, "logs", **arguments)


def test_jsonl_skips_corrupt_lines_reports_counts_and_never_returns_raw_config(registered):
    source, registry, project = registered
    (source / "metrics.jsonl").write_text('{"step":1,"loss":2}\nBROKEN\n{"step":2,"loss":1}\n{', encoding="utf-8")
    result = request(registry, project, "metrics", path="metrics.jsonl")
    assert result["latest_numeric_values"] == {"step": 2, "loss": 1}
    assert result["parse_errors_in_window"] == 2
    (source / "config.json").write_text('{"api_key":"secret-value","max_steps":40}', encoding="utf-8")
    result = request(registry, project, "metrics", path="config.json")
    assert result["latest_numeric_values"] == {"max_steps": 40}
    assert "secret-value" not in json.dumps(result)
    (source / "summary.json").write_text("{" + " " * 70000, encoding="utf-8")
    assert request(registry, project, "metrics", path="summary.json")["status"] == "unreadable_or_incomplete"


def test_inventory_pagination_hidden_files_and_no_model_read(registered):
    source, registry, project = registered
    for i in range(25):
        (source / f"run{i}.log").write_text("log", encoding="utf-8")
    (source / ".secret.log").write_text("secret", encoding="utf-8")
    weights = source / "model.safetensors"
    weights.write_bytes(b"do not read")
    first = request(registry, project, "artifacts")
    second = request(registry, project, "artifacts", offset=20)
    files = first["files"] + second["files"]
    assert len(files) == 26 and first["next_offset"] == 20 and second["next_offset"] is None
    assert not any(f["path"].startswith(".") for f in files)
    with pytest.raises(ValueError):
        request(registry, project, "metrics", path="model.safetensors")
    with pytest.raises(ValueError):
        request(registry, project, "artifacts", offset=1001)


def test_unknown_project_does_not_get_read_access(registered):
    _, registry, project = registered
    with pytest.raises(ValueError, match="not_found"):
        request(registry, {**project, "id": str(uuid4())}, "progress")


def test_hardlinks_rejected(registered):
    source, registry, project = registered
    original = source / "real.log"
    original.write_text("private", encoding="utf-8")
    os.link(original, source / "linked.log")
    with pytest.raises(ValueError, match="path_denied"):
        request(registry, project, "logs", path="linked.log")


def test_symlinks_rejected(registered):
    source, registry, project = registered
    original = source.parent / "outside.log"
    original.write_text("private", encoding="utf-8")
    try:
        (source / "linked.log").symlink_to(original)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(ValueError, match="path_denied"):
        request(registry, project, "logs", path="linked.log")


def test_directory_enumeration_and_jsonl_tail_are_bounded(registered):
    source, registry, project = registered
    for i in range(1005):
        (source / f"run{i}.log").touch()
    assert request(registry, project, "artifacts")["inventory_truncated"]
    (source / "metrics.jsonl").write_text('not-json\n' * 10000 + '{"step":42}\n', encoding="utf-8")
    result = request(registry, project, "metrics", path="metrics.jsonl")
    assert result["latest_numeric_values"] == {"step": 42}
    assert result["partial"] and result["parse_errors_in_window"] <= 255


@pytest.mark.asyncio
async def test_read_only_mode_commands_and_model_tools_work_without_write_permission(registered):
    source, registry, project = registered
    (source / "run.log").write_text("step 4/40", encoding="utf-8")

    class Client:
        async def request(self, operation, arguments):
            return dispatch({"api_version": 1, "root": str(registry.root), "operation": operation, "arguments": arguments})

    service = TrainingService(TrainingConfig(enabled=True), client=Client(), writable=lambda: False)
    tools = ToolRegistry(training_tools(service))
    for verb in ("progress", "files", "logs", "metrics"):
        path = " run.log" if verb == "logs" else " summary.json" if verb == "metrics" else ""
        if verb == "metrics":
            (source / "summary.json").write_text('{"completed":true}', encoding="utf-8")
        result = json.loads(await service.command(f'{verb} {project["id"]}{path}'))
        assert result["read_only"]
    for name in ("training_progress", "training_logs", "training_metrics", "training_artifacts"):
        tool = tools.get(name)
        assert tool.read_only
        arguments = {"project_id": project["id"]}
        if name in {"training_logs", "training_metrics"}:
            arguments["path"] = "run.log" if name == "training_logs" else "summary.json"
        result = await tools.execute(ToolCall(id="read", name=name, arguments=arguments))
        assert not result.is_error and json.loads(result.content)["read_only"]
    with pytest.raises(ToolValidationError):
        tools.get("training_logs").validate_arguments({"project_id": project["id"], "path": "run.log", "limit": 9000})
    await service.command("off")
    assert "training_progress" not in tools.names


def test_fake_proc_associates_only_project_python_and_excludes_zombies(registered, tmp_path):
    source, _, project = registered
    proc = tmp_path / "proc"
    proc.mkdir()
    for pid, script, state in ((100, str(source / "grpo_train.py"), "R"),
                               (101, "/unrelated/train.py", "R"), (102, str(source / "eval.py"), "Z")):
        entry = proc / str(pid)
        entry.mkdir()
        (entry / "comm").write_text("python3")
        (entry / "cmdline").write_bytes(f"python\0{script}\0".encode())
        # Actual proc cwd is a symlink. Resolve follows it only for association, not file reading.
        (entry / "cwd").mkdir()
        (entry / "stat").write_text(f"{pid} (python) {state} " + "0 " * 18 + "1234 0")
    result = observation.processes(project, proc)
    assert [p["pid"] for p in result["matches"]] == [100]
    assert result["matches"][0]["stage_candidate"] == "training"
    assert result["matches"][0]["start_ticks"] == "1234"
    assert "cmdline" not in json.dumps(result)


def test_unrelated_script_with_project_argument_is_not_associated(registered, tmp_path):
    source, _, project = registered
    proc = tmp_path / "proc"
    entry = proc / "123"
    entry.mkdir(parents=True)
    (entry / "comm").write_text("python3")
    (entry / "cmdline").write_bytes(f"python\0/unrelated/script.py\0{source / 'train.py'}\0".encode())
    (entry / "cwd").mkdir()
    (entry / "stat").write_text("123 (python) R " + "0 " * 18 + "1234 0")
    assert observation.processes(project, proc)["matches"] == []


def test_config_numeric_credentials_and_nonfinite_metrics_are_not_exposed(registered):
    source, registry, project = registered
    (source / "config.json").write_text('{"token":123456,"password":654321,"max_tokens":8192,"loss":NaN}', encoding="utf-8")
    result = request(registry, project, "metrics", path="config.json")
    assert result["latest_numeric_values"] == {"max_tokens": 8192}


def test_fifo_is_rejected_without_opening(registered):
    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX FIFO unavailable")
    source, registry, project = registered
    os.mkfifo(source / "named.log")
    with pytest.raises(ValueError, match="path_denied"):
        request(registry, project, "logs", path="named.log")
