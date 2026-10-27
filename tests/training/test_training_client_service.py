import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from lynx_training_worker.__main__ import dispatch

from lynx_agent.tools import ToolRegistry
from lynx_agent.tools.errors import ToolValidationError
from lynx_agent.training.client import (
    TrainingError,
    WSLClient,
    decode_response,
    run_bounded,
)
from lynx_agent.training.config import TrainingConfig
from lynx_agent.training.service import TrainingService
from lynx_agent.training.tools import training_tools

BOOTSTRAP = Path(__file__).resolve().parents[2] / "services" / "training_worker" / "bootstrap.py"


class LocalClient:
    def __init__(self, root):
        self.root = str(root)
        self.calls = []

    async def request(self, operation, arguments):
        self.calls.append(operation)
        return dispatch({"api_version": 1, "root": self.root, "operation": operation, "arguments": arguments})


@pytest.mark.asyncio
async def test_real_isolated_python_worker_persists_registration_and_query(tmp_path):
    source = tmp_path / "source 中文 with spaces"
    source.mkdir()
    root = tmp_path / "state"

    async def request(operation, arguments):
        payload = json.dumps({"api_version": 1, "root": str(root), "operation": operation, "arguments": arguments}).encode()
        return decode_response(await run_bounded([sys.executable, "-I", str(BOOTSTRAP)], payload, 10))

    assert (await request("projects", {}))["projects"] == []
    assert not root.exists()
    registered = await request("register", {"source": str(source), "distribution": "native-test"})
    assert (await request("projects", {}))["projects"][0]["id"] == registered["id"]
    assert (await request("inspect", {"project_id": registered["id"]}))["training_started"] is False


@pytest.mark.asyncio
async def test_duplicate_json_and_protocol_version_rejected_without_db(tmp_path):
    for payload in (b'{"api_version":1,"api_version":1}',
                    json.dumps({"api_version": True, "root": str(tmp_path / "state"),
                                "operation": "projects", "arguments": {}}).encode()):
        response = await run_bounded([sys.executable, "-I", str(BOOTSTRAP)], payload, 10)
        with pytest.raises(TrainingError):
            decode_response(response)
    assert not (tmp_path / "state").exists()


@pytest.mark.asyncio
async def test_short_process_timeout_and_output_limit():
    with pytest.raises(TrainingError, match="超时"):
        await run_bounded([sys.executable, "-c", "import time;time.sleep(5)"], b"", 0.1)
    with pytest.raises(TrainingError, match="超限"):
        await run_bounded([sys.executable, "-c", "import sys;sys.stdout.write('a'*200000)"], b"", 5)


@pytest.mark.asyncio
async def test_cancellation_cleans_up_control_process():
    task = asyncio.create_task(run_bounded([sys.executable, "-c", "import time;time.sleep(5)"], b"", 10))
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_wsl_transport_uses_arguments_not_shell_and_stdin_json():
    calls = []

    async def invoke(argv, payload, timeout):
        calls.append((argv, payload, timeout))
        if "/usr/bin/wslpath" in argv:
            return b"/mnt/e/source with space/bootstrap.py\n"
        return b'{"api_version":1,"ok":true,"data":{"training_started":false},"error":null}'

    client = WSLClient(TrainingConfig(), invoke=invoke)
    result = await client.request("environment", {})
    assert not result["training_started"]
    assert calls[-1][0][-2:] == ["-I", "/mnt/e/source with space/bootstrap.py"]
    assert "bash" not in calls[-1][0]
    assert json.loads(calls[-1][1])["operation"] == "environment"


@pytest.mark.asyncio
async def test_model_tools_disabled_by_default_and_cannot_register(tmp_path):
    client = LocalClient(tmp_path / "state")
    service = TrainingService(TrainingConfig(), client=client)
    tools = ToolRegistry(training_tools(service))
    assert not tools.names
    await service.command("on confirm")
    assert len(tools.names) == 5
    inspect_tool = tools.get("training_inspect")
    with pytest.raises(ToolValidationError):
        inspect_tool.validate_arguments({"mode": "project", "project_id": str(uuid4()), "path": "/home"})
    with pytest.raises(ToolValidationError):
        inspect_tool.validate_arguments({"mode": "environment", "project_id": str(uuid4())})
    assert not client.calls
    for operation in ("register", "record", "submit_job"):
        with pytest.raises(TrainingError):
            await service.read(operation, {})


@pytest.mark.asyncio
async def test_human_registration_requires_confirmation_and_writable_mode(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    service = TrainingService(TrainingConfig(enabled=True), client=LocalClient(tmp_path / "state"))
    assert "confirm" in await service.command(f'register "{source}"')
    with pytest.raises(TrainingError, match="禁止"):
        await service.command(f'register "{source}" confirm')
    assert not (tmp_path / "state").exists()


@pytest.mark.asyncio
async def test_service_marks_manual_records_and_protects_secrets(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    client = LocalClient(tmp_path / "state")
    service = TrainingService(TrainingConfig(enabled=True), client=client, writable=lambda: True, secrets=("known-secret-value",))
    project = await client.request("register", {"source": str(source), "distribution": "native-test"})
    result = json.loads(await service.command(f'record {project["id"]} sft eval-v1 accuracy=0.5 confirm'))
    assert result["verification"] == "user_reported_unverified"
    with pytest.raises(TrainingError):
        await service.command(f'record {project["id"]} sft eval-v1 score=NaN confirm')
    with pytest.raises(TrainingError):
        await service.command(f'record {project["id"]} sft known-secret-value score=1 confirm')
    assert "known-secret-value" not in service.clean({"value": "known-secret-value"})
    with pytest.raises(TrainingError, match="尚未实现"):
        await service.command("submit anything")
