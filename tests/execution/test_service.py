"""真实临时子进程，不调用模型或项目脚本。"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from lynx_agent.execution.config import ExecutionConfig
from lynx_agent.execution.environment import Redactor, build_environment
from lynx_agent.execution.service import ExecutionService
from lynx_agent.filesystem.policy import FilePolicy
from lynx_agent.tools.errors import ToolExecutionError


def service(tmp_path, **config):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    instance = ExecutionService(FilePolicy(root), ExecutionConfig(**config), tmp_path / "logs", lambda: str(uuid), secrets=("my-test-secret",))
    instance.enabled = True
    instance.broker.presenter = lambda item: instance.broker.decide(item.identifier, True) if item else None
    return instance


uuid = uuid4()


@pytest.mark.asyncio
async def test_exit_streams_redaction_and_log(tmp_path):
    instance = service(tmp_path)
    result = await instance.execute({"argv": [sys.executable, "-c", "import sys; print('my-test-secret'); print('oops',file=sys.stderr); sys.exit(3)"]})
    data = json.loads(result.content)
    assert not result.is_error and data["exit_code"] == 3
    assert "[已隐藏]" in data["stdout"] and "my-test-secret" not in result.content
    assert "oops" in data["stderr"]
    assert "my-test-secret" not in next((tmp_path / "logs").rglob("*.log")).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_approval_reject_invalidation_and_disabled(tmp_path):
    instance = service(tmp_path)
    instance.enabled = False
    with pytest.raises(ToolExecutionError):
        await instance.execute({"argv": [sys.executable, "-c", "print(1)"]})
    instance.enabled = True
    ready = asyncio.Event()
    instance.broker.presenter = lambda item: ready.set() if item else None
    task = asyncio.create_task(instance.execute({"argv": [sys.executable, "-c", "print(1)"]}))
    await ready.wait()
    assert not list((tmp_path / "logs").rglob("*.log"))
    instance.disable()
    assert json.loads((await task).content)["status"] == "rejected"
    assert not list((tmp_path / "logs").rglob("*.log"))


@pytest.mark.asyncio
async def test_timeout_cancel_and_subsequent_success(tmp_path):
    instance = service(tmp_path)
    arguments = {"argv": [sys.executable, "-c", "import time; print('started',flush=True); time.sleep(20)"], "timeout_seconds": 1}
    result = await instance.execute(arguments)
    assert result.is_error and json.loads(result.content)["status"] == "timeout"
    task = asyncio.create_task(instance.execute(arguments))
    await asyncio.sleep(0.3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert json.loads(instance.last_result.content)["status"] == "cancelled"
    assert not (await instance.execute({"argv": [sys.executable, "-c", "print('next')"]})).is_error


@pytest.mark.asyncio
async def test_large_output_bounded(tmp_path):
    instance = service(tmp_path)
    result = await instance.execute({"argv": [sys.executable, "-c", "import sys; sys.stdout.write('x' * (20*1024*1024))"]})
    data = json.loads(result.content)
    assert len(result.content) <= instance.max_result_chars
    assert data["total_stdout_bytes"] == 20 * 1024 * 1024
    assert data["stdout_truncated"]
    assert next((tmp_path / "logs").rglob("*.log")).stat().st_size >= 20 * 1024 * 1024
    page = json.loads(instance.read_log(SimpleNamespace(content=result.content, result_id=result.result_id), 100000, 200))
    assert page["content"] == "x" * 200 or "[stdout]" in page["content"]
    assert len(page["content"]) == 200 and page["next_offset"] == 100200


def test_environment_and_chunk_redaction(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "hidden")
    monkeypatch.setenv("RAG_MCP_TOKEN", "hidden")
    monkeypatch.setenv("UV_INDEX_PASSWORD", "hidden")
    monkeypatch.setenv("UV_CACHE_DIR", "cache")
    env = build_environment(("DEEPSEEK_API_KEY",))
    assert not {"DEEPSEEK_API_KEY", "RAG_MCP_TOKEN", "UV_INDEX_PASSWORD"}.intersection(env)
    assert env["UV_CACHE_DIR"] == "cache" and env["PYTHONUTF8"] == "1"
    if os.name == "nt":
        assert any(key.upper() == "SYSTEMROOT" for key in env)
    redact = Redactor(("secret-token",))
    text = redact.feed("a secret-") + redact.feed("token b") + redact.feed("", final=True)
    assert text == "a [已隐藏] b"


@pytest.mark.asyncio
@pytest.mark.parametrize("lowercase", [False, True], ids=["uppercase", "lowercase"])
async def test_commands_inherit_proxies_without_unrelated_variables(tmp_path, monkeypatch, lowercase):
    proxies = {
        "HTTP_PROXY": "http://127.0.0.1:7890",
        "HTTPS_PROXY": "http://127.0.0.1:7890",
        "ALL_PROXY": "socks5://127.0.0.1:7891",
        "NO_PROXY": "127.0.0.1,localhost,::1",
    }
    for key, value in proxies.items():
        monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv(key.lower(), raising=False)
        monkeypatch.setenv(key.lower() if lowercase else key, value)
    monkeypatch.setenv("PRIVATE_API_TOKEN", "must-not-inherit")
    monkeypatch.setenv("UNRELATED_SETTING", "must-not-inherit")
    script = (
        "import json,os; "
        "wanted={'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY',"
        "'PRIVATE_API_TOKEN','UNRELATED_SETTING'}; "
        "print(json.dumps({k.upper():v for k,v in os.environ.items() if k.upper() in wanted}))"
    )
    result = await service(tmp_path).execute({"argv": [sys.executable, "-c", script]})
    data = json.loads(result.content)
    assert data["exit_code"] == 0
    assert json.loads(data["stdout"]) == proxies
