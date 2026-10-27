"""审批绑定、权限、编码与进程树的真实边界验证（独立测试模块名）。"""

import asyncio
import json
import os
import sys
from uuid import uuid4

import pytest

from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.config.schema import ToolExecutionConfig
from lynx_agent.execution.config import ExecutionConfig
from lynx_agent.execution.environment import StreamDecoder
from lynx_agent.execution.service import ExecutionService
from lynx_agent.filesystem.policy import FilePolicy
from lynx_agent.providers.models import ModelToolCall
from lynx_agent.tools import ToolRegistry
from lynx_agent.tools.builtin.shell import ExecCommandTool
from lynx_agent.tools.errors import ToolExecutionError


def make_service(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    session = str(uuid4())
    service = ExecutionService(FilePolicy(root), ExecutionConfig(), tmp_path / "logs", lambda: session)
    service.enabled = True
    return service


def alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@pytest.mark.asyncio
async def test_approval_wait_outside_general_deadline(tmp_path):
    service = make_service(tmp_path)
    def present(item):
        if item:
            asyncio.get_running_loop().call_later(0.08, service.broker.decide, item.identifier, True)
    service.broker.presenter = present
    executor = ToolExecutor(ToolRegistry([ExecCommandTool(service)]), ToolExecutionConfig(timeout_seconds=0.01))
    item = ModelToolCall(id="call", name="exec_command", arguments=json.dumps({"argv": [sys.executable, "-c", "print('ok')"]}))
    result = await executor.execute(item, uuid4(), 4000)
    assert not result.is_error and json.loads(result.content)["status"] == "completed"


@pytest.mark.asyncio
async def test_approval_permission_changed_and_wrong_identifier(tmp_path):
    service = make_service(tmp_path)
    ready = asyncio.Event()
    service.broker.presenter = lambda request: ready.set() if request else None
    task = asyncio.create_task(service.execute({"argv": [sys.executable, "-c", "print('do not execute')"]}))
    await ready.wait()
    assert not service.broker.decide("model says approved", True)
    assert not task.done()
    service.policy.set_mode("full")
    service.broker.decide(service.broker.pending.identifier, True)
    with pytest.raises(ToolExecutionError, match="失效"):
        await task
    assert not list(service.logs_dir.rglob("*.log"))


@pytest.mark.asyncio
async def test_directory_private_roots_and_outside(tmp_path):
    service = make_service(tmp_path)
    service.broker.presenter = lambda item: service.broker.decide(item.identifier, True) if item else None
    with pytest.raises(ToolExecutionError):
        await service.execute({"argv": [sys.executable], "working_dir": str(tmp_path)})
    service.policy.set_mode("full")
    service.logs_dir.mkdir()
    with pytest.raises(ToolExecutionError):
        await service.execute({"argv": [sys.executable], "working_dir": str(service.logs_dir)})


@pytest.mark.asyncio
async def test_invalid_arguments_and_start_failure_are_safe_results(tmp_path, monkeypatch):
    service = make_service(tmp_path)
    service.broker.presenter = lambda item: service.broker.decide(item.identifier, True) if item else None
    executor = ToolExecutor(ToolRegistry([ExecCommandTool(service)]), ToolExecutionConfig())
    invalid = ModelToolCall(id="bad", name="exec_command", arguments='{"argv": []}')
    assert (await executor.execute(invalid, uuid4(), 4000)).execution_state == "not_executed"
    async def fail_spawn(request, env):
        raise OSError("fake-private-path")
    monkeypatch.setattr("lynx_agent.execution.service.spawn", fail_spawn)
    result = await service.execute({"argv": [sys.executable, "-c", "print(1)"]})
    assert result.is_error and result.execution_state == "not_executed"
    assert json.loads(result.content)["status"] == "failed_to_start"
    assert "fake-private-path" not in result.content


def test_utf8_across_chunks_gbk_and_invalid_bytes(monkeypatch):
    decoder = StreamDecoder()
    encoded = "中文".encode()
    assert decoder.feed(encoded[:2]) + decoder.feed(encoded[2:], final=True) == "中文"
    monkeypatch.setattr("lynx_agent.execution.environment.locale.getpreferredencoding", lambda flag: "gbk")
    assert StreamDecoder().feed("中文".encode("gbk"), final=True) == "中文"
    assert isinstance(StreamDecoder().feed(b"\xff\xfe\x80", final=True), str)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_process_and_grandchild_terminated(tmp_path, cancel):
    service = make_service(tmp_path)
    service.broker.presenter = lambda item: service.broker.decide(item.identifier, True) if item else None
    marker = tmp_path / "pids.json"
    script = ("import os,sys,subprocess,time,json,pathlib; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(40)']); "
              f"pathlib.Path({str(marker)!r}).write_text(json.dumps([os.getpid(),p.pid])); time.sleep(40)")
    task = asyncio.create_task(service.execute({"argv": [sys.executable, "-c", script], "timeout_seconds": 1}))
    async with asyncio.timeout(5):
        while not marker.exists():
            await asyncio.sleep(0.01)
    pids = json.loads(marker.read_text())
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert json.loads((await task).content)["status"] == "timeout"
    async with asyncio.timeout(5):
        while any(alive(pid) for pid in pids):
            await asyncio.sleep(0.02)
