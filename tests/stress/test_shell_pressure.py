"""显式开启的50轮真实子进程清理压力，不运行仓库脚本或真实模型。"""

import asyncio
import json
import os
import sys
from uuid import uuid4

import pytest

from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.config.schema import ToolExecutionConfig
from lynx_agent.execution.config import ExecutionConfig
from lynx_agent.execution.service import ExecutionService
from lynx_agent.filesystem.policy import FilePolicy
from lynx_agent.providers.models import ModelToolCall
from lynx_agent.tools import ToolRegistry
from lynx_agent.tools.builtin.shell import ExecCommandTool

pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(
    os.environ.get("LYNX_PRESSURE_TEST") != "1", reason="需显式设置LYNX_PRESSURE_TEST=1")]


def handle_count():
    if os.name != "nt":
        return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    count = wintypes.DWORD()
    assert kernel.GetProcessHandleCount(kernel.GetCurrentProcess(), ctypes.byref(count))
    return count.value


def is_running(pid):
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return open(f"/proc/{pid}/stat", encoding="ascii").read().split()[21] if os.path.isfile(f"/proc/{pid}/stat") else True
        except ProcessLookupError:
            return False
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    process = kernel.OpenProcess(0x1000, False, pid)
    if not process:
        return False
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(process, ctypes.byref(code)) or code.value != 259:
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        assert kernel.GetProcessTimes(process, *(ctypes.byref(item) for item in times))
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        kernel.CloseHandle(process)


async def test_fifty_approved_timeout_cancel_cycles(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    session = str(uuid4())
    service = ExecutionService(FilePolicy(root), ExecutionConfig(), tmp_path / "logs", lambda: session)
    service.enabled = True
    approvals = []
    def present(request):
        if request:
            approvals.append(request.identifier)
            service.broker.decide(request.identifier, True)
    service.broker.presenter = present
    pids, handles = [], []
    for index in range(50):
        marker = root / f"pids-{index}.json"
        script = ("import os,sys,subprocess,time,json,pathlib; "
                  "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                  f"pathlib.Path({str(marker)!r}).write_text(json.dumps([os.getpid(),p.pid])); time.sleep(30)")
        task = asyncio.create_task(service.execute({"argv": [sys.executable, "-c", script], "timeout_seconds": 1}))
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)
        current = [(pid, is_running(pid)) for pid in json.loads(marker.read_text())]
        assert all(identity for _, identity in current)
        pids.extend(current)
        if index % 2:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert json.loads((await task).content)["status"] == "timeout"
        async with asyncio.timeout(5):
            while any(is_running(pid) == identity for pid, identity in current):
                await asyncio.sleep(0.02)
        if (index + 1) % 10 == 0:
            handles.append(handle_count())
    assert len(approvals) == 50 and len(set(approvals)) == 50
    # PID可在50轮或其他并行测试中复用，必须连同创建时间核对。
    assert not any(is_running(pid) == identity for pid, identity in pids)
    if all(value is not None for value in handles):
        assert max(handles) - min(handles) <= 8, handles
    print(f"\nShell压力：50次独立审批，100个进程已退出；句柄采样：{handles}")


async def test_approval_wait_over_forty_seconds_outside_thirty_second_timeout(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    session = str(uuid4())
    service = ExecutionService(FilePolicy(root), ExecutionConfig(), tmp_path / "logs", lambda: session)
    service.enabled = True
    def present(request):
        if request:
            asyncio.get_running_loop().call_later(41, service.broker.decide, request.identifier, True)
    service.broker.presenter = present
    executor = ToolExecutor(ToolRegistry([ExecCommandTool(service)]), ToolExecutionConfig(timeout_seconds=30))
    item = ModelToolCall(id="waited", name="exec_command", arguments=json.dumps({"argv": [sys.executable, "-c", "print('approved')"]}))
    result = await executor.execute(item, uuid4(), 4000)
    assert not result.is_error and json.loads(result.content)["status"] == "completed"
