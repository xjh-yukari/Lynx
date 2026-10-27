import asyncio
import json
import os
import subprocess
import threading

import pytest

from lynx_agent.cli.permissions import (
    handle_approval_command,
    handle_permissions_command,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.filesystem.policy import SensitivePath
from lynx_agent.tools import ToolCall, ToolRegistry
from lynx_agent.tools.builtin.filesystem import create_file_tools
from lynx_agent.tools.errors import ToolExecutionError


@pytest.fixture
def files(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    access = FileAccess(root)
    return access, ToolRegistry(create_file_tools(access))


async def invoke(files, name, **arguments):
    result = await files[1].execute(ToolCall(id="test-file", name=name, arguments=arguments))
    return result, None if result.is_error else json.loads(result.content)


@pytest.mark.asyncio
async def test_list_find_read_search_with_lines_and_limits(files):
    access, _ = files
    root = access.policy.root
    (root / "src").mkdir()
    (root / "src" / "test_code.py").write_text("first\nload_config()\nlast", encoding="utf-8")
    result, listing = await invoke(files, "list_dir")
    assert not result.is_error and listing["results"][0]["name"] == "src"
    _, found = await invoke(files, "find_files", pattern="**/test_*.py")
    assert len(found["results"]) == 1
    _, read = await invoke(files, "read_file", path="src/test_code.py", offset=2, limit=1)
    assert read["content"] == "2: load_config()"
    assert read["next_offset"] == 3 and read["truncated"]
    _, matches = await invoke(files, "search_files", query="LOAD_CONFIG", glob="*.py")
    assert matches["results"][0]["line"] == 2
    assert matches["results"][0]["text"] == "load_config()"
    assert "不可信" in matches["notice"]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args", [
    ("list_dir", {"limit": True}), ("list_dir", {"limit": 201}),
    ("find_files", {"pattern": ""}), ("search_files", {"query": " "}),
    ("read_file", {"path": "x.txt", "offset": 0}), ("read_file", {"path": "x.txt", "limit": 501}),
    ("write_file", {"path": "x.txt", "content": "x", "approved": True}),
    ("edit_file", {"path": "x.txt", "old_text": "", "new_text": "x"}),
])
async def test_strict_arguments_reject_escalation_and_bad_ranges(files, name, args):
    result, _ = await invoke(files, name, **args)
    assert result.is_error


@pytest.mark.asyncio
async def test_all_modes_and_default_workspace(files, tmp_path):
    access, registry = files
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    assert access.policy.mode == "workspace"
    assert len(registry.names) == 6
    for mode in ("workspace", "read"):
        access.set_mode(mode)
        result, _ = await invoke(files, "read_file", path=str(outside))
        assert result.is_error and "超出" in result.content
    assert len(registry.names) == 4
    result, _ = await invoke(files, "write_file", path="denied.txt", content="x")
    assert result.is_error and not (access.policy.root / "denied.txt").exists()
    with pytest.raises(ToolExecutionError, match="仅查看"):
        access.policy.resolve("denied.txt", write=True)
    access.set_mode("full")
    _, read = await invoke(files, "read_file", path=str(outside))
    assert "outside" in read["content"]
    target = tmp_path / "outside-new.txt"
    _, write = await invoke(files, "write_file", path=str(target), content="created")
    assert write["status"] == "created" and target.read_text() == "created"


@pytest.mark.asyncio
async def test_create_edit_preview_confirmation_and_conflict(files):
    access, _ = files
    _, created = await invoke(files, "write_file", path="note.txt", content="old\n")
    assert created["status"] == "created"
    path = access.policy.root / "note.txt"
    _, proposed = await invoke(files, "edit_file", path="note.txt", old_text="old", new_text="new")
    identifier = proposed["approval_id"]
    assert path.read_text() == "old\n"
    preview = access.approve(identifier)
    assert "-old" in preview and "+new" in preview
    assert path.read_text() == "old\n"
    access.approve(identifier)
    assert path.read_text() == "new\n"
    _, next_change = await invoke(files, "write_file", path="note.txt", content="third")
    identifier = next_change["approval_id"]
    access.preview(identifier)
    path.write_text("external", encoding="utf-8")
    with pytest.raises(ValueError, match="已被其他操作修改"):
        access.approve(identifier)
    assert path.read_text() == "external"
    assert not list(path.parent.glob(".lynx-write-*"))


@pytest.mark.asyncio
async def test_sensitive_files_hidden_until_separate_explicit_approval(files):
    access, _ = files
    secret = access.policy.root / ".env"
    secret.write_text("PRIVATE_KEY=never-leak", encoding="utf-8")
    (access.policy.root / "normal.txt").write_text("normal", encoding="utf-8")
    for name, args in [("list_dir", {}), ("find_files", {"pattern": "*"}), ("search_files", {"query": "never-leak"})]:
        result, _ = await invoke(files, name, **args)
        assert "never-leak" not in result.content and '"name": ".env"' not in result.content
    result, _ = await invoke(files, "read_file", path=".env")
    assert result.is_error and "never-leak" not in result.content
    identifier = next(iter(access.pending))
    assert "尚未执行" in access.approve(identifier)
    with pytest.raises(SensitivePath):
        access.policy.resolve(".env")
    access.approve(identifier)
    _, permitted = await invoke(files, "read_file", path=".env")
    assert "never-leak" in permitted["content"]
    access.set_mode("full")
    result, _ = await invoke(files, "read_file", path=".env")
    assert result.is_error


@pytest.mark.parametrize("path", ["../outside.txt", "a/../../outside.txt", "C:relative.txt", "file.txt:stream", "NUL.txt",
    "CON", "COM1.txt", "trailing.txt.", "trailing.txt ", "\\\\server\\share\\x.txt", "\\\\?\\C:\\x.txt"])
def test_special_paths_rejected_even_full(files, path):
    policy = files[0].policy
    policy.set_mode("full")
    with pytest.raises(ToolExecutionError):
        policy.resolve(path)


@pytest.mark.asyncio
async def test_binary_large_and_unsupported_files_rejected(files):
    root = files[0].policy.root
    for name, content in [("binary.txt", b"abc\0secret"), ("large.txt", b"x" * 1048577), ("binary.exe", b"executable")]:
        (root / name).write_bytes(content)
        result, _ = await invoke(files, "read_file", path=name)
        assert result.is_error and "secret" not in result.content
    (root / "gbk.txt").write_bytes("中文文件".encode("gb18030"))
    _, result = await invoke(files, "read_file", path="gbk.txt")
    assert "中文文件" in result["content"]
    reply, _ = await invoke(files, "edit_file", path="gbk.txt", old_text="中文", new_text="新")
    assert reply.is_error and not files[0].pending


@pytest.mark.asyncio
async def test_excluded_directories_and_result_caps(files):
    root = files[0].policy.root
    (root / ".venv").mkdir()
    (root / ".venv" / "private.txt").write_text("hidden", encoding="utf-8")
    for i in range(4):
        (root / f"test{i}.txt").write_text("word\nword", encoding="utf-8")
    _, result = await invoke(files, "search_files", query="word", limit=2)
    assert len(result["results"]) == 2 and result["truncated"]
    reply, _ = await invoke(files, "read_file", path=".venv/private.txt")
    assert reply.is_error


def test_hard_links_and_windows_junction_blocked(files, tmp_path):
    access, _ = files
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "real.txt"
    target.write_text("outside", encoding="utf-8")
    hard = access.policy.root / "hard.txt"
    os.link(target, hard)
    with pytest.raises(ToolExecutionError, match="硬链接"):
        access.policy.resolve(str(hard))
    junction = access.policy.root / "linked"
    if os.name == "nt":
        result = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)], capture_output=True, check=False)
        if result.returncode:
            pytest.skip("系统不允许创建测试目录联接")
    else:
        junction.symlink_to(outside, target_is_directory=True)
    try:
        for mode in ("workspace", "read", "full"):
            access.set_mode(mode)
            with pytest.raises(ToolExecutionError, match="链接|联接"):
                access.policy.resolve(str(junction / "real.txt"))
        assert target.read_text() == "outside"
    finally:
        if os.name == "nt":
            junction.rmdir()
        else:
            junction.unlink()


@pytest.mark.asyncio
async def test_mode_changes_clear_pending_and_reject_does_not_write(files):
    access, _ = files
    path = access.policy.root / "x.txt"
    path.write_text("old", encoding="utf-8")
    _, proposal = await invoke(files, "write_file", path="x.txt", content="new")
    access.reject(proposal["approval_id"])
    assert path.read_text() == "old"
    _, proposal = await invoke(files, "write_file", path="x.txt", content="new")
    access.set_mode("read")
    with pytest.raises(ValueError, match="不存在"):
        access.approve(proposal["approval_id"])
    assert path.read_text() == "old"


@pytest.mark.asyncio
async def test_cancel_before_new_file_commit_waits_for_worker_and_does_not_write(files, monkeypatch):
    access, registry = files
    started = threading.Event()
    commit = access._commit

    def paused(path, data, **kwargs):
        started.set()
        kwargs["cancel"].wait(2)
        return commit(path, data, **kwargs)

    monkeypatch.setattr(access, "_commit", paused)
    task = asyncio.create_task(registry.execute(ToolCall(id="cancel", name="write_file", arguments={"path": "cancel.txt", "content": "x"})))
    async with asyncio.timeout(2):
        while not started.is_set():
            await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not (access.policy.root / "cancel.txt").exists()


def test_permission_commands_require_explicit_full_confirmation(files):
    class Runner:
        file_access = files[0]
        _operation_running = False

        def set_file_mode(self, mode):
            self.file_access.set_mode(mode)

    runner = Runner()
    assert "尚未改变" in handle_permissions_command(runner, "full")
    assert runner.file_access.policy.mode == "workspace"
    assert "完全权限" in handle_permissions_command(runner, "full confirm")
    assert runner.file_access.policy.mode == "full"
    handle_permissions_command(runner, "read")
    assert not runner.file_access.policy.writable
    assert "没有待确认" in handle_approval_command(runner, "/approvals")
