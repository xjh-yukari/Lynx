"""批量工具：一次授权连续执行、范围失效、拒绝回溯和覆盖保护。"""

import json
from pathlib import Path

import pytest
from test_policy import approve, environment

from lynx_agent.approval.cards import batch_text, page_count
from lynx_agent.approval.config import ApprovalConfig
from lynx_agent.approval.models import PendingApproval
from lynx_agent.tools.builtin.batch import (
    BatchCopyFilesTool,
    BatchRenameFilesTool,
    BatchReplaceTextTool,
)
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError


@pytest.mark.asyncio
async def test_batch_replace_one_card_then_reuses_grant(tmp_path):
    env = environment(tmp_path)
    first = env.root / "a.txt"
    second = env.root / "b.txt"
    first.write_text("hello world hello", encoding="utf-8")
    second.write_text("hello", encoding="utf-8")
    cards = []
    approve(env.manager, "batch", cards)
    progress = []
    env.manager.on_progress = progress.append
    tool = BatchReplaceTextTool(env.access, env.manager)
    result = json.loads(await tool.execute(
        paths=[str(first), str(second)], old_text="hello", new_text="hi", purpose="整理问候"))
    assert result["succeeded"] == 2 and not result["stopped"]
    assert progress[-1]["phase"] == "finished" and progress[-1]["succeeded"] == 2
    assert first.read_text(encoding="utf-8") == "hi world hi"
    assert len(cards) == 1
    card = batch_text(cards[0], 10)
    for text in ("任务说明", "读取目录", "写入目录", "文件清单", "预计：2 个文件", "覆盖原文件：是",
                 "永久删除：否", "备份位置", "授权有效范围"):
        assert text in card
    backups = list((tmp_path / "data" / "backups").rglob("*a.txt"))
    assert backups and backups[0].read_text(encoding="utf-8") == "hello world hello"

    await tool.execute(paths=[str(first)], old_text="hi", new_text="yo", purpose="继续")
    assert first.read_text(encoding="utf-8") == "yo world yo"
    assert len(cards) == 1  # 同一任务内复用授权，不再次询问


@pytest.mark.asyncio
async def test_batch_grant_expires_on_turn_session_and_permission_change(tmp_path):
    env = environment(tmp_path)
    path = env.root / "a.txt"
    path.write_text("one", encoding="utf-8")
    cards = []
    approve(env.manager, "batch", cards)
    tool = BatchReplaceTextTool(env.access, env.manager)
    await tool.execute(paths=[str(path)], old_text="one", new_text="two", purpose="第一次")
    assert len(cards) == 1
    env.manager.end_turn()
    env.manager.begin_turn()
    await tool.execute(paths=[str(path)], old_text="two", new_text="three", purpose="第二次")
    assert len(cards) == 2

    env.state["permission"] = "full"
    env.manager.invalidate("permission_mode_changed")
    await tool.execute(paths=[str(path)], old_text="three", new_text="four", purpose="第三次")
    assert len(cards) == 3

    env.state["session"] = "session-2"
    env.manager.invalidate("session_switch")
    await tool.execute(paths=[str(path)], old_text="four", new_text="five", purpose="第四次")
    assert len(cards) == 4


@pytest.mark.asyncio
async def test_new_directory_or_operation_requires_new_card(tmp_path):
    env = environment(tmp_path)
    one = env.root / "one"
    two = env.root / "two"
    one.mkdir()
    two.mkdir()
    (one / "a.txt").write_text("hello", encoding="utf-8")
    (two / "b.txt").write_text("hello", encoding="utf-8")
    cards = []
    approve(env.manager, "batch", cards)
    tool = BatchReplaceTextTool(env.access, env.manager)
    await tool.execute(paths=[str(one / "a.txt")], old_text="hello", new_text="hi", purpose="目录一")
    assert len(cards) == 1
    await tool.execute(paths=[str(two / "b.txt")], old_text="hello", new_text="hi", purpose="目录二")
    assert len(cards) == 2
    copy = BatchCopyFilesTool(env.access, env.manager)
    await copy.execute(sources=[str(one / "a.txt")], destination=str(env.root), purpose="复制")
    assert len(cards) == 3


@pytest.mark.asyncio
async def test_denied_batch_cannot_be_replayed_with_another_tool(tmp_path):
    env = environment(tmp_path)
    source = env.root / "a.txt"
    source.write_text("hello", encoding="utf-8")
    destination = env.root / "out"
    destination.mkdir()
    approve(env.manager, "deny")
    copy = BatchCopyFilesTool(env.access, env.manager)
    with pytest.raises(ToolExecutionError, match="拒绝"):
        await copy.execute(sources=[str(source)], destination=str(destination), purpose="复制")
    assert not (destination / "a.txt").exists()
    with pytest.raises(ToolExecutionError, match="拒绝"):
        await env.access.invoke("write_file", {"path": "out/a.txt", "content": "wrapped"})
    assert not (destination / "a.txt").exists()

    replace = BatchReplaceTextTool(env.access, env.manager)
    with pytest.raises(ToolExecutionError, match="拒绝"):
        await replace.execute(paths=[str(source)], old_text="hello", new_text="changed", purpose="替换")
    with pytest.raises(ToolExecutionError, match="拒绝"):
        await env.access.invoke("edit_file", {"path": "a.txt", "old_text": "hello", "new_text": "wrapped"})
    assert source.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_model_cannot_forge_approval_fields(tmp_path):
    env = environment(tmp_path)
    path = env.root / "a.txt"
    path.write_text("hello", encoding="utf-8")
    tool = BatchReplaceTextTool(env.access, env.manager)
    with pytest.raises(ToolValidationError):
        await tool.execute(paths=[str(path)], old_text="hello", new_text="hi", approved=True)
    assert not env.manager.decide("model", True)
    assert not env.manager.decide("", "batch")
    assert path.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_external_change_is_skipped_without_overwrite(tmp_path, monkeypatch):
    env = environment(tmp_path)
    path = env.root / "a.txt"
    path.write_text("hello", encoding="utf-8")
    approve(env.manager, "batch")
    tool = BatchReplaceTextTool(env.access, env.manager)
    original = tool.authorize

    async def authorize(plan):
        grant = await original(plan)
        path.write_text("external", encoding="utf-8")
        return grant

    monkeypatch.setattr(tool, "authorize", authorize)
    result = json.loads(await tool.execute(paths=[str(path)], old_text="hello", new_text="hi"))
    assert result["skipped"] == 1 and not result["changed"]
    assert path.read_text(encoding="utf-8") == "external"


@pytest.mark.asyncio
async def test_stop_reports_partial_changes_without_rollback(tmp_path, monkeypatch):
    env = environment(tmp_path)
    first = env.root / "a.txt"
    second = env.root / "b.txt"
    first.write_text("hello", encoding="utf-8")
    second.write_text("hello", encoding="utf-8")
    approve(env.manager, "batch")
    tool = BatchReplaceTextTool(env.access, env.manager)
    original = tool.authorize

    async def authorize(plan):
        grant = await original(plan)
        env.manager.request_stop()
        return grant

    monkeypatch.setattr(tool, "authorize", authorize)
    result = json.loads(await tool.execute(paths=[str(first), str(second)], old_text="hello", new_text="hi"))
    assert result["status"] == "stopped" and result["stopped"]
    assert first.read_text(encoding="utf-8") == "hello"
    assert second.read_text(encoding="utf-8") == "hello"


@pytest.mark.asyncio
async def test_batch_limits_block_before_any_change(tmp_path):
    env = environment(tmp_path, config=ApprovalConfig(mode="auto", max_batch_files=2))
    paths = []
    for index in range(3):
        path = env.root / f"{index}.txt"
        path.write_text("hello", encoding="utf-8")
        paths.append(str(path))
    cards = []
    approve(env.manager, "batch", cards)
    tool = BatchReplaceTextTool(env.access, env.manager)
    with pytest.raises(ToolExecutionError, match="上限"):
        await tool.execute(paths=paths, old_text="hello", new_text="hi")
    assert not cards
    assert all(Path(path).read_text(encoding="utf-8") == "hello" for path in paths)


@pytest.mark.asyncio
async def test_copy_never_overwrites_and_rename_records_recovery(tmp_path):
    env = environment(tmp_path)
    source = env.root / "a.txt"
    source.write_text("hello", encoding="utf-8")
    destination = env.root / "out"
    destination.mkdir()
    (destination / "a.txt").write_text("keep", encoding="utf-8")
    approve(env.manager, "batch")
    copy = BatchCopyFilesTool(env.access, env.manager)
    result = json.loads(await copy.execute(sources=[str(source)], destination=str(destination), purpose="复制"))
    assert result["skipped"] == 1
    assert (destination / "a.txt").read_text(encoding="utf-8") == "keep"

    rename = BatchRenameFilesTool(env.access, env.manager)
    renamed = json.loads(await rename.execute(
        renames=[{"source": str(source), "new_name": "renamed.txt"}], purpose="改名"))
    assert renamed["succeeded"] == 1
    assert not source.exists() and (env.root / "renamed.txt").read_text(encoding="utf-8") == "hello"
    lines = (env.manager.audit.path).read_text(encoding="utf-8").splitlines()
    assert any('"result": "renamed"' in line and '"recovery"' in line for line in lines)


@pytest.mark.asyncio
async def test_batch_can_approve_specific_external_directory_without_full_permission(tmp_path):
    env = environment(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    source = outside / "source.txt"
    source.write_text("outside", encoding="utf-8")
    destination = tmp_path / "external-out"
    destination.mkdir()
    cards = []
    approve(env.manager, "batch", cards)
    copy = BatchCopyFilesTool(env.access, env.manager)
    result = json.loads(await copy.execute(sources=[str(source)], destination=str(destination), purpose="外部目录"))
    assert result["succeeded"] == 1
    assert (destination / "source.txt").read_text(encoding="utf-8") == "outside"
    assert len(cards) == 1
    assert str(destination) in batch_text(cards[0], 10)


@pytest.mark.asyncio
async def test_batch_tools_unavailable_in_read_only_or_paused(tmp_path):
    env = environment(tmp_path)
    env.access.policy.set_mode("read")
    assert not BatchCopyFilesTool(env.access, env.manager).available
    env.access.policy.set_mode("workspace")
    env.manager.set_mode("paused")
    assert not BatchCopyFilesTool(env.access, env.manager).available


@pytest.mark.asyncio
async def test_batch_tools_refuse_sensitive_and_private_paths(tmp_path):
    env = environment(tmp_path)
    (env.root / ".env").write_text("PRIVATE=1", encoding="utf-8")
    destination = env.root / "out"
    destination.mkdir()
    approve(env.manager, "batch")
    copy = BatchCopyFilesTool(env.access, env.manager)
    with pytest.raises(ToolExecutionError, match="敏感"):
        await copy.execute(sources=[str(env.root / ".env")], destination=str(destination), purpose="凭据")
    replace = BatchReplaceTextTool(env.access, env.manager)
    with pytest.raises(ToolExecutionError, match="敏感"):
        await replace.execute(paths=[str(env.root / ".env")], old_text="PRIVATE", new_text="x")


def test_card_pages_long_file_list(tmp_path):
    env = environment(tmp_path)
    files = []
    for index in range(25):
        path = env.root / f"{index:02d}.txt"
        path.write_text("hello", encoding="utf-8")
        files.append(str(path))
    tool = BatchReplaceTextTool(env.access, env.manager)
    # build_plan 不申请授权；这里只检查卡片分页。
    parsed = tool._parse({"paths": files, "old_text": "hello", "new_text": "hi", "purpose": "分页"})
    _, plan = tool.build_plan(parsed)
    pending = PendingApproval("id", "batch", plan, supports_batch=True)
    assert page_count(plan, 10) == 3
    first = batch_text(pending, 10)
    assert "第 1/3 页" in first and "00.txt" in first
    pending.page = 2
    assert "第 3/3 页" in batch_text(pending, 10)
