"""真实 Engram 3.4.0 + MCP 验证；每个用例使用独立临时库，不调用模型 API。"""

from pathlib import Path

import pytest

from lynx_agent.memory.commands import handle_memory_command
from lynx_agent.memory.config import MemoryConfig
from lynx_agent.memory.errors import MemoryError
from lynx_agent.memory.service import MemoryService

BINARY = Path("E:/github/lynx_agent_data/bin/engram/3.4.0/engram.exe")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not BINARY.is_file(), reason="需安装已固定版本 Engram；普通环境不联网下载"
    ),
]


def service(tmp_path, *, mode=None, project="projecta", root=None, **kwargs):
    cfg = MemoryConfig(
        enabled=True, executable=str(BINARY), project_id=project, **kwargs
    )
    return MemoryService(
        cfg,
        root or tmp_path / "memory",
        tmp_path / project,
        mode or (lambda: "workspace"),
    )


async def save(memory, kind, body, **kwargs):
    identifier = await memory.propose(kind, body, **kwargs)
    await memory.preview(identifier)
    result = await memory.approve(identifier)
    assert "验证" in result
    return next(
        key
        for index in memory.indexes.values()
        for key, loc in index.items()
        if not loc.deleted and loc.kind == kind
    )


async def test_read_only_record_listing_is_verified_paginated_and_scoped(tmp_path, monkeypatch):
    memory = service(tmp_path)
    await memory.start()
    try:
        first = await save(memory, "fact", "First saved conclusion.", title="First")
        change = await memory.propose("decision", "Second conclusion.", title="Second", source_session="test-session")
        await memory.preview(change)
        await memory.approve(change)
        memory.mode = lambda: "read"
        metadata = await memory.list_records(limit=1)
        assert metadata["total"] == 2 and metadata["next_offset"] == 1
        assert "body" not in metadata["records"][0]
        page = await memory.list_records(limit=1, include_body=True)
        item = page["records"][0]
        assert item["body"] == "Second conclusion." and item["source_session"] == "test-session"
        assert item["origin"] == "user_confirmed"
        assert (await memory.list_records(offset=1))["records"][0]["key"] == first
        rendered = await handle_memory_command(memory, "/memory", "list")
        assert "First saved conclusion." in rendered and "Second conclusion." in rendered
        assert "本轮答复原文依据" in await handle_memory_command(memory, "/memory", "rules")
        for kwargs in ({"offset": -1}, {"limit": 51}, {"limit": True}):
            with pytest.raises(MemoryError):
                await memory.list_records(**kwargs)

        async def unavailable(key):
            raise MemoryError("不得泄漏底层错误")

        monkeypatch.setattr(memory, "get", unavailable)
        failed_page = await memory.list_records(include_body=True)
        assert all(not row["available"] and "body" not in row for row in failed_page["records"])
        assert "底层错误" not in str(failed_page)
    finally:
        await memory.close()
    other = service(tmp_path, project="other")
    await other.start()
    try:
        assert (await other.list_records())["total"] == 0
    finally:
        await other.close()


async def test_real_save_restart_recall_update_and_undo(tmp_path):
    first = service(tmp_path)
    await first.start()
    try:
        assert first.writable, first.status()
        key = await save(
            first,
            "fact",
            "The validation beverage is lapsangunique.",
            title="Validation beverage",
        )
        found = await first.search("lapsangunique")
        assert found[0][0] == key
        assert "lapsangunique" in found[0][1].body
        assert not any(
            name.startswith("mcp__")
            for backend in first.backends.values()
            for name in backend.connection.registry.names
        )
    finally:
        await first.close()
    second = service(tmp_path)
    await second.start()
    try:
        assert (await second.get(key)).revision == 1
        identifier = await second.propose(
            "fact",
            "The validation beverage is mintunique.",
            title="Validation beverage",
            key=key,
        )
        await second.preview(identifier)
        await second.approve(identifier)
        assert (await second.get(key)).revision == 2
        assert "mintunique" in (await second.get(key)).body
        restore = await second.undo(key, 1)
        assert "lapsangunique" in await second.preview(restore)
        await second.approve(restore)
        assert (await second.get(key)).revision == 3
        assert "lapsangunique" in (await second.get(key)).body
    finally:
        await second.close()


async def test_profiles_and_project_isolation(tmp_path):
    first = service(tmp_path)
    await first.start()
    try:
        assert first.writable, first.status()
        await save(first, "user", "偏好：解释代码时先说明整体流程。")
        await save(first, "agents", "项目约定：使用 uv，修改后运行 pytest。")
        key = await save(
            first,
            "decision",
            "Project A uses projectaunique.",
            title="Project A architecture",
        )
        assert "USER" in first.profile_text and "AGENTS" in first.profile_text
    finally:
        await first.close()
    second = service(tmp_path, project="projectb")
    await second.start()
    try:
        assert "USER" in second.profile_text
        assert "AGENTS" not in second.profile_text
        assert not await second.search("projectaunique")
        with pytest.raises(MemoryError):
            await second.get(key)
    finally:
        await second.close()


async def test_read_only_preview_and_mode_change_invalidate(tmp_path):
    mode = ["workspace"]
    memory = service(tmp_path, mode=lambda: mode[0])
    await memory.start()
    try:
        identifier = await memory.propose("user", "默认使用中文。")
        # 未经人工预览的首次批准只展示差异，不写库。
        assert "拟修改内容" in await memory.approve(identifier)
        assert not memory.indexes["profile"]
        mode[0] = "read"
        with pytest.raises(MemoryError):
            await memory.approve(identifier)
        with pytest.raises(MemoryError):
            await memory.propose("fact", "不应保存")
        mode[0] = "full"
        with pytest.raises(MemoryError, match="权限已改变"):
            await memory.approve(identifier)
    finally:
        await memory.close()


async def test_delete_is_verified_and_does_not_return_on_restart(tmp_path):
    memory = service(tmp_path)
    await memory.start()
    try:
        key = await save(memory, "fact", "Forget deleteunique.", title="Delete fixture")
        identifier = await memory.propose_delete(key)
        await memory.preview(identifier)
        assert "已删除" in await memory.approve(identifier)
        assert not await memory.search("deleteunique")
        with pytest.raises(MemoryError):
            await memory.get(key)
    finally:
        await memory.close()
    restored = service(tmp_path)
    await restored.start()
    try:
        assert not await restored.search("deleteunique")
    finally:
        await restored.close()


async def test_uncertain_save_reconciles_without_repeating_write(tmp_path, monkeypatch):
    memory = service(tmp_path)
    await memory.start()
    try:
        backend = memory.backends["project"]
        original = backend.save
        writes = []

        async def uncertain(record):
            writes.append(record.operation_id)
            await original(record)
            raise MemoryError("模拟响应丢失")

        monkeypatch.setattr(backend, "save", uncertain)
        identifier = await memory.propose(
            "fact", "Recovery recoverunique.", title="Recovery fixture"
        )
        await memory.preview(identifier)
        with pytest.raises(MemoryError):
            await memory.approve(identifier)
        assert not memory.writable and memory.uncertain
        await memory.reconcile()
        assert len(writes) == 1
        assert memory.writable
        assert await memory.search("recoverunique")
    finally:
        await memory.close()


async def test_cache_and_corruption_do_not_overwrite_rules(tmp_path, monkeypatch):
    memory = service(tmp_path)
    await memory.start()
    try:
        await save(memory, "agents", "不可覆盖自动生成文件。")

        async def unavailable(*args, **kwargs):
            raise MemoryError("模拟断开")

        monkeypatch.setattr(memory.backends["project"], "get", unavailable)
        await memory.refresh()
        assert "不可覆盖" in memory.profile_text
        assert any("缓存" in warning for warning in memory.warnings)
        (memory.directories["project"] / "profiles.cache.json").unlink()
        await memory.refresh()
        assert memory.rules_unavailable
        with pytest.raises(MemoryError):
            memory.ensure_rules()
    finally:
        await memory.close()


async def test_credentials_budget_and_reserved_profile_protection(tmp_path):
    memory = service(tmp_path, profile_tokens=180)
    await memory.start()
    try:
        with pytest.raises(MemoryError, match="凭据"):
            await memory.propose("fact", "API_KEY=abcdefg123456789")
        with pytest.raises(MemoryError, match="预算"):
            await memory.propose("soul", "长" * 500)
        key = await save(memory, "user", "默认中文。")
        with pytest.raises(MemoryError):
            await memory.propose("fact", "越权覆盖", key=key, scope="user")
    finally:
        await memory.close()


async def test_commands_preview_reject_recall_and_lock(tmp_path):
    memory = service(tmp_path)
    await memory.start()
    duplicate = service(tmp_path)
    try:
        await duplicate.start()
        assert not duplicate.started
        assert memory.writable
        result = await handle_memory_command(
            memory, "/profile", "set user 默认详细解释。"
        )
        assert "拟修改内容" in result
        assert not memory.profiles
        identifier = next(iter(memory.pending))
        await handle_memory_command(memory, "/memory", "reject " + identifier)
        assert not memory.pending
        await handle_memory_command(memory, "/memory", "recall off")
        assert not memory.recall_enabled
    finally:
        await duplicate.close()
        await memory.close()


async def test_chinese_retrieval_append_and_partial_edit_preserve_other_preferences(
    tmp_path,
):
    memory = service(tmp_path)
    await memory.start()
    try:
        key = await save(memory, "user", "偏好：先讲整体流程。\n回答使用中文。")
        identifier = await memory.propose("user", "解释时举例。", edit="append")
        await memory.preview(identifier)
        await memory.approve(identifier)
        body = (await memory.get(key)).body
        assert "使用中文" in body and "举例" in body
        identifier = await memory.propose(
            "user", "偏好：先给结论。", edit="replace_text", find="偏好：先讲整体流程。"
        )
        await memory.preview(identifier)
        await memory.approve(identifier)
        assert "使用中文" in (await memory.get(key)).body
        assert "整体流程" not in (await memory.get(key)).body
        await save(
            memory, "decision", "项目采用上下文压缩，保留近期对话。", title="上下文管理"
        )
        assert await memory.search("上下文压缩")
    finally:
        await memory.close()


async def test_foreign_user_and_same_native_id_do_not_alias(tmp_path):
    first = service(tmp_path)
    await first.start()
    try:
        await save(first, "user", "专属于用户甲。")
        old = await save(first, "fact", "alphaunique", title="alpha")
    finally:
        await first.close()
    second = service(tmp_path, project="projectb", user_id="another")
    await second.start()
    try:
        new = await save(second, "fact", "betaunique", title="beta")
        assert old != new
        with pytest.raises(MemoryError):
            await second.get(old)
        assert "user" not in second.profiles
    finally:
        await second.close()


async def test_uncertain_write_survives_service_restart(tmp_path, monkeypatch):
    memory = service(tmp_path)
    await memory.start()
    backend = memory.backends["project"]
    original = backend.save

    async def lost_reply(record):
        await original(record)
        raise MemoryError("模拟响应丢失")

    monkeypatch.setattr(backend, "save", lost_reply)
    try:
        identifier = await memory.propose(
            "fact", "restartrecoveryunique", title="Restart"
        )
        await memory.preview(identifier)
        with pytest.raises(MemoryError):
            await memory.approve(identifier)
    finally:
        await memory.close()
    restored = service(tmp_path)
    await restored.start()
    try:
        assert restored.uncertain and not restored.writable
        await restored.reconcile()
        assert await restored.search("restartrecoveryunique")
        assert restored.writable
    finally:
        await restored.close()
