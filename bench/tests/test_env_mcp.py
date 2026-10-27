"""测试环境 MCP：业务工具、调用记录、故障注入与路径边界。"""

import pytest
from agent_bench.env_mcp.server import TOOL_NAMES, build_server
from agent_bench.env_mcp.store import EnvStore
from agent_bench.env_mcp.tools import EnvTools, ToolFault


@pytest.fixture
def tools(tmp_path):
    store = EnvStore(tmp_path / "env")
    store.reset()
    return EnvTools(store, tmp_path / "workspace")


@pytest.mark.asyncio
async def test_notes_roundtrip_and_records(tools):
    saved = await tools.notes_set("k1", "v1")
    assert saved["saved"] is True and saved["value_chars"] == 2
    assert (await tools.notes_get("k1"))["value"] == "v1"
    assert (await tools.notes_search("k1"))["count"] == 1
    records = tools.store.records()
    assert [item["tool"] for item in records] == ["notes_set", "notes_get", "notes_search"]
    assert records[0]["arguments"] == {"key": "k1", "value": "v1"}
    assert records[0]["is_error"] is False
    assert [item["seq"] for item in records] == [1, 2, 3]


@pytest.mark.asyncio
async def test_missing_note_raises_fault(tools):
    with pytest.raises(ToolFault) as error:
        await tools.notes_get("nope")
    assert error.value.kind == "not_found"
    assert tools.store.records()[-1]["is_error"] is True


@pytest.mark.asyncio
async def test_documents_read_slices(tools):
    tools.store.mutate(lambda state, _: state["documents"].__setitem__("doc", "x" * 5000))
    payload = await tools.documents_read("doc", offset=1000, limit=500)
    assert payload["total_chars"] == 5000
    assert payload["content"] == "x" * 500
    assert payload["next_offset"] == 1500


@pytest.mark.asyncio
async def test_sandbox_write_and_read(tools):
    await tools.sandbox_write_file("sub/note.txt", "hello")
    payload = await tools.sandbox_read_file("sub/note.txt")
    assert payload["content"] == "hello"
    assert (await tools.sandbox_list_files())["paths"] == ["sub/note.txt"]
    assert tools.state_view()["files"]["sub/note.txt"] == "hello"


@pytest.mark.asyncio
async def test_path_escape_is_rejected(tools):
    with pytest.raises(ToolFault):
        await tools.sandbox_write_file("../outside.txt", "x")
    with pytest.raises(ToolFault):
        await tools.sandbox_read_file("../../etc/passwd")


@pytest.mark.asyncio
async def test_fault_injection_consumes_budget(tools):
    tools.store.set_control({"faults": {"flaky_task": {"mode": "error", "times": 2,
                                                       "message": "坏了"}}})
    with pytest.raises(ToolFault):
        await tools.flaky_task("a")
    with pytest.raises(ToolFault):
        await tools.flaky_task("a")
    assert (await tools.flaky_task("a"))["status"] == "done"
    errors = [item for item in tools.store.records() if item["is_error"]]
    assert len(errors) == 2


@pytest.mark.asyncio
async def test_timeout_fault_sleeps_then_fails(tools):
    tools.store.set_control({"faults": {"flaky_task": {"mode": "timeout", "times": 1,
                                                       "seconds": 0.01}}})
    with pytest.raises(ToolFault) as error:
        await tools.flaky_task()
    assert error.value.kind == "timeout"


@pytest.mark.asyncio
async def test_env_step_merges_state_and_faults(tools):
    tools.apply_env({"kv": {"a": "1"}, "orders": {"X": {"status": "ok"}},
                     "faults": {"flaky_task": {"mode": "error", "times": 1}}})
    assert (await tools.notes_get("a"))["value"] == "1"
    assert (await tools.orders_lookup("X"))["status"] == "ok"


@pytest.mark.asyncio
async def test_reset_clears_state_and_records(tmp_path):
    store = EnvStore(tmp_path / "env")
    store.reset({"kv": {"a": "1"}})
    service = EnvTools(store, tmp_path / "ws")
    await service.notes_get("a")
    assert store.records()
    store.reset()
    assert store.records() == []
    assert store.snapshot()["kv"] == {}


def test_server_exposes_only_business_tools():
    import tempfile
    from pathlib import Path

    root = Path(tempfile.mkdtemp())
    server = build_server(EnvStore(root / "env"), root / "ws")
    # MCPServer 的工具管理器是工具列表的真源；管理操作不在其中。
    tools = server._tool_manager.list_tools()
    names = {tool.name for tool in tools}
    assert names == set(TOOL_NAMES)
    assert "reset" not in names and "snapshot" not in names
