"""真实 stdio 子进程测试；不调用云端服务或模型。"""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from mcp import types
from pydantic import ValidationError

from lynx_agent.agent.tool_executor import ToolExecutor
from lynx_agent.config.schema import ToolExecutionConfig
from lynx_agent.mcp.client import MCPConnection
from lynx_agent.mcp.config import MCPConfig, MCPServerConfig
from lynx_agent.mcp.errors import MCPError
from lynx_agent.mcp.manager import MCPManager
from lynx_agent.mcp.policy import schema_validator, tool_name, validate_arguments
from lynx_agent.mcp.results import result_text
from lynx_agent.providers.models import ModelToolCall
from lynx_agent.tools import ToolCall, ToolRegistry, ToolValidationError

SERVER = Path(__file__).with_name("fixture_server.py").resolve()


def settings(tmp_path, **overrides):
    return MCPServerConfig(
        **(
            {
                "enabled": True,
                "command": sys.executable,
                "args": [str(SERVER)],
                "cwd": str(tmp_path),
                "allowed_tools": [
                    "add",
                    "identity",
                    "slow",
                    "crash",
                    "fail",
                    "large",
                    "change",
                ],
                "read_only_tools": [
                    "add",
                    "identity",
                    "slow",
                    "crash",
                    "fail",
                    "large",
                ],
                "startup_timeout": 10,
                "shutdown_timeout": 5,
            }
            | overrides
        )
    )


@pytest.mark.asyncio
async def test_real_stdio_discovery_call_error_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "must-not-inherit")
    registry = ToolRegistry()
    connection = MCPConnection(
        "demo", settings(tmp_path, env={"LYNX_MCP_TEST": "passed"}), registry
    )
    try:
        await connection.connect()
        assert connection.state == "connected", connection.error
        assert "mcp__demo__add" in registry.names
        assert "mcp__demo__change" not in registry.names
        result = await registry.execute(
            ToolCall(id="1", name="mcp__demo__add", arguments={"a": 2, "b": 3})
        )
        assert not result.is_error
        assert "5" in result.content
        result = await registry.execute(ToolCall(id="2", name="mcp__demo__identity"))
        assert '"secret": null' in result.content
        assert '"explicit": "passed"' in result.content
        result = await registry.execute(ToolCall(id="3", name="mcp__demo__fail"))
        assert result.is_error
        assert connection.state == "connected"
        result = await registry.execute(ToolCall(id="4", name="mcp__demo__large"))
        assert len(result.content) <= connection.config.max_result_chars
        assert "截断" in result.content
        bad = await registry.execute(
            ToolCall(id="5", name="mcp__demo__add", arguments={"a": True, "b": 3})
        )
        assert bad.is_error
        with pytest.raises(MCPError, match="白名单"):
            await connection.call("unlisted", {})
    finally:
        await connection.close()
    assert registry.names == ()
    assert connection._task is None


@pytest.mark.asyncio
async def test_call_cancel_disconnects_without_replay(tmp_path):
    registry = ToolRegistry()
    connection = MCPConnection("demo", settings(tmp_path), registry)
    try:
        await connection.connect()
        assert connection.state == "connected", connection.error
        task = asyncio.create_task(connection.call("slow", {}))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert connection.state == "disconnected"
        assert not registry.names
        await connection.connect()
        assert "3" in await connection.call("add", {"a": 1, "b": 2})
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_timeout_reports_unknown_and_removes_tools(tmp_path):
    registry = ToolRegistry()
    connection = MCPConnection("demo", settings(tmp_path, call_timeout=0.15), registry)
    try:
        await connection.connect()
        assert connection.state == "connected", connection.error
        executor = ToolExecutor(registry, ToolExecutionConfig(timeout_seconds=3))
        result = await executor.execute(
            ModelToolCall(id="slow", name="mcp__demo__slow", arguments="{}"),
            uuid4(),
            2000,
        )
        assert result.is_error
        assert result.execution_state == "unknown"
        assert result.error_kind == "timeout"
        assert not registry.names
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_write_requires_both_local_and_executor_authorization(tmp_path):
    registry = ToolRegistry()
    connection = MCPConnection("demo", settings(tmp_path, allow_write=True), registry)
    try:
        await connection.connect()
        assert connection.state == "connected", connection.error
        call = ModelToolCall(id="write", name="mcp__demo__change", arguments="{}")
        denied = await ToolExecutor(registry, ToolExecutionConfig()).execute(
            call, uuid4(), 2000
        )
        assert denied.is_error and denied.execution_state == "not_executed"
        allowed = await ToolExecutor(
            registry, ToolExecutionConfig(allow_write_tools=True)
        ).execute(call, uuid4(), 2000)
        assert not allowed.is_error
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_start_failure_does_not_disable_other_tools(tmp_path):
    from lynx_agent.tools import CalculatorTool

    registry = ToolRegistry([CalculatorTool()])
    manager = MCPManager(
        MCPConfig(
            servers={
                "missing": settings(tmp_path, command=str(tmp_path / "missing.exe")),
                "off": settings(tmp_path, enabled=False),
            }
        ),
        registry,
    )
    await manager.start()
    try:
        assert registry.names == ("calculator",)
        assert manager.connections["missing"].state == "error"
        assert "disabled" in await manager.command("")
        with pytest.raises(MCPError):
            await manager.command("connect off")
        with pytest.raises(MCPError):
            await manager.command("connect invented")
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_crashed_server_removed_and_manual_reconnect(tmp_path):
    registry = ToolRegistry()
    connection = MCPConnection("demo", settings(tmp_path), registry)
    try:
        await connection.connect()
        assert connection.state == "connected", connection.error
        with pytest.raises(MCPError):
            await connection.call("crash", {})
        assert not registry.names
        await connection.connect()
        assert connection.state == "connected", connection.error
    finally:
        await connection.close()


def test_schema_nested_enum_bounds_and_external_references():
    validator = schema_validator(
        {
            "type": "object",
            "required": ["item"],
            "properties": {
                "item": {
                    "type": "object",
                    "required": ["kind"],
                    "properties": {
                        "kind": {"enum": ["a", "b"]},
                        "n": {"type": "integer", "minimum": 1},
                    },
                }
            },
        }
    )
    validate_arguments(validator, {"item": {"kind": "a", "n": 1}})
    for args in (
        {"item": {"kind": "c"}},
        {"item": {"kind": "a", "n": 0}},
        {"item": {}},
    ):
        with pytest.raises(ToolValidationError):
            validate_arguments(validator, args)
    with pytest.raises(ValueError):
        schema_validator({"type": "object", "$ref": "https://example.invalid/schema"})
    with pytest.raises(ValueError):
        tool_name("demo", "invalid.name")
    with pytest.raises(ValueError):
        tool_name("demo", "a" * 64)


def test_results_errors_structured_and_unsupported():
    result = types.CallToolResult(content=[], structured_content={"ok": True})
    assert json.loads(result_text(result, 128)) == {"ok": True}
    with pytest.raises(MCPError):
        result_text(types.CallToolResult(content=[], is_error=True), 128)
    with pytest.raises(MCPError):
        result_text(
            types.CallToolResult(
                content=[
                    types.ImageContent(type="image", data="", mime_type="image/png")
                ]
            ),
            128,
        )


def test_config_validation_and_secret_representation(tmp_path):
    with pytest.raises(ValidationError):
        settings(tmp_path, command="python")
    with pytest.raises(ValidationError):
        settings(tmp_path, read_only_tools=["not-allowed"])
    with pytest.raises(ValidationError):
        MCPConfig(servers={"bad.name": settings(tmp_path)})
    assert "top-secret" not in repr(settings(tmp_path, env={"TOKEN": "top-secret"}))


@pytest.mark.asyncio
async def test_startup_timeout_and_cancel_cleanup(tmp_path):
    registry = ToolRegistry()
    connection = MCPConnection(
        "hung",
        settings(
            tmp_path,
            args=["-c", "import time; time.sleep(30)"],
            startup_timeout=0.2,
        ),
        registry,
    )
    await connection.connect()
    assert connection.state == "error" and "超时" in connection.error
    assert connection._task is None and not registry.names
    await connection.close()

    connection = MCPConnection(
        "cancel",
        settings(
            tmp_path,
            args=["-c", "import time; time.sleep(30)"],
        ),
        registry,
    )
    task = asyncio.create_task(connection.connect())
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert connection._task is None and not registry.names


@pytest.mark.asyncio
async def test_discovery_rejects_bad_schema_and_names(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    registry = ToolRegistry()
    connection = MCPConnection(
        "demo",
        settings(
            tmp_path,
            allowed_tools=["add", "bad.name", "external"],
            read_only_tools=["add"],
        ),
        registry,
    )
    definitions = [
        types.Tool(name="add", input_schema={"type": "object"}),
        types.Tool(name="bad.name", input_schema={"type": "object"}),
        types.Tool(
            name="external",
            input_schema={"type": "object", "$ref": "https://example.invalid"},
        ),
        types.Tool(name="not_allowed", input_schema={"type": "object"}),
    ]
    client = SimpleNamespace(
        list_tools=AsyncMock(return_value=types.ListToolsResult(tools=definitions))
    )
    tools = await connection._discover(client)
    assert [tool.name for tool in tools] == ["mcp__demo__add"]
    assert connection.rejected == 2


@pytest.mark.asyncio
async def test_discovery_repeated_cursor_is_bounded(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    connection = MCPConnection("demo", settings(tmp_path), ToolRegistry())
    client = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=types.ListToolsResult(tools=[], next_cursor="again")
        )
    )
    with pytest.raises(MCPError, match="游标"):
        await connection._discover(client)
    assert client.list_tools.await_count == 2
