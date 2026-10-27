import asyncio
import socket
from types import SimpleNamespace

import httpx2
import pytest
import uvicorn
from document_rag.server import create_app
from test_boundaries import configuration

from lynx_agent.knowledge.access import KnowledgeAccess
from lynx_agent.knowledge.models import KnowledgeConfig
from lynx_agent.mcp.client import MCPConnection
from lynx_agent.mcp.config import MCPServerConfig
from lynx_agent.tools import ToolRegistry


@pytest.mark.asyncio
@pytest.mark.parametrize("authentication", ["env", "file"])
async def test_real_http_mcp_discovery_auth_origin_and_structured_result(tmp_path, monkeypatch, authentication):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = configuration(tmp_path).model_copy(update={"port": port})
    calls = []

    async def call(query, kb_id, top_k):
        calls.append(query)
        return {"status": "ok", "kb_id": kb_id, "project_id": "test", "index_version": "v1", "hits": []}

    token = "test-token-" * 5
    monkeypatch.setenv("TEST_RAG_TOKEN", token)
    key_path = tmp_path / "token.key"
    key_path.write_text(token, encoding="ascii")
    credentials = {"token_env": "TEST_RAG_TOKEN"} if authentication == "env" else {"token_file": str(key_path)}
    if authentication == "file":
        monkeypatch.delenv("TEST_RAG_TOKEN", raising=False)
    app = create_app(settings, SimpleNamespace(call=call), token)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="critical", access_log=False))
    task = asyncio.create_task(server.serve())
    registry = ToolRegistry()
    connection = MCPConnection("documents", MCPServerConfig(
        enabled=True, transport="streamable_http", url=f"http://127.0.0.1:{port}/mcp", **credentials,
        allowed_tools=("search_knowledge",), read_only_tools=("search_knowledge",)), registry)
    connection.knowledge = KnowledgeAccess(KnowledgeConfig(enabled=True, policy_file=str(settings.policy_file)), "test")
    connection._heartbeat_interval = 0.05
    try:
        async with asyncio.timeout(10):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        async with httpx2.AsyncClient(trust_env=False) as http:
            url = f"http://127.0.0.1:{port}/mcp"
            assert (await http.post(url, json={})).status_code == 401
            response = await http.post(url, headers={"Authorization": "Bearer " + token, "Origin": "https://evil.example"}, json={})
            assert response.status_code == 403
        await connection.connect()
        assert connection.state == "connected", connection.error
        # 必须跨过保活周期；旧send_ping实现会在这里掉线并注销工具。
        await asyncio.sleep(0.2)
        assert connection.state == "connected", connection.error
        assert registry.names == ("mcp__documents__search_knowledge",) or set(registry.names) == {"mcp__documents__search_knowledge"}
        tool = registry.get("mcp__documents__search_knowledge")
        assert "kb_id" in tool.parameters["required"]
        result = await tool.execute(query="文档问题", kb_id="default")
        assert '"hits": []' in result
        assert calls == ["文档问题"]
    finally:
        await connection.close()
        server.should_exit = True
        await asyncio.wait_for(task, 10)
