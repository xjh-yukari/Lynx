"""显式启用的真实本地链路；仅LLM为脚本替身，不调用付费API。"""
import asyncio
import json
import os
import secrets
import socket
import subprocess
from pathlib import Path

import httpx2
import pytest
from document_rag.config import load

from lynx_agent.agent import AgentRunner
from lynx_agent.config import Config
from lynx_agent.memory.errors import MemoryError
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionStore
from lynx_agent.tools import ToolCall

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / "services" / "document_rag"
FIXTURES = ROOT / "tests" / "fixtures" / "document_rag"
BINARY = Path("E:/github/lynx_agent_data/bin/engram/3.4.0/engram.exe")
pytestmark = pytest.mark.skipif(os.environ.get("LYNX_RAG_REAL") != "1" or not BINARY.is_file(),
                               reason="显式设置LYNX_RAG_REAL=1，要求独立验证库已建好及Engram已安装")


class ScriptProvider(ModelProvider):
    def __init__(self):
        self.replies = []

    async def generate(self, request):
        return self.replies.pop(0)

    async def aclose(self):
        pass


def answer(content):
    return ModelResponse(content=content, model="scripted", finish_reason="stop")


def tools(*calls):
    return ModelResponse(content="", model="scripted", finish_reason="tool_calls", tool_calls=calls)


@pytest.mark.asyncio
async def test_real_rag_agent_engram_provenance_and_revoke(tmp_path, monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    policy = tmp_path / "grants.json"
    original = FIXTURES / "grants.json"
    policy.write_bytes(original.read_bytes())
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text((FIXTURES / "settings.toml").read_text(encoding="utf-8")
                             .replace(str(original).replace("/", "\\"), str(policy)).replace("port = 8767", f"port = {port}"), encoding="utf-8")
    assert load(settings_path).policy_file == policy
    token = secrets.token_urlsafe(32)
    monkeypatch.setenv("TEST_RAG_REAL_TOKEN", token)
    environment = os.environ | {"RAG_MCP_TOKEN": token, "PYTHONUTF8": "1"}
    process = await asyncio.create_subprocess_exec(str(SERVICE / ".venv" / "Scripts" / "python.exe"), "-m", "document_rag.server",
        "--config", str(settings_path), env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    url = f"http://127.0.0.1:{port}/mcp"
    provider = ScriptProvider()
    config = Config.model_validate({"providers": {"deepseek": {"api_key": "fake", "base_url": "https://example.invalid"}},
        "agent": {"model": {"name": "scripted"}},
        "memory": {"enabled": True, "executable": str(BINARY), "data_dir": str(tmp_path / "memory"), "project_id": "rag_validation", "auto_save": True},
        "knowledge": {"enabled": True, "policy_file": str(policy)},
        "mcp": {"servers": {"documents": {"enabled": True, "transport": "streamable_http", "url": url,
            "token_env": "TEST_RAG_REAL_TOKEN", "allowed_tools": ["search_knowledge"], "read_only_tools": ["search_knowledge"], "call_timeout": 120}}}})
    try:
        async with asyncio.timeout(100), httpx2.AsyncClient(trust_env=False, timeout=1) as http:
            while True:
                if process.returncode is not None:
                    raise AssertionError("真实RAG服务启动失败，请先运行validate_real.py")
                try:
                    if (await http.post(url, json={})).status_code == 401:
                        break
                except httpx2.HTTPError:
                    pass
                await asyncio.sleep(0.2)
        async with AgentRunner(config, provider, SessionStore(tmp_path / "sessions")) as runner:
            name = "mcp__documents__search_knowledge"
            assert name in runner.tools.names
            runner.memory.auto_save_enabled = False
            provider.replies = [answer("你好。")]
            await runner.ask("你好")  # 无工具路径
            assert not runner._archive.records
            runner.memory.auto_save_enabled = True
            provider.replies = [tools(ModelToolCall(id="r1", name=name, arguments='{"query":"会话保存格式","kb_id":"default"}')),
                                answer("会话采用JSONL保存。"), answer('{"conclusions":[{"title":"会话保存格式","body":"会话采用JSONL保存。"}]}')]
            await runner.ask("查文档确认会话保存格式")  # RAG路径＋真实Engram自动写入
            records = await runner.memory.list_records(include_body=True)
            assert len(records["records"]) == 1 and records["records"][0]["sources"]
            key = records["records"][0]["key"]
            record = await runner.memory.get(key)
            assert all(source.project_id == "rag_validation" and source.index_version for source in record.sources)
            runner.memory.auto_save_enabled = False
            provider.replies = [tools(ModelToolCall(id="m1", name="memory_get", arguments=json.dumps({"key": key}))), answer("历史结论是JSONL。")]
            await runner.ask("此前保存的结论是什么")  # 仅Engram路径
            provider.replies = [tools(ModelToolCall(id="m2", name="memory_get", arguments=json.dumps({"key": key})),
                                     ModelToolCall(id="r2", name=name, arguments='{"query":"会话保存格式","kb_id":"default"}')),
                                answer("历史结论和文档一致。")]
            await runner.ask("把历史结论与文档核对")  # 两者路径；仍复用现有顺序工具循环
            trace = next(trace for trace in runner._archive.records.values() if trace.call.name == name)
            assert runner._archive.read(trace.result_id)
            policy.write_text('{"project_id":"rag_validation","grants":{}}', encoding="utf-8")
            denied = await runner.tools.execute(ToolCall(id="denied", name=name, arguments={"query": "会话", "kb_id": "default"}))
            assert denied.is_error and denied.error_kind == "permission_denied"
            with pytest.raises(Exception, match="授权"):
                runner._archive.read(trace.result_id)
            with pytest.raises(MemoryError, match="撤权"):
                await runner.memory.get(key)
            assert await runner.memory.search("会话") == []
    finally:
        if process.returncode is None:
            process.terminate()
            await asyncio.wait_for(process.wait(), 20)
