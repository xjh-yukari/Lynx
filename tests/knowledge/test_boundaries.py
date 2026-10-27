import asyncio
import json

import pytest
from document_rag.build import build
from document_rag.config import Settings
from document_rag.documents import chunks
from document_rag.retrieval import Retriever
from document_rag.storage import Lease, Store
from mcp import types

from lynx_agent.knowledge.access import KnowledgeAccess
from lynx_agent.knowledge.models import Evidence, KnowledgeConfig
from lynx_agent.mcp.config import MCPServerConfig
from lynx_agent.mcp.errors import MCPError
from lynx_agent.memory.models import MARKER, MemoryRecord


def configuration(tmp_path):
    policy = tmp_path / "grants.json"
    policy.write_text(json.dumps({"project_id": "test", "grants": {"default": {"enabled": True, "alias": "lynx_kb_test"}}}), encoding="utf-8")
    return Settings(policy_file=policy, state_dir=tmp_path, project_id="test", model_revision="a" * 40, chunk_tokens=64, overlap_tokens=8)


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"offset_mapping": [(index, index + 1) for index in range(len(text))]}
    def encode(self, text, **kwargs):
        return list(text)

    def decode(self, tokens, **kwargs):
        return "".join(tokens)


class Encoder:
    tokenizer = Tokenizer()
    tokens = staticmethod(len)
    fingerprint = "a" * 64


def test_http_config_is_local_authenticated():
    valid = {"transport": "streamable_http", "url": "http://127.0.0.1:8766/mcp", "token_env": "RAG_MCP_TOKEN"}
    assert MCPServerConfig(**valid).transport == "streamable_http"
    for url in ("http://0.0.0.0:8766/mcp", "https://example.com/mcp", "http://127.0.0.1/mcp?secret=x"):
        with pytest.raises(ValueError):
            MCPServerConfig(**(valid | {"url": url}))


def test_http_file_auth_is_explicit_and_exclusive(tmp_path):
    common = {"transport": "streamable_http", "url": "http://127.0.0.1:8766/mcp"}
    assert MCPServerConfig(**common, token_file=str(tmp_path / "token.key")).token_env is None
    with pytest.raises(ValueError):
        MCPServerConfig(**common, token_file="relative.key")
    with pytest.raises(ValueError):
        MCPServerConfig(**common, token_file=str(tmp_path / "token.key"), token_env="TEST_TOKEN")


def test_grants_revoke_and_result_validation(tmp_path):
    cfg = configuration(tmp_path)
    access = KnowledgeAccess(KnowledgeConfig(enabled=True, policy_file=str(cfg.policy_file)), "test")
    evidence = Evidence(project_id="test", kb_id="default", index_version="v1", chunk_id="a" * 64, source="doc.md")
    access.check_sources((evidence,))
    body = {"status": "ok", "kb_id": "default", "project_id": "test", "index_version": "v1", "hits": [
        {"chunk_id": "a" * 64, "source": "doc.md", "page": None, "section": None, "text": "证据"}]}
    result = types.CallToolResult(content=[], structured_content=body)
    assert "证据" in access.accept(result, {})
    result.structured_content = body | {"project_id": "other"}
    with pytest.raises(MCPError):
        access.accept(result, {})
    cfg.policy_file.write_text('{"project_id":"test","grants":{}}', encoding="utf-8")
    with pytest.raises(MCPError):
        access.check_sources((evidence,))


def test_memory_serialization_preserves_old_hash_and_new_sources():
    record = MemoryRecord(kind="decision", owner="owner", topic="topic", title="title", body="body")
    old = MARKER + record.model_dump_json(exclude={"sources"})
    assert record.encode() == old
    assert MemoryRecord.model_validate_json(old.removeprefix(MARKER)).encode() == old
    evidence = Evidence(project_id="test", kb_id="default", index_version="v1", chunk_id="a" * 64, source="doc.md")
    with_sources = record.model_copy(update={"sources": (evidence,)})
    assert MemoryRecord.model_validate_json(with_sources.encode().removeprefix(MARKER)).sources == (evidence,)


def test_chunk_ids_json_mapping_empty_and_utf8(tmp_path):
    cfg = configuration(tmp_path)
    path = tmp_path / "test.md"
    path.write_text("# 章节\n\n" + "中文测试" * 100, encoding="utf-8")
    first, _ = chunks(path, cfg, Encoder())
    second, _ = chunks(path, cfg, Encoder())
    assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]
    assert all(len(row["text"]) <= 64 and row["section"] == "章节" for row in first)
    assert all(row["text"] in path.read_text(encoding="utf-8") for row in first)
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="无可提取正文"):
        chunks(path, cfg, Encoder())
    data = tmp_path / "records.json"
    data.write_text('[{"id":"dragon1","title":"白龙","description":"悬崖"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="显式"):
        chunks(data, cfg, Encoder())
    rows, _ = chunks(data, cfg, Encoder(), {"id": "id", "text": ["title", "description"], "title": "title"})
    assert rows[0]["section"] == "record:dragon1"
    assert rows[0]["source"].endswith("records.json")


def test_build_failure_never_creates_collection(tmp_path):
    cfg = configuration(tmp_path)
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"path": str(empty)}]), encoding="utf-8")
    with pytest.raises(ValueError, match="未创建/发布"):
        build(cfg, Encoder(), object(), "default", plan)
    assert (tmp_path / "last_build_failure.json").exists()


def test_offline_online_lease_is_exclusive(tmp_path):
    lease = Lease(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="拒绝并行"):
            Lease(tmp_path)
    finally:
        lease.close()
    Lease(tmp_path).close()


def test_real_version_alias_response_shape_is_respected(tmp_path):
    from unittest.mock import Mock
    settings = configuration(tmp_path)
    store = object.__new__(Store)
    store.settings = settings
    store.manifest = lambda physical, **kwargs: {"kb_id": "default"}
    store.client = Mock()
    store.client.list_collections.return_value = ["old", "new"]
    store.client.list_aliases.side_effect = lambda collection_name: {"collection_name": collection_name,
        "aliases": ["lynx_kb_test"] if collection_name == "old" else []}
    store.publish("default", "new")
    store.client.alter_alias.assert_called_once_with(collection_name="new", alias="lynx_kb_test")
    store.client.create_alias.assert_not_called()


def test_old_encoding_can_be_owned_without_being_query_compatible(tmp_path):
    settings = configuration(tmp_path)
    store = object.__new__(Store)
    store.settings = settings
    physical = "lynx_rag_" + "a" * 24 + "_" + "b" * 8
    store.manifest_path(physical).write_text(json.dumps({"project_id": "test", "status": "ready", "encoding": {"old": True}}), encoding="utf-8")
    with pytest.raises(ValueError, match="编码清单"):
        store.manifest(physical)
    assert store.manifest(physical, compatible=False)["project_id"] == "test"


def test_full_file_permission_cannot_rewrite_knowledge_grants(tmp_path):
    from unittest.mock import AsyncMock

    from lynx_agent.agent import AgentRunner
    from lynx_agent.config import Config
    from lynx_agent.filesystem import FileAccess
    from lynx_agent.providers import ModelProvider
    from lynx_agent.tools.errors import ToolExecutionError
    from lynx_agent.web.setup import create_default_tools

    policy = tmp_path / "grants.json"
    cfg = Config.model_validate({"providers": {"deepseek": {"api_key": "fake", "base_url": "https://example.invalid"}}, "agent": {"model": {"name": "fake"}},
        "knowledge": {"enabled": True, "policy_file": str(policy)}, "mcp": {"servers": {"documents": {
        "enabled": True, "transport": "streamable_http", "url": "http://127.0.0.1:8766/mcp", "token_env": "TEST_TOKEN",
        "allowed_tools": ["search_knowledge"], "read_only_tools": ["search_knowledge"]}}}})
    files = FileAccess(tmp_path)
    AgentRunner(cfg, AsyncMock(spec=ModelProvider), tools=create_default_tools(cfg, file_access=files))
    files.set_mode("full")
    with pytest.raises(ToolExecutionError, match="受保护配置"):
        files.policy.resolve(str(policy), write=True)
    with pytest.raises(ToolExecutionError, match="受保护配置"):
        files.policy.resolve(str(policy.with_name("mcp_token.key")))


@pytest.mark.asyncio
async def test_timeout_keeps_capacity_until_actual_inference_finishes(tmp_path):
    import threading
    cfg = configuration(tmp_path).model_copy(update={"queue_size": 1, "timeout": 0.02})
    release = threading.Event()
    retriever = Retriever(cfg, None, None)
    retriever.search = lambda *args: release.wait(2)
    try:
        with pytest.raises(TimeoutError):
            await retriever.call("test")
        assert retriever.slots == 1
        with pytest.raises(RuntimeError, match="queue full"):
            await retriever.call("test")
        release.set()
        for _ in range(100):
            await asyncio.sleep(0.001)
            if retriever.slots == 0:
                break
        assert retriever.slots == 0
    finally:
        release.set()
        retriever.close()
