"""只操作 tests/fixtures/document_rag/settings.toml 的独立验证库，不连接课程的19530端口。

真实模型、真实数据库和真实HTTP MCP；不调用LLM/Engram，不冒充完整Agent联调。
"""
import asyncio
import json
import secrets
import time
from pathlib import Path

import httpx2
import uvicorn
from document_rag.build import build
from document_rag.config import load
from document_rag.encoding import Encoder, sparse_row
from document_rag.retrieval import Retriever
from document_rag.server import create_app
from document_rag.storage import Lease, Store, atomic_json
from mcp import Client
from mcp.client.streamable_http import streamable_http_client


async def run():
    directory = Path(__file__).resolve().parents[2]
    settings = load(directory / "tests" / "fixtures" / "document_rag" / "settings.toml")
    assert settings.uri == "http://127.0.0.1:19540" and settings.project_id == "rag_validation"
    lease, store = Lease(settings.state_dir), None
    retriever, server_task = None, None
    report = {"kind": "real_components_no_llm", "checks": {}}
    try:
        store = Store(settings)
        started = time.monotonic()
        encoder = Encoder(settings)
        report["model_load_seconds"] = round(time.monotonic() - started, 3)
        from scipy.sparse import csr_array, csr_matrix
        assert sparse_row(csr_array([[0.0, 0.5]]), 0) == {1: 0.5}
        assert sparse_row(csr_matrix([[0.0, 0.0]]), 0) == {}
        report["checks"]["scipy_matrix_array_empty"] = True
        manifest = build(settings, encoder, store, "default", directory / "tests" / "fixtures" / "document_rag" / "plan.json")
        again = build(settings, encoder, store, "default", directory / "tests" / "fixtures" / "document_rag" / "plan.json")
        assert again["physical"] == manifest["physical"]
        report["checks"]["idempotent_build"] = True
        report["physical"] = manifest["physical"]
        report["index_version"] = manifest["index_version"]
        retriever = Retriever(settings, encoder, store)
        query = "关闭程序后会话如何恢复？"
        dense, sparse = encoder.encode([query], query=True)[0]
        # 模拟极高相关的越权记录，dense/sparse/hybrid均应在召回前过滤。
        row = store.client.query(collection_name=manifest["physical"], filter="", limit=1,
                                 output_fields=["*"])[0]
        row.update(chunk_id="f" * 64, project_id="other_project", text="这是其他项目的机密，绝不能返回",
                   dense_vector=dense, sparse_vector=sparse)
        foreign_kb = row | {"chunk_id": "e" * 64, "kb_id": "other", "project_id": settings.project_id}
        store.client.upsert(collection_name=manifest["physical"], data=[row, foreign_kb])
        store.client.flush(collection_name=manifest["physical"])
        for mode in ("dense", "sparse", "hybrid"):
            result = store.search("default", dense, sparse, mode, 5)
            assert result["hits"] and all(hit["chunk_id"] not in {"f" * 64, "e" * 64} for hit in result["hits"])
        report["checks"]["real_pre_retrieval_project_filter"] = True
        empty = store.search("default", dense, {}, "hybrid", 5)
        assert empty["retrieval_mode"] == "dense"
        report["checks"]["empty_sparse_dense_fallback"] = True
        evaluations = []
        for question, source in [(query, "session.md"), ("仅查看权限能不能修改文件", "permissions.md"), ("长期记忆使用什么数据库", "memory.md")]:
            vector, lexical = encoder.encode([question], query=True)[0]
            for mode in ("dense", "hybrid"):
                start = time.monotonic()
                result = store.search("default", vector, lexical, mode, 5)
                evaluations.append({"mode": mode, "query": question, "recall_at_5": int(any(hit["source"].endswith(source) for hit in result["hits"])),
                                    "retrieval_seconds_excluding_embedding": round(time.monotonic() - start, 3)})
        report["evaluation"] = evaluations
        reduced_plan = settings.state_dir / "reduced-validation-plan.json"
        atomic_json(reduced_plan, json.loads((directory / "tests" / "fixtures" / "document_rag" / "plan.json").read_text(encoding="utf-8"))[:2])
        updated = build(settings, encoder, store, "default", reduced_plan)
        assert updated["index_version"] != manifest["index_version"]
        result = retriever.search("长期记忆规范")
        assert all(not hit["source"].endswith("memory.md") for hit in result["hits"])
        report["checks"]["full_rebuild_removes_deleted_documents"] = True
        store.publish("default", manifest["physical"])
        assert store.resolve("default")[0] == manifest["physical"]
        report["checks"]["real_alias_rollback"] = True
        # 在解析alias后模拟管理员切换alias，当前查询必须仍读已固定的物理版本。
        original_resolve = store.resolve
        def resolve_then_switch(kb_id):
            resolved = original_resolve(kb_id)
            store.publish(kb_id, updated["physical"])
            return resolved
        store.resolve = resolve_then_switch
        consistent = store.search("default", dense, sparse, "hybrid", 5)
        assert consistent["index_version"] == manifest["index_version"]
        store.resolve = original_resolve
        store.publish("default", manifest["physical"])
        report["checks"]["one_request_one_physical_version"] = True
        invalid_plan = settings.state_dir / "invalid-validation-plan.json"
        atomic_json(invalid_plan, [{"path": str(settings.state_dir / "missing.pdf")}])
        try:
            build(settings, encoder, store, "default", invalid_plan)
        except ValueError:
            pass
        else:
            raise AssertionError("失败建库不应成功")
        assert store.resolve("default")[0] == manifest["physical"]
        report["checks"]["failed_build_preserves_alias"] = True
        token = secrets.token_urlsafe(32)
        server = uvicorn.Server(uvicorn.Config(create_app(settings, retriever, token), host="127.0.0.1", port=settings.port,
                                               log_level="critical", access_log=False))
        server_task = asyncio.create_task(server.serve())
        async with asyncio.timeout(20):
            while not server.started:
                if server_task.done():
                    await server_task
                await asyncio.sleep(0.01)
        for _ in range(2):
            async with (
                httpx2.AsyncClient(headers={"Authorization": "Bearer " + token}, trust_env=False) as http,
                Client(streamable_http_client(f"http://127.0.0.1:{settings.port}/mcp", http_client=http), read_timeout_seconds=120) as client,
            ):
                tools = await client.list_tools()
                assert [tool.name for tool in tools.tools] == ["search_knowledge"]
                result = await client.call_tool("search_knowledge", {"query": query, "kb_id": "default"})
                assert not result.is_error and result.structured_content["hits"]
        assert retriever.encoder is encoder
        report["checks"]["two_real_http_mcp_clients_one_encoder"] = True
        report["checks"]["only_read_tool_exposed"] = True
        server.should_exit = True
        await asyncio.wait_for(server_task, 20)
        server_task = None
        assert store.resolve("default")[0] == manifest["physical"]
        report["checks"]["shutdown_does_not_release_or_drop_collection"] = True
        report["note"] = "仅三道人工问题；不是通用检索质量结论，未测试付费模型路由或真实Engram写入"
        atomic_json(directory / "tests" / "reports" / "rag-last-real-report.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        if server_task:
            server.should_exit = True
            await asyncio.wait_for(server_task, 20)
        if retriever:
            retriever.close()
        if store:
            store.close()
        lease.close()


if __name__ == "__main__":
    asyncio.run(run())
