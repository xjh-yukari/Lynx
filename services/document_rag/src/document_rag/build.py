import json
from pathlib import Path
from uuid import uuid4

from document_rag.config import digest, grant
from document_rag.documents import chunks
from document_rag.storage import atomic_json


def build(settings, encoder, store, kb_id, plan_path, excludes=()):
    grant(settings, kb_id)
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if not isinstance(plan, list) or not plan:
        raise ValueError("导入计划必须为非空文件列表，不扫描目录")
    excluded = {str(Path(path).resolve()) for path in excludes}
    entries, hashes, failures, used_excludes = [], [], [], set()
    for entry in plan:
        path = Path(entry["path"])
        if not path.is_absolute():
            raise ValueError("导入路径必须为绝对路径")
        if str(path.resolve()) in excluded:
            used_excludes.add(str(path.resolve()))
            continue
        try:
            rows, file_hash = chunks(path, settings, encoder, entry.get("fields"))
            entries.extend(rows)
            hashes.append({"path": str(path.resolve()), "hash": file_hash, "fields": entry.get("fields")})
        except Exception as error:  # noqa: BLE001 - 解析错误只记入失败清单，禁止发布。
            failures.append({"path": str(path), "error": str(error)[:300]})
    if excluded != used_excludes:
        raise ValueError("排除项不属于显式导入计划")
    if failures or not entries:
        atomic_json(settings.state_dir / "last_build_failure.json", {"failures": failures, "chunks": len(entries), "excluded": sorted(excluded)})
        raise ValueError("有导入失败或空库，未创建/发布新库；查看last_build_failure.json并显式--exclude后重试")
    # 相同内容/配置/映射/排除清单获得同一逻辑版本。
    version = digest({"project": settings.project_id, "kb": kb_id, "files": sorted(hashes, key=lambda item: item["path"]),
                      "encoding": settings.manifest(), "excluded": sorted(excluded)})
    try:
        _, current = store.resolve(kb_id)
    except Exception:  # noqa: BLE001 - 新库没有alias；真正发布前仍会验证连接和旧alias归属。
        current = None
    if current and current["index_version"] == version:
        return current
    physical = "lynx_rag_" + version[:24] + "_" + uuid4().hex[:8]
    manifest = {"status": "building", "physical": physical, "index_version": version, "kb_id": kb_id,
                "project_id": settings.project_id, "encoding": settings.manifest(), "files": hashes,
                "excluded": sorted(excluded), "chunks": len(entries)}
    atomic_json(store.manifest_path(physical), manifest)
    store.create(physical)
    for start in range(0, len(entries), settings.batch_size):
        batch = entries[start:start + settings.batch_size]
        vectors = encoder.encode([row["text"] for row in batch])
        data = [row | {"dense_vector": dense, "sparse_vector": sparse, "kb_id": kb_id, "project_id": settings.project_id,
                       "index_version": version, "encoding_hash": encoder.fingerprint}
                for row, (dense, sparse) in zip(batch, vectors, strict=True)]
        store.client.insert(collection_name=physical, data=data)
    store.client.flush(collection_name=physical, timeout=120)
    store.client.load_collection(collection_name=physical, timeout=120)
    count = store.client.query(collection_name=physical, filter="", output_fields=["count(*)"], consistency_level="Strong")[0]["count(*)"]
    if count != len({row["chunk_id"] for row in entries}):
        raise ValueError("落库计数不匹配，旧alias未改变")
    dense, _ = encoder.encode([entries[0]["text"]], query=True)[0]
    probe = store.client.search(collection_name=physical, data=[dense], anns_field="dense_vector", limit=1,
                                search_params={"metric_type": "IP", "params": {}}, output_fields=["kb_id", "project_id"])
    if not probe[0] or probe[0][0]["entity"]["project_id"] != settings.project_id:
        raise ValueError("验证查询失败")
    manifest["status"] = "ready"
    atomic_json(store.manifest_path(physical), manifest)
    store.publish(kb_id, physical)
    return manifest
