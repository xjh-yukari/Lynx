import json
import os
import re
from pathlib import Path

from document_rag.config import digest, grant


class Store:
    def __init__(self, settings):
        # gRPC会读取HTTPS_PROXY；本机数据库不能经系统代理转发。
        bypass = os.environ.get("no_grpc_proxy", "")
        os.environ["no_grpc_proxy"] = ",".join(filter(None, (bypass, "127.0.0.1", "localhost", "::1")))
        from pymilvus import MilvusClient
        self.settings = settings
        self.client = MilvusClient(uri=settings.uri, token=os.environ.get("RAG_MILVUS_TOKEN", ""), timeout=20)

    def close(self):
        self.client.close()

    def manifest_path(self, physical):
        if not re.fullmatch(r"lynx_rag_[a-f0-9]{24}_[a-f0-9]{8}", physical):
            raise PermissionError("拒绝非本服务collection")
        return self.settings.state_dir / (physical + ".json")

    def manifest(self, physical, *, compatible=True):
        data = json.loads(self.manifest_path(physical).read_text(encoding="utf-8"))
        if (data["project_id"] != self.settings.project_id or data["status"] != "ready"
                or compatible and data["encoding"] != self.settings.manifest()):
            raise ValueError("知识库版本、项目或编码清单不匹配")
        return data

    def resolve(self, kb_id):
        entry = grant(self.settings, kb_id)
        physical = self.client.describe_alias(alias=entry["alias"])["collection_name"]
        data = self.manifest(physical)
        if data["kb_id"] != kb_id:
            raise PermissionError("knowledge scope mismatch")
        state = self.client.get_load_state(collection_name=physical)["state"]
        if getattr(state, "name", str(state)) != "Loaded":
            raise RuntimeError("知识库未加载；由离线发布阶段加载，在线禁止自动加载")
        return physical, data

    def create(self, physical):
        from pymilvus import DataType
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        for name, length in {"doc_id": 64, "content_hash": 64, "text": 16000, "source": 2048, "title": 1024,
                             "section": 2048, "asset_path": 2048, "record_id": 512,
                             "kb_id": 32, "project_id": 128, "index_version": 64, "encoding_hash": 64}.items():
            schema.add_field(name, DataType.VARCHAR, max_length=length)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("page", DataType.INT64)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        indexes = self.client.prepare_index_params()
        indexes.add_index("dense_vector", index_type="AUTOINDEX", metric_type="IP")
        indexes.add_index("sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        self.client.create_collection(collection_name=physical, schema=schema, index_params=indexes, consistency_level="Strong")

    def publish(self, kb_id, physical):
        data = self.manifest(physical)
        if data["kb_id"] != kb_id:
            raise PermissionError("知识库不匹配")
        alias = grant(self.settings, kb_id)["alias"]
        # 只有本项目本知识库的旧alias才可替换。
        aliases = {name: collection for collection in self.client.list_collections()
                   for name in self.alias_names(collection)}
        previous = aliases.get(alias)
        if previous is None:
            self.client.create_alias(collection_name=physical, alias=alias)
        else:
            if self.manifest(previous, compatible=False)["kb_id"] != kb_id:
                raise PermissionError("alias归属不明")
            self.client.alter_alias(collection_name=physical, alias=alias)

    def alias_names(self, physical):
        # 2.5.11的类型注解写List[str]，实际公开API返回包含aliases的dict。
        response = self.client.list_aliases(collection_name=physical)
        if not isinstance(response, dict) or response.get("collection_name") != physical or not isinstance(response.get("aliases"), list):
            raise ValueError("Milvus alias响应结构不可验证")
        return tuple(response["aliases"])

    def search(self, kb_id, dense, sparse, mode, top_k):
        from pymilvus import AnnSearchRequest, RRFRanker
        physical, manifest = self.resolve(kb_id)  # 只解析一次；后续全部使用同一物理库。
        version = manifest["index_version"]
        expr = (f"kb_id == {json.dumps(kb_id)} and project_id == {json.dumps(self.settings.project_id)} "
                f"and index_version == {json.dumps(version)}")
        fields = ["chunk_id", "text", "source", "title", "section", "page", "kb_id", "project_id", "index_version", "asset_path", "record_id"]
        params = {"metric_type": "IP", "params": {}}
        if mode == "hybrid" and sparse:
            requests = [AnnSearchRequest([vector], field, params, limit=self.settings.candidates, expr=expr)
                        for vector, field in ((dense, "dense_vector"), (sparse, "sparse_vector"))]
            result = self.client.hybrid_search(collection_name=physical, reqs=requests, ranker=RRFRanker(k=60),
                                              limit=min(self.settings.candidates, 100), output_fields=fields, timeout=20)[0]
            score_type = "rrf"
        else:
            use_sparse = mode == "sparse" and bool(sparse)
            result = self.client.search(collection_name=physical, data=[sparse if use_sparse else dense],
                                        anns_field="sparse_vector" if use_sparse else "dense_vector",
                                        search_params=params, filter=expr, limit=self.settings.candidates,
                                        output_fields=fields, timeout=20)[0]
            score_type = "ip"
        grant(self.settings, kb_id)  # 查询期间撤权也必须生效。
        hits, seen, doc_indices = [], set(), {}
        for hit in result:
            row = hit["entity"]
            if row["kb_id"] != kb_id or row["project_id"] != self.settings.project_id or row["index_version"] != version:
                raise PermissionError("数据库返回了越界记录，拒绝整批返回")
            signature = digest(row["text"].strip())
            # 去重；高度相似片段用字符集合重叠率做保守过滤。
            chars = {row["text"][index:index + 3] for index in range(max(1, len(row["text"]) - 2))}
            if signature in seen or any(len(chars & prior) / max(1, len(chars | prior)) > 0.9 for prior in doc_indices.get(row["source"], [])):
                continue
            seen.add(signature)
            doc_indices.setdefault(row["source"], []).append(chars)
            hits.append({key: row[key] for key in ("chunk_id", "text", "source", "title", "asset_path", "record_id")} | {
                "page": row["page"] or None, "section": row["section"] or None, "rank": len(hits) + 1,
                "score": float(hit["distance"]), "score_type": score_type})
            if len(hits) == top_k:
                break
        return {"status": "ok", "query": "", "kb_id": kb_id, "project_id": self.settings.project_id,
                "retrieval_mode": "hybrid" if score_type == "rrf" else "sparse" if mode == "sparse" and sparse else "dense",
                "index_version": version, "hits": hits}


class Lease:
    """离线建库与在线服务互斥；文件锁随进程退出自动释放。"""
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / "execution.lock").open("a+b")
        self.stream.seek(0)
        self.stream.write(b"0")
        self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("建库或查询服务正在使用同一state_dir，拒绝并行运行") from None

    def close(self):
        self.stream.close()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
