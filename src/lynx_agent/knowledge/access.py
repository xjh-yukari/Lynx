import json
from pathlib import Path

from lynx_agent.knowledge.models import Evidence
from lynx_agent.mcp.errors import MCPError


class KnowledgeAccess:
    def __init__(self, config, project_id):
        self.config, self.project_id = config, project_id

    def grant(self, kb_id):
        try:
            path = Path(self.config.policy_file)
            if not self.config.enabled or not path.is_absolute() or path.stat().st_size > 65536:
                raise ValueError
            data = json.loads(path.read_text(encoding="utf-8"))
            entry = data["grants"][kb_id]
            if data["project_id"] != self.project_id or entry["enabled"] is not True:
                raise ValueError
            return entry
        except (OSError, KeyError, TypeError, ValueError):
            raise MCPError("知识库未授权或授权配置不可验证", error_kind="permission_denied") from None

    def check_sources(self, sources):
        for source in sources:
            self.grant(source.kb_id)
            if source.project_id != self.project_id:
                raise MCPError("知识库项目越界", error_kind="permission_denied")

    def accept(self, result, arguments):
        if getattr(result, "result_type", "complete") != "complete":
            raise MCPError("文档查询尚未完整结束，未作为检索成功", error_kind="invalid_result")
        kb_id = arguments.get("kb_id", "default")
        self.grant(kb_id)
        if result.is_error:
            from lynx_agent.mcp.results import result_text
            return result_text(result, 16000)
        try:
            data = result.structured_content
            if not isinstance(data, dict) or data["status"] != "ok" or data["kb_id"] != kb_id or data["project_id"] != self.project_id:
                raise ValueError
            if not isinstance(data["hits"], list) or len(data["hits"]) > 10:
                raise ValueError
            for hit in data["hits"]:
                source = Evidence(project_id=data["project_id"], kb_id=data["kb_id"], index_version=data["index_version"],
                                  **{key: hit.get(key) for key in ("chunk_id", "source", "page", "section")})
                self.check_sources((source,))
                if not isinstance(hit["text"], str) or not hit["text"].strip():
                    raise ValueError
            content = json.dumps(data, ensure_ascii=False, allow_nan=False)
            if len(content) > 16000:
                raise ValueError
            return content
        except (KeyError, TypeError, ValueError):
            raise MCPError("知识库结果结构、范围或长度不可验证", error_kind="invalid_result") from None

    def sources_from_trace(self, trace):
        expected = f"mcp__{self.config.server_name}__search_knowledge"
        if trace.call.name != expected or trace.is_error or not trace.original_complete:
            return ()
        try:
            data = json.loads(trace.content)
            sources = tuple(Evidence(project_id=data["project_id"], kb_id=data["kb_id"], index_version=data["index_version"],
                                     **{key: hit.get(key) for key in ("chunk_id", "source", "page", "section")}) for hit in data["hits"])
            self.grant(data["kb_id"])
            self.check_sources(sources)
            return sources
        except (KeyError, TypeError, ValueError):
            raise MCPError("历史文档结果不可验证", error_kind="invalid_result") from None

    def bound_result(self, content, max_chars, token_budget):
        from lynx_agent.agent.tokens import estimate_text_tokens
        data = json.loads(content)
        while len(json.dumps(data, ensure_ascii=False)) > max_chars or estimate_text_tokens(json.dumps(data, ensure_ascii=False)) > token_budget:
            if not data["hits"]:
                return '{"error_kind":"result_budget","message":"文档结果预算不足，请缩小查询"}'
            data["hits"].pop()
            data["truncated"] = True
        return json.dumps(data, ensure_ascii=False)
