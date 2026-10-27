import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from lynx_agent.agent.usage import UsageTracker
from lynx_agent.knowledge.access import KnowledgeAccess
from lynx_agent.knowledge.models import Evidence, KnowledgeConfig
from lynx_agent.memory.config import MemoryConfig
from lynx_agent.memory.errors import MemoryError
from lynx_agent.memory.service import MemoryService
from lynx_agent.providers import ModelResponse

BINARY = Path("E:/github/lynx_agent_data/bin/engram/3.4.0/engram.exe")


@pytest.mark.asyncio
@pytest.mark.skipif(not BINARY.is_file(), reason="需要已安装的Engram；不联网下载")
async def test_real_engram_retains_bound_sources_and_revocation_blocks_recall(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text('{"project_id":"ragtest","grants":{"default":{"enabled":true}}}', encoding="utf-8")
    access = KnowledgeAccess(KnowledgeConfig(enabled=True, policy_file=str(policy)), "ragtest")
    evidence = Evidence(project_id="ragtest", kb_id="default", index_version="v1", chunk_id="a" * 64, source="rules.md", page=2)
    config = MemoryConfig(enabled=True, executable=str(BINARY), project_id="ragtest", auto_save=True)
    memory = MemoryService(config, tmp_path / "memory", tmp_path, lambda: "workspace")
    memory.source_references = lambda: (evidence,)
    memory.check_references = access.check_sources
    provider = AsyncMock()
    provider.generate.return_value = ModelResponse(content=json.dumps({"conclusions": [{"title": "会话保存方案", "body": "会话采用JSONL保存。"}]}),
                                                   model="fake", finish_reason="stop")
    await memory.start()
    try:
        await memory.retain_completed(provider, UsageTracker(), "记录本次会话设计结论", "会话采用JSONL保存。")
        records = await memory.list_records(include_body=True)
        assert len(records["records"]) == 1
        key = records["records"][0]["key"]
        assert (await memory.get(key)).sources == (evidence,)
        assert records["records"][0]["sources"][0]["index_version"] == "v1"
        policy.write_text('{"project_id":"ragtest","grants":{}}', encoding="utf-8")
        with pytest.raises(MemoryError, match="撤权"):
            await memory.get(key)
        assert await memory.search("会话") == []
        # 关联来源不可验证时，不能删掉来源转存普通记忆。
        with pytest.raises(MemoryError, match="来源不可验证"):
            await memory.propose("decision", "新结论", title="新任务")
    finally:
        await memory.close()
