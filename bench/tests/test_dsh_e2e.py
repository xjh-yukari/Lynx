"""dsh 端到端自检：真跑 `dsh --profile headless`，但模型换成本地假上游。

默认跳过，避免 CI 与日常测试依赖本机安装的 dsh：

    $env:BENCH_DSH_E2E = '1'
    uv run --locked pytest bench/tests/test_dsh_e2e.py -q

它会验证整条链路：bench 注入 DEEPSEEK_BASE_URL → dsh 走 Anthropic Messages →
录制代理记账 → dsh-mcp-client 挂上测试环境 MCP → 环境侧记录工具调用 → 判分。
"""

import os
import shutil
from pathlib import Path

import pytest
import pytest_asyncio
from agent_bench.config import load_bench_config
from agent_bench.runner import BenchRunner, RunPaths, load_records, load_traces

from tests.fake_anthropic import FakeAnthropic

REPO = Path(__file__).resolve().parents[2]
CASE = "tools.selection.notes"

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        shutil.which("dsh") is None,
        reason="需要本机安装 dsh（npm i -g @deepseek-ai/dsh）"),
]


@pytest_asyncio.fixture
async def upstream():
    server = await FakeAnthropic().start()
    try:
        yield server
    finally:
        await server.stop()


def make_config(tmp_path, upstream):
    base = load_bench_config(REPO / "bench.toml")
    agents = dict(base.agents)
    agents["dsh"] = base.agents["dsh"].model_copy(update={"enabled": True})
    return base.model_copy(update={
        "model": base.model.model_copy(update={
            "base_url": upstream.base_url, "anthropic_base_url": upstream.base_url}),
        "run": base.run.model_copy(update={
            "results_dir": str(tmp_path / "results"),
            "keep_workspaces": False,
            "turn_timeout_seconds": 240,
        }),
        "agents": agents,
    })


@pytest.mark.skipif(os.environ.get("BENCH_DSH_E2E") != "1",
                    reason="设置 BENCH_DSH_E2E=1 才运行真实的 dsh 端到端自检")
async def test_dsh_headless_completes_case_with_mcp_tools(tmp_path, monkeypatch, upstream):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-e2e-not-real")
    config = make_config(tmp_path, upstream)
    runner = BenchRunner(config, repo_root=tmp_path)
    code = await runner.run(agents=["dsh"], only_ids=[CASE], repeats=1)
    assert code == 0, "运行被拒绝，通常是预算或配置问题"

    paths = RunPaths(next((tmp_path / "results").iterdir()))
    traces = load_traces(paths)
    records = load_records(paths)
    assert traces, "没有产生 trace"

    # 1) 记账：模型调用经过录制代理，且拿到了 Anthropic 口径的用量。
    assert records, "代理没有记录到任何模型调用（DEEPSEEK_BASE_URL 可能没生效）"
    assert all(record.prompt_tokens for record in records)
    assert all(record.status == 200 for record in records)
    assert any(record.cached_prompt_tokens for record in records)

    # 2) 会话：来自 session 事件的 sessionId，两轮共用同一会话。
    sessions = {trace.diagnostics.get("session") for trace in traces}
    assert len(sessions) == 1
    assert str(next(iter(sessions))).startswith("session-")

    # 3) 工具：环境 MCP 经 dsh-mcp-client 挂载，工具调用由环境侧记录。
    assert [tool.tool for tool in traces[0].tools] == ["notes_set"]
    assert [tool.tool for tool in traces[1].tools] == ["notes_get"]
    assert all(g.passed for trace in traces for g in trace.grades)

    # 4) 评分文件落盘。
    assert paths.scores.exists()
