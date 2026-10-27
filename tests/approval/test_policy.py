"""审批档位、单步门禁和运行器集成测试；全部使用临时目录。"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from lynx_agent.approval.config import ApprovalConfig
from lynx_agent.approval.manager import ApprovalManager
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.execution.config import ExecutionConfig
from lynx_agent.execution.service import ExecutionService
from lynx_agent.filesystem import FileAccess
from lynx_agent.filesystem.policy import WORKSPACE, FilePolicy
from lynx_agent.tools.errors import ToolExecutionError


def environment(tmp_path, *, mode="auto", config=None, permission="workspace"):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    state = {"session": "session-1", "permission": permission}
    manager = ApprovalManager(
        config or ApprovalConfig(mode=mode), session_id=lambda: state["session"],
        permission_mode=lambda: state["permission"], data_dir=tmp_path / "data",
        backup_dir=tmp_path / "data" / "backups",
    )
    manager.begin_turn()
    access = FileAccess(root)
    access.approval = manager
    return SimpleNamespace(root=root, manager=manager, access=access, state=state)


def approve(manager, decision="once", cards=None):
    def presenter(pending):
        if pending is not None:
            if cards is not None:
                cards.append(pending)
            manager.decide(pending.identifier, decision)
    manager.presenter = presenter


def test_defaults_are_auto_and_labels_are_explicit():
    config = ApprovalConfig()
    assert config.mode == "auto"
    assert config.max_batch_files == 200
    assert config.download_max_bytes > 0
    manager = ApprovalManager(config, session_id=lambda: None, permission_mode=lambda: "workspace")
    assert manager.label == "常用操作自动"
    assert manager.writable
    assert "不是操作系统沙箱" in manager.status_text()


def test_modes_validate_and_paused_blocks_execution_service(tmp_path):
    env = environment(tmp_path)
    with pytest.raises(ValueError):
        env.manager.set_mode("forever")
    service = ExecutionService(FilePolicy(env.root), ExecutionConfig(), tmp_path / "logs",
                               lambda: "session-1")
    service.enabled = True
    service.broker.presenter = lambda item: None
    service.approval = env.manager
    assert service.available
    env.manager.set_mode("paused")
    assert not service.available
    with pytest.raises(ToolExecutionError, match="暂停执行"):
        asyncio.run(service.execute({"argv": ["python"]}))
    env.manager.set_mode("ask")
    assert service.available


@pytest.mark.asyncio
async def test_auto_reads_do_not_ask_and_paused_allows_analysis(tmp_path):
    env = environment(tmp_path)
    (env.root / "a.txt").write_text("hello", encoding="utf-8")
    cards = []
    approve(env.manager, "deny", cards)
    for _ in range(3):
        result = await env.access.invoke("read_file", {"path": "a.txt"})
        assert "hello" in result
    assert not cards
    env.manager.set_mode("paused")
    result = await env.access.invoke("list_dir", {"path": "."})
    assert "a.txt" in result


@pytest.mark.asyncio
async def test_ask_mode_asks_for_reads_and_new_files(tmp_path):
    env = environment(tmp_path, mode="ask")
    (env.root / "a.txt").write_text("hello", encoding="utf-8")
    cards = []
    approve(env.manager, "deny", cards)
    with pytest.raises(ToolExecutionError, match="拒绝|过期"):
        await env.access.invoke("read_file", {"path": "a.txt"})
    assert cards and cards[0].kind == "gate"
    approve(env.manager, "once", cards)
    assert "hello" in await env.access.invoke("read_file", {"path": "a.txt"})
    assert "created" in await env.access.invoke("write_file", {"path": "new.txt", "content": "x"})
    assert (env.root / "new.txt").read_text(encoding="utf-8") == "x"
    assert len(cards) == 3


@pytest.mark.asyncio
async def test_paused_allows_preview_but_refuses_commit_and_new_files(tmp_path):
    env = environment(tmp_path)
    (env.root / "a.txt").write_text("old", encoding="utf-8")
    proposal = await env.access.invoke("edit_file", {"path": "a.txt", "old_text": "old", "new_text": "new"})
    assert "approval_id" in proposal
    identifier = next(iter(env.access.pending))
    env.access.preview(identifier)
    env.manager.set_mode("paused")
    with pytest.raises(ValueError, match="暂停执行"):
        env.access.approve(identifier)
    with pytest.raises(ToolExecutionError, match="暂停执行"):
        await env.access.invoke("write_file", {"path": "b.txt", "content": "x"})
    assert (env.root / "a.txt").read_text(encoding="utf-8") == "old"
    assert not (env.root / "b.txt").exists()


@pytest.mark.asyncio
async def test_no_presenter_denies_ask_and_batch(tmp_path):
    env = environment(tmp_path, mode="ask")
    assert env.manager.presenter is None
    with pytest.raises(ToolExecutionError, match="没有人工审批界面"):
        await env.access.invoke("read_file", {"path": "a.txt"})
    env.manager.set_mode("auto")
    from lynx_agent.approval.models import BatchLimits, BatchPlan, PlanFile
    decision = await env.manager.authorize_batch(BatchPlan(
        tool="t", operation="replace_text", purpose="", read_roots=(), write_roots=(),
        files=(PlanFile(str(env.root / "x"), 1, "a"),), total_bytes=1,
        limits=BatchLimits(10, 100, 10), allow_overwrite=True))
    assert not decision.allowed and "拒绝" in decision.reason


def test_audit_redacts_known_secrets_and_never_records_content(tmp_path):
    env = environment(tmp_path)
    env.manager.secrets = ("super-secret-value",)
    env.manager.record_operation(tool="batch_replace_text", authorization="grant-1",
                                 target=str(env.root / "super-secret-value.txt"),
                                 result="replaced", detail="正文不会写入审计")
    text = env.manager.audit.path.read_text(encoding="utf-8")
    assert "super-secret-value" not in text
    assert "[已隐藏]" in text
    assert "正文不会写入审计" in text


def test_system_and_agent_rule_writes_blocked_even_full(tmp_path):
    policy = FilePolicy(tmp_path)
    policy.set_mode("full")
    system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
    with pytest.raises(ToolExecutionError, match="系统目录"):
        policy.resolve(str(system_root / "lynx-approval-test.txt"), write=True)
    with pytest.raises(ToolExecutionError, match="保护范围"):
        policy.resolve(str(WORKSPACE / "config.toml"), write=True)


def test_runner_defaults_and_independent_switches(tmp_path):
    from lynx_agent.agent import AgentRunner
    from lynx_agent.filesystem import FileAccess
    from lynx_agent.session import SessionStore
    from lynx_agent.web.setup import create_default_tools

    root = tmp_path / "workspace"
    root.mkdir()
    access = FileAccess(root)
    settings = Config(agent=AgentConfig(model=ModelConfig(name="fake")), providers=ProvidersConfig(
        deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))

    class Provider:
        async def aclose(self):
            pass

    store = SessionStore(tmp_path / "sessions")
    runner = AgentRunner(settings, Provider(), store,
                         tools=create_default_tools(settings, file_access=access))
    assert runner.approval_label == "常用操作自动"
    assert runner.permission_label == "工作区修改"
    assert "batch_replace_text" in runner.tools.names and "download_file" in runner.tools.names
    assert "exec_command" not in runner.tools.names  # 没有人类审批界面时不提供命令工具
    runner.set_file_mode("full")
    assert runner.approval_label == "常用操作自动"
    runner.set_approval_mode("paused")
    assert runner.approval_label == "暂停执行" and runner.permission_label == "完全权限"
    assert "batch_replace_text" not in runner.tools.names
    runner.set_approval_mode("auto")
    assert "batch_replace_text" in runner.tools.names
    runner.set_file_mode("read")
    assert "batch_replace_text" not in runner.tools.names
    runner.revoke_approval()
