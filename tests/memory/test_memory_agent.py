"""Agent/CLI 与记忆集成：真实 MCP，模型完全模拟。"""

import asyncio
import json
from pathlib import Path

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.app import LynxTUI
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.memory.config import MemoryConfig
from lynx_agent.memory.errors import MemoryError
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionStore
from lynx_agent.web.setup import create_default_tools

BINARY = Path("E:/github/lynx_agent_data/bin/engram/3.4.0/engram.exe")
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not BINARY.is_file(), reason="需要固定版本 Engram，测试不联网安装"
    ),
]


class Provider(ModelProvider):
    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    async def generate(self, request):
        self.requests.append(request)
        return next(
            self.responses,
            ModelResponse(content="测试回答", model="fake", finish_reason="stop"),
        )

    async def aclose(self):
        self.closed = True


def config(tmp_path, **memory):
    return Config(
        providers=ProvidersConfig(
            deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")
        ),
        agent=AgentConfig(model=ModelConfig(name="fake")),
        memory=MemoryConfig(
            enabled=True,
            executable=str(BINARY),
            data_dir=str(tmp_path / "memory"),
            **memory,
        ),
    )


async def confirm(runner, kind, body, **kwargs):
    identifier = await runner.memory.propose(kind, body, **kwargs)
    await runner.memory.preview(identifier)
    await runner.memory_command("/memory", "approve " + identifier)


def retention_response(body="北京本次降温命名为霜降前哨。"):
    return ModelResponse(
        content=json.dumps({"conclusions": [{"title": "北京降温命名", "body": body}]}, ensure_ascii=False),
        model="fake", finish_reason="stop",
    )


async def test_auto_retention_persists_and_recalls_across_sessions(tmp_path):
    body = "北京本次降温命名为霜降前哨。"
    cfg = config(tmp_path, auto_save=True, project_id="retention")
    provider = Provider([
        ModelResponse(content=body, model="fake", finish_reason="stop"),
        retention_response(body),
        ModelResponse(content=body, model="fake", finish_reason="stop"),
        retention_response(body),
    ])
    store = SessionStore(tmp_path / "sessions")
    phases = []
    async with AgentRunner(cfg, provider, store) as runner:
        result = await runner.ask("给北京本次降温取个名字", on_status=phases.append)
        assert result.content == body
        assert "memorizing" in phases
        assert "已自动保留 1 条" in runner.memory.retention_notice
        locs = list(runner.memory.indexes["project"].values())
        assert len(locs) == 1 and locs[0].topic.startswith("auto/task/")
        original_session = runner.session_id
        key = next(iter(runner.memory.indexes["project"]))
        record = await runner.memory.get(key)
        assert record.source_session == original_session
        page = await runner.memory.list_records(include_body=True)
        assert page["records"][0]["origin"] == "auto_task_conclusion"
        assert page["records"][0]["body"] == body
        assert not runner.memory.profiles and not runner.memory.pending
        assert runner.last_usage.report("memory").calls == 1
        runner.switch_session()
        await runner.ask("北京降温起的什么名字")
        assert runner.session_id != original_session
        assert any("霜降前哨" in m.request_content and "自动任务结论" in m.request_content for m in provider.requests[2].messages)
        assert len(runner.memory.indexes["project"]) == 1
    async with AgentRunner(cfg, Provider(), store) as runner:
        assert (await runner.memory.get(key)).body == body
        assert (await runner.memory.search("北京降温"))[0][0] == key


@pytest.mark.parametrize("failure", ["invalid", "timeout", "cancel"])
async def test_auto_retention_failure_preserves_completed_turn(tmp_path, failure):
    cfg = config(tmp_path, auto_save=True, auto_save_timeout_seconds=0.03)

    class FailingProvider(Provider):
        async def generate(self, request):
            if self.requests:
                if failure == "timeout":
                    await asyncio.sleep(1)
                if failure == "cancel":
                    raise asyncio.CancelledError
            return await super().generate(request)

    provider = FailingProvider()
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(cfg, provider, store) as runner:
        response = await runner.ask("为本次任务给个结论")
        assert response.content == "测试回答"
        assert "回答已完成" in runner.memory.retention_notice
        assert runner.persistence_error is None and not runner.can_retry
        assert not runner.memory.pending and not runner.memory.indexes["project"]
        assert store.load(runner.session_id).events[-1].kind == "turn"


async def test_auto_retention_toggle_permissions_and_opt_out(tmp_path):
    cfg = config(tmp_path, auto_save=True)
    provider = Provider()
    tools = create_default_tools(cfg)
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(cfg, provider, store, tools=tools) as runner:
        await runner.memory_command("/memory", "autosave off")
        await runner.ask("给北京降温取名")
        assert len(provider.requests) == 1
        await runner.memory_command("/memory", "autosave on")
        await runner.ask("本次不要保存任何结论")
        assert len(provider.requests) == 2
        # 确认写权限关闭时连提取请求都不产生。
        runner.memory.mode = lambda: "read"
        await runner.ask("给北京降温取名")
        assert len(provider.requests) == 3
        assert "不可写" in runner.memory.retention_notice


async def test_auto_retention_does_not_overwrite_manual_or_resurrect_deleted(tmp_path):
    body = "北京本次降温命名为霜降前哨。"
    provider = Provider([
        ModelResponse(content=body, model="fake", finish_reason="stop"), retention_response(body),
        ModelResponse(content=body, model="fake", finish_reason="stop"), retention_response(body),
    ])
    cfg = config(tmp_path, auto_save=True)
    async with AgentRunner(cfg, provider, SessionStore(tmp_path / "sessions")) as runner:
        await confirm(runner, "decision", "人工决定保留的旧结论。", title="北京降温命名")
        manual_key = next(iter(runner.memory.indexes["project"]))
        await runner.ask("给北京本次降温取名")
        assert (await runner.memory.get(manual_key)).body == "人工决定保留的旧结论。"
        auto_key = next(key for key, loc in runner.memory.indexes["project"].items() if loc.topic.startswith("auto/task/"))
        change = await runner.memory.propose_delete(auto_key)
        await runner.memory.preview(change)
        await runner.memory.approve(change)
        await runner.ask("再说一下北京本次降温名字")
        assert runner.memory.indexes["project"][auto_key].deleted
        assert len(runner.memory.indexes["project"]) == 2


async def test_auto_retention_uncertain_write_is_not_replayed(tmp_path, monkeypatch):
    body = "北京本次降温命名为霜降前哨。"
    provider = Provider([
        ModelResponse(content=body, model="fake", finish_reason="stop"), retention_response(body),
    ])
    cfg = config(tmp_path, auto_save=True)
    async with AgentRunner(cfg, provider, SessionStore(tmp_path / "sessions")) as runner:
        backend = runner.memory.backends["project"]
        original_save = backend.save
        writes = []

        async def lose_ack(record):
            writes.append(record)
            await original_save(record)
            raise MemoryError("模拟已经保存但响应丢失")

        monkeypatch.setattr(backend, "save", lose_ack)
        result = await runner.ask("给北京降温取名")
        assert result.content == body and len(writes) == 1
        assert runner.memory.uncertain and not runner.memory.writable
        assert not runner.memory.pending and runner.persistence_error is None
        await runner.ask("普通后续任务")
        assert len(provider.requests) == 3  # 后端不可写时，不再提取。
        await runner.memory.reconcile()
        assert len(writes) == 1 and not runner.memory.uncertain
        record = (await runner.memory.search("北京降温"))[0][1]
        assert record.body == body and record.topic.startswith("auto/task/")


async def test_auto_retention_runs_only_after_retry_completes(tmp_path):
    body = "北京本次降温命名为霜降前哨。"
    provider = Provider([
        ModelResponse(content="", reasoning_content="未完成推理", model="fake", finish_reason="length"),
        ModelResponse(content=body, model="fake", finish_reason="stop"), retention_response(body),
    ])
    cfg = config(tmp_path, auto_save=True)
    async with AgentRunner(cfg, provider, SessionStore(tmp_path / "sessions")) as runner:
        response = await runner.ask("给北京降温取名")
        assert response.finish_reason == "length" and len(provider.requests) == 1
        assert runner.can_retry and not runner.memory.indexes["project"]
        response = await runner.retry()
        assert response.content == body and len(provider.requests) == 3
        assert "已自动保留 1 条" in runner.memory.retention_notice


async def test_auto_retention_skips_session_save_failure(tmp_path, monkeypatch):
    cfg = config(tmp_path, auto_save=True)
    provider = Provider()
    async with AgentRunner(cfg, provider, SessionStore(tmp_path / "sessions")) as runner:
        from lynx_agent.session import SessionError

        def fail_save():
            raise SessionError("模拟会话保存失败")

        monkeypatch.setattr(runner, "save_session", fail_save)
        await runner.ask("为任务给个结论")
        assert len(provider.requests) == 1 and runner.persistence_error
        assert not runner.memory.indexes["project"]
        monkeypatch.undo()
        runner.save_session()


async def test_profiles_recall_and_session_history_are_separate(tmp_path):
    cfg = config(tmp_path)
    provider = Provider()
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(cfg, provider, store) as runner:
        await confirm(runner, "user", "默认先说明整体流程，偏好是 userfixtureunique。")
        await confirm(
            runner,
            "fact",
            "Validation budget is budgetunique.",
            title="Budget decision",
        )
        await runner.ask("budgetunique 是什么？")
        request = provider.requests[-1]
        assert "userfixtureunique" in request.messages[0].content
        assert any(
            "budgetunique" in message.request_content and "长期记忆" in message.request_content
            for message in request.messages
        )
        assert not any(
            "userfixtureunique" in message.content for message in runner.context.history
        )
        assert not any(
            message.content.startswith("[相关长期记忆")
            for message in runner.context.history
        )
        assert all(not tool.name.startswith("mcp__memory") for tool in request.tools)
        first_session = runner.session_id
        runner.switch_session()
        await runner.ask("再次开始")
        assert runner.session_id != first_session
        assert "userfixtureunique" in provider.requests[-1].messages[0].content
        assert runner.memory.profiles["user"].body.startswith("默认")
        runner.clear_history()
        assert "userfixtureunique" in runner.context.messages[0].content
    assert provider.closed


async def test_natural_language_proposes_but_cannot_approve(tmp_path):
    proposal = ModelResponse(
        content="",
        model="fake",
        finish_reason="tool_calls",
        tool_calls=[
            ModelToolCall(
                id="m1",
                name="memory_propose",
                arguments='{"kind":"user","body":"默认使用中文。"}',
            ),
        ],
    )
    provider = Provider(
        [
            proposal,
            ModelResponse(
                content="请预览确认候选。", model="fake", finish_reason="stop"
            ),
        ]
    )
    cfg = config(tmp_path)
    async with AgentRunner(
        cfg, provider, SessionStore(tmp_path / "sessions")
    ) as runner:
        await runner.ask("记住，默认使用中文。")
        assert not runner.memory.profiles
        assert len(runner.memory.pending) == 1
        identifier = next(iter(runner.memory.pending))
        assert not runner.memory.pending[identifier].viewed
        assert "尚未保存" in provider.requests[-1].messages[-1].content
        # 首次人工批准只显示预览，第二次才允许提交。
        assert "拟修改内容" in await runner.memory_command(
            "/memory", "approve " + identifier
        )
        await runner.memory_command("/memory", "approve " + identifier)
        assert runner.memory.profiles["user"].body == "默认使用中文。"


async def test_filesystem_cannot_bypass_memory_directory_or_read_mode(tmp_path):
    cfg = config(tmp_path)
    registry = create_default_tools(cfg)
    async with AgentRunner(cfg, Provider(), tools=registry) as runner:
        for mode in ("workspace", "full"):
            runner.set_file_mode(mode)
            with pytest.raises(Exception, match="禁止通过通用文件工具"):
                runner.file_access.policy.resolve(
                    str(runner.memory.directories["project"] / "index.json")
                )
        runner.set_file_mode("read")
        with pytest.raises(MemoryError):
            await runner.memory.propose("user", "不允许修改。")
        assert "memory_propose" not in runner.tools.names


async def test_tui_profile_preview_and_approval_do_not_call_model(tmp_path):
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    cfg = config(tmp_path)
    provider = Provider()
    async with AgentRunner(cfg, provider) as runner:
        tui = LynxTUI(
            runner,
            cfg,
            SessionStore(tmp_path / "sessions"),
            input_stream=DummyInput(),
            output_stream=DummyOutput(),
        )
        await tui._dispatch("/profile set soul 友好、简洁，明确说明不确定性。")
        identifier = next(iter(runner.memory.pending))
        assert "拟修改内容" in tui.state.messages[-1].content
        await tui._dispatch("/memory approve " + identifier)
        await tui._dispatch("/profile show soul")
        assert "不确定性" in tui.state.messages[-1].content
        assert not provider.requests
