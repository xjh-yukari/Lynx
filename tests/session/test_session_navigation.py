"""延迟创建、名称兼容与会话切换测试。"""

import importlib
from unittest.mock import AsyncMock

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import ModelProvider, ModelResponse, ProviderConnectionError
from lynx_agent.session import Session, SessionError, SessionStore


def make_config():
    return Config(
        agent=AgentConfig(model=ModelConfig(name="fake")),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="fake", base_url="https://example.invalid",
        )),
    )


def make_provider():
    provider = AsyncMock(spec=ModelProvider)
    provider.generate.return_value = ModelResponse(content="回答", model="fake", finish_reason="stop")
    return provider


@pytest.mark.asyncio
async def test_lazy_creation_switch_and_rename(tmp_path):
    store = SessionStore(tmp_path)
    mock = make_provider()
    async with AgentRunner(make_config(), mock, store) as runner:
        assert runner.session_id is None
        runner.save_session()
        runner.clear_history()
        assert store.list_sessions() == []
        with pytest.raises(SessionError, match="先发送"):
            runner.rename_session("名称")
        await runner.ask("第一句话\n项目讨论")
        identifier = runner.session_id
        assert store.load(identifier).title == "第一句话 项目讨论"
        runner.rename_session("自定义名称")
        assert store.load(identifier).title == "自定义名称"
        runner.switch_session()
        assert runner.session_id is None
        assert len(runner.context.messages) == 1
        assert len(store.list_sessions()) == 1
        runner.switch_session(identifier)
        assert runner.session_title == "自定义名称"
        assert len(runner.context.messages) == 3
        with pytest.raises(SessionError):
            runner.switch_session("invalid")
        assert runner.session_id == identifier


@pytest.mark.asyncio
async def test_first_api_failure_still_creates_session(tmp_path):
    store = SessionStore(tmp_path)
    mock = make_provider()
    mock.generate.side_effect = ProviderConnectionError("断网")
    async with AgentRunner(make_config(), mock, store) as runner:
        with pytest.raises(ProviderConnectionError):
            await runner.ask("尝试开始")
        assert runner.session_id is not None
        assert store.load(runner.session_id).context_history() == ()
        assert store.load(runner.session_id).pending_reply() is not None


@pytest.mark.asyncio
async def test_failed_save_blocks_switch(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    async with AgentRunner(make_config(), make_provider(), store) as runner:
        await runner.ask("当前内容")
        identifier = runner.session_id
        original_save = store.save

        def fail_save(session):
            raise SessionError("磁盘已满")

        monkeypatch.setattr(store, "save", fail_save)
        with pytest.raises(SessionError):
            runner.switch_session()
        assert runner.session_id == identifier
        assert len(runner.context.messages) == 3
        monkeypatch.setattr(store, "save", original_save)


def test_legacy_metadata_loads_without_title(tmp_path):
    store = SessionStore(tmp_path)
    session = Session()
    path = tmp_path / f"{session.id}.jsonl"
    path.write_text(session.model_dump_json(exclude={"events", "title"}) + "\n", encoding="utf-8")
    assert store.load(session.id).display_title == "未命名会话"


@pytest.mark.asyncio
async def test_help_and_exit_do_not_create_session(tmp_path, monkeypatch, capsys, chat_inputs):
    store = SessionStore(tmp_path)
    runner_module = importlib.import_module("lynx_agent.agent.runner")
    mock = make_provider()
    monkeypatch.setattr(runner_module, "SessionStore", lambda: store)
    monkeypatch.setattr(runner_module, "create_provider", lambda config: mock)
    monkeypatch.setattr(commands, "load_config", make_config)
    chat_inputs([KeyboardInterrupt(), "help", "/tokens", "/tokens 北京天气怎么样", "/save", "/new", "/exit"])
    await commands.chat()
    assert store.list_sessions() == []
    output = capsys.readouterr().out
    assert "↑ / ↓" in output
    assert "输入合计：约" in output
    assert "待发送文本：约" in output
    assert "已取消当前草稿" in output
    mock.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_cli_select_and_rename_restores_history(tmp_path, monkeypatch, capsys, chat_inputs):
    store = SessionStore(tmp_path)
    mock = make_provider()
    async with AgentRunner(make_config(), mock, store) as runner:
        await runner.ask("旧会话的问题")
        identifier = runner.session_id
    runner_module = importlib.import_module("lynx_agent.agent.runner")
    mock = make_provider()
    monkeypatch.setattr(runner_module, "SessionStore", lambda: store)
    monkeypatch.setattr(runner_module, "create_provider", lambda config: mock)
    monkeypatch.setattr(commands, "SessionStore", lambda: store)
    monkeypatch.setattr(commands, "load_config", make_config)
    monkeypatch.setattr(commands, "select_session", AsyncMock(return_value=identifier))
    chat_inputs(["/sessions", "/rename 新名称", "继续", "/exit"])
    await commands.chat()
    assert store.load(identifier).title == "新名称"
    assert len(store.list_sessions()) == 1
    assert len(mock.generate.call_args.args[0].messages) == 4
    assert "旧会话的问题" in capsys.readouterr().out
