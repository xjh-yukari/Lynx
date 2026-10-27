"""验证完整会话跨实例恢复、写入故障和终端重启。"""

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
from lynx_agent.providers import (
    ChatMessage,
    ModelProvider,
    ModelResponse,
    ProviderConnectionError,
)
from lynx_agent.session import SessionError, SessionEvent, SessionStore


def config():
    return Config(
        agent=AgentConfig(model=ModelConfig(name="fake"), system_prompt="当前提示词"),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="not-a-real-key", base_url="https://example.invalid",
        )),
    )


def provider(reply="模拟回复\n第二行  "):
    mock = AsyncMock(spec=ModelProvider)
    mock.generate.return_value = ModelResponse(content=reply, model="fake", finish_reason="stop")
    return mock


@pytest.mark.asyncio
async def test_restart_restores_exact_text_and_continues(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    first_provider = provider()
    async with AgentRunner(config(), first_provider, store, session) as first:
        await first.ask("  我叫小林\n请记住  ")
    restored_provider = provider("你叫小林")
    async with AgentRunner(
        config(), restored_provider, SessionStore(tmp_path), store.load(session.id),
    ) as restored:
        await restored.ask("我叫什么？")
    request = restored_provider.generate.call_args.args[0]
    assert [message.content for message in request.messages] == [
        "当前提示词", "  我叫小林\n请记住  ", "模拟回复\n第二行  ", "我叫什么？",
    ]
    assert len(store.load(session.id).events) == 2
    assert "not-a-real-key" not in next(tmp_path.glob("*.jsonl")).read_text(encoding="utf-8")
    first_provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_reset_keeps_original_events_but_restores_only_new_history(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    async with AgentRunner(config(), provider(), store, session) as agent:
        await agent.ask("旧消息")
        agent.clear_history()
        await agent.ask("新消息")
    loaded = store.load(session.id)
    assert [event.kind for event in loaded.events] == ["turn", "reset", "turn"]
    assert loaded.events[0].messages[0].content == "旧消息"
    assert loaded.history()[0].content == "新消息"


@pytest.mark.asyncio
async def test_failed_api_does_not_save_turn(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create()
    mock = provider()
    mock.generate.side_effect = ProviderConnectionError("断网")
    async with AgentRunner(config(), mock, store, session) as agent:
        with pytest.raises(ProviderConnectionError):
            await agent.ask("失败输入")
        assert len(agent.context.messages) == 1
    assert store.load(session.id).context_history() == ()
    assert store.load(session.id).pending_reply() is not None


@pytest.mark.asyncio
async def test_failed_save_keeps_reply_and_can_retry(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    session = store.create()
    agent = AgentRunner(config(), provider(), store, session)
    original_save = store.save

    def fail_save(session):
        raise SessionError("模拟磁盘已满")

    monkeypatch.setattr(store, "save", fail_save)
    reply = await agent.ask("消息")
    assert reply.content == "模拟回复\n第二行  "
    assert agent.persistence_error is not None
    assert len(agent.context.messages) == 3
    assert store.load(session.id).events == ()
    monkeypatch.setattr(store, "save", original_save)
    agent.save_session()
    assert agent.persistence_error is None
    await agent.aclose()
    assert len(store.load(session.id).events) == 1


def test_atomic_replace_failure_leaves_original_intact(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    session = store.create()
    original = next(tmp_path.glob("*.jsonl")).read_bytes()
    updated = session.with_event(SessionEvent(kind="turn", messages=(
        ChatMessage(role="user", content="问题"), ChatMessage(role="assistant", content="回答"),
    )))

    def fail_replace(source, target):
        raise OSError("模拟替换失败")

    monkeypatch.setattr("lynx_agent.session.store.os.replace", fail_replace)
    with pytest.raises(SessionError, match="无法保存"):
        store.save(updated)
    assert next(tmp_path.glob("*.jsonl")).read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 1


def test_invalid_id_missing_and_corrupt_session(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(SessionError, match="UUID"):
        store.load("../outside")
    session = store.create()
    path = next(tmp_path.glob("*.jsonl"))
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(SessionError, match="格式损坏"):
        store.load(session.id)
    assert path.read_text(encoding="utf-8") == "{broken"
    with pytest.raises(SessionError, match="不存在"):
        store.load("00000000-0000-0000-0000-000000000000")


@pytest.mark.asyncio
async def test_cli_exit_and_resume_with_new_runner(tmp_path, monkeypatch, capsys, chat_inputs):
    runner_module = importlib.import_module("lynx_agent.agent.runner")
    store = SessionStore(tmp_path)
    providers = [provider("你好"), provider("小林")]
    monkeypatch.setattr(runner_module, "SessionStore", lambda: store)
    monkeypatch.setattr(runner_module, "create_provider", lambda config: providers.pop(0))
    monkeypatch.setattr(commands, "load_config", config)
    reader = chat_inputs(["我叫小林", "exit"])
    await commands.chat()
    identifier = str(store.list_sessions()[0].id)
    reader.read.side_effect = ["我叫什么？", "exit"]
    await commands.chat(identifier)
    assert len(store.load(identifier).events) == 2
    assert "小林" in capsys.readouterr().out


def test_list_sessions_does_not_load_api_config(tmp_path, monkeypatch, capsys):
    store = SessionStore(tmp_path)
    first = store.create()
    second = store.create()
    monkeypatch.setattr(commands, "SessionStore", lambda: store)
    monkeypatch.setattr("sys.argv", ["lynx-agent", "--list-sessions"])

    def unexpected_config():
        pytest.fail("列出会话不应加载 API 配置")

    monkeypatch.setattr(commands, "load_config", unexpected_config)
    commands.main()
    output = capsys.readouterr().out
    assert str(first.id) in output
    assert str(second.id) in output
