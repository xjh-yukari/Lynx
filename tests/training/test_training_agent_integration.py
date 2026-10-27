import json

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
    TrainingConfig,
)
from lynx_agent.config.loader import load_config
from lynx_agent.providers import ModelProvider, ModelResponse
from lynx_agent.tools import ToolCall, ToolRegistry


class Provider(ModelProvider):
    def __init__(self):
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        return ModelResponse(content="unused", model="fake", finish_reason="stop")

    async def aclose(self):
        pass


def config():
    return Config(agent=AgentConfig(model=ModelConfig(name="fake")),
                  providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))


@pytest.mark.asyncio
async def test_runner_toggle_and_tools_are_read_only_and_do_not_call_provider():
    provider = Provider()
    runner = AgentRunner(config(), provider)
    try:
        assert "training_inspect" not in runner.tools.names
        await runner.training_command("on confirm")
        assert "training_inspect" in runner.tools.names
        assert runner.tools.get("training_inspect").read_only
        assert "training_submit" not in runner.tools.names

        class Client:
            async def request(self, operation, arguments):
                assert operation == "environment"
                return {"training_started": False}

        runner.training.client = Client()
        result = await runner.tools.execute(ToolCall(id="test", name="training_inspect", arguments={"mode": "environment"}))
        assert not result.is_error
        assert json.loads(result.content)["training_started"] is False
        await runner.training_command("off")
        assert "training_inspect" not in runner.tools.names
        assert provider.calls == 0
    finally:
        await runner.aclose()


@pytest.mark.asyncio
async def test_plain_training_commands_are_handled_without_api(monkeypatch, capsys, chat_inputs):
    current = config()
    provider = Provider()
    runner = AgentRunner(current, provider)
    monkeypatch.setattr(commands, "load_config", lambda: current)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    chat_inputs(["/training", "/training on confirm", "/training submit something", "/training off", "/exit"])
    await commands.chat()
    output = capsys.readouterr().out
    assert "阶段M1" in output
    assert "尚未实现" in output
    assert "已关闭" in output
    assert provider.calls == 0


def test_training_config_loads_from_toml_not_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\n", encoding="utf-8")
    preset = tmp_path / "config.toml"
    preset.write_text('[training]\nenabled = true\ndistribution = "Ubuntu-24.04"\nroot = "~/experiment-state"\n', encoding="utf-8")
    monkeypatch.setattr("lynx_agent.config.loader._collect_environment", lambda values: values)
    loaded = load_config(env_path=env, preset_path=preset)
    assert loaded.training.enabled
    assert loaded.training.root == "~/experiment-state"
    assert TrainingConfig().enabled is False


@pytest.mark.asyncio
async def test_tui_dispatches_training_commands_without_model(tmp_path):
    from prompt_toolkit.input.defaults import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from lynx_agent.cli.app import LynxTUI
    from lynx_agent.cli.layout import transcript_text
    from lynx_agent.session import SessionStore

    provider = Provider()
    current = config()
    store = SessionStore(tmp_path / "sessions")
    async with AgentRunner(current, provider, store) as runner:
        with create_pipe_input() as pipe:
            ui = LynxTUI(runner, current, store, input_stream=pipe, output_stream=DummyOutput(),
                         preferences_path=tmp_path / "prefs.json")
            await ui._dispatch("/training on confirm")
            assert runner.training.enabled
            await ui._dispatch("/training status")
            assert "阶段M1" in transcript_text(ui.state)
            await ui._dispatch("/training off")
            assert not runner.training.enabled
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_reused_registry_rebinds_inactive_training_tools():
    registry = ToolRegistry()
    first = AgentRunner(config(), Provider(), tools=registry)
    old = registry.registered("training_inspect")
    await first.aclose()
    second = AgentRunner(config(), Provider(), tools=registry)
    try:
        assert registry.registered("training_inspect") is not old
        assert registry.registered("training_inspect").service is second.training
        assert "training_inspect" not in registry.names
        await second.training_command("on confirm")
        assert registry.get("training_inspect").service is second.training
    finally:
        await second.aclose()
