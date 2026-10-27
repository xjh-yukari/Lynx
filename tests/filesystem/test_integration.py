import json

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.permissions import handle_approval_command
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.filesystem import FileAccess
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionStore
from lynx_agent.web.setup import create_default_tools


def config():
    return Config(agent=AgentConfig(model=ModelConfig(name="fake")), providers=ProvidersConfig(
        deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")))


class Provider(ModelProvider):
    def __init__(self, action, args):
        self.action = action
        self.args = args
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        if request.messages[-1].role == "tool":
            return ModelResponse(content="工具操作已返回，待审批不代表修改完成。", model="fake", finish_reason="stop")
        return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(
            ModelToolCall(id="file-call", name=self.action, arguments=json.dumps(self.args)),
        ))

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_workspace_write_allowed_without_enabling_arbitrary_write_tools_and_approval_audited(tmp_path):
    access = FileAccess(tmp_path)
    settings = config()
    store = SessionStore(tmp_path / "sessions")
    provider = Provider("write_file", {"path": "test.txt", "content": "old"})
    async with AgentRunner(settings, provider, store, tools=create_default_tools(settings, file_access=access)) as runner:
        assert runner.permission_label == "工作区修改"
        assert not settings.agent.tools.allow_write_tools
        await runner.ask("创建文件")
        assert (tmp_path / "test.txt").read_text() == "old"
        assert json.loads(provider.requests[-1].messages[-1].content)["status"] != "pending_approval"
        provider.action = "edit_file"
        provider.args = {"path": "test.txt", "old_text": "old", "new_text": "new"}
        await runner.ask("修改文件")
        identifier = next(iter(access.pending))
        assert (tmp_path / "test.txt").read_text() == "old"
        assert "仅显示预览" in handle_approval_command(runner, "/approve", identifier)
        handle_approval_command(runner, "/approve", identifier)
        assert (tmp_path / "test.txt").read_text() == "new"
        saved = store.load(runner.session_id)
        assert saved.events[-1].tool.call.name == "file_approval"
        assert '"actor": "user"' in saved.events[-1].tool.call.arguments
        runner.set_file_mode("read")
        provider.args = {"path": "forbidden.txt", "content": "forbidden"}
        provider.action = "write_file"
        await runner.ask("只读不能创建")
        assert not (tmp_path / "forbidden.txt").exists()
        assert "找不到工具" in provider.requests[-1].messages[-1].content
        assert "write_file" not in [tool.name for tool in provider.requests[-1].tools]
        runner.switch_session(None)
        assert runner.permission_label == "仅查看"


@pytest.mark.asyncio
async def test_restart_defaults_workspace_and_mode_change_not_sent_to_model(tmp_path):
    access = FileAccess(tmp_path)
    settings = config()
    async with AgentRunner(settings, Provider("read_file", {"path": "none.txt"}),
                           tools=create_default_tools(settings, file_access=access)) as runner:
        runner.set_file_mode("full")
        assert runner.permission_label == "完全权限"
        assert runner.session_id is None
    fresh = FileAccess(tmp_path)
    assert fresh.policy.mode == "workspace"
