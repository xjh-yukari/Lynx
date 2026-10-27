import json

import pytest

from lynx_agent.agent import AgentRunner
from lynx_agent.cli.formatting import format_tools
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
    clear_config_cache,
    load_config,
)
from lynx_agent.config.loader import _ENVIRONMENT_KEYS
from lynx_agent.providers import ModelProvider, ModelResponse, ModelToolCall
from lynx_agent.session import SessionStore
from lynx_agent.tools import ToolRegistry
from lynx_agent.tools.builtin.web_fetch import WebFetchTool
from lynx_agent.tools.builtin.web_search import WebSearchTool
from lynx_agent.web.anysearch import AnySearch
from lynx_agent.web.http import WebDocument
from lynx_agent.web.models import FetchResponse
from lynx_agent.web.setup import create_default_tools


def config(**values):
    return Config(agent=AgentConfig(model=ModelConfig(name="fake")),
                  providers=ProvidersConfig(deepseek=ProviderConfig(api_key="fake", base_url="https://example.invalid")), **values)


def test_default_factory_and_missing_key_diagnostics(tmp_path):
    default = create_default_tools(config())
    files = {"list_dir", "find_files", "read_file", "search_files", "write_file", "edit_file"}
    assert set(default.names) == {"calculator", "web_fetch", "web_search"} | files
    assert default.get("web_search").config.provider == "anysearch"
    missing = config(web={"search": {"provider": "brave"}})
    registry = create_default_tools(missing)
    assert set(registry.names) == {"calculator", "web_fetch"} | files
    class NoProvider(ModelProvider):
        async def generate(self, request):
            raise AssertionError("不应联网")
        async def aclose(self):
            pass
    runner = AgentRunner(missing, NoProvider(), SessionStore(tmp_path), tools=registry)
    report = format_tools(runner, missing)
    assert "BRAVE_SEARCH_API_KEY" in report
    assert "fake" not in report
    disabled = config(web={"search": {"enabled": False}, "fetch": {"enabled": False}})
    assert set(create_default_tools(disabled).names) == {"calculator"} | files


def test_web_config_environment_overrides_and_secret_masking(tmp_path, monkeypatch):
    for key in _ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)
    clear_config_cache()
    env = tmp_path / ".env"
    env.write_text("DEEPSEEK_API_KEY=fake\nDEEPSEEK_MODEL=fake\nWEB_SEARCH_PROVIDER=brave\n"
                   "BRAVE_SEARCH_API_KEY=brave-private\nWEB_SEARCH_MAX_RESULTS=3\nWEB_FETCH_MAX_CHARS=2000\n"
                   "ANYSEARCH_API_KEY=anysearch-private\n", encoding="utf-8")
    monkeypatch.setenv("WEB_FETCH_ENABLED", "false")
    loaded = load_config(env)
    assert loaded.web.search.provider == "brave"
    assert loaded.web.search.max_results == 3
    assert loaded.web.fetch.enabled is False
    assert loaded.web.fetch.max_chars == 2000
    assert "brave-private" not in repr(loaded)
    assert loaded.web.search.anysearch_api_key.get_secret_value() == "anysearch-private"
    assert "anysearch-private" not in repr(loaded)
    clear_config_cache()


@pytest.mark.asyncio
async def test_search_fetch_answer_chain_and_restore_without_permanent_runtime_prompt(tmp_path):
    class Search:
        async def post(self, url, **kwargs):
            assert url == "https://api.anysearch.com/v1/search"
            assert "Time range:" in kwargs["json"]["query"]
            return WebDocument(url, json.dumps({"code": 0, "data": {"results": [
                {"title": "官方信息", "url": "https://example.com/news", "snippet": "近期更新"},
            ]}}).encode(), "application/json")

    class Fetch:
        async def fetch(self, url):
            assert url == "https://example.com/news"
            return FetchResponse(url=url, final_url=url, title="官方信息", content_type="text/html", content="已核实的网页正文")

    class Provider(ModelProvider):
        def __init__(self):
            self.requests = []

        async def generate(self, request):
            self.requests.append(request)
            assert "当前本机时间" not in request.messages[0].content
            assert any("当前本机时间" in message.request_content for message in request.messages if message.role == "user")
            assert "不可信资料" in request.messages[0].content
            if len(self.requests) == 1:
                call = ModelToolCall(id="search", name="web_search", arguments='{"query":"近期更新","freshness":"week"}')
            elif len(self.requests) == 2:
                results = json.loads(request.messages[-1].content)
                assert results["results"][0]["url"] == "https://example.com/news"
                call = ModelToolCall(id="fetch", name="web_fetch", arguments='{"url":"https://example.com/news"}')
            else:
                assert "已核实的网页正文" in request.messages[-1].content
                return ModelResponse(content="近期更新已核实。[官方信息](https://example.com/news)", model="fake", finish_reason="stop")
            return ModelResponse(content="", model="fake", finish_reason="tool_calls", tool_calls=(call,))

        async def aclose(self):
            pass

    settings = config()
    registry = ToolRegistry([WebSearchTool(settings.web.search, service=AnySearch(settings.web.search, client=Search())),
                             WebFetchTool(settings.web.fetch, fetcher=Fetch())])
    store = SessionStore(tmp_path)
    async with AgentRunner(settings, Provider(), store, tools=registry) as runner:
        await runner.ask("最近有什么更新？", on_stream=lambda event: None)
        assert [message.role for message in runner.context.history] == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
        assert runner.context.messages[0].content == settings.agent.system_prompt
        assert runner.last_usage.report().calls == 3
        saved = store.load(runner.session_id)
        assert [event.kind for event in saved.events] == ["tool", "tool", "turn"]
    async with AgentRunner(settings, Provider(), store, saved, tools=registry) as restored:
        assert len(restored.context.history) == 6
        assert restored.context.history[-1].content.endswith("(https://example.com/news)")
