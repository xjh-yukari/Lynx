import json

import pytest

from lynx_agent.config import WebFetchConfig, WebSearchConfig
from lynx_agent.tools import ToolCall, ToolRegistry, ToolValidationError
from lynx_agent.tools.builtin.web_fetch import WebFetchTool
from lynx_agent.tools.builtin.web_search import WebSearchTool
from lynx_agent.web.brave import BraveSearch
from lynx_agent.web.errors import WebError
from lynx_agent.web.fetcher import WebFetcher
from lynx_agent.web.http import WebDocument
from lynx_agent.web.search import create_search_service


class Client:
    def __init__(self, body, media="text/html"):
        self.document = WebDocument("https://example.com/article", body.encode() if isinstance(body, str) else body, media)
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.document




@pytest.mark.asyncio
async def test_brave_auth_is_separate_and_never_returned():
    client = Client(json.dumps({"web": {"results": [{"title": "文档", "url": "https://example.com/", "description": "<b>内容</b>", "page_age": "2026-10-18"}]}}), "application/json")
    config = WebSearchConfig(provider="brave", api_key="fake-brave-secret")
    service = create_search_service(config, client)
    reply = await service.search("测试", 3, "day")
    assert isinstance(service, BraveSearch)
    assert client.calls[0][0] == "https://api.search.brave.com/res/v1/web/search"
    assert client.calls[0][1]["headers"] == {"X-Subscription-Token": "fake-brave-secret"}
    assert client.calls[0][1]["params"]["freshness"] == "pd"
    assert reply.results[0].snippet == "内容"
    assert "fake-brave-secret" not in reply.model_dump_json()
    assert "fake-brave-secret" not in repr(config)


def test_missing_brave_key_is_not_silently_switched_to_another_provider():
    with pytest.raises(WebError, match="BRAVE_SEARCH_API_KEY"):
        create_search_service(WebSearchConfig(provider="brave"))


@pytest.mark.asyncio
async def test_fetch_extracts_article_without_scripts_and_marks_truncation():
    client = Client('''<html><head><title>文章标题</title><meta property="article:published_time" content="2026-10-18"></head>
    <body><nav>导航噪声</nav><article><h1>重要新闻</h1><p>''' + "正文内容" * 100 + '''</p><script>leak_secret()</script></article>
    <footer>页脚</footer></body></html>''')
    fetcher = WebFetcher(WebFetchConfig(max_chars=128), client=client)
    reply = await fetcher.fetch("https://example.com/source")
    assert reply.title == "文章标题"
    assert reply.url == "https://example.com/source"
    assert reply.final_url == "https://example.com/article"
    assert reply.published_at == "2026-10-18"
    assert len(reply.content) == 128 and reply.truncated
    assert "导航" not in reply.content and "leak_secret" not in reply.content
    assert "不可信" in reply.notice


@pytest.mark.asyncio
async def test_fetch_plain_text_and_charset_fallback():
    client = Client("纯文本\n第二行", "text/plain")
    result = await WebFetcher(WebFetchConfig(), client=client).fetch("https://example.com")
    assert result.content == "纯文本\n第二行"
    assert not result.truncated


@pytest.mark.asyncio
async def test_fetch_tool_blocks_private_url_before_client_call():
    client = Client("正文")
    tool = WebFetchTool(WebFetchConfig(), fetcher=WebFetcher(WebFetchConfig(), client=client))
    result = await ToolRegistry([tool]).execute(ToolCall(id="fetch", name="web_fetch", arguments={"url": "http://127.0.0.1"}))
    assert result.is_error
    assert not client.calls
    assert "禁止访问" in result.content


@pytest.mark.parametrize("arguments", [{"query": " "}, {"query": "x", "count": True}, {"query": "x", "count": 6},
                                      {"query": "x", "freshness": "century"}, {"query": "x", "unknown": 1}])
def test_search_parameters_are_strict(arguments):
    tool = WebSearchTool(WebSearchConfig(), service=object())
    with pytest.raises(ToolValidationError):
        tool.validate_arguments(arguments)


@pytest.mark.asyncio
async def test_errors_are_tool_results_not_fake_articles():
    class BrokenSearch:
        async def search(self, *args):
            raise WebError("搜索服务限流")

    tool = WebSearchTool(WebSearchConfig(), service=BrokenSearch())
    reply = await ToolRegistry([tool]).execute(ToolCall(id="search", name="web_search", arguments={"query": "新闻"}))
    assert reply.is_error and reply.content == "搜索服务限流"
