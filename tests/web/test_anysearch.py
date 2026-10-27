import json
from importlib.metadata import version

import pytest

from lynx_agent.config import WebSearchConfig
from lynx_agent.web.anysearch import AnySearch
from lynx_agent.web.errors import WebError
from lynx_agent.web.http import WebDocument
from lynx_agent.web.search import create_search_service


class Client:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return WebDocument(url, json.dumps(self.payload).encode(), "application/json")


@pytest.mark.asyncio
async def test_anonymous_search_uses_fixed_endpoint_and_normalizes_results():
    client = Client({"code": 0, "data": {"results": [
        {"title": "文档 &amp; 信息", "url": "https://example.com/doc", "snippet": "摘要<b>内容</b>"},
        {"title": "重复", "url": "https://example.com/doc#part"},
        {"title": "私网", "url": "http://127.0.0.1/private"},
        {"title": "正文回退", "url": "https://docs.python.org/", "content": "正文" * 1000},
        {"title": "超出数量", "url": "https://www.python.org/"},
    ]}})
    service = create_search_service(WebSearchConfig(), client)
    assert isinstance(service, AnySearch)
    result = await service.search("近期文档", 2)
    assert [row.url for row in result.results] == ["https://example.com/doc", "https://docs.python.org/"]
    assert result.results[0].title == "文档 & 信息"
    assert result.results[0].snippet == "摘要 内容"
    assert len(result.results[1].snippet) == 1000
    assert result.results[0].published_at is None
    assert result.searched_at.tzinfo is not None
    assert result.freshness_note is None
    endpoint, request = client.calls[0]
    assert endpoint == "https://api.anysearch.com/v1/search"
    assert request["json"] == {"query": "近期文档", "max_results": 2}
    assert "Authorization" not in request["headers"]
    assert request["headers"]["X-Anysearch-Client"] == f"lynx-agent/{version('lynx-agent')}"


@pytest.mark.asyncio
async def test_optional_key_is_separate_from_brave_and_not_returned():
    client = Client({"code": 0, "message": "sensitive ignored", "data": {"results": []}})
    config = WebSearchConfig(api_key="brave-secret", anysearch_api_key="anysearch-secret")
    result = await AnySearch(config, client=client).search("test", 3)
    assert client.calls[0][1]["headers"]["Authorization"] == "Bearer anysearch-secret"
    assert "secret" not in repr(config)
    assert "secret" not in result.model_dump_json()
    assert "sensitive" not in result.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("freshness", ["day", "week", "month", "year"])
async def test_freshness_is_query_hint_not_claimed_as_filter(freshness):
    client = Client({"code": 0, "data": {"results": []}})
    result = await AnySearch(WebSearchConfig(), client=client).search("news", 3, freshness)
    assert client.calls[0][1]["json"]["query"].startswith("news\nTime range: ")
    assert "freshness" not in client.calls[0][1]["json"]
    assert "不保证" in result.freshness_note
    assert result.query == "news"


@pytest.mark.parametrize("payload", [None, [], {}, {"code": False, "data": {"results": []}},
    {"code": -1, "message": "password=secret api_key=secret"},
    {"code": 0, "data": None}, {"code": 0, "data": {"results": {}}},
    {"code": 0, "data": {"results": [{"title": "unsafe", "url": "file:///secret"}]}},
])
def test_business_errors_or_malformed_responses_never_leak_message(payload):
    document = WebDocument("https://api.anysearch.com/v1/search", json.dumps(payload).encode(), "application/json")
    with pytest.raises(WebError) as caught:
        AnySearch._parse(document, "test", 3)
    assert "secret" not in str(caught.value)


def test_invalid_json_is_safe():
    with pytest.raises(WebError):
        AnySearch._parse(WebDocument("https://api.anysearch.com", b"not-json-private", "application/json"), "test", 3)
