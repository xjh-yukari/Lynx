import asyncio
import gzip

import aiohttp
import pytest

from lynx_agent.web.errors import UnsafeURLError, WebError
from lynx_agent.web.http import SafeHTTPClient
from lynx_agent.web.security import PublicResolver


class Content:
    def __init__(self, body, gate=None):
        self.body = body
        self.gate = gate
        self.started = False

    async def iter_chunked(self, size):
        self.started = True
        if self.gate is not None:
            await self.gate.wait()
        for offset in range(0, len(self.body), size):
            yield self.body[offset:offset + size]


class Response:
    def __init__(self, body=b"text", *, status=200, headers=None, media="text/plain", gate=None):
        self.status = status
        self.headers = headers or {}
        self.content_type = media
        self.charset = "utf-8"
        self.content_length = None
        self.content = Content(body, gate)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True


class Session:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def get(self, url, **kwargs):
        self.calls.append((str(url), kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def post(self, url, **kwargs):
        return self.get(url, **kwargs)


def mock_session(monkeypatch, replies):
    captured = {}
    session = Session(replies)

    def connector(**kwargs):
        captured["connector"] = kwargs
        return object()

    def client(**kwargs):
        captured["session"] = kwargs
        return session

    monkeypatch.setattr(aiohttp, "TCPConnector", connector)
    monkeypatch.setattr(aiohttp, "ClientSession", client)
    return session, captured


@pytest.mark.asyncio
async def test_get_is_pinned_no_proxy_no_cookies_and_closes(monkeypatch):
    response = Response()
    session, captured = mock_session(monkeypatch, [response])
    document = await SafeHTTPClient().get("https://example.com/a", params={"q": "测试"}, content_types={"text/plain"})
    assert document.body == b"text"
    assert isinstance(captured["connector"]["resolver"], PublicResolver)
    assert captured["connector"]["use_dns_cache"] is False
    assert captured["session"]["trust_env"] is False
    assert captured["session"]["auto_decompress"] is False
    assert isinstance(captured["session"]["cookie_jar"], aiohttp.DummyCookieJar)
    assert session.calls[0][1]["allow_redirects"] is False
    assert session.closed and response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["http://127.0.0.1/private", "//169.254.169.254/latest", "file:///secret"])
async def test_redirect_cannot_enter_private_network(monkeypatch, location):
    response = Response(status=302, headers={"Location": location})
    session, _ = mock_session(monkeypatch, [response])
    with pytest.raises(UnsafeURLError):
        await SafeHTTPClient().get("https://example.com")
    assert len(session.calls) == 1
    assert session.closed and response.closed


@pytest.mark.asyncio
async def test_relative_redirect_and_redirect_limit(monkeypatch):
    session, _ = mock_session(monkeypatch, [Response(status=302, headers={"Location": "/next"}), Response()])
    document = await SafeHTTPClient().get("https://example.com/start")
    assert document.url == "https://example.com/next"
    session, _ = mock_session(monkeypatch, [Response(status=302, headers={"Location": "/again"}) for _ in range(2)])
    with pytest.raises(WebError, match="重定向"):
        await SafeHTTPClient(max_redirects=1).get("https://example.com")
    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_api_key_is_not_forwarded_on_redirect(monkeypatch):
    session, _ = mock_session(monkeypatch, [Response(status=302, headers={"Location": "https://other.example/path"})])
    with pytest.raises(WebError):
        await SafeHTTPClient().get("https://api.search.brave.com/a", headers={"X-Subscription-Token": "secret"})
    assert len(session.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["plain", "gzip_bomb", "header", "media"])
async def test_size_decompression_and_type_limits(monkeypatch, kind):
    response = Response(b"x" * 2000)
    if kind == "gzip_bomb":
        response = Response(gzip.compress(b"x" * 100000), headers={"Content-Encoding": "gzip"})
    elif kind == "header":
        response.content_length = 2000
    elif kind == "media":
        response.content_type = "application/pdf"
    session, _ = mock_session(monkeypatch, [response])
    with pytest.raises(WebError):
        await SafeHTTPClient(max_bytes=1024).get("https://example.com", content_types={"text/plain"})
    assert session.closed and response.closed
    if kind in {"header", "media"}:
        assert not response.content.started


@pytest.mark.asyncio
async def test_valid_gzip_is_decoded(monkeypatch):
    mock_session(monkeypatch, [Response(gzip.compress(b"hello"), headers={"Content-Encoding": "gzip"})])
    assert (await SafeHTTPClient().get("https://example.com")).body == b"hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_and_cancel_close_active_response(monkeypatch, cancel):
    response = Response(gate=asyncio.Event())
    session, _ = mock_session(monkeypatch, [response])
    task = asyncio.create_task(SafeHTTPClient(timeout=1 if cancel else 0.01).get("https://example.com"))
    while not response.content.started:
        await asyncio.sleep(0)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else WebError):
        await task
    assert session.closed and response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 402, 403, 429, 500])
async def test_http_errors_do_not_read_or_leak_error_body(monkeypatch, status):
    response = Response(b"private-key-server-details", status=status)
    session, _ = mock_session(monkeypatch, [response])
    with pytest.raises(WebError) as caught:
        await SafeHTTPClient().get("https://example.com")
    assert "private-key" not in str(caught.value)
    assert not response.content.started
    assert len(session.calls) == 1


@pytest.mark.asyncio
async def test_post_json_preserves_network_guards_and_closes(monkeypatch):
    response = Response(b'{"code":0}', media="application/json")
    session, captured = mock_session(monkeypatch, [response])
    result = await SafeHTTPClient().post("https://api.anysearch.com/v1/search", json={"query": "test"},
                                         content_types={"application/json"})
    assert result.body == b'{"code":0}'
    assert session.calls[0][1]["json"] == {"query": "test"}
    assert session.calls[0][1]["allow_redirects"] is False
    assert captured["session"]["trust_env"] is False
    assert isinstance(captured["connector"]["resolver"], PublicResolver)
    assert session.closed and response.closed


@pytest.mark.asyncio
async def test_post_never_redirects_even_without_auth(monkeypatch):
    response = Response(status=307, headers={"Location": "https://other.example/"})
    session, _ = mock_session(monkeypatch, [response])
    with pytest.raises(WebError, match="重定向"):
        await SafeHTTPClient().post("https://api.anysearch.com/v1/search", json={"query": "test"})
    assert len(session.calls) == 1
    assert session.closed and response.closed


@pytest.mark.asyncio
async def test_post_quota_response_never_read_or_credentials_saved(monkeypatch):
    response = Response(b'username=user password=secret api_key=secret', status=402)
    session, _ = mock_session(monkeypatch, [response])
    with pytest.raises(WebError, match="额度") as caught:
        await SafeHTTPClient().post("https://api.anysearch.com/v1/search", json={"query": "test"})
    assert "secret" not in str(caught.value)
    assert not response.content.started
    assert session.closed and response.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_post_timeout_or_cancel_closes_response(monkeypatch, cancel):
    response = Response(gate=asyncio.Event(), media="application/json")
    session, _ = mock_session(monkeypatch, [response])
    task = asyncio.create_task(SafeHTTPClient(timeout=1 if cancel else 0.01).post(
        "https://api.anysearch.com/v1/search", json={"query": "test"}))
    while not response.content.started:
        await asyncio.sleep(0)
    if cancel:
        task.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else WebError):
        await task
    assert session.closed and response.closed
