import socket

import pytest

from lynx_agent.web.errors import UnsafeURLError
from lynx_agent.web.security import PublicResolver, check_ip, public_url


@pytest.mark.parametrize("url", [
    "file:///etc/passwd", "ftp://example.com/a", "http://127.0.0.1", "http://10.0.0.1",
    "http://169.254.169.254/latest", "http://100.100.100.200", "http://localhost",
    "http://a.local", "http://metadata.google.internal", "http://[::1]", "http://[::ffff:127.0.0.1]",
    "http://2130706433", "http://0177.0.0.1", "http://0x7f000001", "http://%31%32%37.0.0.1",
    "http://example.com:8080", "https://user:secret@example.com", "https://example.com\\@127.0.0.1",
    "https://example.com/\npath", "http://[fe80::1%25eth0]", "http://[2002:7f00:1::]",
    "http://224.0.0.1", "http://0.0.0.0", "http://192.168.1.1", "https://printer",
])
def test_unsafe_targets_rejected(url):
    with pytest.raises(UnsafeURLError):
        public_url(url)


def test_public_urls_normalized_without_fragment():
    assert str(public_url("https://Example.COM./article#section")) == "https://example.com/article"
    assert str(public_url("https://cafe.be/news")) == "https://cafe.be/news"
    check_ip("8.8.8.8")
    check_ip("2606:4700:4700::1111")


@pytest.mark.asyncio
async def test_resolver_rejects_mixed_public_private_and_rebinding(monkeypatch):
    responses = [["8.8.8.8"], ["8.8.8.8", "127.0.0.1"]]
    calls = []

    async def lookup(host, port, **kwargs):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port)) for ip in responses.pop(0)]

    import asyncio
    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", lookup)
    resolver = PublicResolver()
    resolved = await resolver.resolve("example.com", 443)
    assert resolved[0]["host"] == "8.8.8.8"
    assert resolved[0]["hostname"] == "example.com"
    assert resolved[0]["flags"] == socket.AI_NUMERICHOST
    with pytest.raises(UnsafeURLError):
        await resolver.resolve("example.com", 443)
    assert calls == ["example.com", "example.com"]
