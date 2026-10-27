"""有界公网 GET/JSON POST；无环境代理、无 Cookie、无自动重定向或错误重试。"""

import asyncio
import zlib
from dataclasses import dataclass

import aiohttp
from yarl import URL

from lynx_agent.web.errors import WebError
from lynx_agent.web.security import PublicResolver, public_url


@dataclass(frozen=True)
class WebDocument:
    url: str
    body: bytes
    content_type: str
    charset: str | None = None


async def read_bounded(response, limit: int) -> bytes:
    if response.content_length is not None and response.content_length > limit:
        raise WebError("下载内容超过大小限制")
    encoding = response.headers.get("Content-Encoding", "identity").lower().strip()
    if encoding not in {"identity", "", "gzip", "deflate"}:
        raise WebError("不支持的网页压缩格式，未读取正文")
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS) if encoding in {"gzip", "deflate"} else None
    parts = []
    raw_size = decoded_size = 0
    async for chunk in response.content.iter_chunked(16384):
        raw_size += len(chunk)
        if raw_size > limit:
            raise WebError("下载内容超过大小限制")
        try:
            data = decoder.decompress(chunk, limit - decoded_size + 1) if decoder else chunk
        except zlib.error:
            raise WebError("网页压缩内容不合法") from None
        decoded_size += len(data)
        if decoded_size > limit or (decoder and decoder.unconsumed_tail):
            raise WebError("解压后内容超过大小限制")
        parts.append(data)
    if decoder and (not decoder.eof or decoder.unused_data):
        raise WebError("压缩内容不完整或包含多段压缩流")
    return b"".join(parts)


class SafeHTTPClient:
    def __init__(self, timeout: float = 15, max_bytes: int = 2097152, max_redirects: int = 3):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_redirects = max_redirects

    async def get(self, url: str, *, params=None, headers=None, content_types=None) -> WebDocument:
        return await self._request("GET", url, params=params, headers=headers, content_types=content_types)

    async def post(self, url: str, *, json, headers=None, content_types=None) -> WebDocument:
        return await self._request("POST", url, json=json, headers=headers, content_types=content_types)

    async def _request(self, method, url, *, params=None, json=None, headers=None, content_types=None):
        target = public_url(url)
        if params:
            target = target.update_query(params)
        request_headers = {"User-Agent": "LynxAgent/0.1 (read-only web tools)", "Accept-Encoding": "identity"}
        request_headers.update(headers or {})
        try:
            async with asyncio.timeout(self.timeout):
                connector = aiohttp.TCPConnector(resolver=PublicResolver(), use_dns_cache=False, limit=4)
                async with aiohttp.ClientSession(connector=connector, trust_env=False, auto_decompress=False,
                                                cookie_jar=aiohttp.DummyCookieJar(),
                                                timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                    for hop in range(self.max_redirects + 1):
                        target = public_url(str(target))
                        request = (session.get(target, headers=request_headers, allow_redirects=False) if method == "GET"
                                   else session.post(target, json=json, headers=request_headers, allow_redirects=False))
                        async with request as response:
                            if response.status in {301, 302, 303, 307, 308}:
                                location = response.headers.get("Location")
                                if method == "POST" or headers or hop == self.max_redirects or not location:
                                    raise WebError("重定向被拒绝或超过跳转次数限制")
                                target = public_url(str(target.join(URL(location))))
                                continue
                            if response.status in {401, 403}:
                                raise WebError("目标拒绝访问或认证失败；请检查服务配置，不要绕过访问限制")
                            if response.status == 429:
                                raise WebError("搜索或网页服务限流，请稍后再试；未自动重试")
                            if response.status == 402:
                                raise WebError("搜索服务额度已耗尽，请在服务控制台检查额度；未读取或保存响应中的凭据")
                            if response.status >= 400:
                                raise WebError(f"目标返回 HTTP {response.status}，未读取错误正文")
                            media = response.content_type.lower()
                            if content_types and media not in content_types:
                                raise WebError("不支持的内容类型，第一版不处理 PDF、图片或文件下载")
                            body = await read_bounded(response, self.max_bytes)
                            return WebDocument(str(target), body, media, response.charset)
        except TimeoutError:
            raise WebError("联网请求超时，已关闭连接") from None
        except aiohttp.ClientError:
            raise WebError("联网连接、TLS 或响应读取失败，请检查网络；未自动重试") from None
        raise WebError("未能取得有效网页")
