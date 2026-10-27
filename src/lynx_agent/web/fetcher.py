from lynx_agent.web.extractor import extract
from lynx_agent.web.http import SafeHTTPClient
from lynx_agent.web.security import public_url


class WebFetcher:
    def __init__(self, config, *, client=None):
        self.config = config
        self.client = client or SafeHTTPClient(timeout=config.timeout_seconds, max_bytes=config.max_response_bytes,
                                               max_redirects=config.max_redirects)

    async def fetch(self, url):
        target = str(public_url(url))
        document = await self.client.get(target, content_types={"text/html", "application/xhtml+xml", "text/plain", "application/json"})
        return await asyncio.to_thread(extract, document, target, self.config.max_chars)
import asyncio
