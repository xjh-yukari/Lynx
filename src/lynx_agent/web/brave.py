"""Brave 官方 Web Search API，密钥仅发送给固定官方端点。"""

import asyncio
import json

from lynx_agent.web.errors import UnsafeURLError, WebError
from lynx_agent.web.extractor import clean_text
from lynx_agent.web.http import SafeHTTPClient
from lynx_agent.web.models import SearchResponse, SearchResult
from lynx_agent.web.security import public_url


class BraveSearch:
    def __init__(self, config, *, client=None):
        self.key = config.api_key
        self.client = client or SafeHTTPClient(timeout=config.timeout_seconds, max_bytes=1048576, max_redirects=0)

    async def search(self, query, count, freshness=None):
        params = {"q": query, "count": count}
        if freshness:
            params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[freshness]
        document = await self.client.get("https://api.search.brave.com/res/v1/web/search", params=params,
                                         headers={"X-Subscription-Token": self.key.get_secret_value()},
                                         content_types={"application/json"})
        return await asyncio.to_thread(self._parse, document, query, count)

    @staticmethod
    def _parse(document, query, count):
        try:
            data = json.loads(document.body)
            items = data.get("web", {}).get("results", [])
            if not isinstance(items, list):
                raise TypeError
            results = []
            seen = set()
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                    continue
                try:
                    url = str(public_url(item["url"]))
                except UnsafeURLError:
                    continue
                if url in seen:
                    continue
                seen.add(url)
                date = item.get("page_age")
                results.append(SearchResult(title=clean_text(str(item.get("title", "")))[:300], url=url,
                                             snippet=clean_text(str(item.get("description", "")))[:1000],
                                             published_at=date[:100] if isinstance(date, str) else None))
                if len(results) == count:
                    break
            return SearchResponse(query=query, provider="brave", results=results)
        except (ValueError, TypeError, AttributeError):
            raise WebError("Brave 返回了无法解析的搜索结果，未回传原始响应") from None
