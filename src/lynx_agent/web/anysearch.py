"""AnySearch 官方接口；默认匿名，可选独立搜索密钥。"""

import asyncio
import json
from datetime import datetime, timedelta

from lynx_agent.web.errors import UnsafeURLError, WebError
from lynx_agent.web.extractor import clean_text
from lynx_agent.web.http import SafeHTTPClient
from lynx_agent.web.models import SearchResponse, SearchResult
from lynx_agent.web.security import public_url


class AnySearch:
    def __init__(self, config, *, client=None):
        self.key = config.anysearch_api_key
        self.client = client or SafeHTTPClient(timeout=config.timeout_seconds, max_bytes=1048576, max_redirects=0)

    async def search(self, query, count, freshness=None):
        search_query = query
        note = None
        if freshness:
            days = {"day": 1, "week": 7, "month": 30, "year": 365}[freshness]
            now = datetime.now().astimezone()
            start = now - timedelta(days=days)
            search_query = f"{query}\nTime range: {start.isoformat(timespec='seconds')} to {now.isoformat(timespec='seconds')}"
            note = "AnySearch 通用接口无已确认的原生时间过滤；时间范围仅加入查询提示，不保证结果在范围内，需阅读原文核实日期。"
        headers = {"Accept": "application/json", "X-Anysearch-Client": "lynx-agent/0.1.0"}
        if self.key and self.key.get_secret_value().strip():
            headers["Authorization"] = f"Bearer {self.key.get_secret_value()}"
        document = await self.client.post(
            "https://api.anysearch.com/v1/search",
            json={"query": search_query, "max_results": count},
            headers=headers, content_types={"application/json"},
        )
        return await asyncio.to_thread(self._parse, document, query, count, note)

    @staticmethod
    def _parse(document, query, count, note=None):
        try:
            data = json.loads(document.body)
            if not isinstance(data, dict) or type(data.get("code")) is not int or data["code"] != 0:
                raise ValueError
            payload = data.get("data")
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise TypeError
            items = payload["results"]
            results = []
            seen = set()
            for item in items:
                if not isinstance(item, dict) or not isinstance(item.get("url"), str) or not isinstance(item.get("title"), str):
                    continue
                try:
                    url = str(public_url(item["url"]))
                except (UnsafeURLError, ValueError):
                    continue
                if url in seen:
                    continue
                seen.add(url)
                snippet = item.get("snippet") or item.get("content") or ""
                results.append(SearchResult(
                    title=clean_text(item["title"])[:300], url=url,
                    snippet=clean_text(snippet)[:1000] if isinstance(snippet, str) else "",
                ))
                if len(results) >= count:
                    break
            if items and not results:
                raise ValueError
            return SearchResponse(query=query, provider="anysearch", results=results, freshness_note=note)
        except (ValueError, TypeError, AttributeError, RecursionError):
            # 不回传服务 message、原始响应或可能生成的临时账号/密码/API Key。
            raise WebError("AnySearch 返回业务错误或无效搜索数据，未回传原始响应；请检查服务状态和额度") from None
