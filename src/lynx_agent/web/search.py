"""搜索服务契约和选择，服务凭据不进入模型工具参数。"""

from typing import Protocol

from lynx_agent.config.schema import WebSearchConfig
from lynx_agent.web.errors import WebError
from lynx_agent.web.models import SearchResponse


class SearchService(Protocol):
    async def search(self, query: str, count: int, freshness: str | None = None) -> SearchResponse: ...


def create_search_service(config: WebSearchConfig, client=None) -> SearchService:
    if config.provider == "anysearch":
        from lynx_agent.web.anysearch import AnySearch
        return AnySearch(config, client=client)
    if config.api_key is None or not config.api_key.get_secret_value().strip():
        raise WebError("Brave 搜索缺少 BRAVE_SEARCH_API_KEY，请先配置后重启")
    from lynx_agent.web.brave import BraveSearch
    return BraveSearch(config, client=client)
