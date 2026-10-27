from lynx_agent.tools import CalculatorTool, ToolRegistry
from lynx_agent.tools.builtin.web_fetch import WebFetchTool
from lynx_agent.tools.builtin.web_search import WebSearchTool


def search_unavailable_reason(config) -> str | None:
    if not config.enabled:
        return "配置已关闭"
    if config.provider == "brave" and (config.api_key is None or not config.api_key.get_secret_value().strip()):
        return "缺少 BRAVE_SEARCH_API_KEY；未发送请求"
    return None


def create_default_tools(config, *, file_access=None) -> ToolRegistry:
    from lynx_agent.filesystem import FileAccess
    from lynx_agent.tools.builtin.filesystem import create_file_tools
    registry = ToolRegistry([CalculatorTool()])
    for tool in create_file_tools(file_access or FileAccess()):
        registry.register(tool)
    if search_unavailable_reason(config.web.search) is None:
        registry.register(WebSearchTool(config.web.search))
    if config.web.fetch.enabled:
        registry.register(WebFetchTool(config.web.fetch))
    return registry
