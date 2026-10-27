"""根据配置显式选择模型实现。"""

from lynx_agent.config.schema import Config
from lynx_agent.providers.base import ModelProvider
from lynx_agent.providers.deepseek import DeepSeekProvider
from lynx_agent.providers.errors import ProviderConfigurationError


def create_provider(config: Config) -> ModelProvider:
    """创建客户端，不发送 API 请求。"""
    if config.agent.model.provider == "deepseek":
        return DeepSeekProvider(config.providers.deepseek, config.agent.model)
    raise ProviderConfigurationError("当前只支持 deepseek 供应商")
