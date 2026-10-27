"""模型调用的公共异常，不携带服务端原始响应。"""


class ProviderError(Exception):
    """模型供应商错误。"""


class ProviderConfigurationError(ProviderError):
    """供应商或模型配置不受支持。"""


class ProviderAuthenticationError(ProviderError):
    """认证失败。"""


class ProviderRateLimitError(ProviderError):
    """供应商限制请求或额度。"""


class ProviderTimeoutError(ProviderError):
    """请求超时。"""


class ProviderConnectionError(ProviderError):
    """无法连接供应商。"""


class ProviderResponseError(ProviderError):
    """请求遭拒或响应不符合当前文本接口。"""


class ProviderToolMarkupError(ProviderResponseError):
    """工具协议标记泄漏到正文；不得解释为可执行调用。"""


class ProviderContextOverflowError(ProviderResponseError):
    """供应商明确确认上下文溢出；不用于其他 HTTP 400 错误。"""
