class WebError(Exception):
    """可安全显示给用户及模型的联网错误，不包含密钥或原始异常。"""


class UnsafeURLError(WebError):
    """访问目标未通过网络安全检查，禁止改换其他工具绕过。"""
