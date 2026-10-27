class ConfigError(Exception):
    """配置模块的基础异常。"""


class ConfigLoadError(ConfigError):
    """配置文件或环境变量无法正确加载。"""


class ConfigValidationError(ConfigError):
    """配置字段缺失或内容不符合要求。"""