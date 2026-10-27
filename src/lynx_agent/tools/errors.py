"""工具模块异常。"""


class ToolError(Exception):
    """所有工具异常的基类。"""


class ToolRegistrationError(ToolError):
    """工具名称冲突或定义无效。"""


class ToolNotFoundError(ToolError):
    """请求了尚未注册的工具。"""


class ToolValidationError(ToolError):
    """工具参数不符合 JSON Schema。"""


class ToolExecutionError(ToolError):
    """工具执行期间发生错误。"""
