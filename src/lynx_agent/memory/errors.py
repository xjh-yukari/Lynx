"""不包含服务原始异常或密钥的记忆边界错误。"""

from lynx_agent.tools.errors import ToolExecutionError


class MemoryError(ToolExecutionError):
    """拒绝、不确定写入或后端不可用。"""


class MemoryNotFound(MemoryError):
    """已连接服务明确报告记录不存在。"""
