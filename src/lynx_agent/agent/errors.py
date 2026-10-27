"""上下文预算与压缩错误。"""


class ContextError(Exception):
    """请求超出预算或摘要无法安全提交。"""


class SummaryError(ContextError):
    """候选摘要不可用；仅软压力下允许保留原文降级。"""
