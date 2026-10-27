"""MCP 边界只向用户返回安全的诊断，不回显命令参数或环境变量。"""

from lynx_agent.tools.errors import ToolExecutionError


class MCPError(ToolExecutionError):
    """连接、协议或不支持的结果。"""

    def __init__(self, message, *, execution_state="returned", error_kind="mcp_error"):
        super().__init__(message)
        self.execution_state = execution_state
        self.error_kind = error_kind
