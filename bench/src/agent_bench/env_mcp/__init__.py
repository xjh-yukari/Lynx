"""测试环境：业务工具、共享状态与调用记录。

管理通道是这个状态目录本身（业务状态、故障控制、调用记录三个文件），
只由 bench 主进程读写，永远不出现在被测 agent 的工具列表里。被测 agent
只能通过 stdio MCP 调用业务工具，因此拿不到用例预期与判分规则。
"""

from agent_bench.env_mcp.store import EnvStore

__all__ = ["EnvStore"]
