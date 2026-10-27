"""工具注册、查询和执行。"""

from collections.abc import Iterable

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRegistrationError,
    ToolValidationError,
)
from lynx_agent.tools.schema import ToolCall, ToolDefinition, ToolResult


class ToolRegistry:
    """集中管理 Agent 当前能够使用的工具。"""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """注册工具；重复名称视为配置错误。"""
        if tool.name in self._tools:
            raise ToolRegistrationError(f"工具 {tool.name!r} 已经注册")
        definition = tool.definition
        if definition.name != tool.name:
            raise ToolRegistrationError("工具定义名称与工具名称不一致")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """移除工具；不存在时保持幂等。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        """按精确名称查找工具。"""
        try:
            if name in self._tools and not self._tools[name].available:
                raise KeyError(name)
            return self._tools[name]
        except KeyError as error:
            available = ", ".join(self.names) or "无"
            raise ToolNotFoundError(
                f"找不到工具 {name!r}；当前可用工具：{available}"
            ) from error

    @property
    def names(self) -> tuple[str, ...]:
        """按名称排序返回工具列表，保证输出稳定。"""
        return tuple(sorted(name for name, tool in self._tools.items() if tool.available))

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """返回可发送给模型的工具定义。"""
        return tuple(self._tools[name].definition for name in self.names)

    async def execute(self, call: ToolCall) -> ToolResult:
        """验证并执行一次调用，把失败也转换成统一结果。"""
        try:
            tool = self.get(call.name)
            tool.validate_arguments(call.arguments)
            content = await tool.execute(**call.arguments)
            if not isinstance(content, str):
                raise ToolExecutionError("工具必须返回字符串")
        except TimeoutError:
            # 让统一执行器标记为超时与执行状态未知，而不是普通工具错误。
            raise
        except (ToolNotFoundError, ToolValidationError, ToolExecutionError) as error:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=str(error),
                is_error=True,
                error_kind=getattr(error, "error_kind", None),
                execution_state=getattr(error, "execution_state", "returned"),
            )
        except Exception as error:  # noqa: BLE001 - 工具边界必须隔离第三方工具异常
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=f"工具执行失败：{type(error).__name__}（内部异常详情未回传模型）",
                is_error=True,
            )
        return ToolResult(call_id=call.id, name=call.name, content=content)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names
