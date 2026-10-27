"""工具抽象接口和基础参数验证。"""

from abc import ABC, abstractmethod
from typing import Any

from lynx_agent.tools.errors import ToolValidationError
from lynx_agent.tools.schema import ToolDefinition


class Tool(ABC):
    """所有 Agent 工具必须遵守的接口。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型调用工具时使用的唯一名称。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """帮助模型判断何时使用工具的说明。"""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """描述工具参数的 JSON Schema。"""

    @property
    def available(self) -> bool:
        """运行时权限是否允许向模型提供此工具。"""
        return True

    @property
    def read_only(self) -> bool:
        """工具是否不会修改外部状态。"""
        return False

    @property
    def definition(self) -> ToolDefinition:
        """返回供应商无关的工具定义。"""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """验证最小工具 Schema 支持的对象参数。"""
        schema = self.parameters
        if schema.get("type") != "object":
            raise ToolValidationError(f"工具 {self.name!r} 的参数 Schema 必须是 object")

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        missing = [name for name in required if name not in arguments]
        if missing:
            raise ToolValidationError(f"缺少必填参数：{', '.join(missing)}")

        if schema.get("additionalProperties") is False:
            unknown = [name for name in arguments if name not in properties]
            if unknown:
                raise ToolValidationError(f"存在未知参数：{', '.join(unknown)}")

        for name, value in arguments.items():
            property_schema = properties.get(name)
            if property_schema is None:
                continue
            self._validate_type(name, value, property_schema.get("type"))

    @staticmethod
    def _validate_type(name: str, value: Any, expected: str | None) -> None:
        """验证常用 JSON 基础类型，不执行隐式类型转换。"""
        type_map: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        python_type = type_map.get(expected or "")
        if python_type is None:
            return
        if expected in {"integer", "number"} and isinstance(value, bool):
            raise ToolValidationError(f"参数 {name!r} 必须是 {expected}")
        if not isinstance(value, python_type):
            raise ToolValidationError(f"参数 {name!r} 必须是 {expected}")

    @abstractmethod
    async def execute(self, **arguments: Any) -> str:
        """执行工具并返回可发送给模型的文本。"""
