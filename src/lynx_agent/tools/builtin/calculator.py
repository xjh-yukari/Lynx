"""不使用 eval 的安全四则运算工具。"""

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolExecutionError

Number = int | float
BinaryOperation = Callable[[Number, Number], Number]
UnaryOperation = Callable[[Number], Number]

_BINARY_OPERATIONS: dict[type[ast.operator], BinaryOperation] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATIONS: dict[type[ast.unaryop], UnaryOperation] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class CalculatorTool(Tool):
    """计算只含数字和算术运算符的表达式。"""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "计算数学表达式，支持 +、-、*、/、//、%、** 和括号。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "需要计算的数学表达式，例如 (12 + 3) * 4。",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **arguments: Any) -> str:
        expression = str(arguments["expression"])
        if len(expression) > 200:
            raise ToolExecutionError("表达式不能超过 200 个字符")
        try:
            tree = ast.parse(expression, mode="eval")
            if sum(1 for _ in ast.walk(tree)) > 64:
                raise ValueError("表达式过于复杂")
            result = self._evaluate(tree.body)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
            raise ToolExecutionError(f"无法计算表达式：{error}") from error
        return str(result)

    def _evaluate(self, node: ast.AST) -> Number:
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("只允许数字")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("只支持有限实数结果")
            return value
        if isinstance(node, ast.BinOp):
            operation = _BINARY_OPERATIONS.get(type(node.op))
            if operation is None:
                raise ValueError("包含不支持的运算符")
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("指数绝对值不能超过 100")
            if isinstance(node.op, ast.Pow) and isinstance(left, int) and right > 0 and left.bit_length() * right > 8192:
                raise ValueError("幂运算结果超过安全大小")
            result = operation(left, right)
            if isinstance(result, int) and result.bit_length() > 8192:
                raise ValueError("运算结果超过安全大小")
            if not isinstance(result, (int, float)) or (isinstance(result, float) and not math.isfinite(result)):
                raise ValueError("只支持有限实数结果")
            return result
        if isinstance(node, ast.UnaryOp):
            operation = _UNARY_OPERATIONS.get(type(node.op))
            if operation is None:
                raise ValueError("包含不支持的一元运算符")
            return operation(self._evaluate(node.operand))
        raise ValueError("表达式中只能包含数字和算术运算符")
