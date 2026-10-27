"""工具注册表和内置计算器测试。"""

import pytest

from lynx_agent.tools import (
    CalculatorTool,
    ToolCall,
    ToolRegistrationError,
    ToolRegistry,
)


def test_registry_exposes_stable_definitions() -> None:
    registry = ToolRegistry([CalculatorTool()])

    assert registry.names == ("calculator",)
    definition = registry.definitions()[0]
    assert definition.name == "calculator"
    assert definition.parameters["required"] == ["expression"]
    assert definition.to_openai_schema()["type"] == "function"


@pytest.mark.asyncio
@pytest.mark.parametrize("expression", ["((10**100)**100)**100", "1e999", "(-1)**0.5"])
async def test_calculator_rejects_unsafe_numeric_results(expression):
    registry = ToolRegistry([CalculatorTool()])
    result = await registry.execute(ToolCall(id="large", name="calculator", arguments={"expression": expression}))
    assert result.is_error


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry([CalculatorTool()])

    with pytest.raises(ToolRegistrationError, match="已经注册"):
        registry.register(CalculatorTool())


@pytest.mark.asyncio
async def test_registry_executes_calculator() -> None:
    registry = ToolRegistry([CalculatorTool()])

    result = await registry.execute(
        ToolCall(id="call-1", name="calculator", arguments={"expression": "(12 + 3) * 4"})
    )

    assert result.content == "60"
    assert result.is_error is False


@pytest.mark.asyncio
async def test_registry_returns_validation_error() -> None:
    registry = ToolRegistry([CalculatorTool()])

    result = await registry.execute(
        ToolCall(id="call-2", name="calculator", arguments={"wrong": "1 + 1"})
    )

    assert result.is_error is True
    assert "缺少必填参数" in result.content


@pytest.mark.asyncio
async def test_registry_returns_unknown_tool_error() -> None:
    registry = ToolRegistry()

    result = await registry.execute(
        ToolCall(id="call-3", name="missing", arguments={})
    )

    assert result.is_error is True
    assert "找不到工具" in result.content


@pytest.mark.asyncio
async def test_calculator_rejects_python_code() -> None:
    registry = ToolRegistry([CalculatorTool()])

    result = await registry.execute(
        ToolCall(
            id="call-4",
            name="calculator",
            arguments={"expression": "__import__('os').getcwd()"},
        )
    )

    assert result.is_error is True
    assert "只能包含数字和算术运算符" in result.content
