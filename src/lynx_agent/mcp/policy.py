"""本地授权和完整 JSON Schema 校验，不信任服务的只读注解。"""

import re

from jsonschema import validators
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import NoSuchResource

from lynx_agent.tools.errors import ToolValidationError


def tool_name(server, remote):
    name = f"mcp__{server}__{remote}"
    if len(name) > 64 or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name) is None:
        raise ValueError("MCP 工具名称不兼容（不自动改名）")
    return name


def _deny_remote(uri):
    raise NoSuchResource(ref=uri)


def schema_validator(schema):
    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(item, str) or not item.startswith("#")
                ):
                    raise ValueError("不支持外部 Schema 引用")
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)

    try:
        check(schema)
        if schema.get("type") != "object":
            raise ValueError("工具参数必须为 object")
        cls = validators.validator_for(schema)
        if schema.get("$schema") and schema["$schema"] not in validators.SPECIFICATIONS:
            raise ValueError("不支持的 Schema 版本")
        cls.check_schema(schema)
        return cls(schema, registry=Registry(retrieve=_deny_remote))
    except (SchemaError, TypeError, RecursionError) as error:
        raise ValueError("MCP 参数 Schema 不合法") from error


def validate_arguments(validator, arguments):
    try:
        validator.validate(arguments)
    except Exception as error:  # noqa: BLE001 - 引用解析异常也必须拒绝执行
        if isinstance(error, ValidationError):
            raise ToolValidationError("MCP 参数不符合完整 JSON Schema") from None
        raise ToolValidationError("MCP 参数 Schema 无法安全解析") from None
