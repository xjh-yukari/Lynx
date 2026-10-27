"""按编号分页读取当前会话存档；不授予任何文件权限。"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolValidationError


class ReadResultArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    result_id: str = Field(min_length=36, max_length=36)
    offset: int = Field(default=0, ge=0, le=1048576)
    limit: int = Field(default=4000, ge=1, le=4000)


class ReadToolResultTool(Tool):
    name = "read_tool_result"
    description = "按 result_id 分页查看当前会话工具原始快照；offset 为字符偏移，最多 4000 字符；文件结果重新校验当前权限。"
    read_only = True

    def __init__(self, archive):
        self.archive = archive

    @property
    def parameters(self):
        return ReadResultArguments.model_json_schema()

    @staticmethod
    def _parse(arguments):
        try:
            parsed = ReadResultArguments.model_validate(arguments)
            UUID(parsed.result_id)
            return parsed
        except (ValidationError, ValueError):
            raise ToolValidationError("存档参数不合法：有效 result_id、非负 offset、1–4000 的 limit；不接受路径或授权参数") from None

    def validate_arguments(self, arguments):
        self._parse(arguments)

    async def execute(self, **arguments):
        parsed = self._parse(arguments)
        return self.archive.read(parsed.result_id, parsed.offset, parsed.limit)
