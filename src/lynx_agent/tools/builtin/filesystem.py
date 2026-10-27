"""六个文件工具；权限管理和审批只能由人通过终端指令操作。"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolValidationError


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    path: str = Field(default=".", min_length=1, max_length=4096)


class ListArguments(Arguments):
    limit: int = Field(default=50, ge=1, le=200)


class FindArguments(ListArguments):
    pattern: str = Field(min_length=1, max_length=256)


class SearchArguments(ListArguments):
    query: str = Field(min_length=1, max_length=512)
    glob: str = Field(default="*", min_length=1, max_length=256)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("关键词不能为空")
        return value


class ReadArguments(Arguments):
    path: str = Field(min_length=1, max_length=4096)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=200, ge=1, le=500)


class WriteArguments(Arguments):
    path: str = Field(min_length=1, max_length=4096)
    content: str = Field(max_length=1048576)


class EditArguments(Arguments):
    path: str = Field(min_length=1, max_length=4096)
    old_text: str = Field(min_length=1, max_length=131072)
    new_text: str = Field(max_length=131072)


class FileTool(Tool):
    name = ""
    description = ""
    arguments_model = Arguments
    writes = False

    def __init__(self, service):
        self.service = service

    @property
    def available(self):
        return not self.writes or self.service.policy.writable

    @property
    def read_only(self):
        return not self.writes

    @property
    def parameters(self):
        return self.arguments_model.model_json_schema()

    def validate_arguments(self, arguments):
        self._parse(arguments)

    def _parse(self, arguments):
        try:
            return self.arguments_model.model_validate(arguments).model_dump()
        except ValidationError:
            raise ToolValidationError("文件工具参数不合法，请检查路径、必填字段及数量限制；不接受未知字段或审批参数") from None

    async def execute(self, **arguments):
        return await self.service.invoke(self.name, self._parse(arguments))


class ListDirTool(FileTool):
    name = "list_dir"
    description = "列出允许访问目录的普通文件与子目录；排除敏感和无关目录，不递归。"
    arguments_model = ListArguments


class FindFilesTool(FileTool):
    name = "find_files"
    description = "在允许访问目录递归按文件名或通配符查找文件；深度、时间与数量有界，不读取正文。"
    arguments_model = FindArguments


class ReadFileTool(FileTool):
    name = "read_file"
    description = "读取允许访问的普通文本文件并返回行号；offset 从 1 开始，默认 200 行，最多 500 行、1 MiB。"
    arguments_model = ReadArguments


class SearchFilesTool(FileTool):
    name = "search_files"
    description = "在允许访问的文本文件中按普通关键词搜索，不区分大小写；返回路径、行号、有限片段。不执行正则或命令。"
    arguments_model = SearchArguments


class WriteFileTool(FileTool):
    name = "write_file"
    description = "在权限范围内新建 UTF-8 文本文件；父目录须存在。覆盖已有文件仅生成待用户确认的差异，不立即修改。"
    arguments_model = WriteArguments
    writes = True


class EditFileTool(FileTool):
    name = "edit_file"
    description = "精确替换 UTF-8 文件中唯一匹配的 old_text，保留其余内容；只生成差异，需用户在终端审批后写入。"
    arguments_model = EditArguments
    writes = True


def create_file_tools(service):
    return [tool(service) for tool in (ListDirTool, FindFilesTool, ReadFileTool, SearchFilesTool, WriteFileTool, EditFileTool)]
