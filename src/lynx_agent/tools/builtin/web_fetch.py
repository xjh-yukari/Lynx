from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError
from lynx_agent.web.errors import WebError
from lynx_agent.web.fetcher import WebFetcher


class FetchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    url: str = Field(min_length=1, max_length=4096)


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "读取已知公网 URL 的网页正文以核实来源；不执行脚本、不绕过登录或验证码。失败时说明限制，不得伪称已读全文。"

    def __init__(self, config, *, fetcher=None):
        self.fetcher = fetcher or WebFetcher(config)

    @property
    def read_only(self):
        return True

    @property
    def parameters(self):
        return {"type": "object", "properties": {"url": {"type": "string", "maxLength": 4096}},
                "required": ["url"], "additionalProperties": False}

    def validate_arguments(self, arguments):
        try:
            FetchArguments.model_validate(arguments)
        except ValidationError:
            raise ToolValidationError("网页读取参数不合法：必须提供长度不超过 4096 的 url 字符串") from None

    async def execute(self, **arguments):
        self.validate_arguments(arguments)
        try:
            result = await self.fetcher.fetch(arguments["url"])
            return result.model_dump_json()
        except WebError as error:
            raise ToolExecutionError(str(error)) from None
