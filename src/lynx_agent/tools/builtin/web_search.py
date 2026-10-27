from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError
from lynx_agent.web.errors import WebError
from lynx_agent.web.search import create_search_service


class SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    query: str = Field(min_length=1, max_length=512)
    count: int | None = Field(default=None, ge=1, le=10)
    freshness: Literal["day", "week", "month", "year"] | None = None

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        if not value.strip():
            raise ValueError("查询不能为空")
        return value.strip()


class WebSearchTool(Tool):
    name = "web_search"
    description = ("搜索公网资料、近期新闻和官方文档；返回来源链接和摘要，不是网页全文。"
                   "freshness 的实际效果见结果说明；AnySearch 仅将时间范围加入查询，不保证严格过滤，近期信息需读取原文核实日期。")

    def __init__(self, config, *, service=None):
        self.config = config
        self.service = service or create_search_service(config)

    @property
    def read_only(self):
        return True

    @property
    def parameters(self):
        return {"type": "object", "properties": {
            "query": {"type": "string", "minLength": 1, "maxLength": 512},
            "count": {"type": "integer", "minimum": 1, "maximum": self.config.max_results},
            "freshness": {"type": "string", "enum": ["day", "week", "month", "year"]},
        }, "required": ["query"], "additionalProperties": False}

    def _arguments(self, arguments):
        try:
            parsed = SearchArguments.model_validate(arguments)
            if parsed.count is not None and parsed.count > self.config.max_results:
                raise ValueError
            return parsed
        except (ValidationError, ValueError):
            raise ToolValidationError(f"搜索参数不合法：query 为非空字符串；count 为 1–{self.config.max_results} 整数；freshness 为 day/week/month/year") from None

    def validate_arguments(self, arguments):
        self._arguments(arguments)

    async def execute(self, **arguments):
        args = self._arguments(arguments)
        try:
            result = await self.service.search(args.query, args.count or self.config.max_results, args.freshness)
            return result.model_dump_json()
        except WebError as error:
            raise ToolExecutionError(str(error)) from None
