"""模型只能提出参数，不能开启Shell或批准执行。"""

from pydantic import ValidationError

from lynx_agent.execution.models import CommandArguments
from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolValidationError


class ExecCommandTool(Tool):
    name = "exec_command"
    description = ("请求人工逐次审批后在本机运行程序参数数组，不经过Shell解释；不是沙箱。"
                   "禁止交互、管道、重定向、后台常驻和自动安装依赖。被拒绝后不能换写法绕过。"
                   "非零退出码是实际执行结果，不代表工具调用失败；不要自动重跑。")

    def __init__(self, service):
        self.service = service

    @property
    def available(self):
        return self.service.available

    @property
    def parameters(self):
        schema = CommandArguments.model_json_schema()
        # DeepSeek最小参数形式：可选字段可省略，超时不是任意JSON值。
        schema["properties"]["timeout_seconds"] = {"type": "integer", "minimum": 1, "maximum": 300, "description": "默认60秒，最大300秒"}
        return schema

    def validate_arguments(self, arguments):
        try:
            CommandArguments.model_validate(arguments)
        except ValidationError:
            raise ToolValidationError("命令参数无效；需非空argv数组，超时1–300秒，不支持拼接或重定向") from None

    async def execute_result(self, arguments):
        self.validate_arguments(arguments)
        return await self.service.execute(arguments)

    async def execute(self, **arguments):
        return (await self.execute_result(arguments)).content
