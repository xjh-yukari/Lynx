"""面向模型的只读训练基础工具；不暴露注册、审批、任意命令或SQL。"""

from uuid import UUID

from lynx_agent.tools.base import Tool
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError
from lynx_agent.training.client import TrainingError


class TrainingReadTool(Tool):
    name = ""
    description = ""
    read_only = True

    def __init__(self, service, name):
        self.service = service
        self.name = name
        self.description = {
            "training_inspect": "M1只读环境/已注册训练项目静态发现。不执行项目代码，不启动训练；入口和恢复能力均需进一步验证。",
            "training_status": "列出用户显式注册的训练项目。M1没有任务调度，不能据此声称训练正在运行。",
            "training_records": "查询指定项目的SQLite人工实验台账。人工数值未自动验证，不是训练执行证明。",
            "training_plan": "为已注册项目生成分阶段接入计划和阻塞项；计划不可执行，不代表训练获批或已运行。",
            "training_compare": "对比同项目两条人工实验记录，只算同协议名的数值差。数据和训练未核验，不能证明效果提升。",
        }[name]

    @property
    def available(self):
        return self.service.enabled

    @property
    def parameters(self):
        if self.name == "training_status":
            return {"type": "object", "properties": {}, "additionalProperties": False}
        properties = {"project_id": {"type": "string", "format": "uuid"}}
        required = ["project_id"]
        if self.name == "training_compare":
            properties["experiment_ids"] = {"type": "array", "items": {"type": "string", "format": "uuid"},
                                             "minItems": 2, "maxItems": 2, "uniqueItems": True}
            required.append("experiment_ids")
        if self.name == "training_inspect":
            properties["mode"] = {"type": "string", "enum": ["environment", "project"]}
            required = ["mode"]
        return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}

    def validate_arguments(self, arguments):
        super().validate_arguments(arguments)
        try:
            if self.name == "training_status":
                return
            if self.name == "training_inspect":
                if arguments["mode"] not in {"environment", "project"}:
                    raise ValueError
                if arguments["mode"] == "environment":
                    if "project_id" in arguments:
                        raise ValueError
                    return
            if str(UUID(arguments["project_id"])) != arguments["project_id"]:
                raise ValueError
            if self.name == "training_compare":
                values = arguments["experiment_ids"]
                if (len(values) != 2 or not all(isinstance(value, str) for value in values)
                        or len(set(values)) != 2 or any(str(UUID(value)) != value for value in values)):
                    raise ValueError
        except (ValueError, KeyError, TypeError):
            raise ToolValidationError("训练只读查询参数无效；项目必须由终端用户先注册") from None

    async def execute(self, **arguments):
        self.validate_arguments(arguments)
        operations = {"training_status": "projects", "training_records": "experiments", "training_plan": "plan",
                      "training_compare": "compare"}
        operation = operations.get(self.name, "environment" if arguments.get("mode") == "environment" else "inspect")
        payload = {"project_id": arguments["project_id"]} if "project_id" in arguments else {}
        if "experiment_ids" in arguments:
            payload["experiment_ids"] = arguments["experiment_ids"]
        try:
            return self.service.clean(await self.service.read(operation, payload))
        except TrainingError as error:
            raise ToolExecutionError(str(error)) from None


def training_tools(service):
    return [TrainingReadTool(service, name) for name in
            ("training_inspect", "training_status", "training_records", "training_plan", "training_compare")]
