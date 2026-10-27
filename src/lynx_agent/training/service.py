"""M1宿主服务：只有真实终端命令可以注册项目或登记实验。"""

import json
import shlex
from uuid import UUID, uuid4

from lynx_agent.training.client import TrainingError, WSLClient

HELP = (
    "训练助手 M1（尚不提交或控制训练）\n"
    "/training on confirm | off | status\n"
    "/training environment | projects\n"
    "/training register <WSL源码绝对路径> confirm\n"
    "/training inspect <项目ID> | plan <项目ID> | records <项目ID>\n"
    "/training progress <项目ID> | files <项目ID> [分页偏移]\n"
    "/training logs <项目ID> <相对日志路径> [距末尾字节偏移]\n"
    "/training metrics <项目ID> <相对指标路径>\n"
    "/training compare <项目ID> <基线实验ID> <候选实验ID>\n"
    "/training record <项目ID> <sft/grpo/other> <评测协议名> <指标=数值>... confirm\n"
    "登记值标注为人工报告、未验证；不会冒充已运行训练。路径有空格请加引号。"
)


class TrainingService:
    def __init__(self, config, *, client=None, writable=lambda: False, secrets=()):
        self.config = config
        self.enabled = config.enabled
        self.client = client or WSLClient(config)
        self.writable = writable
        self.secrets = tuple(value for value in secrets if value)

    def clean(self, value):
        text = json.dumps(value, ensure_ascii=False, allow_nan=False)
        for secret in sorted(self.secrets, key=len, reverse=True):
            text = text.replace(secret, "[已隐藏]")
        return text

    async def read(self, operation, arguments):
        if not self.enabled:
            raise TrainingError("训练助手未开启；使用 /training on confirm")
        if operation not in {"environment", "projects", "inspect", "plan", "experiments", "compare",
                             "progress", "logs", "metrics", "artifacts"}:
            raise TrainingError("M1模型工具仅支持只读发现、计划和查询")
        return await self._request(operation, arguments)

    async def _request(self, operation, arguments):
        try:
            return await self.client.request(operation, arguments)
        except (OSError, UnicodeError, ValueError) as error:
            if isinstance(error, TrainingError):
                raise
            raise TrainingError("训练基础请求失败；未启动训练，未回传原始异常或配置") from None

    async def command(self, argument):
        try:
            parts = shlex.split(argument)
        except ValueError:
            raise TrainingError("训练命令引号不完整") from None
        if not parts or parts == ["status"]:
            return f"训练助手：{'开启' if self.enabled else '关闭'}；阶段M1，持久训练调度/暂停续训尚未开放。\n" + HELP
        if parts == ["off"]:
            self.enabled = False
            return "训练助手已关闭；不会停止任何已有训练。"
        if parts == ["on", "confirm"]:
            self.enabled = True
            return "训练助手 M1 已开启；支持已授权项目只读监测，无需逐次审批；不会启动训练。"
        if parts[0] in {"submit", "pause", "resume", "cancel"}:
            raise TrainingError("M1尚未实现训练审批、持久调度和框架续训；没有执行任务")
        if not self.enabled:
            raise TrainingError("请先 /training on confirm")
        if len(parts) == 1 and parts[0] in {"environment", "projects"}:
            return self.clean(await self.read(parts[0], {}))
        if parts[0] in {"progress", "files", "logs", "metrics"}:
            valid = {"progress": {2}, "files": {2, 3}, "logs": {3, 4}, "metrics": {3}}
            if len(parts) not in valid[parts[0]]:
                raise TrainingError(HELP)
            payload = {"project_id": str(UUID(parts[1]))}
            if parts[0] in {"logs", "metrics"}:
                payload["path"] = parts[2]
            if parts[0] == "files" and len(parts) == 3:
                payload["offset"] = int(parts[2])
            if parts[0] == "logs" and len(parts) == 4:
                payload["offset"] = int(parts[3])
            return self.clean(await self.read("artifacts" if parts[0] == "files" else parts[0], payload))
        if len(parts) == 2 and parts[0] in {"inspect", "plan", "records"}:
            identifier = str(UUID(parts[1]))
            return self.clean(await self.read("experiments" if parts[0] == "records" else parts[0], {"project_id": identifier}))
        if len(parts) == 4 and parts[0] == "compare":
            return self.clean(await self.read("compare", {"project_id": str(UUID(parts[1])),
                                                          "experiment_ids": [str(UUID(value)) for value in parts[2:]]}))
        if parts[0] in {"register", "record"}:
            if parts[-1] != "confirm":
                return "此操作会写入WSL本地SQLite；只有你明确登记的目录才会允许后续静态读取。\n" + HELP
            if not self.writable():
                raise TrainingError("仅查看或暂停执行状态禁止注册项目与登记实验")
            if parts[0] == "register" and len(parts) == 3:
                if not parts[1].startswith("/"):
                    raise TrainingError("M1注册请使用WSL内绝对路径；不会猜测Windows盘符映射")
                result = await self._request("register", {"source": parts[1], "distribution": self.config.distribution})
                return self.clean(result)
            if parts[0] == "record" and len(parts) >= 6:
                try:
                    metrics = {}
                    for metric in parts[4:-1]:
                        name, value = metric.split("=", 1)
                        if name in metrics:
                            raise ValueError
                        metrics[name] = float(value)
                    payload = {"project_id": str(UUID(parts[1])), "method": parts[2], "protocol": parts[3],
                               "metrics": metrics, "evidence": "terminal_user_reported_not_verified",
                               "experiment_id": str(uuid4())}
                    # 将已知密钥排除在台账外，而不仅在输出时隐藏。
                    raw = json.dumps(payload, allow_nan=False)
                    if any(secret in raw for secret in self.secrets):
                        raise TrainingError("登记内容含已知密钥，未保存")
                except (ValueError, OverflowError):
                    raise TrainingError("实验登记需唯一指标名和有限数值，不接受NaN/Inf") from None
                return self.clean(await self._request("record", payload))
        raise TrainingError(HELP)
