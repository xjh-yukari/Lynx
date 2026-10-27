"""bench.toml 的解析与校验；价格、预算与模型参数全部来自配置。"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from agent_bench.models import BenchModel

DEFAULT_CONFIG_NAME = "bench.toml"


class ConfigError(Exception):
    """配置缺失或不合法；错误信息不带密钥。"""


class ModelSettings(BenchModel):
    name: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    # dsh 走 Anthropic Messages 协议；缺省为 <base_url>/anthropic。
    anthropic_base_url: str | None = None
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=8192, gt=0)
    reasoning_effort: str | None = "high"
    api_key_env: str = "DEEPSEEK_API_KEY"
    env_file: str = ".env"
    request_timeout_seconds: float = Field(default=180, gt=0)


class PriceSettings(BenchModel):
    """单位：每百万 Token 的人民币价格；由使用者按实际报价填写。"""

    currency: str = "CNY"
    input_per_million: float = Field(default=2.0, ge=0)
    cached_input_per_million: float = Field(default=0.2, ge=0)
    output_per_million: float = Field(default=8.0, ge=0)


class BudgetSettings(BenchModel):
    # 必填：没有预算上限就不启动真实调用。
    max_cost_cny: float = Field(gt=0)


class RunSettings(BenchModel):
    seed: int = 20251023
    repeats: int = Field(default=1, ge=1, le=10)
    concurrency: int = Field(default=1, ge=1, le=8)
    align_context_window: int = Field(default=131_072, ge=8_192)
    context_fill_ratio: float = Field(default=0.85, gt=0.1, le=0.95)
    results_dir: str = "results"
    turn_timeout_seconds: float = Field(default=600, gt=0)
    case_timeout_seconds: float = Field(default=3600, gt=0)
    retries: int = Field(default=3, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=2.0, ge=0, le=60)
    keep_workspaces: bool = False


class ProxySettings(BenchModel):
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=0, ge=0, le=65535)


class EnvSettings(BenchModel):
    enabled: bool = True
    document_chars: int = Field(default=40_000, ge=1_000, le=400_000)
    page_chars: int = Field(default=6_000, ge=500, le=100_000)


class EfficiencySettings(BenchModel):
    """效率维度的目标值；分数按“达到目标即满分、超标按比例”计算。"""

    max_input_tokens_per_turn: int = Field(default=40_000, gt=0)
    min_cached_ratio: float = Field(default=0.25, ge=0, le=1)
    max_latency_ms: float = Field(default=30_000, gt=0)
    max_cost_cny_per_case: float = Field(default=1.0, gt=0)


class ScoringSettings(BenchModel):
    bootstrap_iterations: int = Field(default=2000, ge=100, le=20000)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    pass_k: int = Field(default=3, ge=1, le=10)
    efficiency: EfficiencySettings = Field(default_factory=EfficiencySettings)


class JudgeSettings(BenchModel):
    enabled: bool = False
    model: str = "deepseek-v4-pro"
    max_tokens: int = Field(default=512, gt=0)


class AgentSettings(BenchModel):
    """一个被测 agent 的接入配置；不同 kind 只用其中一部分字段。"""

    kind: Literal["lynx", "cli", "openai_chat"]
    enabled: bool = True
    display_name: str | None = None

    # 通用：声明能力，缺省按 kind 给默认值。
    capabilities: tuple[str, ...] | None = None
    # 是否自带工具（native 赛道需要真实原生工具）。
    native_tools: bool = False
    # 独立工作目录相对路径；缺省为临时目录下的 agent 名。
    workspace_subdir: str | None = None

    # ---- cli / openai_chat 的模型注入方式 ----
    base_url_env: str = "DEEPSEEK_BASE_URL"
    api_key_env: str = "DEEPSEEK_API_KEY"
    extra_env: dict[str, str] = Field(default_factory=dict)

    # ---- cli ----
    command: tuple[str, ...] = ()
    resume_args: tuple[str, ...] = ()
    # stdin=true 时任务通过标准输入传入，`{task}` 展开成 stdin_marker。
    cwd: str | None = None
    stdin: bool = False
    # stdin 模式下 {task} 展开成什么（约定 `-` 表示从标准输入读任务）。
    stdin_marker: str = "-"
    stdout: Literal["text", "ndjson"] = "text"
    answer_event_type: str | None = None
    text_field: str = "text"
    event_type_field: str = "type"
    session_field: str = "sessionId"
    session_event_type: str | None = None
    tool_event_type: str | None = None
    # reset_memory 时清空的目录（相对 cwd 或绝对路径）。
    memory_dirs: tuple[str, ...] = ()
    # 关闭会话日志上传等插件用的额外参数。
    extra_args: tuple[str, ...] = ()
    # 启动前写入工作区的 patch 文件内容（如关闭插件）；用 {patch} 引用路径。
    patch: str | None = None

    # ---- Lynx 专用（其它 kind 忽略）----
    memory_enabled: bool = True
    memory_executable: str | None = None

    # ---- openai_chat ----
    url: str | None = None
    health_path: str = "/health"
    request_timeout_seconds: float = Field(default=300, gt=0)

    @field_validator("capabilities")
    @classmethod
    def check_capabilities(cls, value):
        if value is None:
            return None
        allowed = {"multi_turn", "cross_session_memory", "mcp", "tool_result_recall", "native_tools"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"未知能力：{', '.join(sorted(unknown))}")
        return value

    @model_validator(mode="after")
    def check_kind_fields(self):
        if self.kind == "cli" and not self.command:
            raise ValueError("cli 适配器必须配置 command")
        if self.kind == "openai_chat" and not self.url:
            raise ValueError("openai_chat 适配器必须配置 url")
        uses_patch = any("{patch}" in item for item in (*self.command, *self.resume_args,
                                                        *self.extra_args))
        if uses_patch and not self.patch:
            raise ValueError("命令模板引用了 {patch}，但没有配置 patch 内容")
        return self

    @property
    def label(self) -> str:
        return self.display_name or self.kind

    def default_capabilities(self) -> set[str]:
        if self.capabilities is not None:
            return set(self.capabilities)
        base = {"multi_turn"}
        if self.kind == "lynx":
            base |= {"cross_session_memory", "mcp", "tool_result_recall"}
            if self.native_tools:
                base |= {"native_tools"}
        return base


class BenchConfig(BenchModel):
    model: ModelSettings = Field(default_factory=ModelSettings)
    price: PriceSettings = Field(default_factory=PriceSettings)
    budget: BudgetSettings
    run: RunSettings = Field(default_factory=RunSettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    env: EnvSettings = Field(default_factory=EnvSettings)
    scoring: ScoringSettings = Field(default_factory=ScoringSettings)
    llm_judge: JudgeSettings = Field(default_factory=JudgeSettings)
    agents: dict[str, AgentSettings] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_agents(self):
        if not self.agents:
            raise ValueError("至少配置一个被测 agent")
        for name in self.agents:
            if not name or not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("agent 名称只能使用字母、数字、下划线和连字符")
        return self


def _format_error(error: ValidationError) -> str:
    parts = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(item) for item in detail.get("loc", ()))
        kind = str(detail.get("type", ""))
        message = "缺少必要字段" if kind == "missing" else str(detail.get("msg", "取值不合法"))
        parts.append(f"{location}: {message}")
    return "bench 配置校验失败：" + "；".join(parts)


def load_bench_config(path: str | Path = DEFAULT_CONFIG_NAME) -> BenchConfig:
    """读取并校验 bench.toml；路径相对当前工作目录。"""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"找不到评测配置：{file}")
    try:
        data = tomllib.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError("评测配置无法读取或不是合法 TOML") from error
    try:
        return BenchConfig.model_validate(data)
    except ValidationError as error:
        raise ConfigError(_format_error(error)) from error


def read_env_file(path: Path) -> dict[str, str]:
    """极简 .env 读取：只支持 KEY=VALUE，不修改当前进程环境。"""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if not entry or entry.startswith("#") or "=" not in entry:
                continue
            key, _, raw = entry.partition("=")
            key = key.strip()
            value = raw.strip().strip('"').strip("'")
            if key:
                values[key] = value
    except OSError as error:
        raise ConfigError(f"无法读取环境文件：{path}") from error
    return values


def resolve_api_key(config: BenchConfig, *, root: Path | None = None) -> str:
    """优先进程环境变量，其次 .env；错误信息绝不包含取值。"""
    name = config.model.api_key_env
    value = os.environ.get(name)
    if value:
        return value.strip()
    base = root if root is not None else Path.cwd()
    values = read_env_file(base / config.model.env_file)
    value = values.get(name, "")
    if not value.strip():
        raise ConfigError(f"缺少模型密钥：请在环境变量或 {config.model.env_file} 中设置 {name}")
    return value.strip()


def config_fingerprint(config: BenchConfig) -> str:
    """配置哈希：用于把结果绑定到当时的参数，不包含任何密钥。"""
    payload = config.model_dump(mode="json", exclude={"model": {"api_key_env"}})
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def model_parameters(config: BenchConfig) -> dict[str, Any]:
    """记录到 run.json 的模型参数（不含密钥）。"""
    return {
        "name": config.model.name,
        "base_url": config.model.base_url,
        "temperature": config.model.temperature,
        "max_tokens": config.model.max_tokens,
        "reasoning_effort": config.model.reasoning_effort,
        "price": config.price.model_dump(mode="json"),
        "budget_max_cost_cny": config.budget.max_cost_cny,
    }
