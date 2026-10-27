"""配置加载与验证。"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field, ValidationError

from lynx_agent.approval.config import ApprovalConfig
from lynx_agent.config.errors import ConfigLoadError, ConfigValidationError
from lynx_agent.config.paths import get_env_path
from lynx_agent.config.schema import (
    DEFAULT_SYSTEM_PROMPT,
    Config,
    ConfigModel,
    ModelConfig,
)
from lynx_agent.execution.config import ExecutionConfig
from lynx_agent.knowledge.models import KnowledgeConfig
from lynx_agent.mcp.config import MCPConfig
from lynx_agent.memory.config import MemoryConfig
from lynx_agent.training.config import TrainingConfig


class PresetFile(ConfigModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    approval: ApprovalConfig = Field(default_factory=ApprovalConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    model_presets: dict[str, ModelConfig] = Field(default_factory=dict)
    model_preset: str | None = None

_ENVIRONMENT_KEYS = (
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_BASE_URL",
    "DEEPSEEK_MODEL",
    "DEEPSEEK_TIMEOUT_SECONDS",
    "MODEL_TEMPERATURE",
    "MODEL_MAX_TOKENS",
    "AGENT_SYSTEM_PROMPT",
    "AGENT_MAX_ITERATIONS",
    "MODEL_CONTEXT_WINDOW_TOKENS",
    "CONTEXT_RESERVE_TOKENS",
    "CONTEXT_KEEP_RECENT_TOKENS",
    "CONTEXT_SUMMARY_MAX_TOKENS",
    "CONTEXT_SUMMARY_GENERATION_MAX_TOKENS",
    "CONTEXT_SUMMARY_INPUT_MAX_TOKENS",
    "CONTEXT_COMPACTION_TRIGGER_RATIO",
    "CONTEXT_CALIBRATE_TOKENS",
    "RECOVERY_MAX_EXTRA_REQUESTS",
    "RECOVERY_MAX_SECONDS",
    "RECOVERY_MAX_IDENTICAL_TOOL_FAILURES",
    "CONTEXT_SAFETY_TOKENS",
    "CONTEXT_AUTO_COMPACT",
    "TOOL_PRUNING_ENABLED",
    "TOOL_PRUNING_PRESSURE_RATIO",
    "TOOL_PRUNING_THRESHOLD_CHARS",
    "TOOL_PRUNING_HEAD_CHARS",
    "TOOL_PRUNING_TAIL_CHARS",
    "MODEL_PRESET",
    "MODEL_MAX_OUTPUT_TOKENS",
    "REPLY_REWRITE_ENABLED",
    "REPLY_REWRITE_MAX_ATTEMPTS",
    "TOOLS_TIMEOUT_SECONDS",
    "TOOLS_MAX_CALLS_PER_RESPONSE",
    "TOOLS_MAX_ARGUMENT_CHARS",
    "TOOLS_MAX_RESULT_CHARS",
    "TOOLS_ALLOW_WRITE",
    "APPROVAL_MODE",
    "WEB_SEARCH_ENABLED",
    "WEB_SEARCH_PROVIDER",
    "ANYSEARCH_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "WEB_SEARCH_MAX_RESULTS",
    "WEB_SEARCH_TIMEOUT_SECONDS",
    "WEB_FETCH_ENABLED",
    "WEB_FETCH_TIMEOUT_SECONDS",
    "WEB_FETCH_MAX_RESPONSE_BYTES",
    "WEB_FETCH_MAX_CHARS",
    "WEB_FETCH_MAX_REDIRECTS",
)


def _read_env_file(path: Path) -> dict[str, str]:
    """读取 .env，但不修改当前进程的环境变量。"""

    if not path.exists():
        return {}

    try:
        values = dotenv_values(path, encoding="utf-8")
    except OSError as exc:
        raise ConfigLoadError(f"无法读取配置文件：{path}") from exc

    return {
        key: value
        for key, value in values.items()
        if isinstance(value, str)
    }


def _collect_environment(file_values: Mapping[str, str]) -> dict[str, str]:
    """合并 .env 和系统环境变量。

    系统环境变量优先于 .env。
    """

    result = dict(file_values)

    for key in _ENVIRONMENT_KEYS:
        value = os.environ.get(key)
        if value is not None:
            result[key] = value

    return result


def _required(values: Mapping[str, str], name: str) -> str:
    """读取必填配置，不在错误信息中暴露配置值。"""

    value = values.get(name)

    if value is None or not value.strip():
        raise ConfigValidationError(f"缺少必要配置：{name}")

    return value


def _build_config_data(values: Mapping[str, str], presets: PresetFile | None = None) -> dict[str, object]:
    """把扁平环境变量转换成 schema.py 所需的嵌套结构。"""

    provider: dict[str, object] = {
        "api_key": _required(values, "DEEPSEEK_API_KEY"),
        "base_url": values.get(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
        "timeout_seconds": values.get(
            "DEEPSEEK_TIMEOUT_SECONDS",
            "60",
        ),
    }

    presets = presets or PresetFile()
    selected = values.get("MODEL_PRESET", presets.model_preset)
    if selected is not None and selected not in presets.model_presets:
        raise ConfigValidationError("MODEL_PRESET 指定的模型预设不存在")
    model: dict[str, object] = {
        "provider": "deepseek",
    }
    if selected is not None:
        model.update(presets.model_presets[selected].model_dump())
    for key, field in (
        ("DEEPSEEK_MODEL", "name"), ("MODEL_TEMPERATURE", "temperature"),
        ("MODEL_CONTEXT_WINDOW_TOKENS", "context_window_tokens"),
        ("MODEL_MAX_TOKENS", "max_tokens"), ("MODEL_MAX_OUTPUT_TOKENS", "max_output_tokens"),
    ):
        if key in values:
            model[field] = values[key]
    if "name" not in model:
        model["name"] = _required(values, "DEEPSEEK_MODEL")

    agent: dict[str, object] = {
        "model": model,
        "context": {
            "reserve_tokens": values.get("CONTEXT_RESERVE_TOKENS", "16384"),
            "keep_recent_tokens": values.get("CONTEXT_KEEP_RECENT_TOKENS", "20000"),
            "summary_max_tokens": values.get("CONTEXT_SUMMARY_MAX_TOKENS", "2000"),
            "summary_generation_max_tokens": values.get("CONTEXT_SUMMARY_GENERATION_MAX_TOKENS") or None,
            "summary_input_max_tokens": values.get("CONTEXT_SUMMARY_INPUT_MAX_TOKENS") or None,
            "compaction_trigger_ratio": values.get("CONTEXT_COMPACTION_TRIGGER_RATIO", "0.9"),
            "calibrate_tokens": values.get("CONTEXT_CALIBRATE_TOKENS", "false"),
            "safety_tokens": values.get("CONTEXT_SAFETY_TOKENS", "1024"),
            "auto_compact": values.get("CONTEXT_AUTO_COMPACT", "true"),
            "tool_result_pruning": {
                "enabled": values.get("TOOL_PRUNING_ENABLED", "true"),
                "pressure_ratio": values.get("TOOL_PRUNING_PRESSURE_RATIO", "0.9"),
                "threshold_chars": values.get("TOOL_PRUNING_THRESHOLD_CHARS", "8192"),
                "head_chars": values.get("TOOL_PRUNING_HEAD_CHARS", "4096"),
                "tail_chars": values.get("TOOL_PRUNING_TAIL_CHARS", "1024"),
            },
        },
        "system_prompt": values.get(
            "AGENT_SYSTEM_PROMPT",
            DEFAULT_SYSTEM_PROMPT,
        ),
        "max_iterations": values.get(
            "AGENT_MAX_ITERATIONS",
            "8",
        ),
        "reply_rewrite": {
            "enabled": values.get("REPLY_REWRITE_ENABLED", "true"),
            "max_attempts": values.get("REPLY_REWRITE_MAX_ATTEMPTS", "2"),
        },
        "tools": {
            "timeout_seconds": values.get("TOOLS_TIMEOUT_SECONDS", "30"),
            "max_calls_per_response": values.get("TOOLS_MAX_CALLS_PER_RESPONSE", "8"),
            "max_argument_chars": values.get("TOOLS_MAX_ARGUMENT_CHARS", "16384"),
            "max_result_chars": values.get("TOOLS_MAX_RESULT_CHARS", "16000"),
            "allow_write_tools": values.get("TOOLS_ALLOW_WRITE", "false"),
        },
        "recovery": {
            "max_extra_requests": values.get("RECOVERY_MAX_EXTRA_REQUESTS", "2"),
            "max_seconds": values.get("RECOVERY_MAX_SECONDS", "90"),
            "max_identical_tool_failures": values.get("RECOVERY_MAX_IDENTICAL_TOOL_FAILURES", "2"),
        },
    }

    return {
        "providers": {
            "deepseek": provider,
        },
        "agent": agent,
        "model_presets": presets.model_presets,
        "model_preset": selected,
        "mcp": presets.mcp,
        "memory": presets.memory,
        "knowledge": presets.knowledge,
        "execution": presets.execution,
        "training": presets.training,
        "approval": {
            **presets.approval.model_dump(),
            **({"mode": values["APPROVAL_MODE"]} if "APPROVAL_MODE" in values else {}),
        },
        "web": {
            "search": {
                "enabled": values.get("WEB_SEARCH_ENABLED", "true"),
                "provider": values.get("WEB_SEARCH_PROVIDER", "anysearch"),
                "anysearch_api_key": values.get("ANYSEARCH_API_KEY") or None,
                "api_key": values.get("BRAVE_SEARCH_API_KEY") or None,
                "max_results": values.get("WEB_SEARCH_MAX_RESULTS", "5"),
                "timeout_seconds": values.get("WEB_SEARCH_TIMEOUT_SECONDS", "10"),
            },
            "fetch": {
                "enabled": values.get("WEB_FETCH_ENABLED", "true"),
                "timeout_seconds": values.get("WEB_FETCH_TIMEOUT_SECONDS", "15"),
                "max_response_bytes": values.get("WEB_FETCH_MAX_RESPONSE_BYTES", "2097152"),
                "max_chars": values.get("WEB_FETCH_MAX_CHARS", "12000"),
                "max_redirects": values.get("WEB_FETCH_MAX_REDIRECTS", "3"),
            },
        },
    }


def _format_validation_error(error: ValidationError) -> str:
    """生成不包含输入值的安全错误信息。"""

    messages: list[str] = []

    for detail in error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    ):
        location = ".".join(str(part) for part in detail.get("loc", ()))
        error_type = str(detail.get("type", ""))

        if error_type == "missing":
            message = "缺少必要字段"
        elif error_type == "extra_forbidden":
            message = "存在未知字段"
        else:
            message = "字段格式或取值不合法"

        messages.append(f"{location}: {message}")

    return "配置验证失败：" + "; ".join(messages)


@lru_cache(maxsize=8)
def _load_config_cached(env_path: str, preset_path: str) -> Config:
    """读取并缓存指定 .env 文件对应的配置。"""

    path = Path(env_path)
    file_values = _read_env_file(path)
    values = _collect_environment(file_values)
    try:
        preset_file = Path(preset_path)
        presets = PresetFile.model_validate(tomllib.loads(preset_file.read_text(encoding="utf-8"))) if preset_file.exists() else PresetFile()
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise ConfigLoadError("模型预设文件无法读取或格式不合法") from exc
    data = _build_config_data(values, presets)

    try:
        return Config.model_validate(data)
    except ValidationError as exc:
        message = _format_validation_error(exc)
        raise ConfigValidationError(message) from exc


def load_config(env_path: str | Path | None = None, *, preset_path: str | Path | None = None) -> Config:
    """加载并返回经过验证的应用配置。"""

    path = Path(env_path) if env_path is not None else get_env_path()
    resolved_path = path.expanduser().resolve()

    preset_file = Path(preset_path) if preset_path is not None else resolved_path.with_name("config.toml")
    if preset_path is not None and not preset_file.is_file():
        raise ConfigLoadError("指定的模型预设文件不存在")
    return _load_config_cached(str(resolved_path), str(preset_file.resolve()))


def clear_config_cache() -> None:
    """清除配置缓存，主要用于测试和配置重新加载。"""

    _load_config_cached.cache_clear()
