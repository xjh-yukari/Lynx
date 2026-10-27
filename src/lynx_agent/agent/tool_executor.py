"""完整 JSON 参数验证、顺序执行、超时、取消和结果长度控制。"""

import asyncio
import json
import math
from collections.abc import Callable
from uuid import UUID, uuid4

from lynx_agent.agent.tokens import estimate_text_tokens
from lynx_agent.agent.tool_result_adapters import bound_structured_result
from lynx_agent.config.schema import ToolExecutionConfig
from lynx_agent.memory.tools import MemoryTool
from lynx_agent.providers.models import ModelToolCall, ToolTrace
from lynx_agent.session.tool_results import (
    FILE_READ_TOOLS,
    MAX_ARCHIVE_BYTES,
    content_hash,
)
from lynx_agent.tools import ToolCall, ToolNotFoundError, ToolRegistry, ToolResult
from lynx_agent.tools.builtin.controlled import ControlledTool
from lynx_agent.tools.builtin.filesystem import FileTool
from lynx_agent.tools.builtin.shell import ExecCommandTool
from lynx_agent.tools.errors import ToolExecutionError, ToolValidationError


def _reject_constant(value):
    raise ValueError("JSON 不允许非有限数字")


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON 不允许非有限数字")
    return result


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON 字段重复")
        result[key] = value
    return result


class RepeatedToolCallError(ValueError):
    """调用未执行；避免使用不同 ID 绕过同参数失败上限。"""


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, config: ToolExecutionConfig, on_trace: Callable[[ToolTrace], None] | None = None, max_failures=2):
        self.registry = registry
        self.config = config
        self.on_trace = on_trace
        self.max_failures = max_failures
        self.write_guard = None
        self.start_turn()

    def start_turn(self):
        self._failures = {}
        self._writes = set()

    async def execute(self, call: ModelToolCall, turn_id: UUID, token_budget: int) -> ToolResult:
        cancelled = False
        tool = None
        fingerprint = (call.name, content_hash(call.arguments))
        try:
            if self._failures.get(fingerprint, 0) >= self.max_failures:
                raise RepeatedToolCallError
            if len(call.arguments) > self.config.max_argument_chars:
                raise ValueError("工具参数超过长度限制")
            try:
                arguments = json.loads(call.arguments, parse_constant=_reject_constant,
                                       parse_float=_finite_float, object_pairs_hook=_object)
            except (ValueError, RecursionError):
                raise ValueError("工具参数不是合法且完整的 JSON 对象") from None
            if not isinstance(arguments, dict):
                raise TypeError("工具参数必须是 JSON 对象")
            fingerprint = (call.name, content_hash(json.dumps(arguments, sort_keys=True, ensure_ascii=False)))
            if fingerprint in self._writes or self._failures.get(fingerprint, 0) >= self.max_failures:
                raise RepeatedToolCallError
            tool = self.registry.get(call.name)
            if not tool.read_only and self.write_guard is not None:
                self.write_guard()
            scoped_write = isinstance(tool, FileTool) and tool.service.policy.writable
            scoped_memory = isinstance(tool, MemoryTool) and tool.service.writable
            scoped_shell = isinstance(tool, ExecCommandTool) and tool.available
            scoped_controlled = isinstance(tool, ControlledTool) and tool.available
            if (not tool.read_only and not self.config.allow_write_tools and not scoped_write
                    and not scoped_memory and not scoped_shell and not scoped_controlled):
                raise ValueError("当前只允许只读工具，未执行有写入能力的工具")
            if not tool.read_only and not scoped_shell:
                self._writes.add(fingerprint)
            if scoped_shell:
                # 审批等待在通用工具超时之外；服务自己负责执行期限与清理。
                result = (await tool.execute_result(arguments)).model_copy(update={"call_id": call.id})
            else:
                async with asyncio.timeout(self.config.timeout_seconds):
                    result = await self.registry.execute(ToolCall(id=call.id, name=call.name, arguments=arguments))
            if not tool.read_only and result.is_error and not scoped_shell:
                result = result.model_copy(update={"execution_state": "unknown", "content": json.dumps({
                    "error_kind": result.error_kind or "tool_error", "execution_state": "unknown", "retryable": False,
                    "message": result.content, "notice": "写入工具报错不证明外部操作未发生；请先核实实际状态，不能自动重做。",
                }, ensure_ascii=False)})
        except RepeatedToolCallError:
            result = ToolResult(call_id=call.id, name=call.name, content=json.dumps({
                "error_kind": "repeated_call_blocked", "execution_state": "not_executed", "retryable": False,
                "message": "相同参数已失败过多次或写操作已尝试；未重复执行。请使用已有结果、缩小范围或明确说明限制。",
            }, ensure_ascii=False), is_error=True, error_kind="repeated_call_blocked", execution_state="not_executed")
        except (ValueError, TypeError, ToolNotFoundError, ToolValidationError, ToolExecutionError) as error:
            result = ToolResult(call_id=call.id, name=call.name, content=str(error), is_error=True,
                                error_kind="validation_or_permission", execution_state="not_executed")
        except TimeoutError:
            result = ToolResult(call_id=call.id, name=call.name, content=json.dumps({
                "error_kind": "timeout", "execution_state": "unknown", "retryable": False,
                "message": "工具执行超时；外部操作不保证已撤销。读取可缩小范围，写入须先核实实际状态，不能盲目重试。",
            }, ensure_ascii=False), is_error=True, error_kind="timeout", execution_state="unknown")
        except asyncio.CancelledError:
            cancelled = True
            result = ToolResult(call_id=call.id, name=call.name, content="工具执行已取消；外部操作可能已经发生。", is_error=True,
                                error_kind="cancelled", execution_state="unknown")
            if isinstance(tool, ExecCommandTool) and tool.service.last_result is not None:
                result = tool.service.last_result.model_copy(update={"call_id": call.id})
        approved_execution = isinstance(tool, ExecCommandTool) and result.execution_state != "not_executed"
        if result.is_error and fingerprint is not None and not approved_execution:
            self._failures[fingerprint] = self._failures.get(fingerprint, 0) + 1
        # 有界原文先存档；不能为了回收无限保存第三方输出。
        raw = result.content
        complete = len(raw.encode("utf-8")) <= MAX_ARCHIVE_BYTES
        if not complete:
            raw = raw.encode("utf-8")[:MAX_ARCHIVE_BYTES].decode("utf-8", errors="ignore")
        scope = ()
        if not result.is_error and isinstance(tool, FileTool):
            paths = [str(tool.service.policy.resolve(arguments.get("path", ".")))]
            try:
                data = json.loads(result.content)
                paths.extend(item["path"] for item in data.get("results", []) if isinstance(item, dict) and isinstance(item.get("path"), str))
                if isinstance(data.get("path"), str):
                    paths.append(data["path"])
                scope = tuple(dict.fromkeys(str(tool.service.policy.resolve(path)) for path in paths))
            except (ValueError, TypeError, AttributeError):
                scope = ()
        identifier = result.result_id or uuid4()
        trace_call = call
        if isinstance(tool, ExecCommandTool):
            from lynx_agent.execution.environment import Redactor
            redactor = Redactor(tool.service.secrets)
            raw = redactor.clean(raw)
            def clean(value):
                if isinstance(value, str):
                    return redactor.clean(value)
                if isinstance(value, list):
                    return [clean(item) for item in value]
                if isinstance(value, dict):
                    return {key: clean(item) for key, item in value.items()}
                return value
            try:
                trace_call = call.model_copy(update={"arguments": json.dumps(clean(json.loads(call.arguments)), ensure_ascii=False)})
            except ValueError:
                trace_call = call.model_copy(update={"arguments": "{}"})
        trace = ToolTrace(turn_id=turn_id, call=trace_call, content=raw, is_error=result.is_error,
                          result_id=identifier, original_complete=complete, scope_paths=scope,
                          error_kind=result.error_kind or ("tool_error" if result.is_error else None),
                          execution_state=result.execution_state)
        if self.on_trace is not None:
            self.on_trace(trace)
        # 模型可见版本仍受即时长度限制，不将超大结果直接塞入上下文。
        marker = "\n[工具结果已截断，不是完整结果]"
        content = result.content
        maximum = min(len(content), self.config.max_result_chars)
        low, high = 0, maximum
        while low < high:
            middle = (low + high + 1) // 2
            sample = content[:middle] + (marker if middle < len(content) else "")
            if len(sample) <= self.config.max_result_chars and estimate_text_tokens(sample) <= max(1, token_budget):
                low = middle
            else:
                high = middle - 1
        text = content if low == len(content) else content[:low] + marker
        if isinstance(tool, ExecCommandTool):
            try:
                data = json.loads(raw)
            except ValueError:
                data = {"status": "rejected", "message": raw, "log_result_id": None}
            while len(json.dumps(data, ensure_ascii=False)) > self.config.max_result_chars or estimate_text_tokens(json.dumps(data, ensure_ascii=False)) > max(1, token_budget):
                field = max(("stdout", "stderr", "notice"), key=lambda key: len(data.get(key, "")))
                if len(data.get(field, "")) < 16:
                    data = {"status": data.get("status"), "exit_code": data.get("exit_code"), "log_result_id": str(identifier)}
                    break
                data[field] = data[field][:len(data[field]) // 2]
                if field in {"stdout", "stderr"}:
                    data[field + "_truncated"] = True
            text = json.dumps(data, ensure_ascii=False)
        if call.name in FILE_READ_TOOLS | {"web_search", "web_fetch"} and not result.is_error:
            text = bound_structured_result(content, identifier, self.config.max_result_chars, token_budget)
        knowledge = getattr(getattr(tool, "connection", None), "knowledge", None) if not result.is_error else None
        if knowledge is not None:
            text = knowledge.bound_result(content, self.config.max_result_chars, token_budget)
        result = result.model_copy(update={"content": text, "result_id": identifier})
        if cancelled:
            raise asyncio.CancelledError
        return result
