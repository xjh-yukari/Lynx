"""展开工具历史快照；不重跑工具、不读取当前源文件。"""

import json
import re

from lynx_agent.cli.message_actions import clean_text
from lynx_agent.tools.errors import ToolExecutionError

PAGE_CHARS = 2000


def redact_arguments(arguments):
    def clean(value):
        if isinstance(value, dict):
            return {key: "[已隐藏]" if re.search(r"key|token|password|secret|authorization|cookie", key, re.IGNORECASE) else clean(item)
                    for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "[已隐藏]", value)
        return value
    try:
        return json.dumps(clean(json.loads(arguments)), ensure_ascii=False, indent=2)[:1000]
    except (ValueError, RecursionError):
        return "[参数无法安全展示]"


def inspect_trace(runner, trace):
    if not any(item.result_id == trace.result_id and item == trace for item in runner.tool_traces):
        raise ToolExecutionError("该工具快照不属于当前可访问会话")
    file_tools = {"read_file", "search_files", "find_files", "list_dir", "write_file", "edit_file"}
    if trace.call.name in file_tools and not trace.scope_paths:
        raise ToolExecutionError("工具快照缺少文件权限范围，拒绝展开")
    for path in trace.scope_paths:
        if runner.file_access is None:
            raise ToolExecutionError("当前未启用文件权限管理")
        try:
            runner.file_access.policy.resolve(path)
        except OSError:
            raise ToolExecutionError("当前文件权限无法核实，拒绝展开快照") from None
    if trace.call.name == "read_tool_result":
        arguments = json.loads(trace.call.arguments)
        target = next((item for item in runner.tool_traces if str(item.result_id) == arguments.get("result_id")), None)
        if target is None or target.call.name == "read_tool_result":
            raise ToolExecutionError("恢复快照缺少可直接核实的来源，拒绝展开")
        inspect_trace(runner, target)
    return trace.content


def detail_text(message, runner):
    parts = []
    for trace in message.tools:
        body = clean_text(inspect_trace(runner, trace))
        parts.append(f"{trace.call.name} · {'错误/未核实' if trace.is_error or trace.execution_state != 'returned' else '已返回'}\n"
                     f"参数：{redact_arguments(trace.call.arguments)}\n历史结果（不是源文件最新内容）：\n{body}")
    source = "\n\n".join(parts)
    pages = max(1, (len(source) + PAGE_CHARS - 1) // PAGE_CHARS)
    message.detail_page = min(message.detail_page, pages - 1)
    start = message.detail_page * PAGE_CHARS
    return source[start:start + PAGE_CHARS], pages
