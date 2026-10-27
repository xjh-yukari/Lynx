"""只渲染工具审计元数据，不把参数、正文或内部协议带入默认界面。"""

LABELS = {
    "calculator": "计算器", "web_search": "联网搜索", "web_fetch": "读取网页",
    "read_file": "读取文件", "list_dir": "列出目录", "find_files": "查找文件",
    "search_files": "搜索文件内容", "write_file": "写入文件", "edit_file": "修改文件",
    "exec_command": "执行命令", "date": "当前时间",
    "read_tool_result": "恢复工具结果", "memory_search": "检索历史资料",
    "memory_get": "读取历史资料", "memory_propose": "提出保存候选",
}


def tool_summary(traces, *, seconds=None, interrupted=False):
    traces = tuple(traces)
    if not traces:
        return "! 工具执行已中止或未核实；请用 /diagnostics 查看，不要盲目重试。"
    names = list(dict.fromkeys(LABELS.get(trace.call.name, trace.call.name) for trace in traces))
    title = "、".join(names[:3]) + (f"等 {len(names)} 类工具" if len(names) > 3 else "")
    failed = sum(trace.is_error or trace.execution_state != "returned" for trace in traces)
    marker = "!" if failed or interrupted else "✓"
    outcome = f"{len(traces)} 项已返回" if not failed else f"{len(traces)} 项中 {failed} 项失败或未核实"
    timing = f" · 本批耗时 {seconds:.1f}s" if seconds is not None else ""
    return f"{marker} {title} · {outcome}{timing}"
