"""可复用的文本报告，不直接向终端打印。"""

from math import ceil

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.tokens import (
    estimate_context_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
)
from lynx_agent.agent.usage import UsageTracker
from lynx_agent.config import Config


def format_diagnostics(agent):
    records = agent.diagnostics[-12:]
    if not records:
        return "当前会话暂无故障诊断记录。"
    labels = {"summary": "历史摘要", "chat": "对话", "rewrite": "精简重写", "recovery": "恢复"}
    lines = ["最近故障/恢复诊断（不包含请求正文、密钥或推理原文）："]
    for item in records:
        lines.append(f"{labels[item.stage]}：{item.error_kind}；动作={item.action}；结束={item.finish_reason or '未获得'}；"
                     f"生成上限={item.max_tokens or '未记录'}；正文 {item.text_chars} 字符，推理 {item.reasoning_chars} 字符；{item.elapsed_seconds:.2f}s。")
    if agent.can_retry:
        lines.append("当前有未完成任务；/retry 只整理已有结果，不重新执行工具。")
    return "\n".join(lines)


def format_tools(agent: AgentRunner, config: Config) -> str:
    lines = ["当前注册的模型工具："]
    for name in agent.tools.names:
        tool = agent.tools.get(name)
        lines.append(f"  {name}：{tool.description}（{'只读' if tool.read_only else '有写入能力'}）")
    if not agent.tools.names:
        lines.append("  无")
    if "web_search" in agent.tools.names:
        service = agent.tools.get("web_search")
        name = service.config.provider if hasattr(service, "config") else "自定义"
        lines.append(f"搜索后端：{name}；只有实际调用时才联网，配置状态不代表网络已连通。")
    else:
        from lynx_agent.web.setup import search_unavailable_reason
        reason = "已通过 /search off 关闭，可用 /search on 开启" if agent.search_disabled_by_user else (
            search_unavailable_reason(config.web.search) or "当前为自定义工具注册器")
        lines.append("web_search 未注册：" + reason)
    if "web_fetch" not in agent.tools.names:
        lines.append("web_fetch 未注册：" + ("配置已关闭" if not config.web.fetch.enabled else "当前为自定义工具注册器"))
    limits = config.agent.tools
    if agent.file_access:
        lines.append(f"文件权限：{agent.permission_label}；固定工作区：{agent.file_access.policy.root}")
    lines.append(f"每次最多 {limits.max_calls_per_response} 个调用；单工具超时 {limits.timeout_seconds:g} 秒；模型最多 {config.agent.max_iterations} 轮。")
    if "exec_command" in agent.tools.names:
        lines.append("exec_command 单独计时：审批最多10分钟，执行默认60秒、最大300秒；逐次人工批准，本机无沙箱。")
    lines.append("其他有写入能力的工具：" + ("允许，请自行核实注册工具的安全性。" if limits.allow_write_tools else "默认禁止；文件写入单独受权限和审批控制。"))
    return "\n".join(lines)


def format_api_usage(tracker: UsageTracker) -> str:
    total = tracker.report()
    if not total.calls:
        return "API 实际用量：本次运行的当前会话尚无模型请求记录。"
    lines = ["API 实际用量（最近一次模型操作，非当前上下文估算）："]
    for kind, label in (("chat", "对话生成"), ("summary", "历史摘要"), ("rewrite", "回复精简重写"), ("memory", "任务结论提取")):
        report = tracker.report(kind)
        if not report.calls:
            continue
        if report.usage is None:
            lines.append(f"  {label}：未获得 usage（{report.calls} 次调用尝试），无法确认实际用量。")
        else:
            usage = report.usage
            lines.append(
                f"  {label}：输入 {usage.prompt_tokens:,}，输出 {usage.completion_tokens:,}，"
                f"合计 {usage.total_tokens:,} Tokens；已返回 usage {report.reported_calls}/{report.calls} 次。"
            )
    if total.usage is not None:
        lines.append(f"  本次操作已知合计：{total.usage.total_tokens:,} Tokens（包括已返回用量的摘要、重写和记忆提取请求）。")
    if total.reported_calls < total.calls:
        lines.append("  部分请求失败、被取消或未返回 usage；统计不完整，不以本地估算补齐，也不代表零消耗。")
    return "\n".join(lines)


def format_tokens(agent: AgentRunner, config: Config, pending: str | None = None) -> str:
    estimate = estimate_context_tokens(agent.context_meter_messages, pending)
    budget = config.agent.input_budget
    tools = estimate_tool_tokens(agent.tools.definitions())
    runtime = estimate_text_tokens(agent.runtime_context)
    factor = getattr(agent, "token_estimate_factor", 1.0)
    total = ceil((estimate.total + tools + runtime) * factor)
    lines = [
        "本地估算（当前准备发送的上下文）：",
        f"模型：{config.agent.model.name}",
        f"系统提示词：约 {estimate.system:,} Tokens",
        f"历史消息：约 {estimate.history:,} Tokens",
        f"工具定义：约 {tools:,} Tokens",
        f"固定运行与安全规则：约 {runtime:,} Tokens（请求时附加；动态快照已计入消息）",
    ]
    if pending is not None:
        lines.append(f"待发送文本：约 {estimate.pending:,} Tokens（仅预估，未发送）")
    if agent.memory.enabled:
        lines.append(f"长期档案：约 {estimate_text_tokens(agent.context.memory_profiles):,} Tokens（已计入系统）；最近检索资料：约 {estimate_text_tokens(agent.context.memory_recall):,} Tokens（历史快照按实际消息计入，不另外累加）。")
    lines.extend([
        f"消息封装开销：约 {estimate.framing:,} Tokens",
        f"输入合计：约 {estimate.total + tools + runtime:,} Tokens，共 {estimate.message_count} 条消息",
        f"输出上限：{config.agent.output_limit}",
        f"本地窗口预算：{config.agent.model.context_window_tokens:,}；压缩触发线：{config.agent.compaction_trigger:,}；硬输入预算：{budget:,}",
        f"保守校正系数：{factor:.3f}；校正后输入合计：约 {total:,} Tokens。",
        f"输入占用：{total / budget:.1%}；自动压缩：{'开启' if config.agent.context.auto_compact else '关闭'}",
    ])
    if agent.context.summary:
        lines.append("当前上下文包含历史摘要。")
    lines.append(f"摘要目标上限：{config.agent.context.summary_max_tokens:,}；摘要生成预算：{config.agent.summary_generation_limit:,}；单批摘要输入上限：{config.agent.summary_input_limit:,} Tokens。")
    lines.append(f"自动恢复额外请求上限：{config.agent.recovery.max_extra_requests} 次；存在未完成任务：{'是' if agent.can_retry else '否'}。")
    pruning = config.agent.context.tool_result_pruning
    lines.append(f"工具结果回收：{'开启' if pruning.enabled else '关闭'}；压力触发线：约 {int(budget * pruning.pressure_ratio):,} Tokens。")
    if agent.last_pruning is not None:
        result = agent.last_pruning
        lines.append(f"最近一次回收：{len(result.replacements)} 条工具结果；完整请求估算 {result.before_tokens:,} → {result.after_tokens:,} Tokens（非 API 用量）。")
    lines.extend([
        "本地经验估算，非精确计数；不含模型回复。窗口预算不代表供应商实际最大窗口。",
        format_api_usage(agent.last_usage),
    ])
    return "\n".join(lines)
