"""服务端用量与费用面板；与本地上下文估算分开。"""

from lynx_agent.billing.pricing import period_label, rates_for, utc_now


def breakdown_text(title, usage):
    lines = [title]
    if not usage.calls:
        return title + "\n尚无请求记录。"
    lines.append(f"请求 {usage.calls} 次；返回 usage {usage.reported_calls} 次；尚无结果/中断 {usage.pending} 次。")
    if not usage.reported_calls:
        lines.append("输入、输出、缓存命中与未命中：未知（尚未获得 usage）。")
        return "\n".join(lines)
    lines.append(f"输入 {usage.prompt_tokens:,} · 输出 {usage.completion_tokens:,} · 合计 {usage.total_tokens:,} Tokens")
    if usage.cached_tokens is None:
        lines.append("输入缓存：命中未知 · 未命中未知（接口未完整返回缓存拆分）。")
    else:
        percent = f"{usage.cache_percent:.2f}%" if usage.cache_percent is not None else "—（输入为零）"
        lines.append(f"输入缓存：命中 {usage.cached_tokens:,} · 未命中 {usage.uncached_tokens:,} Tokens · 命中率 {percent}")
    if usage.reported_calls < usage.calls:
        lines.append("* 仅累计已返回的实际用量；缺失请求不代表零消耗。")
    return "\n".join(lines)


def usage_details(runner, config, view="tokens"):
    current = breakdown_text("本轮用量（最近一次模型操作，含工具循环、摘要、重写与记忆提取）", runner.last_usage.breakdown())
    session = breakdown_text("会话累计用量（仅本次程序运行）", runner.costs.usage(runner.session_id))
    instant = utc_now()
    rates = rates_for(config.agent.model.name, instant, runner.costs.official)
    if rates is None:
        unit_price = "当前单价：未计价（接口、模型或计费时段未核验）。"
    else:
        hit, miss, output = rates
        unit_price = (f"当前单价（{config.agent.model.name} · {period_label(instant)}时 · 元/百万 Tokens）\n"
                      f"输入缓存命中 ¥{hit.normalize():f} · 输入未命中 ¥{miss.normalize():f} · 输出 ¥{output.normalize():f}")
    cost = runner.costs.details(runner.session_id) + "\n\n" + unit_price
    sections = [cost, current, session] if view == "cost" else [current, session, cost]
    return "\n\n".join(sections) + "\n\n命中率按输入 Token 加权计算；上下文进度条另为本地估算，不等于 API 消耗。\n↑↓ / PgUp/PgDn 滚动 · Enter / Esc 关闭"
