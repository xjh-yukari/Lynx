"""底栏度量的纯函数；usage_metrics 独立一行保留 Token/缓存与费用入口。

metrics/gap 保留旧的同行降级约定，兼容外部调用。
"""

from prompt_toolkit.utils import get_cwidth


def cache_text(percent):
    """没有用量时给出占位符：这一行始终存在，避免底栏高度跳动。"""
    return "缓存 —" if percent is None else f"缓存 {percent:.1f}%"


def metrics(cache_percent, cost, *, width=1000):
    """返回放得下的 (样式, 文本) 段；费用优先于缓存命中率。"""
    cost_segment = [("class:status.value", str(cost))] if cost else []
    cache_segment = []
    cache = cache_text(cache_percent)
    if cache:
        cache_segment = [("class:status.metric", cache + " · ")]

    def used(segments):
        return sum(get_cwidth(text) for _, text in segments)

    if used(cache_segment + cost_segment) <= width:
        return cache_segment + cost_segment
    if used(cost_segment) <= width:
        return cost_segment
    return []


def gap(left_width, right_width, width):
    """左右两块之间的空格数；放不下时返回负数，调用方据此丢弃右块。"""
    return width - left_width - right_width


def usage_metrics(usage, cost, *, width, selected=""):
    """独立一行保留 Token/缓存与费用入口；窄屏缩写，不直接隐藏。"""
    from lynx_agent.cli.metrics import compact_tokens

    token_arrow = "▴" if selected == "tokens" else "▾"
    cost_arrow = "▴" if selected == "cost" else "▾"
    total = compact_tokens(usage.total_tokens) if usage.reported_calls else "—"
    percent = f"{usage.cache_percent:.1f}%" if usage.cache_percent is not None else "—"
    mark = "*" if usage.calls and usage.incomplete else ""
    variants = [
        (f"本轮 Token {total}{mark} · 缓存 {percent} {token_arrow}", f"会话 {cost} {cost_arrow}"),
        (f"Token {total}{mark} · 缓存 {percent} {token_arrow}", f"{cost} {cost_arrow}"),
        (f"Token/缓存 {token_arrow}", f"费用 {cost_arrow}"),
        ("Token/缓存", "费用"),
        ("T/C", "¥"),
    ]
    for left, right in variants:
        padding = width - get_cwidth(left) - get_cwidth(right)
        if padding >= 1:
            return [("class:usage.tokens", left, "tokens"),
                    ("class:hint", " " * padding, ""),
                    ("class:usage.cost", right, "cost")]
    return []  # 少于五列的终端无法同时容纳两个可读入口。
