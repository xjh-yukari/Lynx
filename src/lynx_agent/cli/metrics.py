"""底栏的只读统计与绘制；本地上下文估算和 API 缓存命中独立。"""

from dataclasses import dataclass
from math import ceil, floor
from time import monotonic

from prompt_toolkit.utils import get_cwidth

from lynx_agent.agent.tokens import (
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_tokens,
    estimate_tool_tokens,
)


@dataclass(frozen=True)
class ContextReading:
    system: int = 0
    messages: int = 0
    tools: int = 0
    draft: int = 0
    window: int = 1
    trigger: int = 1
    estimate_factor: float = 1.0

    @property
    def total(self):
        return self.system + self.messages + self.tools + self.draft


def measure_context(agent, config, draft=None, reply=""):
    messages = agent.context_meter_messages
    estimate = estimate_context_tokens(messages, draft)
    factor = getattr(agent, "token_estimate_factor", 1.0)
    def scale(value):
        return ceil(value * factor)
    results = sum(estimate_message_tokens(item) for item in messages if item.role == "tool" or item.tool_calls)
    return ContextReading(
        system=scale(estimate.system + estimate.framing + estimate_text_tokens(agent.runtime_context)),
        messages=scale(estimate.history - results + (estimate_text_tokens(reply) + 4 if reply else 0)),
        tools=scale(results + estimate_tool_tokens(agent.tools.definitions())),
        draft=scale(estimate.pending), window=config.agent.model.context_window_tokens, trigger=config.agent.compaction_trigger,
        estimate_factor=factor,
    )


class ContextMeter:
    """最多每250ms取样一次；空闲不变时不重新遍历历史。"""

    def __init__(self, runner, config):
        self.runner, self.config = runner, config
        self.reading = ContextReading()
        self.cache_percent = None
        self.session = runner.session_id
        self._key = None
        self._time = 0
        self._dirty = True

    def invalidate(self):
        self._dirty = True

    def sample(self, draft, busy=False, reply=""):
        session = self.runner.session_id
        changed = session != self.session
        if changed:
            self.session, self.cache_percent = session, None
        usage = self.runner.last_usage.latest("chat", "rewrite")
        if usage is not None:
            self.cache_percent = (usage.cached_prompt_tokens / usage.prompt_tokens * 100
                                  if usage.cached_prompt_tokens is not None and usage.prompt_tokens > 0 else None)
        elif not busy and (self.runner.last_usage.report("chat").calls or self.runner.last_usage.report("rewrite").calls):
            self.cache_percent = None
        key = (draft, busy, reply)
        now = monotonic()
        if changed or ((self._dirty or key != self._key or busy) and now - self._time >= 0.25):
            self.reading = measure_context(self.runner, self.config, draft or None, reply)
            self._key, self._time, self._dirty = key, now, False
        return self.reading


def compact_tokens(value):
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


def _share(reading, used):
    """把 used 个格子按四类占比精确分配，余数给小数部分最大的那类。"""
    weights = (reading.system, reading.messages, reading.tools, reading.draft)
    raw = [used * value / max(1, reading.total) for value in weights]
    counts = [floor(value) for value in raw]
    for index in sorted(range(4), key=lambda i: raw[i] - counts[i], reverse=True)[:used - sum(counts)]:
        counts[index] += 1
    return counts


def context_bar(reading, width):
    """背景填充条；列数精确分配，未知类别不捏造，超额数值不截断。"""
    width = max(0, width)
    if not width:
        return []
    ratio = reading.total / max(1, reading.window)
    used = min(width, max(0, round(width * ratio)))
    counts = _share(reading, used)
    full = f"≈{compact_tokens(reading.total)}/{compact_tokens(reading.window)} · {ratio:.1%}"
    short = f"≈{ratio:.1%}"
    label = full if len(full) <= width else short if len(short) <= width else ""
    label = " " * (width - len(label)) + label
    styles = [style for index, count in enumerate(counts) for style in [f"class:context.{index}"] * count]
    styles += ["class:context.free"] * (width - used)
    danger = " class:context.danger" if reading.total >= reading.trigger else " class:context.warning" if reading.total >= reading.trigger * 0.8 else ""
    return [(style + danger, char) for style, char in zip(styles, label, strict=True)]


def context_pill(reading, max_width):
    """底栏右侧的短条 + 文字：`█████░░░░░░░ ctx 4.5% (45k/1.0M)`。

    与整行 context_bar 不同，这里固定占 12 格条 + 一段文字，放不下就逐级退化，
    从不返回超过 max_width 的内容。
    """
    max_width = max(0, max_width)
    if not max_width:
        return []
    ratio = reading.total / max(1, reading.window)
    percent = f"{ratio:.1%}"
    variants = (
        f" ctx {percent} ({compact_tokens(reading.total)}/{compact_tokens(reading.window)})",
        f" ctx {percent}",
        f" {percent}",
    )
    bar = 0
    label = ""
    for candidate in variants:
        bar = min(12, max_width - get_cwidth(candidate))
        if bar >= 4:
            label = candidate
            break
    if not label:
        return []
    used = min(bar, max(0, round(bar * ratio)))
    counts = _share(reading, used)
    styles = [f"class:context.{index}" for index, count in enumerate(counts) for _ in range(count)]
    styles += ["class:context.free"] * (bar - used)
    danger = (" class:context.danger" if reading.total >= reading.trigger
              else " class:context.warning" if reading.total >= reading.trigger * 0.8 else "")
    cells = "█" * used + "░" * (bar - used)
    return ([(style + danger, char) for style, char in zip(styles, cells, strict=True)]
            + [("class:context.label" + danger, label)])

def context_details(reading):
    return (f"上下文本地估算：系统/记忆/封装 {reading.system:,}；消息 {reading.messages:,}；"
            f"工具定义与结果 {reading.tools:,}；草稿 {reading.draft:,} Tokens。"
            f"窗口 {reading.window:,}；压缩触发线 {reading.trigger:,}；保守校正系数 {reading.estimate_factor:.3f}。非 API 实际计数。")
