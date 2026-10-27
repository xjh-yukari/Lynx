"""按种子生成与事实无关的填充对话；大段文本在运行时产生，不进仓库。"""

from __future__ import annotations

import random
from math import ceil

from agent_bench.cases.models import FillerSpec

SERVICES = ("gateway", "indexer", "renderer", "scheduler", "collector", "notifier")
FAULTS = ("连接池耗尽", "队列积压", "缓存击穿", "磁盘写入抖动", "下游超时")
ACTIONS = ("重启滚动实例", "扩容两个副本", "回滚到上一版本", "调低并发上限", "清理过期索引")
SECTIONS = ("部署日志", "值班备注", "容量评估", "发布检查", "故障复盘")


def estimate_tokens(text: str) -> int:
    """与 Lynx 本地估算同口径的粗估：ASCII 4 字符 ≈ 1 Token，其他 1.5 Token/字符。"""
    ascii_count = sum(character.isascii() for character in text)
    return ceil(ascii_count / 4 + (len(text) - ascii_count) * 1.5)


def generate_filler(spec: FillerSpec, rng: random.Random, *, turn: int) -> str:
    """生成一段该轮次的填充文本，总长度接近 tokens/turns。"""
    target = max(40, spec.tokens // max(1, spec.turns))
    lines: list[str] = []
    used = 0
    index = 0
    while used < target:
        index += 1
        line = _line(spec.topic, rng, turn, index)
        lines.append(line)
        used = estimate_tokens("\n".join(lines))
        if index > 20_000:  # 防御异常配置
            break
    text = "\n".join(lines)
    if spec.tools:
        text += ("\n\n以上为背景资料。整理时请先用 sandbox_list_files 查看沙箱文件，"
                 "再用 notes_set 记录本轮的检查结论。")
    return text


def _line(topic: str, rng: random.Random, turn: int, index: int) -> str:
    section = rng.choice(SECTIONS)
    service = f"{rng.choice(SERVICES)}-{rng.randint(1, 9)}"
    fault = rng.choice(FAULTS)
    return (
        f"[{topic}·{section} 第{turn + 1}轮-{index}] "
        f"{service} 在 {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d} 触发{fault}，"
        f"错误率 {rng.uniform(0, 3):.2f}%，P95 延迟 {rng.randint(60, 480)}ms，"
        f"排队深度 {rng.randint(0, 900)}。处置：{rng.choice(ACTIONS)}；"
        f"复核项包括实例健康、连接复用与缓存命中，结论是{rng.choice(('继续观察', '已恢复', '需要跟进'))}。"
    )


def filler_rng(seed: int, case_id: str, turn: int) -> random.Random:
    return random.Random(f"{seed}:{case_id}:filler:{turn}")


def generate_document(*, chars: int, topic: str, facts: dict[str, str],
                      rng: random.Random) -> str:
    """生成指定长度的文档正文，并把关键事实按百分比位置插进去。

    用例 YAML 里只写 `{$text: {chars, topic, facts}}`，正文在运行时生成，
    仓库里不存放大段文本。
    """
    target = max(500, int(chars))
    lines: list[str] = []
    length = 0
    index = 0
    while length < target and index < 5000:
        index += 1
        line = _line(topic, rng, 0, index)
        lines.append(line)
        length += len(line) + 1
    for position, text in sorted(facts.items()):
        try:
            ratio = max(0.0, min(0.99, float(position) / 100))
        except (TypeError, ValueError):
            ratio = 0.5
        offset = int(len(lines) * ratio)
        lines.insert(max(0, min(len(lines), offset)), f"【关键信息】{text}")
    return "\n".join(lines)
