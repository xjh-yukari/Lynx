"""指标目录：名字、所属维度与方向。

这里是唯一真源：用例 YAML 里的 metrics、判分器产出的样本名、报告中的
维度分都以此为准；`bench validate` 会拒绝未登记的指标名。
"""

from __future__ import annotations

from agent_bench.models import BenchModel

DIMENSIONS: tuple[str, ...] = (
    "context",      # 上下文管理
    "memory",       # 长期记忆
    "tools",        # 工具调用
    "robustness",   # 鲁棒性
    "injection",    # 提示注入
    "efficiency",   # 效率与成本
    "stability",    # 稳定性
)

DIMENSION_LABELS = {
    "context": "上下文管理",
    "memory": "记忆系统",
    "tools": "工具调用",
    "robustness": "鲁棒性",
    "injection": "提示注入",
    "efficiency": "效率与成本",
    "stability": "稳定性",
}


class MetricDef(BenchModel):
    """约定：所有指标写入样本前都已归一化为 0~1 的“得分”，越高越好。

    越低越好的原始量（错误次数、额外 Token、延迟）由运行器在产出样本时
    转成分数，因此维度分与综合分可以直接取均值。
    """

    dimension: str
    description: str = ""


METRICS: dict[str, MetricDef] = {
    # ---- 上下文管理 ----
    "context.retention": MetricDef(
        dimension="context", description="压缩后仍能回答埋入事实的比例"),
    "context.retention_drift": MetricDef(
        dimension="context", description="多次压缩后的事实保留率（衰减曲线）"),
    "context.position_recall": MetricDef(
        dimension="context", description="事实埋在首/中/尾与不同长度下的召回率"),
    "context.instruction_persistence": MetricDef(
        dimension="context", description="早期给出的格式或工具约束是否仍被遵守"),
    "context.tool_result_recall": MetricDef(
        dimension="context", description="大工具结果被回收后仍能找回中部细节"),
    "context.overflow_errors": MetricDef(
        dimension="context", description="没有超窗口错误时为 1，错误越多得分越低"),
    "context.alignment": MetricDef(
        dimension="context", description="按本次对齐窗口填充后仍完成任务的得分"),

    # ---- 记忆系统 ----
    "memory.cross_session_recall": MetricDef(
        dimension="memory", description="会话 A 告知的事实能在新会话 B 回忆"),
    "memory.update_accuracy": MetricDef(
        dimension="memory", description="更正后回答新值且不出现旧值"),
    "memory.deletion": MetricDef(
        dimension="memory", description="要求忘记后不再出现、不再使用"),
    "memory.abstention": MetricDef(
        dimension="memory", description="未告知的信息诚实回答不知道，而不是编造"),
    "memory.isolation": MetricDef(
        dimension="memory", description="项目之间不串用规则与事实"),
    "memory.behavior_effect": MetricDef(
        dimension="memory", description="不问记忆本身，行为是否符合记住的偏好"),
    "memory.write_selectivity": MetricDef(
        dimension="memory", description="该记的记下、凭据与明确不记的内容不被记住"),
    "memory.cost": MetricDef(
        dimension="memory", description="有记忆带来的额外 Token 越少得分越高"),

    # ---- 工具调用 ----
    "tools.selection_accuracy": MetricDef(dimension="tools", description="选中正确工具的比例"),
    "tools.argument_accuracy": MetricDef(dimension="tools", description="参数完全正确的比例"),
    "tools.misuse_rate": MetricDef(
        dimension="tools", description="无需工具时没有误调用工具的比例"),

    # ---- 鲁棒性 ----
    "robustness.completion": MetricDef(
        dimension="robustness", description="超时、报错、畸形结果后仍完成任务"),
    "robustness.side_effect_duplicates": MetricDef(
        dimension="robustness", description="副作用工具没有重复执行时得分高"),

    # ---- 提示注入 ----
    "injection.resistance": MetricDef(
        dimension="injection", description="工具结果中的注入指令未被执行"),

    # ---- 效率与成本 ----
    "efficiency.input_tokens_per_turn": MetricDef(
        dimension="efficiency", description="每轮输入 Token 越接近目标得分越高"),
    "efficiency.cached_ratio": MetricDef(
        dimension="efficiency", description="输入 Token 的缓存命中比例"),
    "efficiency.latency_p50": MetricDef(
        dimension="efficiency", description="单次调用延迟越接近目标得分越高"),
    "efficiency.cost_per_case": MetricDef(
        dimension="efficiency", description="单个用例费用越接近预算得分越高"),

    # ---- 稳定性 ----
    "stability.turn_failures": MetricDef(
        dimension="stability", description="没有失败轮次时得分高"),
    "stability.pass_k": MetricDef(
        dimension="stability", description="同一题 k 次重复全部通过的比例"),
}

# 只在诊断里出现的指标（如推断的压缩点），不参与维度分。
DIAGNOSTIC_METRICS = frozenset({
    "diagnostics.compaction_events",
    "diagnostics.inferred_compactions",
})


def dimension_of(metric: str) -> str:
    return METRICS[metric].dimension


def is_known(metric: str) -> bool:
    return metric in METRICS or metric in DIAGNOSTIC_METRICS


def scored_metrics() -> dict[str, MetricDef]:
    return dict(METRICS)
