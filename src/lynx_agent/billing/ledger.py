"""按请求唯一编号记账，独立于最近一轮 usage，切换会话不重复累计。"""

from dataclasses import dataclass
from decimal import Decimal

from lynx_agent.billing.pricing import (
    SOURCE,
    VERSION,
    official_endpoint,
    period_label,
    price_components,
    price_usage,
)


@dataclass(frozen=True)
class CostReport:
    amount: Decimal
    calls: int
    priced: int
    pending: int
    unpriced: int


class CostLedger:
    def __init__(self, provider, endpoint):
        self.official = provider == "deepseek" and official_endpoint(endpoint)
        self.sessions = {}
        self.components = {}

    def observe(self, session, model, record):
        amount = price_usage(model, record.usage, record.started_at, self.official) if record.finished else None
        self.sessions.setdefault(session, {})[record.id] = (record, amount)
        parts = price_components(model, record.usage, record.started_at, self.official) if record.finished else None
        self.components.setdefault(session, {})[record.id] = parts

    def usage(self, session):
        from lynx_agent.agent.usage import usage_breakdown

        return usage_breakdown(record for record, _ in self.sessions.get(session, {}).values())

    def report(self, session, kind=None):
        entries = [item for item in self.sessions.get(session, {}).values() if kind is None or item[0].kind == kind]
        return CostReport(
            sum((amount for _, amount in entries if amount is not None), Decimal(0)), len(entries),
            sum(amount is not None for _, amount in entries), sum(not record.finished for record, _ in entries),
            sum(record.finished and amount is None for record, amount in entries),
        )

    def label(self, session):
        report = self.report(session)
        if not self.official:
            return "费用未计价"
        if not report.priced and report.calls:
            value = "费用—"
        elif 0 < report.amount < Decimal("0.0001"):
            value = "≈¥<0.0001"
        else:
            value = f"≈¥{report.amount:.4f}"
        return value + ("*" if report.pending or report.unpriced else "") + " · " + period_label()

    def details(self, session):
        report = self.report(session)
        amount = "未知" if report.calls and not report.priced else f"¥{report.amount:.8f}"
        lines = ["模型费用估算（本次运行内的当前会话，非账单/账户余额）：",
                 (f"已知估算：{amount}；已计价 {report.priced}/{report.calls} 次；"
                  f"未返回/中断 {report.pending} 次；未计价/缺少用量 {report.unpriced} 次。"),
                 f"当前时段：{period_label()}（北京时间）；价格快照：{VERSION}。"]
        if report.priced:
            parts = [item for item in self.components.get(session, {}).values() if item is not None]
            totals = [sum((item[index] for item in parts), Decimal(0)) for index in range(3)]
            lines.append(f"已计价部分：缓存命中输入 ¥{totals[0]:.8f}；未命中输入 ¥{totals[1]:.8f}；输出 ¥{totals[2]:.8f}。")
        for kind, name in (("chat", "对话与工具循环"), ("summary", "历史摘要"), ("rewrite", "回复重写"), ("memory", "记忆提取")):
            part = self.report(session, kind)
            if part.calls:
                lines.append(f"{name}：¥{part.amount:.8f}；已计价 {part.priced}/{part.calls} 次。")
        lines.extend([
            "按每次请求发起时间估算峰谷费率，不随当前时段重算旧费用；跨边界请求以平台账单为准。",
            "* 表示统计尚不完整。缺 usage、取消或未返回缓存拆分不按零消耗处理。",
            "仅含模型 Token 费，不含 AnySearch、其他工具服务费或 MCP 费用。",
            "切回会话保留本次运行记录；重启后不追算历史费用。未知模型/第三方接口不套用官方价格。",
            "峰时：北京时间周一至周五09–12、14–18点，排除中国假期；周末为谷时。日历仅核验2026年。",
            "单价按官方人民币表，谷时为峰时一半；本地按公布假期估算，不保证和账单完全一致。",
            f"价格来源：{SOURCE}",
        ])
        return "\n".join(lines)
