"""记录一次模型操作的服务端用量，缺失值不会用本地估算补齐。"""

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from lynx_agent.providers import TokenUsage

CallKind = Literal["chat", "summary", "rewrite", "memory"]


@dataclass(frozen=True)
class UsageRecord:
    kind: CallKind
    usage: TokenUsage | None = None
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished: bool = False


@dataclass(frozen=True)
class UsageReport:
    calls: int
    reported_calls: int
    usage: TokenUsage | None


@dataclass(frozen=True)
class UsageBreakdown:
    calls: int
    reported_calls: int
    pending: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None
    uncached_tokens: int | None

    @property
    def incomplete(self):
        return self.reported_calls < self.calls or self.cached_tokens is None

    @property
    def cache_percent(self):
        return self.cached_tokens / self.prompt_tokens * 100 if self.cached_tokens is not None and self.prompt_tokens else None


def usage_breakdown(records):
    records = tuple(records)
    usages = [record.usage for record in records if record.usage is not None]
    prompt = sum(usage.prompt_tokens for usage in usages)
    split_known = bool(usages) and all(usage.cached_prompt_tokens is not None or usage.prompt_tokens == 0 for usage in usages)
    cached = sum(usage.cached_prompt_tokens or 0 for usage in usages) if split_known else None
    return UsageBreakdown(len(records), len(usages), sum(not record.finished for record in records),
                          prompt, sum(usage.completion_tokens for usage in usages), sum(usage.total_tokens for usage in usages),
                          cached, prompt - cached if cached is not None else None)


class UsageTracker:
    """请求前登记，收到响应后填入 usage；失败和取消保留未知状态。"""

    def __init__(self, on_record=None) -> None:
        self._records: list[UsageRecord] = []
        self._on_record = on_record

    def start(self, kind: CallKind) -> int:
        self._records.append(UsageRecord(kind))
        if self._on_record:
            self._on_record(self._records[-1])
        return len(self._records) - 1

    def finish(self, index: int, usage: TokenUsage | None) -> None:
        self._records[index] = replace(self._records[index], usage=usage, finished=True)
        if self._on_record:
            self._on_record(self._records[index])

    def latest(self, *kinds: CallKind) -> TokenUsage | None:
        """仅返回最近匹配调用的样本，不用更早样本填补未知值。"""
        return next((record.usage for record in reversed(self._records) if record.kind in kinds), None)

    def breakdown(self):
        return usage_breakdown(self._records)

    def report(self, kind: CallKind | None = None) -> UsageReport:
        records = [record for record in self._records if kind is None or record.kind == kind]
        usages = [record.usage for record in records if record.usage is not None]
        total = None
        if usages:
            total = TokenUsage(
                prompt_tokens=sum(usage.prompt_tokens for usage in usages),
                completion_tokens=sum(usage.completion_tokens for usage in usages),
                total_tokens=sum(usage.total_tokens for usage in usages),
                cached_prompt_tokens=(sum(usage.cached_prompt_tokens for usage in usages)
                                      if all(usage.cached_prompt_tokens is not None for usage in usages) else None),
            )
        return UsageReport(len(records), len(usages), total)
