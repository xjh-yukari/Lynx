"""模型调用的落盘记录与费用累计。"""

from __future__ import annotations

import json
from pathlib import Path

from agent_bench.config import PriceSettings
from agent_bench.models import ModelCallRecord


class ProxyRecorder:
    """内存 + JSONL 双写；报告只从这里取数，不再重新调用 API。"""

    def __init__(self, run_id: str, path: Path | None = None, *, truncate: bool = True) -> None:
        self.run_id = run_id
        self.path = path
        self._records: list[ModelCallRecord] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # 断点续跑时不能清空已有记录，否则已完成的用例会丢用量。
            if truncate or not self.path.exists():
                self.path.write_text("", encoding="utf-8")

    def add(self, record: ModelCallRecord) -> None:
        self._records.append(record)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(record.model_dump_json() + "\n")
                stream.flush()

    @property
    def records(self) -> tuple[ModelCallRecord, ...]:
        return tuple(self._records)

    def for_scope(self, agent: str, case_id: str) -> list[ModelCallRecord]:
        return [item for item in self._records
                if item.agent == agent and item.case_id == case_id]

    def totals(self) -> dict[str, int]:
        prompt = sum(item.prompt_tokens or 0 for item in self._records)
        completion = sum(item.completion_tokens or 0 for item in self._records)
        cached = sum(item.cached_prompt_tokens or 0 for item in self._records)
        reported = sum(1 for item in self._records if item.prompt_tokens is not None)
        return {
            "calls": len(self._records),
            "reported_calls": reported,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_prompt_tokens": cached,
        }


def cost_of(records: tuple[ModelCallRecord, ...] | list[ModelCallRecord],
            price: PriceSettings) -> float:
    """按配置价格计算人民币费用；缺失 usage 的调用不计入，但会被单独统计。"""
    total = 0.0
    for record in records:
        prompt = record.prompt_tokens or 0
        cached = record.cached_prompt_tokens or 0
        completion = record.completion_tokens or 0
        fresh = max(0, prompt - cached)
        total += (fresh * price.input_per_million
                  + cached * price.cached_input_per_million
                  + completion * price.output_per_million) / 1_000_000
    return total


def estimate_cost(tokens_in: int, tokens_out: int, price: PriceSettings,
                  *, cached_ratio: float = 0.0) -> float:
    cached = int(tokens_in * cached_ratio)
    fresh = max(0, tokens_in - cached)
    return (fresh * price.input_per_million
            + cached * price.cached_input_per_million
            + tokens_out * price.output_per_million) / 1_000_000


def write_jsonl(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for entry in entries:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
