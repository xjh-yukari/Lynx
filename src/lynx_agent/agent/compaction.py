"""请求前预算检查与分批摘要；候选结果不直接修改 Context。"""

import json
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from time import monotonic

from lynx_agent.agent.context import AgentContext
from lynx_agent.agent.errors import ContextError, SummaryError
from lynx_agent.agent.events import StatusCallback
from lynx_agent.agent.tokens import (
    TokenBudget,
)
from lynx_agent.agent.usage import UsageTracker
from lynx_agent.config import AgentConfig
from lynx_agent.providers import ChatMessage, ModelProvider, ModelRequest
from lynx_agent.providers.errors import ProviderError
from lynx_agent.providers.models import RequestDiagnostic, split_turns
from lynx_agent.providers.output_guard import contains_tool_markup


@dataclass(frozen=True)
class CompactionResult:
    summary: str
    kept: tuple[ChatMessage, ...]
    summarized_messages: int
    before_tokens: int
    after_tokens: int
    summary_calls: int


class ContextCompactor:
    """按完整回合切分旧历史，并在输入预算内增量生成摘要。"""

    def __init__(self, provider: ModelProvider, config: AgentConfig, on_diagnostic=None, *, token_budget=None) -> None:
        self.provider = provider
        self.config = config
        self.on_diagnostic = on_diagnostic
        self.tokens = token_budget if token_budget is not None else TokenBudget()

    async def prepare(
        self, context: AgentContext, pending: str | None = None, *, force: bool = False,
        on_status: StatusCallback | None = None,
        usage_tracker: UsageTracker | None = None,
        overhead_tokens: int = 0,
    ) -> CompactionResult | None:
        original_factor = self.tokens.factor
        budget = self.config.input_budget - overhead_tokens
        before = self.tokens.context(context.messages, pending)
        if self.tokens.context(context.project("", ()), pending) > budget:
            raise ContextError("系统提示词和本次输入已超过输入预算，请缩短输入或调整配置")
        if not force and before <= budget:
            return None
        if not force and not self.config.context.auto_compact:
            raise ContextError("上下文超过输入预算，自动压缩已关闭；请使用 /compact")

        history = context.history
        try:
            turns = split_turns(history)
        except ValueError as error:
            raise ContextError("历史不是完整回合，无法安全压缩工具调用") from error
        cut = len(history)
        recent_tokens = 0
        for turn in reversed(turns):
            index = cut - len(turn)
            pair_tokens = self.tokens.context(turn)
            if recent_tokens + pair_tokens > self.config.context.keep_recent_tokens:
                break
            # 保证未来摘要上限、系统提示词和当前输入也能放下。
            candidate = context.project("", history[index:])
            if self.tokens.context(candidate, pending) + self.config.context.summary_max_tokens + 40 > budget:
                break
            cut = index
            recent_tokens += pair_tokens
        if cut == 0:
            if before > budget:
                raise ContextError("没有可压缩的旧历史；请减少近期保留量或缩短输入")
            return None

        source = "\n".join(item.model_dump_json() for item in history[:cut])
        if on_status is not None:
            on_status("compacting")
        tracker = usage_tracker if usage_tracker is not None else UsageTracker()
        summary, calls = await self._summarize(source, context.summary, tracker)
        kept = history[cut:]
        # 摘要 usage 可能上调估算系数；按新系数重新校验近期完整回合，不能沿用旧预算。
        kept_turns = list(split_turns(kept))
        while kept_turns and sum(self.tokens.context(turn) for turn in kept_turns) > self.config.context.keep_recent_tokens:
            oldest = kept_turns.pop(0)
            extra_source = "\n".join(item.model_dump_json() for item in oldest)
            summary, extra_calls = await self._summarize(extra_source, summary, tracker, max_calls=128 - calls)
            calls += extra_calls
            cut += len(oldest)
        kept = history[cut:]
        budget = self.config.input_budget - ceil(overhead_tokens * self.tokens.factor / original_factor)
        before = self.tokens.context(context.messages, pending)
        after = self.tokens.context(context.project(summary, kept), pending)
        if after >= before:
            raise SummaryError("候选摘要未减少上下文，原上下文保持不变")
        if after > budget:
            raise SummaryError("压缩后仍超过输入预算，原上下文保持不变")
        return CompactionResult(summary, kept, cut, before, after, calls)

    async def _summarize(self, source: str, previous: str, tracker: UsageTracker, *, max_calls=128) -> tuple[str, int]:
        try:
            prompt = Path(__file__).with_name("templates").joinpath("compaction.md").read_text(encoding="utf-8")
        except OSError as error:
            raise ContextError("无法读取摘要提示词模板，原上下文保持不变") from error
        summary = previous
        offset = 0
        calls = 0

        def request_for(fragment: str) -> ModelRequest:
            return ModelRequest(messages=(
                ChatMessage(role="system", content=prompt + f"\n最终摘要目标不超过 {self.config.context.summary_max_tokens} Tokens，直接给出摘要，不调用工具。"),
                ChatMessage(role="user", content=json.dumps({
                    "previous_summary": summary, "transcript_fragment": fragment,
                }, ensure_ascii=False)),
            ), temperature=0, max_tokens=self.config.summary_generation_limit)

        while offset < len(source):
            # 二分选择可放入一次摘要请求的最大片段，超长单消息也能拆分。
            low, high = 0, min(len(source) - offset, self.config.summary_input_limit * 4)
            while low < high:
                middle = (low + high + 1) // 2
                request = request_for(source[offset:offset + middle])
                if self.tokens.context(request.messages) <= self.config.summary_input_limit:
                    low = middle
                else:
                    high = middle - 1
            if low == 0 or calls >= max_calls:
                raise ContextError("摘要输入无法容纳或分批次数过多，请调整预算；原上下文未修改")
            request_index = tracker.start("summary")
            started = monotonic()
            request = request_for(source[offset:offset + low])
            try:
                response = await self.provider.generate(request)
            except ProviderError as error:
                if self.on_diagnostic:
                    self.on_diagnostic(RequestDiagnostic(stage="summary", error_kind=type(error).__name__,
                        max_tokens=self.config.summary_generation_limit, elapsed_seconds=monotonic() - started))
                raise
            tracker.finish(request_index, response.usage)
            if self.tokens.observe(request, response.usage) and self.on_diagnostic:
                self.on_diagnostic(RequestDiagnostic(stage="summary", error_kind="token_estimate_low", action="raised_input_estimate"))
            calls += 1
            summary = response.content.strip()
            invalid = (response.finish_reason != "stop" or not summary or response.tool_calls
                       or contains_tool_markup(summary) or self.tokens.text(summary) > self.config.context.summary_max_tokens)
            if self.on_diagnostic:
                self.on_diagnostic(RequestDiagnostic(stage="summary", error_kind="invalid_summary" if invalid else "none",
                    action="candidate_rejected" if invalid else "summary_received", finish_reason=response.finish_reason,
                    max_tokens=self.config.summary_generation_limit, text_chars=len(response.content),
                    reasoning_chars=len(response.reasoning_content or ""), elapsed_seconds=monotonic() - started, usage=response.usage))
            if response.finish_reason != "stop" or not summary or response.tool_calls or contains_tool_markup(summary):
                raise SummaryError("摘要生成未完整结束，原上下文保持不变")
            if self.tokens.text(summary) > self.config.context.summary_max_tokens:
                raise SummaryError("摘要超过本地估算上限，原上下文保持不变")
            offset += low
        return summary, calls
