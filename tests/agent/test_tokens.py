"""估算能覆盖中英文并且不会修改对话历史。"""

from lynx_agent.agent import AgentContext
from lynx_agent.agent.tokens import estimate_context_tokens, estimate_text_tokens


def test_fast_count_matches_original_weights_for_mixed_text():
    from math import ceil
    for text in ("", "abcd" * 100000, "中文🙂\nabc\x00", "\ud800", "é中" * 10000):
        ascii_count = sum(char.isascii() for char in text)
        assert estimate_text_tokens(text) == ceil(ascii_count / 4 + (len(text) - ascii_count) * 1.5)


def test_message_cache_reuses_immutable_messages_and_recounts_changed_copy(monkeypatch):
    import gc
    import weakref

    from lynx_agent.agent import tokens
    from lynx_agent.providers import ChatMessage

    original = tokens.estimate_text_tokens
    calls = []
    def count(text):
        calls.append(len(text))
        return original(text)
    monkeypatch.setattr(tokens, "estimate_text_tokens", count)
    message = ChatMessage(role="user", content="中文abc" * 100000)
    first = tokens.estimate_message_tokens(message)
    assert tokens.estimate_message_tokens(message) == first
    assert len(calls) == 1
    changed = message.model_copy(update={"content": "short"})
    assert tokens.estimate_message_tokens(changed) < first
    assert len(calls) == 2
    reference = weakref.ref(message)
    del message
    gc.collect()
    assert reference() is None  # 缓存不延长包含原文的消息生命周期。


def test_budget_calibration_only_increases_and_does_not_change_usage():
    from lynx_agent.agent.tokens import TokenBudget, estimate_request_tokens
    from lynx_agent.providers import ChatMessage, ModelRequest, TokenUsage

    request = ModelRequest(messages=(ChatMessage(role="user", content="abcd" * 100),))
    raw = estimate_request_tokens(request)
    usage = TokenUsage(prompt_tokens=raw * 2, completion_tokens=8, total_tokens=raw * 2 + 8)
    budget = TokenBudget(enabled=True)
    assert budget.observe(request, usage)
    assert budget.factor == 2.1 and budget.request(request) >= usage.prompt_tokens
    assert usage.prompt_tokens == raw * 2
    assert not budget.observe(request, usage)
    assert not budget.observe(request, None)
    assert not budget.observe(request, TokenUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1))
    assert budget.factor == 2.1
    assert not TokenBudget().observe(request, usage)
    budget.reset()
    assert budget.factor == 1


def test_estimate_handles_empty_unicode_and_longer_text():
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("中文🙂") > 0
    assert estimate_text_tokens("hello " * 100) > estimate_text_tokens("hello")


def test_pending_input_increases_estimate_without_changing_history():
    context = AgentContext("system instruction")
    context.add_user_message("我的问题")
    context.add_assistant_message("我的回复")
    snapshot = context.messages
    current = estimate_context_tokens(snapshot)
    projected = estimate_context_tokens(snapshot, "下一条输入")
    assert current.system > 0
    assert current.history > 0
    assert current.pending == 0
    assert projected.total > current.total
    assert projected.message_count == current.message_count + 1
    assert projected.total == sum((projected.system, projected.history, projected.pending, projected.framing))
    assert context.messages == snapshot
