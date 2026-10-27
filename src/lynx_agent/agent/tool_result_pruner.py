"""可测试的压力触发回收计划；持久化后才应用替换。"""

from dataclasses import dataclass

from lynx_agent.agent.tokens import estimate_request_tokens
from lynx_agent.agent.tool_result_adapters import prune_content
from lynx_agent.session.tool_results import ToolResultReplacement, content_hash
from lynx_agent.tools.errors import ToolExecutionError


@dataclass(frozen=True)
class PruningResult:
    replacements: tuple[ToolResultReplacement, ...]
    before_tokens: int
    after_tokens: int


def plan_pruning(request, config, archive, protected=(), *, force=False, estimator=None):
    estimate = estimator or estimate_request_tokens
    before = estimate(request)
    limit = int(config.input_budget * config.context.tool_result_pruning.pressure_ratio)
    policy = config.context.tool_result_pruning
    if not policy.enabled or (not force and before < limit):
        return PruningResult((), before, before)
    messages = list(request.messages)
    replacements = []
    for index, message in enumerate(messages):
        identifier = message.result_id
        trace = archive.records.get(identifier)
        if identifier in protected or trace is None or trace.is_error or not trace.original_complete:
            continue
        try:
            archive.check_access(trace)
        except (ToolExecutionError, OSError):
            continue
        candidate = prune_content(trace.call.name, message.content, identifier, policy)
        if candidate is None:
            continue
        updated = message.model_copy(update={"content": candidate})
        trial = request.model_copy(update={"messages": (*messages[:index], updated, *messages[index + 1:])})
        if estimate(trial) >= estimate(request.model_copy(update={"messages": tuple(messages)})):
            continue
        replacements.append(ToolResultReplacement(result_id=identifier, source_hash=content_hash(message.content), content=candidate))
        messages[index] = updated
        if estimate(trial) < limit:
            break
    after = estimate(request.model_copy(update={"messages": tuple(messages)}))
    return PruningResult(tuple(replacements), before, after)
