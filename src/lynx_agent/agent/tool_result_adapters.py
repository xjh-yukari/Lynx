"""按内置工具字段裁剪；不把截断 JSON 伪装成合法完整结果。"""

import json

from lynx_agent.agent.tokens import estimate_text_tokens
from lynx_agent.session.tool_results import FILE_READ_TOOLS, WEB_TOOLS

MARKER = "\n[中间工具正文已回收；非完整证据]\n"


def bound_structured_result(content, identifier, max_chars, token_budget):
    """入口长度限制仍生效，但已知 JSON 工具不返回语法截断片段。"""
    def fits(text):
        return len(text) <= max_chars and estimate_text_tokens(text) <= max(1, token_budget)

    if fits(content):
        return content
    try:
        data = json.loads(content)
    except (ValueError, RecursionError):
        data = None
    if isinstance(data, dict):
        data["initial_truncation"] = {"result_id": str(identifier), "notice": "工具结果已截断，不是完整结果；可分页查看存档。"}
        while True:
            candidate = json.dumps(data, ensure_ascii=False)
            if fits(candidate):
                return candidate
            if isinstance(data.get("results"), list) and len(data["results"]) > 1:
                data["results"].pop()
                continue
            strings = [(data, "content")] if isinstance(data.get("content"), str) else []
            if isinstance(data.get("results"), list):
                strings.extend((item, field) for item in data["results"] if isinstance(item, dict)
                               for field in ("snippet", "text") if isinstance(item.get(field), str))
            if not strings or max(len(obj[field]) for obj, field in strings) < 64:
                break
            obj, field = max(strings, key=lambda pair: len(pair[0][pair[1]]))
            obj[field] = obj[field][:len(obj[field]) // 2]
    # 极小预算也不返回伪造的成功 JSON；最低预留不足时由调用侧明确停止。
    fallback = json.dumps({"error": "工具结果已截断，预算不足", "result_id": str(identifier)}, ensure_ascii=False)
    return fallback if fits(fallback) else "{}"


def _shorten(text, head, tail, *, lines=False):
    if len(text) <= head + tail + len(MARKER):
        return text
    first = text[:head]
    last = text[-tail:] if tail else ""
    if lines:
        first = first.rsplit("\n", 1)[0] if "\n" in first else ""
        last = last.split("\n", 1)[1] if "\n" in last else ""
    return first + MARKER + last


def prune_content(name, content, result_id, config):
    if name not in FILE_READ_TOOLS | WEB_TOOLS | {"exec_command"} or len(content) <= config.threshold_chars:
        return None
    try:
        data = json.loads(content)
    except (ValueError, RecursionError):
        return None
    if not isinstance(data, dict) or "recovery" in data or data.get("pending_approval"):
        return None
    if name == "exec_command":
        for field in ("stdout", "stderr"):
            if isinstance(data.get(field), str):
                data[field] = _shorten(data[field], config.head_chars // 2, config.tail_chars // 2)
                data[field + "_truncated"] = True
    elif name in {"read_file", "web_fetch"}:
        if not isinstance(data.get("content"), str):
            return None
        data["content"] = _shorten(data["content"], config.head_chars, config.tail_chars, lines=name == "read_file")
    else:
        results = data.get("results")
        if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
            return None
        # 搜索摘要优先缩短；标题、URL、日期、路径与行号不切碎。
        for item in results:
            for field in ("snippet", "text"):
                if isinstance(item.get(field), str):
                    item[field] = _shorten(item[field], 160, 64)
        data["returned_count_before_recycling"] = len(results)
    data["recovery"] = {"result_id": str(result_id), "notice":
        "工具结果已回收，仅保留部分资料；需要完整证据时用 read_tool_result 分页查阅原始快照，仍受当前权限限制。"}

    def serialize():
        return json.dumps(data, ensure_ascii=False)

    # 元数据太大时不丢失来源字段；列表仅保留可完整容纳的条目。
    while len(serialize()) >= config.threshold_chars and isinstance(data.get("results"), list) and len(data["results"]) > 1:
        data["results"].pop()
    candidate = serialize()
    if len(candidate) >= config.threshold_chars or len(candidate) >= len(content):
        return None
    return candidate
