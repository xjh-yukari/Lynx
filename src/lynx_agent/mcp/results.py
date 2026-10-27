"""只接收文本和 JSON；不访问服务返回的资源链接。"""

import json

from lynx_agent.mcp.errors import MCPError


def result_text(result, limit):
    if getattr(result, "result_type", "complete") != "complete":
        raise MCPError("MCP 服务要求额外交互，第一版不支持，未当成成功结果")
    texts = []
    used = 0
    truncated = False
    for block in result.content:
        if block.type != "text":
            raise MCPError("MCP 返回了不支持的图片、音频或资源内容，未读取资源")
        text = block.text
        texts.append(text[: max(0, limit - used)])
        used += len(text) + 1
        truncated = truncated or used > limit
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        text = json.dumps(structured, ensure_ascii=False, allow_nan=False)
        texts.append(text[: max(0, limit - min(used, limit))])
        truncated = truncated or used + len(text) > limit
    content = "\n".join(texts)[:limit]
    if truncated:
        marker = "\n[MCP 结果已截断，不是完整结果]"
        content = content[: limit - len(marker)] + marker
    if result.is_error:
        kind = structured.get("error_kind") if isinstance(structured, dict) else None
        kind = kind if kind in {"invalid_argument", "permission_denied", "timeout", "unavailable"} else "mcp_error"
        raise MCPError("MCP 服务报告工具错误：" + content, error_kind=kind)
    return content or "[MCP 工具成功，未返回正文]"
