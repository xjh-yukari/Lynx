"""内容过滤、身份与受保护范围，不是操作系统沙箱。"""

import re

from lynx_agent.memory.errors import MemoryError


def check_body(body):
    if not body.strip() or len(body) > 6000 or "LYNX_MEMORY_V1:" in body:
        raise MemoryError("内容为空、过长或含保留标记，未保存")
    if re.search(
        r"sk-[A-Za-z0-9_-]{12,}|-----BEGIN .*PRIVATE KEY-----|(?i:api[_ -]?key|password|密码|token)\s*[:=]\s*\S{6,}",
        body,
    ):
        raise MemoryError("检测到可能的凭据，禁止写入长期记忆；请移除敏感值")
    if any(ord(char) < 32 and char not in "\n\r\t" for char in body):
        raise MemoryError("内容含不可见控制字符，未保存")


def keywords(text):
    # 不把整段会话发给检索器；FTS5 不是语义检索，中文使用有界相邻片段。
    terms = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]+", text[:500])
    words = []
    for term in terms:
        if re.fullmatch(r"[\u4e00-\u9fff]+", term):
            words.extend(
                term[index : index + 2] for index in range(min(8, len(term) - 1))
            )
        else:
            words.append(term)
    return " ".join(dict.fromkeys(words))[:120]


def search_index(text):
    terms = []
    for word in re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]+", text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", word):
            terms.extend(word[index : index + 2] for index in range(len(word) - 1))
        else:
            terms.append(word)
    return " ".join(list(dict.fromkeys(terms))[:256])
