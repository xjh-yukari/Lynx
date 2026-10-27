"""把统一 diff 文本映射成语义样式；只影响显示，不改变内容。"""

ADD = "class:diff.add"
DEL = "class:diff.del"
HUNK = "class:diff.hunk"
HEADER = "class:diff.header"


def diff_style(line):
    """按行前缀返回样式：文件头 / hunk / 增删 / 普通上下文。

    只认行首单字符，避免把正文里以 + - 开头的说明误判成 diff。
    """
    if not isinstance(line, str) or not line:
        return ""
    if line.startswith(("+++ ", "--- ")):
        return HEADER
    if line.startswith("@@"):
        return HUNK
    if line.startswith("+"):
        return ADD
    if line.startswith("-"):
        return DEL
    return ""
