"""工具名 → 语义样式类。只决定渲染颜色，不影响工具执行或权限。"""

READ = frozenset({"read_file", "list_dir", "find_files", "search_files", "read_tool_result"})
WRITE = frozenset({"write_file", "edit_file"})
EXEC = frozenset({"exec_command"})
WEB = frozenset({"web_search", "web_fetch"})
MEMORY = frozenset({"memory_search", "memory_get", "memory_propose"})

DEFAULT = "class:tool.name.other"


def tool_style(name):
    """按工具类型返回样式类；未知工具退回中性色，不做猜测。"""
    if not isinstance(name, str) or not name:
        return DEFAULT
    if name in READ:
        return "class:tool.name.read"
    if name in WRITE:
        return "class:tool.name.write"
    if name in EXEC:
        return "class:tool.name.exec"
    if name in WEB:
        return "class:tool.name.web"
    if name in MEMORY:
        return "class:tool.name.memory"
    return DEFAULT
