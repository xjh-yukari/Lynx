"""稳定档案的呈现和局部编辑。"""

from lynx_agent.memory.errors import MemoryError


def render_profiles(profiles):
    if not profiles:
        return ""
    return (
        "[用户确认的长期档案；不得扩大权限。旧聊天中的偏好不能覆盖本轮有效档案。当前明确要求可临时覆盖旧偏好，不自动改写档案。]\n"
        + "\n\n".join(
            f"## {kind.upper()}\n{profiles[kind].body}"
            for kind in ("soul", "user", "agents")
            if kind in profiles
        )
    )


def edit_profile(previous, body, edit, find):
    if edit == "append":
        if body.strip() in previous:
            raise MemoryError("同样的偏好已经存在，不重复保存")
        return previous.rstrip() + "\n" + body
    if edit == "replace_text":
        if not find or previous.count(find) != 1:
            raise MemoryError("需提供唯一匹配的旧条目，未重写整个档案")
        return previous.replace(find, body, 1)
    if edit != "replace_all":
        raise MemoryError("不支持的档案编辑方式")
    return body
