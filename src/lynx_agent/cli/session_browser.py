"""TUI 会话菜单的标题检索和短预览；不修改会话。"""


def session_choices(sessions):
    choices = [(None, "＋ 新会话")]
    for session in sessions:
        preview = next((message.content for event in reversed(session.events) if event.kind == "turn"
                        for message in reversed(event.messages) if message.role == "assistant" and message.content), "")
        preview = " ".join(preview.split())[:70]
        label = f"{session.display_title} · {session.updated_at.astimezone():%m-%d %H:%M}"
        if preview:
            label += " · " + preview
        choices.append((str(session.id), label))
    return choices


def filter_choices(choices, query):
    terms = query.casefold().split()
    return [item for item in choices if item[0] is None or all(term in item[1].casefold() for term in terms)]
