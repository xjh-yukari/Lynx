"""使用上下键和回车选择会话。"""

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl

from lynx_agent.session import Session

NEW_SESSION = "__new__"


def build_picker(sessions: list[Session], *, input_stream=None, output_stream=None):
    """构造独立选择器；注入输入输出用于真实按键测试。"""
    choices = [(NEW_SESSION, "＋ 新会话")]
    choices.extend((
        str(session.id),
        f"{session.display_title}  {session.updated_at.astimezone():%m-%d %H:%M}  [{str(session.id)[:8]}]",
    ) for session in sessions)
    selected = 0
    bindings = KeyBindings()

    def render():
        start = max(0, selected - 8)
        visible = choices[start:start + 12]
        fragments = []
        for index, (_, label) in enumerate(visible, start):
            active = index == selected
            fragments.append(("reverse" if active else "", f"{' > ' if active else '   '}{label}\n"))
        return fragments

    @bindings.add("up")
    def move_up(event):
        nonlocal selected
        selected = (selected - 1) % len(choices)

    @bindings.add("down")
    def move_down(event):
        nonlocal selected
        selected = (selected + 1) % len(choices)

    @bindings.add("enter")
    def accept(event):
        event.app.exit(result=choices[selected][0])

    @bindings.add("escape", eager=True)
    @bindings.add("c-c")
    @bindings.add("c-d")
    def cancel(event):
        event.app.exit(result=None)

    control = FormattedTextControl(render, focusable=True)
    layout = Layout(HSplit([
        Window(FormattedTextControl("选择会话：↑↓ 移动，Enter 进入，Esc 返回"), height=1),
        Window(control, height=min(len(choices), 12)),
    ]), focused_element=control)
    return Application(
        layout=layout, key_bindings=bindings, full_screen=False,
        input=input_stream, output=output_stream,
    )


async def select_session(sessions: list[Session]) -> str | None:
    return await build_picker(sessions).run_async()
