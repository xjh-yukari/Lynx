"""统一 TUI 的标题、对话、输入和状态区域。"""

from time import monotonic

from prompt_toolkit.application import get_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.layout import (
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.widgets import Frame, TextArea

from lynx_agent.cli import statusbar
from lynx_agent.cli.markdown import render_markdown, wrap_fragments
from lynx_agent.cli.message_actions import clean_text
from lynx_agent.cli.metrics import context_bar, context_pill
from lynx_agent.cli.permissions import (
    APPROVAL_OPTIONS,
    APPROVAL_SHORT,
    PERMISSION_OPTIONS,
)
from lynx_agent.cli.scrolling import ScrollbarControl, TranscriptControl
from lynx_agent.cli.state import UIState


def transcript_text(state: UIState) -> str:
    return "\n\n".join(f"{message.role}：{message.content}" for message in state.messages if message.content)


def transcript_fragments(state: UIState):
    fragments = []
    for message in state.messages:
        if not message.content:
            continue
        if message.role.startswith("你"):
            suffix = message.role.removeprefix("你")
            fragments.append(("class:message.marker", f"› {suffix + ' ' if suffix else ''}"))
            fragments.append(("class:message.user", message.content))
        elif message.role.startswith("Lynx"):
            suffix = message.role.removeprefix("Lynx")
            fragments.append(("class:message.assistant", f"● {suffix + ' ' if suffix else ''}"))
            rendered = list(render_markdown(message.content))
            while rendered and not rendered[-1][1].rstrip("\n"):
                rendered.pop()
            if rendered:
                rendered[-1] = (rendered[-1][0], rendered[-1][1].rstrip("\n"))
            fragments.extend(rendered)
        elif message.role == "工具":
            style = "class:message.warning" if message.content.startswith("!") else "class:message.tool"
            fragments.append(("class:marker.tool", " • "))
            fragments.append((style, message.content))
        elif message.role == "问候":
            fragments.append(("class:message.assistant", "● "))
            fragments.append(("class:welcome", message.content))
        else:
            warning = any(word in message.content for word in ("失败", "未核实", "未保存", "已取消"))
            style = "class:message.warning" if warning else "class:message.notice"
            fragments.append((style, "· " + message.content))
        fragments.append(("", "\n\n"))
    return wrap_fragments(fragments, get_app().output.get_size().columns - 1)


def displayed_text(state: UIState) -> str:
    return fragment_list_to_text(transcript_fragments(state))


def clip_text(text: str, width: int) -> str:
    """按终端列宽裁剪，保留中文宽度；标题不允许换行侵占对话区。"""
    text = " ".join(clean_text(text).split())
    width = max(0, width)
    if get_cwidth(text) <= width:
        return text
    if width < 2:
        return "…" if width else ""
    result = ""
    for char in text:
        if get_cwidth(result + char) > width - 1:
            break
        result += char
    return result + "…"


def package_version() -> str:
    """取安装包版本；取不到就当作空串，界面不因版本信息缺失而报错。"""
    try:
        from importlib.metadata import version

        return version("lynx-agent")
    except Exception:  # noqa: BLE001 - 版本只用于展示
        return ""


def header_fragments(state: UIState, width: int):
    """单行顶栏：品牌与版本靠左，工作区与会话居中，模型与档位靠右（对齐参考图 01）。"""
    if width <= 0:
        return [("class:title", "")]
    mark = "◆ " if width > 6 else ""
    left = clip_text(("Lynx " + package_version()).rstrip(), max(0, width - get_cwidth(mark)))
    right = f"{state.model} · {state.effort or '默认'}"
    room = width - get_cwidth(mark) - get_cwidth(left) - get_cwidth(right) - 6
    middle = clip_text(f"{state.project}  ·  会话：{state.session_title}", max(0, room)) if room >= 8 else ""
    parts = [("class:status.model", mark), ("class:title", left)]
    if middle:
        parts += [("class:status.metric", "  ·  "), ("class:path", middle)]
    pad = width - sum(get_cwidth(text) for _, text in parts) - get_cwidth(right)
    if pad >= 1:
        parts += [("class:hint", " " * pad), ("class:status.model", clip_text(right, width))]
    return parts

def status_text(state: UIState, width: int | None = None) -> str:
    timing = state.timing_text()
    prefix = f"{timing} · " if timing else ""
    browsing = " · 浏览中，Ctrl+End 回到最新" if not state.follow_tail else ""
    full = (f"{prefix}{state.status} · 权限：{state.permissions} · "
            f"Shell：{'开启（无沙箱）' if state.shell_enabled else '关闭'} · 联网搜索：{state.search_backend}{browsing}")
    if width is None or get_cwidth(full) <= width:
        return full
    permission = f"权限：{state.permissions}"
    available = max(0, width - get_cwidth(permission) - 3)
    compact = clip_text(state.status, min(16, available)) + " · " + permission
    for extra in (f"联网搜索：{state.search_backend}", timing):
        if extra and get_cwidth(compact + " · " + extra) <= width:
            compact += " · " + extra
    return clip_text(compact, width)


def build_layout(state: UIState, editor: TextArea, scroll_to, scroll_by, jump_to_latest, *,
                 projection=None, interaction=None, search_editor=None, session_click=None, copy_selection=None,
                 meter=None, meter_sample=None, meter_click=None, effort_click=None, effort_pick=None,
                 permissions_click=None, permissions_pick=None,
                 approval_click=None, approval_pick=None, stop_click=None,
                 cost_label=None, cost_click=None, approval_editor=None,
                 usage_sample=None, usage_click=None, usage_editor=None) -> tuple[Layout, Window]:
    def header():
        return header_fragments(state, get_app().output.get_size().columns)

    def cursor():
        lines = projection.rows if projection else displayed_text(state).split("\n")
        if state.follow_tail:
            return Point(x=len(lines[-1]), y=len(lines) - 1)
        return Point(x=0, y=min(state.scroll_line, len(lines) - 1))

    transcript = Window(
        TranscriptControl(lambda: transcript_fragments(state), scroll_by, get_cursor_position=cursor,
                          projection=projection, interaction=interaction),
        wrap_lines=False,
    )
    scrollbar = Window(ScrollbarControl(state, transcript, scroll_to, scroll_by), width=1)

    def footer():
        width = get_app().output.get_size().columns
        if state.usage_panel:
            hints = "↑↓ / PgUp/PgDn 滚动明细 · Enter/Esc 关闭"
        elif state.permissions_open:
            hints = "←/→ 或点击切换权限 · Enter/Esc 完成"
        elif state.approval_open:
            hints = ("B 允许本批次 · Y 允许一次 · N 拒绝" if state.approval_supports_batch
                     else "Y 允许一次 · N 拒绝 · Esc 拒绝并取消本轮")
        elif state.approval_selector_open:
            hints = "←/→ 或点击切换审批档位 · Enter/Esc 完成"
        elif state.effort_open:
            hints = "←/→ 或点击调整 · Enter/Esc 完成 · 下一次请求生效"
        elif state.picker_open:
            hints = "↑↓ 选择 · Enter 进入 · Esc 返回"
        elif state.draft_expanded:
            hints = "Enter 换行 · Alt+S 发送 · Esc 收起保留草稿"
        elif state.batch_running:
            hints = f"{state.batch_progress or '批量任务执行中'} · Ctrl+X 请求停止"
        elif state.busy:
            hints = "Ctrl+C 取消 · 滚轮/拖动滚动条浏览"
        else:
            hints = "Enter 发送 · Alt+Enter 换行 · Ctrl+D 退出 · /help 帮助"
        if state.toast and monotonic() < state.toast_until:
            hints = state.toast
        elif state.selection_available and not state.picker_open:
            hints = "[复制所选] · Alt+C 复制 · Esc 取消选区"

        elif get_cwidth(hints) > width:
            if state.picker_open:
                hints = "↑↓ · Enter 进入 · Esc 返回"
            elif state.draft_expanded:
                hints = "Enter 换行 · Alt+S 发送 · Esc 收起"
            else:
                hints = "Ctrl+C 取消 · PgUp/PgDn 浏览" if state.busy else "Enter 发送 · Alt+Enter 换行 · /help"
        def jump(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT:
                if state.selection_available and copy_selection:
                    copy_selection()
                elif not state.follow_tail:
                    jump_to_latest()
        # 状态行（就绪/交互状态）已并入底栏；这里只留一行快捷键提示，并带上耗时。
        timing = state.timing_text()
        if timing:
            hints = f"{timing} · {hints}"
        return [("class:hint", clip_text(hints, width), jump)]

    def choices():
        count = max(1, min(14, get_app().output.get_size().rows - 9))
        start = max(0, state.selected - count // 2)
        fragments = []
        for index, (_, label) in enumerate(state.session_choices[start:start + count], start):
            def pick(event, selected=index):
                if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and session_click:
                    session_click(selected)
            fragments.append((
                "class:selected" if index == state.selected else "",
                clip_text(f"{'❯' if index == state.selected else ' '} {label}", max(2, get_app().output.get_size().columns - 6)) + "\n", pick,
            ))
        return fragments

    picker = ConditionalContainer(
        Frame(HSplit([
            *([search_editor] if search_editor else []),
            Window(FormattedTextControl("↑↓ 选择，Enter 进入，Esc 返回"), height=1),
            Window(FormattedTextControl(choices), height=Dimension(min=2, max=15)),
        ]), title="选择会话", modal=True),
        filter=Condition(lambda: state.picker_open),
    )
    approval = ConditionalContainer(
        Frame(approval_editor or Window(), title="审批：允许一次/本批次/拒绝（不是沙箱）", modal=True,
              height=Dimension(min=8, max=22)),
        filter=Condition(lambda: state.approval_open),
    )
    usage = ConditionalContainer(
        # 保持底栏可点击，允许再次点击收起或直接切到另一项；输入由状态锁定。
        Frame(usage_editor or Window(), title="Token / 缓存 / 费用明细 · Esc 关闭"),
        filter=Condition(lambda: bool(state.usage_panel) and not state.approval_open),
    )

    def border(top):
        width = get_app().output.get_size().columns
        left, right = ("╭", "╮") if top else ("╰", "╯")
        return [("class:input.border", (left + "─" * max(0, width - 2) + right)[:width])]

    composer = HSplit([
        Window(FormattedTextControl(lambda: border(True)), height=1),
        VSplit([Window(width=1, char="│", style="class:input.border"), editor,
                Window(width=1, char="│", style="class:input.border")]),
        Window(FormattedTextControl(lambda: border(False)), height=1),
    ])
    old_draft = HSplit([Window(height=1, char="─", style="class:separator"), editor,
                        Window(height=1, char="─", style="class:separator")])

    def meter_fragments():
        reading = meter_sample()

        def show(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and meter_click:
                meter_click()
        return [(style, text, show) for style, text in context_bar(reading, get_app().output.get_size().columns)]

    def model_line():
        """单行底栏：左边身份/权限/审批，右边上下文短条（对齐参考图 01）。"""
        reading = meter_sample()
        width = get_app().output.get_size().columns
        # 当前权限与审批优先，窄屏不会把完全权限或暂停状态挤掉。
        suffix = f" · 权限：{state.permissions}" + (" ▾" if permissions_click else "")
        approval_text = f" · 审批：{APPROVAL_SHORT.get(state.approval, state.approval)}" + (" ▾" if approval_click else "")
        stop_text = " · [停止]" if state.batch_running else ""
        level = state.effort or "默认"
        name = clip_text(f"{state.model} · {level} ▾",
                         max(0, width - get_cwidth(suffix) - get_cwidth(approval_text) - get_cwidth(stop_text)))
        style = "class:permission-danger" if state.permissions == "完全权限" else "class:action"

        def choose(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and effort_click:
                effort_click()

        def choose_permissions(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and permissions_click:
                permissions_click()

        def choose_approval(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and approval_click:
                approval_click()

        def choose_stop(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and stop_click:
                stop_click()

        def show_context(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and meter_click:
                meter_click()

        left = [("class:action", name, choose),
                (style, clip_text(suffix, max(0, width - get_cwidth(name))), choose_permissions)]
        used = get_cwidth(name) + get_cwidth(suffix)
        if used + get_cwidth(approval_text) <= width:
            left.append(("class:action", approval_text, choose_approval))
            used += get_cwidth(approval_text)
        if stop_text and used + get_cwidth(stop_text) <= width:
            left.append(("class:message.warning", stop_text, choose_stop))
            used += get_cwidth(stop_text)
        used = sum(get_cwidth(item[1]) for item in left)
        pill = context_pill(reading, max(0, width - used - 2))
        pill_width = sum(get_cwidth(text) for _, text in pill)
        # 空白与短条都响应“查看上下文详情”，所以底栏右侧整片都能点。
        if not pill or used + pill_width + 1 > width:
            return left
        space = width - used - pill_width
        return (left
                + [("class:hint", " " * space, show_context)]
                + [(pill_style, pill_text, show_context) for pill_style, pill_text in pill])

    def metrics_line():
        def show_tokens(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and usage_click:
                usage_click()

        def show_cost(event):
            if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and cost_click:
                cost_click()

        return [(style, text, show_tokens if target == "tokens" else show_cost) if target else (style, text)
                for style, text, target in statusbar.usage_metrics(
                    usage_sample(), cost_label() if cost_label else "费用—",
                    width=get_app().output.get_size().columns, selected=state.usage_panel)]

    def permission_choices():
        width = max(0, get_app().output.get_size().columns)
        fragments = [("class:title", "文件权限\n")]
        for index, (_, label) in enumerate(PERMISSION_OPTIONS):
            def pick(event, chosen=index):
                if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and permissions_pick:
                    permissions_pick(chosen)
            fragments.append(("class:hint", " · " if index else ""))
            fragments.append(("class:selected" if index == state.permissions_selected else "class:action",
                              label + (" ✓" if label == state.permissions else ""), pick))
        descriptions = {
            "仅查看": "只读工作区内文件；命令工具不可用。",
            "工作区修改": "读写工作区内文件；Shell 每条命令仍需审批。",
            "完全权限": "允许读写工作区外的本地文件；Shell 每条命令仍需审批。",
        }
        fragments.append(("class:permission-danger" if state.permissions == "完全权限" else "class:hint",
                          "\n" + clip_text(descriptions.get(state.permissions, ""), width)))
        fragments.append(("class:hint", "\n" + clip_text("←/→ 或点击立即切换 · Enter/Esc 完成 · 重启恢复工作区修改", width)))
        return fragments

    def approval_choices():
        width = max(0, get_app().output.get_size().columns)
        fragments = [("class:title", "审批档位（与文件权限独立）\n")]
        for index, (_, label) in enumerate(APPROVAL_OPTIONS):
            def pick(event, chosen=index):
                if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and approval_pick:
                    approval_pick(chosen)
            fragments.append(("class:hint", " · " if index else ""))
            fragments.append(("class:selected" if index == state.approval_selected else "class:action",
                              label + (" ✓" if label == state.approval else ""), pick))
        descriptions = {
            "每次询问": "只读、新建、批量任务和命令都先询问；仍是应用层审批。",
            "常用操作自动": "内置只读与授权范围内新建自动；批量任务一次授权；命令逐次审批。",
            "暂停执行": "可分析和生成预览，但不修改、不下载、不执行命令。",
        }
        fragments.append(("class:hint", "\n" + clip_text(descriptions.get(state.approval, ""), width)))
        fragments.append(("class:hint", "\n" + clip_text(
            "←/→ 或点击立即切换 · Enter/Esc 完成 · 完全权限不取消审批 · 不是沙箱", width)))
        return fragments

    def effort_choices():
        width = max(0, get_app().output.get_size().columns)
        fragments = [("class:title", "推理强度\n")]
        for index, option in enumerate(state.effort_options):
            def pick(event, chosen=index):
                if event.event_type == MouseEventType.MOUSE_UP and event.button == MouseButton.LEFT and effort_pick:
                    effort_pick(chosen)
            fragments.append(("class:hint", " ── " if index else ""))
            label = option.title() + (" ✓" if option == state.effort else "")
            fragments.append(("class:selected" if index == state.effort_selected else "class:action", label, pick))
        descriptions = {"off": "关闭思考；不等于关闭工具。", "low": "较轻推理，适合简单任务。",
                        "high": "较深入推理，默认平衡档位。", "max": "最高推理档位，适合复杂任务；生成上限仍为原配置。"}
        fragments.append(("class:hint", "\n" + clip_text(descriptions.get(state.effort, ""), width)))
        fragments.append(("class:hint", "\n" + clip_text("←/→ 或点击调整 · Enter/Esc 完成（已选档位保留）", width)))
        return fragments

    body = HSplit([
        Window(
            FormattedTextControl(header), height=1,
            wrap_lines=False,
        ),
        ConditionalContainer(VSplit([transcript, scrollbar]), filter=Condition(lambda: not state.draft_expanded)),
        ConditionalContainer(Window(FormattedTextControl(lambda: f"全屏草稿 · {editor.text.count(chr(10)) + 1} 行 · {len(editor.text)} 字符"), height=1),
                             filter=Condition(lambda: state.draft_expanded)),
        ConditionalContainer(Window(FormattedTextControl(effort_choices), height=4),
                             filter=Condition(lambda: state.effort_open)),
        ConditionalContainer(Window(FormattedTextControl(permission_choices), height=4),
                             filter=Condition(lambda: state.permissions_open)),
        ConditionalContainer(Window(FormattedTextControl(approval_choices), height=4),
                             filter=Condition(lambda: state.approval_selector_open)),
        DynamicContainer(lambda: old_draft if state.draft_expanded else composer),
        *([ConditionalContainer(Window(FormattedTextControl(model_line), height=1, wrap_lines=False),
                                filter=Condition(lambda: not state.draft_expanded))]
          if meter and meter_sample else []),
        *([ConditionalContainer(Window(FormattedTextControl(metrics_line), height=1, wrap_lines=False),
                                filter=Condition(lambda: not state.draft_expanded))]
          if usage_sample else []),
        Window(
            FormattedTextControl(footer), height=1,
            wrap_lines=False,
        ),
    ])
    return Layout(FloatContainer(body, floats=[
        Float(content=CompletionsMenu(max_height=8, scroll_offset=1,
                                     extra_filter=Condition(lambda: not state.busy and not state.picker_open and not state.effort_open and not state.permissions_open and not state.draft_expanded and not state.usage_panel)),
              xcursor=True, ycursor=True),
        Float(content=picker, left=1, right=1, top=4),
        Float(content=usage, left=1, right=1, top=2, bottom=5),
        Float(content=approval, left=1, right=1, top=2, bottom=4),
    ], style="class:app"), focused_element=editor), transcript
