"""按当前界面状态解释快捷键。"""

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


def build_bindings(ui):
    keys = KeyBindings()
    picker = Condition(lambda: ui.state.picker_open)
    effort = Condition(lambda: ui.state.effort_open)
    permissions = Condition(lambda: ui.state.permissions_open)
    approval = Condition(lambda: ui.state.approval_open)
    approval_selector = Condition(lambda: ui.state.approval_selector_open)
    usage = Condition(lambda: bool(ui.state.usage_panel) and not ui.state.approval_open)
    overlay = picker | effort | permissions | approval | approval_selector | usage

    @keys.add("y", filter=approval, eager=True)
    @keys.add("Y", filter=approval, eager=True)
    def approve_execution(event):
        if ui.state.approval_kind == "batch":
            ui.decide_batch("once")
        else:
            ui.decide_execution(True)

    @keys.add("b", filter=approval, eager=True)
    @keys.add("B", filter=approval, eager=True)
    def approve_batch(event):
        if ui.state.approval_kind == "batch" and ui.state.approval_supports_batch:
            ui.decide_batch("batch")

    @keys.add("n", filter=approval, eager=True)
    @keys.add("N", filter=approval, eager=True)
    def reject_execution(event):
        if ui.state.approval_kind == "batch":
            ui.decide_batch("deny")
        else:
            ui.decide_execution(False)

    @keys.add("escape", filter=approval, eager=True)
    def cancel_execution(event):
        ui.cancel()

    @keys.add("c-x")
    def stop_batch(event):
        if ui.state.batch_running or (ui.state.approval_open and ui.state.approval_kind == "batch"):
            ui.stop_batch()

    @keys.add("enter")
    def enter(event):
        if ui.state.usage_panel and not ui.state.approval_open:
            ui.close_usage()
        elif ui.state.approval_selector_open:
            ui.close_approval()
        elif ui.state.effort_open:
            ui.close_effort()
        elif ui.state.permissions_open:
            ui.close_permissions()
        elif ui.state.picker_open:
            ui.accept_session()
        elif not ui.state.busy:
            if ui.state.draft_expanded:
                ui.editor.buffer.insert_text("\n")
            elif ui.editor.buffer.complete_state and ui.editor.buffer.complete_state.completions:
                buffer = ui.editor.buffer
                completion = buffer.complete_state.current_completion or buffer.complete_state.completions[0]
                command = buffer.text.startswith("/")
                buffer.apply_completion(completion)
                if command:
                    ui.submit()
            else:
                ui.submit()

    @keys.add("tab", filter=~overlay)
    def complete(event):
        if ui.state.busy:
            return
        if ui.state.draft_expanded:
            ui.editor.buffer.insert_text("    ")
        elif ui.editor.text.startswith("/") and (not ui.editor.buffer.complete_state or not ui.editor.buffer.complete_state.completions):
            # 命令表在内存中；避免自动补全线程启动瞬间吞掉 Tab。
            from lynx_agent.cli.completion import AgentCompleter
            values = list(AgentCompleter(ui.runner).get_completions(ui.editor.buffer.document, CompleteEvent(completion_requested=True)))
            if values:
                ui.editor.buffer.apply_completion(values[0])
        elif ui.editor.buffer.complete_state and ui.editor.buffer.complete_state.completions:
            ui.editor.buffer.complete_next()
        else:
            ui.editor.buffer.start_completion(select_first=True)

    @keys.add("escape", "c", filter=~overlay)
    def copy(event):
        ui.copy_selection()

    @keys.add("escape", "e", filter=~overlay)
    def expand(event):
        ui.toggle_editor()

    @keys.add("c-o", filter=~overlay)
    def tools(event):
        ui.toggle_last_tool()

    @keys.add("escape", "s", filter=~overlay)
    def send_draft(event):
        if ui.state.draft_expanded and not ui.state.busy:
            ui.state.draft_expanded = False
            ui.submit()

    @keys.add("escape", "enter", filter=~overlay)
    def newline(event):
        if not ui.state.busy:
            ui.editor.buffer.insert_text("\n")

    @keys.add(Keys.BracketedPaste)
    def paste(event):
        if (not ui.state.busy and not ui.state.picker_open and not ui.state.effort_open
                and not ui.state.permissions_open and not ui.state.approval_selector_open
                and not ui.state.usage_panel and not ui.state.approval_open):
            ui.editor.buffer.insert_text(event.data.replace("\r\n", "\n").replace("\r", "\n"))

    @keys.add("c-c")
    def cancel(event):
        ui.cancel()

    @keys.add("c-d")
    def exit_app(event):
        if ui.state.approval_open:
            ui.cancel()
        elif ui.state.usage_panel:
            ui.close_usage()
        elif ui.state.approval_selector_open:
            ui.close_approval()
        elif ui.state.effort_open:
            ui.close_effort()
        elif ui.state.permissions_open:
            ui.close_permissions()
        elif ui.state.picker_open:
            ui.close_picker()
        elif not ui.state.busy:
            ui.application.exit()

    @keys.add("escape", filter=picker, eager=True)
    def close_picker(event):
        ui.close_picker()

    @keys.add("escape", filter=effort, eager=True)
    def close_effort(event):
        ui.close_effort()

    @keys.add("escape", filter=permissions, eager=True)
    def close_permissions(event):
        ui.close_permissions()

    @keys.add("escape", filter=approval_selector, eager=True)
    def close_approval_selector(event):
        ui.close_approval()

    @keys.add("escape", filter=usage, eager=True)
    def close_usage(event):
        ui.close_usage()

    @keys.add("up", filter=usage)
    def usage_up(event):
        ui.usage_editor.buffer.cursor_up()

    @keys.add("down", filter=usage)
    def usage_down(event):
        ui.usage_editor.buffer.cursor_down()

    @keys.add("left", filter=permissions)
    def lower_permission(event):
        ui.pick_permission(max(0, ui.state.permissions_selected - 1))

    @keys.add("right", filter=permissions)
    def higher_permission(event):
        from lynx_agent.cli.permissions import PERMISSION_OPTIONS
        ui.pick_permission(min(len(PERMISSION_OPTIONS) - 1, ui.state.permissions_selected + 1))

    @keys.add("left", filter=approval_selector)
    def lower_approval(event):
        ui.pick_approval(max(0, ui.state.approval_selected - 1))

    @keys.add("right", filter=approval_selector)
    def higher_approval(event):
        from lynx_agent.cli.permissions import APPROVAL_OPTIONS
        ui.pick_approval(min(len(APPROVAL_OPTIONS) - 1, ui.state.approval_selected + 1))

    @keys.add("left", filter=effort)
    def lower_effort(event):
        ui.pick_effort(max(0, ui.state.effort_selected - 1))

    @keys.add("right", filter=effort)
    def higher_effort(event):
        ui.pick_effort(min(len(ui.state.effort_options) - 1, ui.state.effort_selected + 1))

    @keys.add(Keys.Any, filter=effort | permissions | approval_selector | usage)
    def ignore_editor_text(event):
        pass  # 选择档位或权限时不允许文字穿透到被保护的草稿。

    @keys.add("escape", filter=~overlay)
    def close_overlay(event):
        if ui.editor.buffer.complete_state:
            ui.editor.buffer.cancel_completion()
        elif ui.selection.start:
            ui.clear_selection()
        elif ui.state.draft_expanded:
            ui.state.draft_expanded = False

    @keys.add("up", filter=picker)
    def up(event):
        ui.move_selection(-1)

    @keys.add("down", filter=picker)
    def down(event):
        ui.move_selection(1)

    @keys.add("pageup")
    def page_up(event):
        if ui.state.usage_panel and not ui.state.approval_open:
            ui.scroll_usage(-1)
            return
        if ui.state.approval_open and ui.state.approval_kind == "batch":
            ui.runner.approval.set_page(-1)
            ui._refresh_batch_card()
            return
        ui.scroll(-1)

    @keys.add("pagedown")
    def page_down(event):
        if ui.state.usage_panel and not ui.state.approval_open:
            ui.scroll_usage(1)
            return
        if ui.state.approval_open and ui.state.approval_kind == "batch":
            ui.runner.approval.set_page(1)
            ui._refresh_batch_card()
            return
        ui.scroll(1)

    @keys.add("c-end")
    def latest(event):
        if ui.state.usage_panel and not ui.state.approval_open:
            ui.usage_editor.buffer.cursor_position = len(ui.usage_editor.text)
            return
        ui.jump_to_latest()

    return keys
