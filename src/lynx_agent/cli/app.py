"""统一 TUI：一份布局管理输入、流式回复、会话和运行状态。"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from pathlib import Path
from time import monotonic

from prompt_toolkit.application import Application
from prompt_toolkit.completion import ThreadedCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.widgets import TextArea

from lynx_agent.agent import AgentRunner
from lynx_agent.agent.errors import ContextError
from lynx_agent.agent.events import AgentPhase, ReplyStreamEvent
from lynx_agent.approval.cards import pending_text
from lynx_agent.cli.activities import tool_summary
from lynx_agent.cli.bindings import build_bindings
from lynx_agent.cli.clipboard import copy_text
from lynx_agent.cli.commands import HELP_TEXT
from lynx_agent.cli.completion import AgentCompleter
from lynx_agent.cli.formatting import format_tokens, format_tools
from lynx_agent.cli.layout import build_layout
from lynx_agent.cli.message_actions import clean_text, copy_source
from lynx_agent.cli.metrics import ContextMeter, context_details
from lynx_agent.cli.permissions import (
    APPROVAL_OPTIONS,
    PERMISSION_OPTIONS,
    handle_approval_command,
    handle_approval_mode_command,
    handle_permissions_command,
)
from lynx_agent.cli.projection import TranscriptProjection, session_messages
from lynx_agent.cli.search import handle_search_command
from lynx_agent.cli.selection import Anchor, TextSelection
from lynx_agent.cli.session_browser import filter_choices, session_choices
from lynx_agent.cli.state import DisplayMessage, UIState
from lynx_agent.cli.themes import (
    load_effort,
    load_theme,
    save_effort,
    save_theme,
    theme_style,
)
from lynx_agent.cli.tool_cards import detail_text, inspect_trace
from lynx_agent.cli.usage_details import usage_details
from lynx_agent.config import Config
from lynx_agent.config.paths import get_home_path, get_project_root
from lynx_agent.providers import ProviderError
from lynx_agent.session import SessionError, SessionStore
from lynx_agent.tools.errors import ToolExecutionError


class LynxTUI:
    def __init__(
        self, runner: AgentRunner, config: Config, store: SessionStore, *,
        project: Path | None = None, input_stream=None, output_stream=None,
        clipboard_writer=None, preferences_path=None,
    ) -> None:
        self.runner = runner
        self.config = config
        self.store = store
        actual_project = runner.file_access.policy.root if runner.file_access else (project or Path.cwd())
        self.state = UIState(project=actual_project.resolve(), model=config.agent.model.name)
        self.context_meter = ContextMeter(runner, config)
        self.preferences_path = Path(preferences_path) if preferences_path else get_home_path() / "tui" / "preferences.json"
        self.state.theme = load_theme(self.preferences_path)
        saved_effort = load_effort(self.preferences_path, config.agent.model.name)
        if config.agent.model.reasoning_effort is None and saved_effort in runner.reasoning_efforts:
            runner.set_reasoning_effort(saved_effort)
        self.clipboard_writer = clipboard_writer or copy_text
        self.selection = TextSelection()
        self.projection = TranscriptProjection(self.state, runner, self.selection)
        self._pressed_action = None
        self._drag_origin = None
        self._selection_scroll_task = None
        self._selection_scroll_direction = 0
        self._selection_scroll_x = 0
        self.state.mcp_status = lambda: self.runner.mcp.label
        if self.runner.memory.enabled:
            self.state.memory_status = lambda: self.runner.memory.label
        self.state.search_backend = runner.search_backend
        self._task: asyncio.Task[None] | None = None
        self._active_reply: DisplayMessage | None = None
        self._tool_activity = None
        self.editor = TextArea(
            multiline=True, wrap_lines=True,
            style="class:input",
            height=Dimension(min=1, max=6), history=InMemoryHistory(),
            dont_extend_height=True,
            completer=ThreadedCompleter(AgentCompleter(runner)), complete_while_typing=True,
            read_only=Condition(lambda: self.state.busy or self.state.picker_open or self.state.effort_open
                                or self.state.permissions_open or self.state.approval_selector_open
                                or self.state.approval_open or bool(self.state.usage_panel)),
            get_line_prefix=lambda line, wrap: "  " if wrap else ("› " if line == 0 else "… "),
        )
        self.editor.window.height = lambda: Dimension(min=3, weight=1) if self.state.draft_expanded else Dimension(min=1, max=6)
        self.editor.window.dont_extend_height = Condition(lambda: not self.state.draft_expanded)
        self.search_editor = TextArea(height=1, multiline=False, prompt="搜索会话：")
        self.approval_editor = TextArea(read_only=True, scrollbar=True, wrap_lines=True)
        self.usage_editor = TextArea(read_only=True, scrollbar=True, wrap_lines=True)
        self.search_editor.buffer.on_text_changed += self._filter_sessions
        layout, self.transcript = build_layout(
            self.state, self.editor, self.scroll_to, self.scroll_lines, self.jump_to_latest,
            projection=self.projection, interaction=self._transcript_mouse,
            search_editor=self.search_editor, session_click=self.pick_session, copy_selection=self.copy_selection,
            meter=self.context_meter, meter_sample=self._sample_context, meter_click=self._show_context,
            effort_click=self.open_effort, effort_pick=self.pick_effort,
            permissions_click=self.open_permissions, permissions_pick=self.pick_permission,
            approval_click=self.open_approval, approval_pick=self.pick_approval, stop_click=self.stop_batch,
            cost_label=lambda: self.runner.costs.label(self.runner.session_id), cost_click=lambda: self.toggle_usage("cost"),
            approval_editor=self.approval_editor,
            usage_sample=lambda: self.runner.last_usage.breakdown(), usage_click=lambda: self.toggle_usage("tokens"),
            usage_editor=self.usage_editor,
        )
        self.application: Application[None] = Application(
            layout=layout, key_bindings=build_bindings(self), full_screen=True,
            input=input_stream, output=output_stream, mouse_support=True,
            refresh_interval=0.1,
            before_render=self._refresh_usage_panel,
            style=theme_style(self.state.theme),
        )
        self._load_transcript()
        if self.runner.approval is not None:
            self.runner.approval.presenter = self._present_batch
            self.runner.approval.on_progress = self._batch_progress
        if self.runner.execution is not None:
            self.runner.execution.broker.presenter = self._present_approval
            try:
                self.runner.shell_command("on confirm")
            except (ValueError, OSError) as error:
                self.notice(f"Shell 默认开启失败：{error}")
            self._sync_header()

    def _present_approval(self, request):
        from lynx_agent.cli.execution import approval_text
        self.state.approval_kind = "command"
        self.state.approval_supports_batch = False
        self.state.approval_open = request is not None
        if request is not None:
            self.state.usage_panel = ""
            self.approval_editor.text = approval_text(self.runner.execution, request)
            self.application.layout.focus(self.approval_editor)
            self.state.status = "等待命令审批（尚未执行）…"
        else:
            self.application.layout.focus(self.editor)
            if self.state.busy:
                self.state.status = "正在执行命令或处理审批结果…"
        self.application.invalidate()

    def _present_batch(self, pending):
        self.state.approval_kind = "batch" if pending is not None else "command"
        self.state.approval_supports_batch = bool(pending and pending.supports_batch)
        self.state.approval_open = pending is not None
        if pending is not None:
            self.state.usage_panel = ""
            self.approval_editor.text = pending_text(pending, self.runner.approval.config.preview_page_size)
            self.application.layout.focus(self.approval_editor)
            self.state.status = "等待批量任务审批（尚未执行）…"
        else:
            self.application.layout.focus(self.editor)
            if self.state.busy:
                self.state.status = "正在执行批量任务或处理审批结果…"
        self.application.invalidate()

    def _refresh_batch_card(self):
        pending = self.runner.approval.pending if self.runner.approval is not None else None
        if pending is not None and pending.kind == "batch":
            self.approval_editor.text = pending_text(pending, self.runner.approval.config.preview_page_size)
            self.application.invalidate()

    def _batch_progress(self, progress):
        self.state.batch_running = progress["phase"] not in {"finished", "stopped", "failed"}
        self.state.batch_progress = (
            f"{progress['phase']} {progress['done']}/{progress['total']} · "
            f"成功 {progress['succeeded']} · 失败 {progress['failed']} · 跳过 {progress['skipped']}"
        )
        self.state.status = self.state.batch_progress
        self.application.invalidate()

    def stop_batch(self):
        if self.runner.approval is None:
            return
        self.notice(self.runner.request_stop())
        self.application.invalidate()

    def decide_batch(self, decision):
        manager = self.runner.approval
        if manager is not None and manager.pending is not None:
            manager.decide(manager.pending.identifier, decision)

    def decide_execution(self, approved):
        service = self.runner.execution
        if service and service.broker.pending:
            service.broker.decide(service.broker.pending.identifier, approved)

    def _sync_header(self) -> None:
        self.context_meter.invalidate()
        self.state.permissions = self.runner.permission_label
        self.state.search_backend = self.runner.search_backend
        self.state.mcp = self.runner.mcp.label
        self.state.session_title = self.runner.session_title
        self.state.session_id = self.runner.session_id
        self.state.effort = self.runner.reasoning_effort
        self.state.effort_options = self.runner.reasoning_efforts
        self.state.approval = self.runner.approval_label
        self.state.shell_enabled = bool(self.runner.execution and self.runner.execution.enabled
                                        and self.runner.approval_label != "暂停执行")

    def open_approval(self):
        if self.state.busy or self.runner._operation_running:
            self.toast("请先等待或取消当前请求，再切换审批档位。")
            return
        if self.runner.approval is None:
            self.toast("当前入口未启用审批策略。")
            return
        self.editor.buffer.cancel_completion()
        self.state.usage_panel = ""
        self.state.effort_open = False
        self.state.picker_open = False
        self.state.permissions_open = False
        self.state.approval_selector_open = False
        self.state.approval_open = False
        self.state.approval_selected = next(
            (index for index, (_, label) in enumerate(APPROVAL_OPTIONS) if label == self.state.approval), 1)
        self.state.approval_selector_open = True
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def close_approval(self):
        self.state.approval_selector_open = False
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def pick_approval(self, index):
        if self.state.busy or self.runner._operation_running or not 0 <= index < len(APPROVAL_OPTIONS):
            return
        if self.runner.approval is None:
            return
        mode, _ = APPROVAL_OPTIONS[index]
        if mode != self.runner.approval.mode:
            self.runner.set_approval_mode(mode)
            self.clear_selection()
        self.state.approval_selected = index
        self._sync_header()
        self.application.invalidate()

    def open_effort(self, *, allow_dispatch=False):
        if self.state.busy and not allow_dispatch:
            self.toast("请先等待或取消当前请求，再调整推理强度。")
            return
        if not self.runner.reasoning_efforts:
            self.notice("当前模型尚未声明可调整的推理档位，未修改请求参数。")
            return
        if not self.state.busy:
            self.editor.buffer.cancel_completion()
        self.state.usage_panel = ""
        self.state.permissions_open = False
        self.state.picker_open = False
        self.state.approval_selector_open = False
        self.state.effort_open = True
        self.state.effort_selected = self.state.effort_options.index(self.state.effort) if self.state.effort in self.state.effort_options else 0
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def close_effort(self):
        self.state.effort_open = False
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def open_permissions(self):
        if self.state.busy or self.runner._operation_running:
            self.toast("请先等待或取消当前请求，再切换权限。")
            return
        if self.runner.file_access is None:
            self.toast("当前入口未启用本地文件权限。")
            return
        self.editor.buffer.cancel_completion()
        self.state.usage_panel = ""
        self.state.effort_open = False
        self.state.picker_open = False
        self.state.approval_selector_open = False
        self.state.permissions_open = True
        mode = self.runner.file_access.policy.mode
        self.state.permissions_selected = next(index for index, (value, _) in enumerate(PERMISSION_OPTIONS) if value == mode)
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def close_permissions(self):
        self.state.permissions_open = False
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def pick_permission(self, index):
        if self.state.busy or self.runner._operation_running or not 0 <= index < len(PERMISSION_OPTIONS):
            return
        if self.runner.file_access is None:
            return
        mode, _ = PERMISSION_OPTIONS[index]
        if mode != self.runner.file_access.policy.mode:
            self.runner.set_file_mode(mode)
            self.clear_selection()
        self.state.permissions_selected = index
        self._sync_header()
        self.application.invalidate()

    def pick_effort(self, index):
        if self.state.busy or not 0 <= index < len(self.state.effort_options):
            return
        self._apply_effort(self.state.effort_options[index])
        self.state.effort_selected = index
        self.application.invalidate()

    def _apply_effort(self, effort):
        self.runner.set_reasoning_effort(effort)
        self._sync_header()
        try:
            save_effort(self.preferences_path, self.state.model, effort)
        except (OSError, ValueError):
            self.notice("推理强度已应用，但偏好保存失败；下次启动可能恢复原档位。")

    def _sample_context(self):
        # 已经提交到上下文的回复不能作为流式草稿再计算一次。
        reply = self._active_reply.content if self._active_reply and self.runner.context_meter_inflight else ""
        return self.context_meter.sample(self.editor.text, self.state.busy, reply)

    def _show_context(self):
        self.notice(context_details(self._sample_context()))
        self.application.invalidate()

    def toggle_usage(self, view="tokens"):
        if self.state.approval_open:
            return  # 审批卡始终优先，统计不能盖住待审批操作。
        if self.state.usage_panel == view:
            self.close_usage()
            return
        self.editor.buffer.cancel_completion()
        self.state.picker_open = False
        self.state.effort_open = False
        self.state.permissions_open = False
        self.state.approval_selector_open = False
        self.state.usage_panel = view
        self.usage_editor.text = usage_details(self.runner, self.config, view)
        self.application.layout.focus(self.usage_editor)
        self.application.invalidate()

    def close_usage(self):
        self.state.usage_panel = ""
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def _refresh_usage_panel(self, _application=None):
        if not self.state.usage_panel:
            return
        text = usage_details(self.runner, self.config, self.state.usage_panel)
        if self.usage_editor.text != text:
            from prompt_toolkit.document import Document

            position = min(self.usage_editor.buffer.cursor_position, len(text))
            self.usage_editor.buffer.set_document(Document(text, position), bypass_readonly=True)

    def scroll_usage(self, direction):
        info = self.usage_editor.window.render_info
        lines = max(1, info.window_height - 1) if info else 10
        if direction < 0:
            self.usage_editor.buffer.cursor_up(count=lines)
        else:
            self.usage_editor.buffer.cursor_down(count=lines)

    def _load_transcript(self) -> None:
        self.state.usage_panel = ""
        self.projection.clear()
        self.state.selection_available = False
        self.state.history_end_id = None
        self.state.messages.clear()
        if self.runner.session_id is not None:
            session = self.store.load(self.runner.session_id)
            self.state.messages.extend(session_messages(session))
        if not self.state.messages:
            # 开场由 agent 本人发一条问候（助手身份，不是系统提示）；
            # 用独立角色，避免污染“第一条 Lynx 消息就是模型回答”的既有语义。
            self.state.messages.append(DisplayMessage(
                "问候", "你好，我是 Lynx。有什么要做的直接说：读写文件、执行命令、查资料、记事情；"
                        "需要时用 /help 看命令与快捷键。"))
        if self.runner.can_retry:
            self.notice("当前会话有未完成任务：/retry 仅整理已有结果；/diagnostics 查看诊断。发送新任务会放弃该恢复入口。")
        self._sync_header()
        self.state.follow_tail = True
        self.state.unread_activity = False
        self.state.scroll_line = 0

    def _open_link(self, url) -> None:
        """打开正文里的链接：只接受 http(s)，失败只提示不抛错。"""
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            return
        import webbrowser
        try:
            webbrowser.open(url)
        except OSError:
            self.notice(f"无法打开链接：{url}")
        self.application.invalidate()

    def _show_cost(self) -> None:
        self.notice(self.runner.costs.details(self.runner.session_id))
        self.application.invalidate()

    def notice(self, text: str) -> None:
        self.state.messages.append(DisplayMessage("系统", text))
        if not self.state.follow_tail:
            self.state.unread_activity = True

    def _finish_tool_activity(self, *, interrupted=False):
        if self._tool_activity is None:
            return
        message, started, known = self._tool_activity
        traces = [trace for trace in self.runner.tool_traces if (trace.turn_id, trace.call.id) not in known]
        message.content = tool_summary(traces, seconds=max(0, monotonic() - started), interrupted=interrupted)
        message.tools = tuple(traces)
        self._tool_activity = None
        if not self.state.follow_tail:
            self.state.unread_activity = True

    def _status(self, phase: AgentPhase) -> None:
        if phase == "tooling" and self._tool_activity is None and self._active_reply is not None:
            message = DisplayMessage("工具", "… 正在执行工具")
            index = next(index for index, item in enumerate(self.state.messages) if item is self._active_reply)
            self.state.messages.insert(index, message)
            if not self.state.follow_tail:
                self.state.unread_activity = True
            known = {(trace.turn_id, trace.call.id) for trace in self.runner.tool_traces}
            self._tool_activity = (message, monotonic(), known)
        elif phase != "tooling":
            self._finish_tool_activity()
        if not self.state.cancelling:
            self.state.status = {
                "remembering": "正在加载档案与检索长期记忆…",
                "memorizing": "正在保留重要任务结论…",
                "compacting": "正在压缩历史…", "generating": "正在生成…",
                "rewriting": "回复达到上限，正在精简重写…",
                "tooling": "正在执行工具…",
                "recovering": "检测到工具协议异常，正在整理最终答复…",
                "pruning": "正在回收工具结果…",
                "pruned": "工具结果已回收，正在重新检查预算…",
                "summary_degraded": "摘要未完成，当前请求仍在预算内，继续处理…",
                "output_budget_exhausted": "生成预算不足，已保留进度…",
            }[phase]
            if phase == "pruned" and self.runner.last_pruning is not None:
                self.state.status = f"已回收 {len(self.runner.last_pruning.replacements)} 条工具结果，正在重新检查预算…"
            if phase == "summary_degraded":
                self.notice("历史摘要未完成；当前请求仍在输入预算内，保留原历史继续处理。本轮不重复尝试同一摘要。")
        self._sync_header()
        self.application.invalidate()

    def _launch(self, operation: Callable[[], Awaitable[None]]) -> None:
        self.state.busy = True
        self.state.status = "正在处理…"
        self._task = self.application.create_background_task(self._guard(operation))
        self._task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[None]) -> None:
        if task is not self._task:
            return
        # 覆盖任务进入 guard 前就被取消的情况。
        if self.state.busy:
            self.state.finish_request()
            self.state.busy = False
            self.state.cancelling = False
            self.state.status = "就绪"
            self.notice("操作已取消。")
        if task.cancelled():
            self.application.invalidate()

    async def _guard(self, operation: Callable[[], Awaitable[None]]) -> None:
        previous = self.runner.last_compaction
        try:
            await operation()
        except asyncio.CancelledError:
            self.notice("已取消本次操作，未完成的回合未写入历史。")
            if self.runner.last_compaction is not None and self.runner.last_compaction is not previous:
                self.notice("本次已完成的历史压缩已保留。")
        except (ProviderError, ContextError, SessionError, ValueError, ToolExecutionError) as error:
            self.notice(f"操作失败：{error}")
            if self.runner.can_retry:
                self.notice("未完成任务保留在内存；请先 /save 成功后再 /retry。" if self.runner.persistence_error else
                            "未完成任务已保留，可用 /retry 仅整理答复，不重做工具；详情用 /diagnostics 查看。")
        except Exception as error:  # noqa: BLE001 - 隔离 UI 后台任务，避免异常日志破坏终端布局
            self.notice(f"发生内部错误：{type(error).__name__}，可退出后检查代码。")
        finally:
            self.state.finish_request()
            self.state.busy = False
            self.state.cancelling = False
            self.state.batch_running = False
            self.state.status = "就绪"
            self._sync_header()
            self.application.invalidate()

    def submit(self) -> None:
        text = self.editor.text
        if not text.strip() or self.state.busy:
            return
        self.editor.buffer.append_to_history()
        self.editor.buffer.reset()
        self.clear_selection()
        self.state.history_end_id = None
        self.state.follow_tail = True
        self.state.unread_activity = False
        self._launch(lambda: self._dispatch(text))

    async def _dispatch(self, text: str) -> None:
        command = text.strip() if "\n" not in text and "\r" not in text else ""
        verb, _, argument = command.partition(" ")
        if command.lower() in {"exit", "quit", "/exit"}:
            self.application.exit()
        elif command.lower() in {"help", "/help"}:
            self.notice(HELP_TEXT + "\nTUI 导航\n  滚轮 / 拖动右侧滚动条 / PageUp / PageDown 浏览对话\n  Ctrl+End 回到最新\n  拖动正文选择；Alt+C 或点击底栏复制所选；回答下方 [复制] 一键复制该条回答\n  点击工具卡详情可展开、翻页和复制历史快照，不重新执行工具\n  /copy [plain|markdown|code 1] 复制最近回答；不调用模型\n  输入 / 显示命令菜单，Tab 补全；@ 插入受权限限制的文件路径引用，不自动附加正文\n  /sessions 可搜索标题和预览，点击或回车进入；Esc 返回并保留草稿\n  Alt+E 或 /editor 展开草稿；Enter 换行，Alt+S 发送，Esc 收起\n  /theme [ansi|dark|light] 设置并保存界面主题\n  /speed [0–10000] 设置显示字符/秒；0 跟随 API\n  /workspace 查看完整目录；长历史顶部可加载更早记录\n  /approval [ask|auto|paused|revoke|stop] 独立审批档位；完全权限不取消审批\n  /approvals 查看批量任务卡片；Y 允许一次、B 允许本批次、N 拒绝、PgUp/PgDn 翻页")
        elif verb == "/speed":
            if argument:
                speed = int(argument)
                if not 0 <= speed <= 10000:
                    raise ValueError("显示速度范围为 0–10000；0 表示跟随 API。")
                self.state.stream_speed = speed
            speed = self.state.stream_speed
            self.notice(f"流式显示：{speed} 字符/秒。" if speed else "流式显示：直接跟随 API，不额外限速。")
        elif verb == "/effort":
            if not argument:
                self.open_effort(allow_dispatch=True)
            elif argument == "status":
                self.notice(f"当前推理强度：{self.runner.reasoning_effort or '服务端默认（未声明档位）'}；可选：{' / '.join(self.runner.reasoning_efforts) or '无'}。")
            else:
                self._apply_effort(argument.lower())
                self.notice(f"推理强度已设为 {self.runner.reasoning_effort}，从下一次请求生效。")
        elif verb == "/theme":
            if argument:
                self.application.style = theme_style(argument)
                self.state.theme = argument
                try:
                    save_theme(self.preferences_path, argument)
                except (OSError, ValueError):
                    self.notice("主题已应用，但偏好保存失败；下次启动可能恢复旧主题。")
            self.notice(f"当前主题：{self.state.theme}；可选 ansi / dark / light。")
        elif command == "/editor":
            self.toggle_editor()
        elif verb == "/copy":
            self.copy_selection() if not argument and self.selection.snapshot else self.copy_latest(argument or "plain")
        elif verb == "/tokens":
            self.notice(format_tokens(self.runner, self.config, argument or None))
        elif verb == "/cost":
            self._show_cost()
        elif command == "/tools":
            self.notice(format_tools(self.runner, self.config))
        elif verb == "/mcp":
            self.state.status = "正在处理 MCP 连接…"
            self.notice(await self.runner.mcp.command(argument))
        elif verb == "/training":
            self.state.status = "正在处理训练基础查询…"
            self.notice(await self.runner.training_command(argument))
        elif verb in {"/memory", "/profile"}:
            self.state.status = "正在处理长期记忆…"
            self.notice(await self.runner.memory_command(verb, argument))
        elif command == "/diagnostics":
            from lynx_agent.cli.formatting import format_diagnostics
            self.notice(format_diagnostics(self.runner))
        elif command == "/retry":
            await self._ask("继续上次未完成任务（仅整理已有结果）", retry=True)
        elif verb == "/search":
            self.notice(handle_search_command(self.runner, argument))
        elif verb == "/shell":
            self.notice(self.runner.shell_command(argument))
        elif verb == "/permissions":
            self.notice(handle_permissions_command(self.runner, argument))
        elif verb == "/approval":
            self.notice(handle_approval_mode_command(self.runner, argument))
        elif verb in {"/approvals", "/approve", "/reject"}:
            self.notice(handle_approval_command(self.runner, verb, argument))
        elif command == "/workspace":
            self.notice(
                f"当前项目目录：{self.state.project}\n源码目录：{get_project_root()}\n"
                f"运行数据目录：{get_home_path()}\n文件工具工作区：{self.runner.file_access.policy.root if self.runner.file_access else '未启用'}\n"
                "文件工具只按当前权限和固定工作区操作，不随终端启动位置变化。"
            )
        elif command == "/sessions":
            self.state.usage_panel = ""
            sessions = await asyncio.to_thread(self.store.list_sessions)
            self.state.session_all_choices = session_choices(sessions)
            self.search_editor.text = ""
            self._filter_sessions()
            self.state.selected = 0
            self.state.picker_open = True
            self.application.layout.focus(self.search_editor)
        elif command == "/new":
            await self._switch(None)
        elif verb == "/rename":
            self.runner.rename_session(argument)
            self.notice("会话名称已更新。")
        elif command == "/clear":
            self.runner.clear_history()
            self.notice("对话上下文已清空，之前的原文仍在当前 Session 中。")
        elif command == "/save":
            self.runner.save_session()
            self.notice("会话已保存。" if self.runner.session_id else "尚未创建会话，请先发送消息。")
        elif command == "/compact":
            self.state.start_request()
            result = await self.runner.compact(on_status=self._status)
            if result is None:
                self.notice("没有超出近期保留范围的旧历史，无需压缩。")
            else:
                self.notice(f"压缩完成：约 {result.before_tokens:,} → {result.after_tokens:,} Tokens。")
        elif command.startswith("/"):
            self.notice("未知命令，输入 /help 查看帮助。")
        else:
            await self._ask(text)

    async def _ask(self, text: str, *, retry=False) -> None:
        self.state.start_request()
        user = DisplayMessage("你", text)
        assistant = DisplayMessage("Lynx", "")
        self._active_reply = assistant
        self.state.messages.extend([user, assistant])
        pending = ""

        async def animate() -> None:
            nonlocal pending
            credit = 0.0
            while True:
                await asyncio.sleep(1 / 30)
                credit = credit + self.state.stream_speed / 30 if pending else 0.0
                count = int(credit)
                credit -= count
                assistant.content += pending[:count]
                pending = pending[count:]
                self.application.invalidate()

        def on_text(fragment: str) -> None:
            nonlocal pending
            if fragment and not self.state.follow_tail:
                self.state.unread_activity = True
            if self.state.stream_speed:
                pending += fragment
            else:
                assistant.content += pending + fragment
                pending = ""
            self.application.invalidate()

        def on_stream(event: ReplyStreamEvent) -> None:
            nonlocal pending
            if event.kind == "start":
                bounds = self.selection.bounds(self.state.messages)
                if bounds:
                    order = {message.id: index for index, message in enumerate(self.state.messages)}
                    if order[bounds[0].message_id] <= order[assistant.id] <= order[bounds[1].message_id]:
                        self.clear_selection()
                pending = ""
                assistant.content = ""
                assistant.role = "Lynx（正在精简重写）" if event.attempt else "Lynx"
                self.application.invalidate()
            else:
                on_text(event.text)

        animation = asyncio.create_task(animate())
        try:
            response = (await self.runner.retry(on_status=self._status, on_stream=on_stream) if retry else
                        await self.runner.ask(text, on_status=self._status, on_stream=on_stream))
        except asyncio.CancelledError:
            user.role = "你（已取消，未保存）"
            assistant.role = "Lynx（部分内容，未保存）"
            raise
        except (ProviderError, ContextError, SessionError):
            user.role = "你（本轮失败，未保存）"
            assistant.role = "Lynx（部分内容，未保存）"
            raise
        finally:
            self._finish_tool_activity(interrupted=True)
            self._active_reply = None
            animation.cancel()
            with suppress(asyncio.CancelledError):
                await animation
            # API 完成或取消时立即显示已收到的尾部，不延迟保存和中断。
            assistant.content += pending
        assistant.content = response.content
        assistant.role = "Lynx"
        if self.runner.memory.retention_notice:
            self.notice(self.runner.memory.retention_notice)
        if self.runner.memory.enabled and self.runner.memory.warnings:
            self.notice(self.runner.memory.status())
        if self.runner.last_pruning is not None:
            self.notice(f"已回收 {len(self.runner.last_pruning.replacements)} 条工具结果；估算变化可用 /tokens 查看。")
        if self.runner.last_compaction is not None:
            result = self.runner.last_compaction
            self.notice(f"已自动压缩：约 {result.before_tokens:,} → {result.after_tokens:,} Tokens。")
        if self.runner.persistence_error is not None:
            self.notice(f"回答已生成，但保存失败：{self.runner.persistence_error}；可使用 /save 重试。")
        if response.finish_reason == "length":
            user.role = "你（本轮未完成）"
            assistant.role = "Lynx（截断草稿，未加入上下文）"
            self.notice("回复仍达到输出上限，自动重写已停止。本轮未加入上下文；草稿已记录，请拆分任务或提高上限。")
            if self.runner.can_retry:
                self.notice("工具执行记录和未完成任务已保存。可用 /diagnostics 查原因，调整配置后 /retry 仅整理答复，不重做工具。")

    async def _switch(self, identifier: str | None) -> None:
        self.runner.switch_session(identifier)
        self._load_transcript()

    def accept_session(self) -> None:
        if not self.state.session_choices:
            return
        identifier, _ = self.state.session_choices[self.state.selected]
        self.close_picker()
        self._launch(lambda: self._switch(identifier))

    def move_selection(self, direction: int) -> None:
        if self.state.session_choices:
            self.state.selected = (self.state.selected + direction) % len(self.state.session_choices)

    def cancel(self) -> None:
        if self.state.approval_open:
            if self.state.approval_kind == "batch":
                self.decide_batch("deny")
            else:
                self.decide_execution(False)
                if self._task is not None and not self._task.done():
                    self.state.cancelling = True
                    self._task.cancel()
        elif self.state.usage_panel:
            self.close_usage()
        elif self.state.approval_selector_open:
            self.close_approval()
        elif self.state.effort_open:
            self.close_effort()
        elif self.state.permissions_open:
            self.close_permissions()
        elif self.state.picker_open:
            self.close_picker()
        elif self.state.busy:
            if self._task is not None and not self._task.done() and not self.state.cancelling:
                self.state.cancelling = True
                self.state.status = "正在取消并清理请求…"
                self._task.cancel()
        else:
            self.editor.buffer.reset()
            self.notice("已取消当前草稿。")

    def scroll(self, direction: int) -> None:
        info = self.transcript.render_info
        height = max(1, info.window_height - 1) if info else 10
        self.scroll_lines(direction * height)

    def jump_to_latest(self) -> None:
        self.state.history_end_id = None
        self.clear_selection()
        self.state.follow_tail = True
        self.state.unread_activity = False
        self.application.invalidate()

    def scroll_lines(self, amount: int) -> None:
        info = self.transcript.render_info
        start = info.first_visible_line() if info and self.state.follow_tail else self.state.scroll_line
        self.scroll_to(start + amount)

    def scroll_to(self, line: int) -> None:
        info = self.transcript.render_info
        height = info.window_height if info else 10
        self.projection.update(self.application.output.get_size().columns - 1)
        last = len(self.projection.rows)
        maximum = max(0, last - height)
        self.state.scroll_line = max(0, min(maximum, line))
        self.state.follow_tail = self.state.scroll_line >= maximum
        if self.state.follow_tail:
            self.state.unread_activity = False
        self.transcript.vertical_scroll = self.state.scroll_line
        self.application.invalidate()

    def toast(self, text):
        self.state.toast = text
        self.state.toast_until = monotonic() + 4
        self.application.invalidate()

    def clear_selection(self):
        self._selection_scroll_direction = 0
        if self._selection_scroll_task is not None and not self._selection_scroll_task.done():
            self._selection_scroll_task.cancel()
        self.selection.clear()
        self.state.selection_available = False
        self.application.invalidate()

    def _transcript_mouse(self, event):
        if event.button != MouseButton.LEFT:
            return NotImplemented
        point = (event.position.x, event.position.y)
        glyph = self.projection.hit(*point)
        anchor = glyph.anchor if glyph else None
        if anchor and anchor.offset < 0:
            anchor = None
        if event.event_type == MouseEventType.MOUSE_DOWN:
            self.clear_selection()
            self._pressed_action = glyph.action if glyph else None
            self._drag_origin = point
            if anchor:
                info = self.transcript.render_info
                if info:
                    self.state.scroll_line = info.first_visible_line()
                self.state.follow_tail = False
                self.selection.begin(anchor)
        elif event.event_type == MouseEventType.MOUSE_MOVE and self.selection.dragging:
            if anchor:
                self.selection.extend(Anchor(anchor.message_id, anchor.offset + 1))
            info = self.transcript.render_info
            direction = -1 if info and event.position.y <= info.first_visible_line() else 1 if info and event.position.y >= info.last_visible_line() else 0
            self._selection_scroll_direction = direction
            self._selection_scroll_x = event.position.x
            if direction and (self._selection_scroll_task is None or self._selection_scroll_task.done()):
                self._selection_scroll_task = self.application.create_background_task(self._selection_autoscroll())
        elif event.event_type == MouseEventType.MOUSE_UP:
            action = glyph.action if glyph else None
            if self._pressed_action and action == self._pressed_action and point == self._drag_origin:
                self._perform_action(action)
            elif self.selection.dragging:
                if point == self._drag_origin:
                    self.clear_selection()
                else:
                    if anchor:
                        self.selection.extend(Anchor(anchor.message_id, anchor.offset + 1))
                    try:
                        self.selection.finish(self.state.messages, self.projection.plain)
                        self.state.selection_available = bool(self.selection.snapshot)
                    except (ToolExecutionError, ValueError):
                        self.clear_selection()
                        self.toast("当前权限下无法复制该快照")
            self._pressed_action = None
        self.application.invalidate()
        return None

    async def _selection_autoscroll(self):
        while self.selection.dragging and self._selection_scroll_direction:
            self.scroll_lines(self._selection_scroll_direction)
            self.state.follow_tail = False
            info = self.transcript.render_info
            if info:
                row = info.first_visible_line() if self._selection_scroll_direction < 0 else info.last_visible_line()
                glyph = self.projection.hit(self._selection_scroll_x, row)
                if glyph and glyph.anchor and glyph.anchor.offset >= 0:
                    self.selection.extend(Anchor(glyph.anchor.message_id, glyph.anchor.offset + 1))
            self.application.invalidate()
            await asyncio.sleep(0.08)

    def _copy(self, text):
        async def perform():
            try:
                await self.clipboard_writer(text)
            except Exception:  # noqa: BLE001 -- 剪贴板是外部桌面边界，失败不能破坏正在运行的 TUI。
                self.toast("复制失败：系统剪贴板不可用，请稍后重试")
            else:
                self.clear_selection()
                self.toast(f"已复制 {len(text)} 个字符")
        self.application.create_background_task(perform())

    def copy_selection(self):
        if not self.selection.snapshot:
            self.toast("请先拖动选择正文，或点击回答下方的复制按钮")
            return
        bounds = self.selection.bounds(self.state.messages)
        if bounds is None:
            self.clear_selection()
            return
        selected = False
        try:
            for message in self.state.messages:
                if message.id == bounds[0].message_id:
                    selected = True
                if selected and message.tools:
                    detail_text(message, self.runner)  # 复制前复查权限和归属。
                if message.id == bounds[1].message_id:
                    break
        except (ToolExecutionError, ValueError):
            self.toast("当前权限下无法复制该工具快照")
            return
        self._copy(self.selection.snapshot)

    def copy_latest(self, kind="plain"):
        message = next((message for message in reversed(self.state.messages) if message.role.startswith("Lynx") and message.content), None)
        if message is None:
            self.toast("没有可以复制的模型回答")
            return
        if kind.startswith("code "):
            self._perform_action(("code", message.id, int(kind.split()[1]) - 1))
        elif kind in {"plain", "markdown"}:
            self._perform_action(("copy", message.id, kind))
        else:
            self.toast("用法：/copy [plain|markdown|code 1]")

    def _perform_action(self, action):
        try:
            kind = action[0]
            if kind == "older":
                self.state.history_end_id = self.projection.start_id
                self.state.follow_tail = False
                self.state.scroll_line = 0
                self.clear_selection()
                return
            if kind == "latest":
                self.jump_to_latest()
                return
            message = next((message for message in self.state.messages if message.id == action[1]), None)
            if message is None:
                raise ValueError("消息已经失效")
            if kind == "copy":
                self._copy(copy_source(message, action[2]))
            elif kind == "code":
                self._copy(copy_source(message, "code", action[2]))
            elif kind in {"tool", "tool_page", "tool_copy", "tool_copy_all"}:
                if kind == "tool":
                    message.expanded = not message.expanded
                elif kind == "tool_page":
                    message.detail_page = max(0, message.detail_page + action[2])
                text, _ = detail_text(message, self.runner)
                if kind == "tool_copy":
                    self._copy(text)
                elif kind == "tool_copy_all":
                    self._copy("\n\n".join(clean_text(inspect_trace(self.runner, trace)) for trace in message.tools))
                self.clear_selection()
            elif kind == "open":
                self._open_link(action[1])
            elif kind == "more":
                message.visible_chars += 12000
            self.application.invalidate()
        except (ToolExecutionError, ValueError, IndexError):
            self.toast("当前操作不可用：消息、代码块或权限已改变")

    def toggle_editor(self):
        if self.state.busy or self.state.picker_open:
            return
        self.state.draft_expanded = not self.state.draft_expanded
        self.editor.buffer.cancel_completion()
        self.application.invalidate()

    def toggle_last_tool(self):
        message = next((message for message in reversed(self.state.messages) if message.tools), None)
        if message is None:
            self.toast("没有可以展开的工具快照")
        else:
            self._perform_action(("tool", message.id))

    def _filter_sessions(self, _=None):
        self.state.session_query = self.search_editor.text
        self.state.session_choices = filter_choices(self.state.session_all_choices, self.search_editor.text)
        self.state.selected = 0
        self.application.invalidate() if hasattr(self, "application") else None

    def close_picker(self):
        self.state.picker_open = False
        self.application.layout.focus(self.editor)
        self.application.invalidate()

    def pick_session(self, index):
        if not self.state.busy and 0 <= index < len(self.state.session_choices):
            self.state.selected = index
            self.accept_session()

    async def run(self) -> None:
        try:
            await self.application.run_async(handle_sigint=False)
        except EOFError:
            pass
        finally:
            if self.runner.execution:
                self.runner.execution.disable()
            if self._task is not None and not self._task.done():
                self._task.cancel()
                with suppress(asyncio.CancelledError):
                    await self._task
