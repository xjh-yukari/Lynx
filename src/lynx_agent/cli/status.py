"""运行可取消的异步操作，独占终端显示等待状态。"""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from itertools import cycle
from typing import TypeVar

from prompt_toolkit.application import Application
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension

from lynx_agent.agent.events import AgentPhase, StatusCallback

T = TypeVar("T")


class OperationCancelled(Exception):
    """当前操作已取消并完成清理，终端可继续接收输入。"""


async def run_with_status(
    operation: Callable[[StatusCallback], Awaitable[T]], *,
    input_stream=None, output_stream=None,
    stream_fragments: list[str] | None = None,
    execution=None, approval=None,
) -> T:
    """等待任务真正结束后再返回，避免残留请求与下一条输入并发。"""
    bindings = KeyBindings()
    task: asyncio.Task[T] | None = None
    cancelling = False
    label = "正在准备请求"
    spinner = cycle("|/-\\")

    def render():
        if approval is not None and approval.pending is not None:
            from lynx_agent.approval.cards import pending_text
            return pending_text(approval.pending, approval.config.preview_page_size)
        if execution and execution.broker.pending:
            from lynx_agent.cli.execution import approval_text
            return approval_text(execution, execution.broker.pending)
        return f"{next(spinner)} {label}…  Ctrl+C 取消，完成后返回输入框"

    @bindings.add("y")
    @bindings.add("Y")
    def approve(event):
        if approval is not None and approval.pending is not None:
            approval.decide(approval.pending.identifier, "once")
        elif execution and execution.broker.pending:
            execution.broker.decide(execution.broker.pending.identifier, True)

    @bindings.add("b")
    @bindings.add("B")
    def approve_batch(event):
        if approval is not None and approval.pending is not None and approval.pending.supports_batch:
            approval.decide(approval.pending.identifier, "batch")

    @bindings.add("n")
    @bindings.add("N")
    def reject(event):
        if approval is not None and approval.pending is not None:
            approval.decide(approval.pending.identifier, "deny")
        elif execution and execution.broker.pending:
            execution.broker.decide(execution.broker.pending.identifier, False)

    @bindings.add("pageup")
    def page_up(event):
        if approval is not None and approval.pending is not None:
            approval.set_page(-1)
            app.invalidate()

    @bindings.add("pagedown")
    def page_down(event):
        if approval is not None and approval.pending is not None:
            approval.set_page(1)
            app.invalidate()

    def streamed_text():
        return "Lynx：" + "".join(stream_fragments or [])

    def stream_cursor():
        lines = streamed_text().split("\n")
        return Point(x=len(lines[-1]), y=len(lines) - 1)

    body = Window(
        FormattedTextControl(streamed_text, get_cursor_position=stream_cursor),
        height=Dimension(max=12), wrap_lines=True,
    )

    app: Application[T] = Application(
        layout=Layout(HSplit([
            ConditionalContainer(body, Condition(lambda: bool(stream_fragments))),
            Window(FormattedTextControl(render), wrap_lines=True),
        ])),
        key_bindings=bindings, full_screen=False, erase_when_done=True,
        refresh_interval=0.15, input=input_stream, output=output_stream,
    )

    def report(phase: AgentPhase) -> None:
        nonlocal label
        if not cancelling:
            label = {
                "remembering": "正在加载档案与检索长期记忆",
                "memorizing": "正在保留重要任务结论",
                "compacting": "正在压缩历史", "generating": "正在生成",
                "rewriting": "回复达到上限，正在精简重写",
                "tooling": "正在执行工具",
                "recovering": "检测到工具协议异常，正在整理最终答复",
                "pruning": "正在回收工具结果",
                "pruned": "工具结果已回收，正在重新检查预算",
                "summary_degraded": "摘要未完成，保留历史继续处理",
                "output_budget_exhausted": "生成预算不足，已保留进度",
            }[phase]
            app.invalidate()

    @bindings.add("c-c")
    @bindings.add("escape", filter=Condition(
        lambda: bool((execution and execution.broker.pending) or (approval is not None and approval.pending))), eager=True)
    def cancel(event):
        nonlocal cancelling, label
        if task is not None and not task.done() and not cancelling:
            cancelling = True
            label = "正在取消并清理请求"
            task.cancel()
            app.invalidate()

    def finished(completed: asyncio.Task[T]) -> None:
        # done 回调也覆盖任务尚未开始就被 Ctrl+C 取消的情况。
        if completed.cancelled():
            if app.is_running and not app.is_done:
                app.exit(exception=OperationCancelled())
            return
        error = completed.exception()
        if app.is_running and not app.is_done:
            if error is not None:
                app.exit(exception=error)
            else:
                app.exit(result=completed.result())

    def start() -> None:
        nonlocal task
        task = asyncio.create_task(operation(report))
        task.add_done_callback(finished)

    try:
        return await app.run_async(pre_run=start, handle_sigint=False)
    finally:
        if task is not None:
            if not task.done():
                task.cancel()
            with suppress(asyncio.CancelledError):
                await task
