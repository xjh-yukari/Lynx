"""支持多行编辑和粘贴的异步聊天输入框。"""

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys

from lynx_agent.agent.events import ReplyStreamEvent
from lynx_agent.cli.status import run_with_status


class ChatInput:
    """复用一个终端输入会话；输入历史只保存在当前进程内。"""

    def __init__(self, *, input_stream=None, output_stream=None) -> None:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event):
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def newline(event):
            event.current_buffer.insert_text("\n")

        @bindings.add(Keys.BracketedPaste)
        def paste(event):
            # 粘贴事件整体插入，内含的换行不会作为 Enter 发送。
            text = event.data.replace("\r\n", "\n").replace("\r", "\n")
            event.current_buffer.insert_text(text)

        @bindings.add("c-c")
        def cancel(event):
            event.app.exit(exception=KeyboardInterrupt())

        @bindings.add("c-d")
        def exit_chat(event):
            event.app.exit(exception=EOFError())

        self.session: PromptSession[str] = PromptSession(
            multiline=True,
            key_bindings=bindings,
            prompt_continuation=lambda width, line, soft_wrap: " " * width if soft_wrap else "… ",
            input=input_stream,
            output=output_stream,
            enable_open_in_editor=False,
        )
        self.execution = None
        self.approval = None

    async def read(self) -> str:
        """等待整条消息；保留正文空白及换行，Ctrl+C 抛出中断。"""
        return await self.session.prompt_async("你：")

    async def run_operation(self, operation):
        """等待操作期间复用同一终端，Ctrl+C 只取消该操作。"""
        return await run_with_status(
            operation, input_stream=self.session.input, output_stream=self.session.output,
            execution=self.execution, approval=self.approval,
        )

    async def run_stream_operation(self, operation):
        """逐段刷新文本区域，同时保持状态行和取消按键可用。"""
        fragments: list[str] = []

        def on_text(text: str | ReplyStreamEvent) -> None:
            if isinstance(text, ReplyStreamEvent):
                if text.kind == "start":
                    fragments.clear()
                else:
                    fragments.append(text.text)
            else:
                fragments.append(text)
            get_app().invalidate()

        return await run_with_status(
            lambda status: operation(status, on_text),
            input_stream=self.session.input, output_stream=self.session.output,
            stream_fragments=fragments,
            execution=self.execution, approval=self.approval,
        )
