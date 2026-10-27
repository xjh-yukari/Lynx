"""通过真实终端按键验证状态显示、取消和清理。"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from lynx_agent.agent import AgentRunner
from lynx_agent.cli import commands
from lynx_agent.cli.input import ChatInput
from lynx_agent.cli.status import OperationCancelled, run_with_status
from lynx_agent.config import (
    AgentConfig,
    Config,
    ModelConfig,
    ProviderConfig,
    ProvidersConfig,
)
from lynx_agent.providers import ModelProvider, ModelResponse
from lynx_agent.session import SessionStore


class RecordingOutput(DummyOutput):
    def __init__(self, expected):
        self.expected = expected
        self.text = ""
        self.visible = asyncio.Event()

    def write(self, data):
        self.text += data
        if self.expected in self.text:
            self.visible.set()

    def write_raw(self, data):
        self.write(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase, label", [("generating", "正在生成"), ("compacting", "正在压缩历史")])
async def test_cancel_cleans_task_and_returns_to_input(phase, label):
    cleaned = asyncio.Event()
    # renderer 只输出发生变化的字符，“正在”前缀可能被复用而不重绘。
    output = RecordingOutput(label.removeprefix("正在"))

    async def operation(report):
        report(phase)
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    with create_pipe_input() as pipe:
        async def cancel_when_visible():
            await output.visible.wait()
            pipe.send_text("\x03")

        controller = asyncio.create_task(cancel_when_visible())
        with pytest.raises(OperationCancelled):
            await asyncio.wait_for(run_with_status(
                operation, input_stream=pipe, output_stream=output,
            ), timeout=3)
        await controller
        assert cleaned.is_set()
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        pipe.send_text("取消后继续提问\r")
        assert await asyncio.wait_for(reader.read(), timeout=3) == "取消后继续提问"


@pytest.mark.asyncio
async def test_early_ctrl_c_does_not_hang():
    async def operation(report):
        await asyncio.Event().wait()

    with create_pipe_input() as pipe:
        pipe.send_text("\x03")
        with pytest.raises(OperationCancelled):
            await asyncio.wait_for(run_with_status(
                operation, input_stream=pipe, output_stream=DummyOutput(),
            ), timeout=3)


@pytest.mark.asyncio
async def test_second_ctrl_c_does_not_interrupt_cleanup():
    running = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def operation(report):
        report("generating")
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()

    with create_pipe_input() as pipe:
        async def controller():
            await running.wait()
            pipe.send_text("\x03")
            await cleaning.wait()
            pipe.send_text("\x03")
            await asyncio.sleep(0.05)
            release.set()

        control_task = asyncio.create_task(controller())
        with pytest.raises(OperationCancelled):
            await asyncio.wait_for(run_with_status(
                operation, input_stream=pipe, output_stream=DummyOutput(),
            ), timeout=3)
        await control_task
        assert finished.is_set()


@pytest.mark.asyncio
async def test_completed_result_is_not_reported_as_cancelled():
    running = asyncio.Event()

    async def operation(report):
        running.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # 模拟操作已提交，返回实际完成结果，UI 不得谎报取消成功。
            return "本轮已保存"

    with create_pipe_input() as pipe:
        async def controller():
            await running.wait()
            pipe.send_text("\x03")

        control_task = asyncio.create_task(controller())
        result = await asyncio.wait_for(run_with_status(
            operation, input_stream=pipe, output_stream=DummyOutput(),
        ), timeout=3)
        await control_task
        assert result == "本轮已保存"


@pytest.mark.asyncio
async def test_original_error_propagates_and_next_operation_works():
    async def fail(report):
        raise ValueError("模拟失败")

    async def success(report):
        report("generating")
        return "正常回复"

    with create_pipe_input() as pipe:
        with pytest.raises(ValueError, match="模拟失败"):
            await run_with_status(fail, input_stream=pipe, output_stream=DummyOutput())
        assert await run_with_status(success, input_stream=pipe, output_stream=DummyOutput()) == "正常回复"


@pytest.mark.asyncio
async def test_cli_cancel_then_send_again_in_same_session(tmp_path, monkeypatch, capsys):
    config = Config(
        agent=AgentConfig(model=ModelConfig(name="fake")),
        providers=ProvidersConfig(deepseek=ProviderConfig(
            api_key="unused", base_url="https://example.invalid",
        )),
    )
    store = SessionStore(tmp_path)
    provider = AsyncMock(spec=ModelProvider)
    started = asyncio.Event()
    cleaned = asyncio.Event()
    calls = 0

    async def generate(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()
        assert cleaned.is_set()
        return ModelResponse(content="第二次回答成功", model="fake", finish_reason="stop")

    provider.generate.side_effect = generate
    runner = AgentRunner(config, provider, store)
    monkeypatch.setattr(commands, "load_config", lambda: config)
    monkeypatch.setattr(commands.AgentRunner, "from_config", lambda config: runner)
    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=DummyOutput())
        monkeypatch.setattr(commands, "ChatInput", lambda: reader)
        original_read = reader.read
        reads = 0

        async def read():
            nonlocal reads
            reads += 1
            if reads == 2:
                pipe.send_text("第二次输入\r")
            elif reads == 3:
                pipe.send_text("/exit\r")
            return await original_read()

        monkeypatch.setattr(reader, "read", read)

        async def cancel_request():
            await started.wait()
            pipe.send_text("\x03")

        controller = asyncio.create_task(cancel_request())
        pipe.send_text("第一次输入\r")
        await asyncio.wait_for(commands.chat(), timeout=3)
        await controller
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert len(sessions[0].events) == 1
    assert sessions[0].history()[0].content == "第二次输入"
    assert [message.content for message in provider.generate.call_args.args[0].messages] == [
        config.agent.system_prompt, "第二次输入",
    ]
    assert "已取消本次请求" in capsys.readouterr().out
    provider.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_text_is_visible_before_operation_finishes():
    output = RecordingOutput("首段内容")
    release = asyncio.Event()
    completed = False

    async def operation(report, on_text):
        nonlocal completed
        report("generating")
        on_text("首段内容\n")
        await release.wait()
        on_text("第二段内容")
        completed = True
        return "首段内容\n第二段内容"

    with create_pipe_input() as pipe:
        reader = ChatInput(input_stream=pipe, output_stream=output)

        async def controller():
            await output.visible.wait()
            assert not completed
            assert "第二段内容" not in output.text
            release.set()

        control_task = asyncio.create_task(controller())
        result = await asyncio.wait_for(reader.run_stream_operation(operation), timeout=3)
        await control_task
    assert result == "首段内容\n第二段内容"
