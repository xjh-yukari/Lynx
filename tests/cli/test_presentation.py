"""TUI 信息层次与窄屏布局，不访问真实 API。"""

from pathlib import Path
from uuid import uuid4

import pytest
from prompt_toolkit.application.current import create_app_session
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

from lynx_agent.cli.activities import tool_summary
from lynx_agent.cli.layout import (
    clip_text,
    header_fragments,
    status_text,
    transcript_fragments,
)
from lynx_agent.cli.state import DisplayMessage, UIState
from lynx_agent.providers.models import ModelToolCall, ToolTrace


@pytest.mark.parametrize("width", [0, 1, 24, 40, 80])
def test_header_and_footer_keep_cell_width_and_prioritize_permissions(width):
    state = UIState(project=Path("E:/很长的工作区/" + "子目录/" * 30), model="deepseek-flash")
    state.session_title = "超长会话名称" * 40
    state.status = "正在保留重要任务结论…"
    state.permissions = "完全权限"
    state.search_backend = "anysearch"
    text = fragment_list_to_text(header_fragments(state, width))
    # 顶栏已压成一行（对齐参考图 01）：模型与权限都在这一行里。
    assert "\n" not in text
    assert get_cwidth(text) <= width
    assert get_cwidth(status_text(state, width)) <= width
    if width >= 24:
        assert "权限：完全权限" in status_text(state, width)
    assert clip_text("中文abc", 5) == "中文…"


def test_transcript_markers_and_markdown_do_not_mutate_raw_messages():
    state = UIState(project=Path.cwd(), model="fake", messages=[
        DisplayMessage("你", "问题"), DisplayMessage("工具", "✓ 读取文件 · 1 项已返回"),
        DisplayMessage("Lynx", "**答案**\n\n```python\na = '**literal**'\n```"),
        DisplayMessage("系统", "操作失败：服务不可用"),
    ])
    with create_app_session(output=DummyOutput()):
        fragments = transcript_fragments(state)
    text = fragment_list_to_text(fragments)
    assert "› 问题" in text and "● 答案" in text
    assert "**答案**" not in text and "**literal**" in text
    assert any(style == "class:message.warning" for style, _ in fragments)
    assert state.messages[2].content.startswith("**答案**")


@pytest.mark.parametrize("failed,execution_state", [(False, "returned"), (True, "returned"), (False, "unknown")])
def test_tool_summary_shows_only_verified_metadata(failed, execution_state):
    trace = ToolTrace(turn_id=uuid4(), call=ModelToolCall(
        id="sample", name="web_search", arguments='{"query":"SECRET_QUERY"}',
    ), content="SECRET_RESULT", is_error=failed, execution_state=execution_state)
    text = tool_summary([trace], seconds=1.25)
    assert "联网搜索" in text and "本批耗时 1.2s" in text
    assert "SECRET" not in text
    assert text.startswith("!" if failed or execution_state != "returned" else "✓")
    assert tool_summary([], interrupted=True).startswith("!")
