"""文字层级、代码高亮与展示留白不得改变复制源或会话原文。"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth

from lynx_agent.cli.markdown import render_markdown
from lynx_agent.cli.message_actions import copy_source
from lynx_agent.cli.projection import TranscriptProjection
from lynx_agent.cli.selection import TextSelection
from lynx_agent.cli.state import DisplayMessage, UIState
from lynx_agent.cli.themes import theme_style


def project(message, width=60):
    state = UIState(project=Path.cwd(), model="fake", messages=[message])
    view = TranscriptProjection(state, SimpleNamespace(file_access=None), TextSelection())
    view.update(width)
    return view


@pytest.mark.parametrize("theme", ["ansi", "light", "dark"])
def test_body_is_neutral_and_markers_have_separate_emphasis(theme):
    style = theme_style(theme)
    assert not style.get_attrs_for_style_str("class:message.user").bold
    assert not style.get_attrs_for_style_str("class:status").bold
    assert style.get_attrs_for_style_str("class:message.marker").bold
    assert style.get_attrs_for_style_str("class:md.heading").bold
    assert style.get_attrs_for_style_str("class:md.heading").color == style.get_attrs_for_style_str("").color


def test_code_highlighting_changes_only_styles_and_falls_back_for_unknown_language():
    text = "if True:\n    print('你好') # comment\n"
    fragments = render_markdown(f"```python\n{text}```")
    assert fragment_list_to_text(fragments) == text.rstrip("\n") + "\n\n"
    assert any("syntax.keyword" in style for style, _ in fragments)
    assert any("syntax.string" in style for style, _ in fragments)
    assert any("syntax.comment" in style for style, _ in fragments)
    unknown = render_markdown("```not-a-real-language\n**literal**\n```")
    assert fragment_list_to_text(unknown) == "**literal**\n\n"


def test_answer_has_one_copy_action_and_no_code_or_markdown_buttons():
    message = DisplayMessage("Lynx", "正文\n\n```python\nx = 1\n```\n\n```\nsecond\n```")
    view = project(message)
    rows = ["".join(g.char for g in row) for row in view.rows]
    actions = {g.action for row in view.rows for g in row if g.action}
    assert actions == {("copy", message.id, "plain")}
    text = "\n".join(rows)
    assert text.count("⧉") == 1
    assert "[Markdown]" not in text and "[复制正文]" not in text and "[复制代码]" not in text
    assert "复制代码" not in copy_source(message)
    assert copy_source(message, "code", 0) == "x = 1\n"
    assert copy_source(message, "code", 1) == "second\n"
    plain = copy_source(message)
    for row in view.rows:
        for glyph in row:
            if glyph.anchor:
                assert plain[glyph.anchor.offset] == glyph.char


@pytest.mark.parametrize("width", [2, 7, 24, 40, 80])
def test_cjk_wrapping_margins_and_source_anchors(width):
    message = DisplayMessage("Lynx", "1. " + "中文列表abc" * 15 + "\n\n下一段")
    view = project(message, width)
    assert all(sum(get_cwidth(g.char) for g in row) <= width for row in view.rows)
    if width >= 24:
        assert all("".join(g.char for g in row).startswith("  ") for row in view.rows if row)
        # 续行越过角色标记与列表编号，与首行列表正文对齐。
        continuation = next(row for row in view.rows[1:] if any(g.anchor and g.char == "中" for g in row))
        assert "".join(g.char for g in continuation).startswith(" " * 7)
    assert message.content.startswith("1. 中文列表")


def test_partial_stream_markdown_and_empty_code_blocks_are_safe():
    for source in ("**正在生成", "```python\nif", "```\n```\n\n```python\nx=1\n```", "[链接](https://example.com"):
        message = DisplayMessage("Lynx", source)
        view = project(message)
        assert view.rows
        assert copy_source(message, "markdown") == source


def test_failure_only_emphasizes_status_marker_not_entire_message():
    message = DisplayMessage("系统", "操作失败：这是详细说明")
    view = project(message)
    assert any(g.style == "class:message.error" for row in view.rows for g in row)
    assert all(not g.style for row in view.rows for g in row if g.anchor)
