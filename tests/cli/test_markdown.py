"""显示层保留代码原文，并按中英文实际列宽换行。"""

from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth

from lynx_agent.cli.markdown import render_markdown, wrap_fragments


def test_markdown_styles_and_code_are_not_plain_marker_replacement():
    source = "# 标题\n\n**加粗** 与 `a ** b`\n\n```python\na = '**原样**'\n```\n\n[官网](https://example.com)"
    fragments = render_markdown(source)
    text = fragment_list_to_text(fragments)
    assert "# 标题" not in text
    assert "标题" in text
    assert ("bold", "加粗") in fragments
    assert "a ** b" in text
    assert "a = '**原样**'" in text
    assert "官网 (https://example.com)" in text
    assert "```" not in text


def test_wrapping_uses_terminal_columns_and_keeps_styles():
    result = wrap_fragments([("bold", "你好abc世界\n下一行")], 5)
    assert all(get_cwidth(line) <= 5 for line in fragment_list_to_text(result).split("\n"))
    assert all(style == "bold" for style, _ in result)


def test_lists_and_escaped_markers():
    text = fragment_list_to_text(render_markdown("3. 三\n4. 四\n\n- 条目\n\n\\*原文\\*"))
    assert "3. 三" in text
    assert "4. 四" in text
    assert "• 条目" in text
    assert "*原文*" in text
