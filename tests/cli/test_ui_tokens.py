"""L0 界面语言：语义令牌、工具分类配色、底栏右对齐度量。"""

import pytest
from prompt_toolkit.utils import get_cwidth

from lynx_agent.cli import statusbar
from lynx_agent.cli.diff_style import diff_style
from lynx_agent.cli.themes import BASE, theme_style
from lynx_agent.cli.tool_style import tool_style

NEW_TOKENS = (
    "marker.tool", "tool.name.read", "tool.name.write", "tool.name.exec",
    "tool.name.web", "tool.name.memory", "tool.name.other",
    "diff.add.bg", "diff.add.fg", "diff.del.bg", "diff.del.fg",
    "status.model", "status.metric", "status.value", "status.warn", "shimmer",
)


def test_every_token_resolves_in_every_theme():
    for theme in ("ansi", "dark", "light"):
        style = theme_style(theme)
        for token in BASE:
            style.get_attrs_for_style_str(f"class:{token}")


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_new_tokens_are_not_left_at_placeholder(theme):
    style = theme_style(theme)
    # 工具分类必须真的分色，否则等于没加令牌。
    colors = {style.get_attrs_for_style_str(f"class:tool.name.{kind}").color
              for kind in ("read", "write", "exec", "web", "memory")}
    assert len(colors) == 5
    assert style.get_attrs_for_style_str("class:diff.add.bg").bgcolor
    assert style.get_attrs_for_style_str("class:diff.del.bg").bgcolor


def test_tool_style_maps_by_type_and_falls_back():
    assert tool_style("read_file") == "class:tool.name.read"
    assert tool_style("write_file") == "class:tool.name.write"
    assert tool_style("exec_command") == "class:tool.name.exec"
    assert tool_style("web_fetch") == "class:tool.name.web"
    assert tool_style("memory_search") == "class:tool.name.memory"
    assert tool_style("calculator") == "class:tool.name.other"
    assert tool_style("") == "class:tool.name.other"
    assert tool_style(None) == "class:tool.name.other"


def test_cache_text_keeps_placeholder_when_usage_is_unknown():
    assert statusbar.cache_text(None) == "缓存 —"
    assert statusbar.cache_text(96.66) == "缓存 96.7%"


def test_metrics_drop_cache_then_everything_as_width_shrinks():
    wide = statusbar.metrics(96.7, "≈¥0.0642", width=80)
    assert [text for _, text in wide] == ["缓存 96.7% · ", "≈¥0.0642"]

    middle = statusbar.metrics(96.7, "≈¥0.0642", width=10)
    assert [text for _, text in middle] == ["≈¥0.0642"]

    narrow = statusbar.metrics(96.7, "≈¥0.0642", width=3)
    assert narrow == []


def test_metrics_keep_placeholder_cache_and_cost_together():
    segments = statusbar.metrics(None, "≈¥0.0000", width=80)
    assert [text for _, text in segments] == ["缓存 — · ", "≈¥0.0000"]


def test_gap_is_what_is_left_over():
    assert statusbar.gap(20, 15, 80) == 45
    assert statusbar.gap(70, 15, 80) == -5


def test_status_line_never_exceeds_width():
    for width in (0, 1, 12, 20, 40, 80, 160):
        right = statusbar.metrics(96.7, "≈¥0.0642", width=width)
        left = get_cwidth("deepseek-flash · 默认 ▾ · 工作区修改")
        space = statusbar.gap(left, sum(get_cwidth(text) for _, text in right), width)
        if right and space >= 1:
            assert left + space + sum(get_cwidth(text) for _, text in right) == width
        else:
            assert left <= max(width, left)


def test_diff_style_only_reacts_to_line_prefixes():
    assert diff_style("+++ b/storage.py") == "class:diff.header"
    assert diff_style("--- a/storage.py") == "class:diff.header"
    assert diff_style("@@ -1,2 +1,3 @@") == "class:diff.hunk"
    assert diff_style("+    hits = store.query(kb, top_k)") == "class:diff.add"
    assert diff_style("-def search(kb, top_k=5):") == "class:diff.del"
    assert diff_style("     return tuple(") == ""
    assert diff_style('参数：{"top_k": 5}') == ""
    assert diff_style("") == ""
    assert diff_style(None) == ""