import pytest

from lynx_agent.providers.errors import ProviderToolMarkupError
from lynx_agent.providers.output_guard import TextOutputGuard, reject_tool_markup


@pytest.mark.parametrize("marker", ["<|DSML|calls>", "<｜DSML｜tool_calls>", "<｜｜DSML｜｜ calls>", "< | dsml | invoke>"])
def test_detects_markers_at_every_stream_boundary(marker):
    for split in range(len(marker) + 1):
        guard = TextOutputGuard()
        visible = ""
        with pytest.raises(ProviderToolMarkupError):
            visible += guard.feed("说明文字" + marker[:split])
            visible += guard.feed(marker[split:])
            visible += guard.finish()
        assert visible in {"", "说明文字"}
    with pytest.raises(ProviderToolMarkupError):
        reject_tool_markup(marker)


@pytest.mark.parametrize("text", ["正常回复", "a < b", "<html><body>code</body></html>", "<", "代码中的 DSML 单词不是标记"])
def test_ordinary_text_unchanged_even_character_by_character(text):
    guard = TextOutputGuard()
    assert "".join(guard.feed(char) for char in text) + guard.finish() == text


def test_incomplete_marker_does_not_leak_at_stream_end():
    guard = TextOutputGuard()
    assert guard.feed("<｜DS") == ""
    with pytest.raises(ProviderToolMarkupError):
        guard.finish()
