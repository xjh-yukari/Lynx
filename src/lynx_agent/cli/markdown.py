"""Markdown 仅用于显示，不改写会话和发给模型的原文。"""

from functools import lru_cache

from markdown_it import MarkdownIt
from prompt_toolkit.utils import get_cwidth
from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
from pygments.util import ClassNotFound

_parser = MarkdownIt("commonmark", {"html": False})


def code_fragments(content, language):
    """只添加样式；禁止高亮器补换行、去缩进或改写代码。"""
    try:
        lexer = get_lexer_by_name(language, stripnl=False, ensurenl=False)
    except ClassNotFound:
        return [("class:md.codeblock", content)]
    result = []
    for token, value in lex(content, lexer):
        role = next((name for group, name in (
            (Token.Keyword, "keyword"), (Token.Literal.String, "string"),
            (Token.Comment, "comment"), (Token.Literal.Number, "number"),
        ) if token in group), "name")
        result.append((f"class:md.codeblock class:md.syntax.{role}", value))
    # 某些 lexer 会规范化源文本；显示和选区必须与复制源一致。
    return result if "".join(value for _, value in result) == content else [("class:md.codeblock", content)]


@lru_cache(maxsize=128)
def render_markdown(text: str) -> tuple[tuple[str, str], ...]:
    fragments = []
    lists = []
    heading = False
    for token in _parser.parse(text):
        kind = token.type
        if kind in {"bullet_list_open", "ordered_list_open"}:
            lists.append(int(token.attrGet("start") or 1) if kind == "ordered_list_open" else None)
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            lists.pop()
        elif kind == "list_item_open":
            number = lists[-1]
            fragments.append(("", "  " * (len(lists) - 1) + ("• " if number is None else f"{number}. ")))
            if number is not None:
                lists[-1] += 1
        elif kind == "heading_open":
            heading = True
        elif kind in {"paragraph_close", "heading_close"}:
            fragments.append(("", "\n" if lists else "\n\n"))
            heading = False
        elif kind in {"fence", "code_block"}:
            fragments.append(("class:md.code-start", ""))
            content = token.content.expandtabs(4).rstrip("\n")
            fragments.extend(code_fragments(content, token.info.split()[0] if token.info else "text"))
            fragments.append(("", "\n\n"))
        elif kind == "hr":
            fragments.append(("class:separator", "────────────\n"))
        elif kind == "inline":
            styles = ["class:md.heading"] if heading else []
            links = []
            for child in token.children or []:
                if child.type in {"strong_open", "em_open", "link_open"}:
                    styles.append({"strong_open": "bold", "em_open": "italic", "link_open": "underline"}[child.type])
                    if child.type == "link_open":
                        links.append(child.attrGet("href"))
                elif child.type in {"strong_close", "em_close", "link_close"}:
                    if styles:
                        styles.pop()
                    if child.type == "link_close":
                        fragments.append(("class:md.link", f" ({links.pop()})"))
                elif child.type in {"softbreak", "hardbreak"}:
                    fragments.append(("", "\n"))
                elif child.type == "code_inline":
                    fragments.append(("class:md.code", child.content))
                else:
                    fragments.append((" ".join(styles), child.content))
    return tuple(fragments)


def wrap_fragments(fragments, width: int):
    """提前按终端列宽折行，使滚轮、拖动、翻页使用同一套屏幕行坐标。"""
    result = []
    column = 0
    for style, text in fragments:
        buffer = ""
        for char in text.expandtabs(4):
            size = max(0, get_cwidth(char))
            if char != "\n" and column and column + size > max(2, width):
                buffer += "\n"
                column = 0
            buffer += char
            column = 0 if char == "\n" else column + size
        if buffer:
            result.append((style, buffer))
    return result
