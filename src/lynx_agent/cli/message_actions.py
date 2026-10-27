"""显示文本和代码复制源；不包含角色标记、按钮或自动折行。"""

import re
from functools import lru_cache

from markdown_it import MarkdownIt
from prompt_toolkit.formatted_text import fragment_list_to_text

from lynx_agent.cli.markdown import render_markdown


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)", "", text)
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    return "".join(char for char in text if char in "\n\r\t" or (ord(char) >= 32 and not 127 <= ord(char) <= 159))


@lru_cache(maxsize=32)
def plain_answer(source):
    return fragment_list_to_text(render_markdown(clean_text(source))).rstrip("\n")


def code_blocks(source):
    return [token.content for token in MarkdownIt("commonmark").parse(clean_text(source)) if token.type in {"fence", "code_block"}]


def copy_source(message, kind="plain", index=0):
    if kind == "markdown":
        return clean_text(message.content)
    if kind == "code":
        blocks = code_blocks(message.content)
        if not 0 <= index < len(blocks):
            raise ValueError("该代码块不存在")
        return blocks[index]
    return plain_answer(message.content) if message.role.startswith("Lynx") else clean_text(message.content)
