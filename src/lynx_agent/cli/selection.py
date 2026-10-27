"""以消息标识和原始显示文字位置锚定选区，不以折行后的屏幕行锚定。"""

import unicodedata
from dataclasses import dataclass


def whole_character_range(text, left, right):
    """避免只复制组合音标、变体选择符或 ZWJ 表情的一部分。"""
    def extension(char):
        return bool(unicodedata.combining(char)) or "\ufe00" <= char <= "\ufe0f" or "\U0001f3fb" <= char <= "\U0001f3ff"
    while 0 < left < len(text) and (extension(text[left]) or text[left] == "\u200d" or text[left - 1] == "\u200d"):
        left -= 1
    while right < len(text) and (extension(text[right]) or text[right] == "\u200d" or (right and text[right - 1] == "\u200d")):
        right += 1
    return left, right


@dataclass(frozen=True)
class Anchor:
    message_id: str
    offset: int


class TextSelection:
    def __init__(self):
        self.clear()

    def clear(self):
        self.start = None
        self.end = None
        self.dragging = False
        self.snapshot = ""

    def begin(self, anchor):
        self.clear()
        self.start = self.end = anchor
        self.dragging = True

    def extend(self, anchor):
        if self.dragging and anchor is not None:
            self.end = anchor

    def bounds(self, messages, order=None):
        if self.start is None or self.end is None:
            return None
        order = order if order is not None else {message.id: index for index, message in enumerate(messages)}
        if self.start.message_id not in order or self.end.message_id not in order:
            return None
        return sorted((self.start, self.end), key=lambda anchor: (order[anchor.message_id], anchor.offset))

    def finish(self, messages, plain):
        self.dragging = False
        bounds = self.bounds(messages)
        if bounds is None or bounds[0] == bounds[1]:
            self.clear()
            return
        first, last = bounds
        collecting = False
        pieces = []
        for message in messages:
            if message.id == first.message_id:
                collecting = True
            if collecting:
                text = plain(message)
                left = first.offset if message.id == first.message_id else 0
                right = last.offset if message.id == last.message_id else len(text)
                left, right = whole_character_range(text, left, right)
                pieces.append(text[left:right])
            if message.id == last.message_id:
                break
        self.snapshot = "\n\n".join(pieces)

    def contains(self, anchor, messages):
        bounds = self.bounds(messages)
        if bounds is None or anchor is None:
            return False
        order = {message.id: index for index, message in enumerate(messages)}
        if anchor.message_id not in order:
            return False
        key = lambda item: (order[item.message_id], item.offset)
        return key(bounds[0]) <= key(anchor) < key(bounds[1])
