"""统一消息投影：稳定文字锚点、分页历史、有界折行缓存、按需绘制屏幕行。"""

import re
from collections import OrderedDict
from dataclasses import dataclass

from prompt_toolkit.utils import get_cwidth

from lynx_agent.cli.activities import tool_summary
from lynx_agent.cli.diff_style import diff_style
from lynx_agent.cli.markdown import render_markdown
from lynx_agent.cli.message_actions import clean_text, copy_source
from lynx_agent.cli.selection import Anchor
from lynx_agent.cli.state import DisplayMessage
from lynx_agent.cli.tool_cards import detail_text, inspect_trace
from lynx_agent.cli.tool_style import tool_style
from lynx_agent.providers.output_guard import contains_tool_markup
from lynx_agent.tools.errors import ToolExecutionError


@dataclass(frozen=True)
class Glyph:
    char: str
    style: str = ""
    anchor: Anchor | None = None
    action: tuple | None = None


URL_PATTERN = re.compile(r"https?://[^\s（）()<>\"'，。；、】]+")


def linkify_rows(rows, source):
    """把正文里的 http(s) 链接标成可点击：蓝色下划线 + 鼠标点击打开浏览器。"""
    spans = [(match.start(), match.end(), match.group()) for match in URL_PATTERN.finditer(source)]
    if not spans:
        return rows
    for row in rows:
        for index, glyph in enumerate(row):
            anchor = glyph.anchor
            if anchor is None:
                continue
            for start, end, url in spans:
                if start <= anchor.offset < end:
                    row[index] = Glyph(glyph.char, "class:hyperlink", anchor, ("open", url))
                    break
    return rows


def plain_message(message):
    return copy_source(message)


def session_messages(session):
    """历史只还原为同一种展示消息，后续样式、选择、工具卡与实时消息共用投影。"""
    result = []
    by_turn, by_result, shown = {}, {}, set()
    for event in session.events:
        if event.tool is not None:
            by_turn.setdefault(event.tool.turn_id, []).append(event.tool)
            if event.tool.result_id is not None:
                by_result[event.tool.result_id] = event.tool
    for event in session.events:
        if event.kind == "turn":
            traces = [by_result[message.result_id] for message in event.messages
                      if message.role == "tool" and message.result_id in by_result]
            inserted = False
            for message in event.messages:
                if message.role == "tool" or message.tool_calls:
                    continue
                if message.role == "assistant" and traces and not inserted:
                    result.append(DisplayMessage("工具", tool_summary(traces), tools=tuple(traces)))
                    shown.update(trace.turn_id for trace in traces)
                    inserted = True
                if message.role == "assistant" and contains_tool_markup(message.content):
                    result.append(DisplayMessage("系统", "该历史回复含异常工具协议，已隐藏；原始会话记录仍保留。"))
                else:
                    result.append(DisplayMessage("你" if message.role == "user" else "Lynx", message.content))
        elif event.kind == "reset":
            result.append(DisplayMessage("系统", "此处已清空模型上下文；之前的原文仍保留。"))
        elif event.kind == "compact":
            result.append(DisplayMessage("系统", f"历史压缩：约 {event.before_tokens:,} → {event.after_tokens:,} Tokens。"))
        elif event.kind == "prune":
            result.append(DisplayMessage("系统", f"已回收 {len(event.replacements)} 条工具结果；原始快照仍保留。"))
        elif event.attempt is not None:
            result.append(DisplayMessage("系统", f"回复尝试 {event.attempt.index + 1} 达到上限；截断草稿保存在会话文件中，不参与模型上下文。"))
    for turn_id, traces in by_turn.items():
        if turn_id not in shown:
            result.append(DisplayMessage("工具", tool_summary(traces, interrupted=True), tools=tuple(traces)))
    return result


class TranscriptProjection:
    def __init__(self, state, runner, selection):
        self.state, self.runner, self.selection = state, runner, selection
        self.rows = [[]]
        self.cache = OrderedDict()
        self.cache_chars = 0
        self.start_id = None
        self.width = 0
        self._signature = None
        self.order = {}
        self._end_anchor = None

    def clear(self):
        self.cache.clear()
        self.cache_chars = 0
        self._signature = None
        self._end_anchor = None
        self.selection.clear()

    def plain(self, message):
        if message.role == "工具" and message.expanded:
            details, _ = detail_text(message, self.runner)
            return clean_text(message.content) + "\n" + details
        return plain_message(message)

    def _wrap(self, message, width, policy):
        key = (message.id, message.role, message.content, message.expanded, message.detail_page,
               message.visible_chars, tuple((trace.result_id, trace.is_error) for trace in message.tools), width, policy)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        rows = [[]]
        column = 0
        margin = 2 if width >= 24 else 0
        edge = max(2, width - margin)
        hanging = 2
        list_indents = {match.start(): get_cwidth(match.group()) for match in
                        re.finditer(r"(?m)^[ ]*(?:• |\d+\. )", copy_source(message))}

        def emit(text, style="", *, source=False, action=None, offset=0):
            nonlocal column, hanging
            for index, char in enumerate(text):
                anchor = Anchor(message.id, offset + index) if source else None
                if source and offset + index in list_indents:
                    hanging = list_indents[offset + index] + (2 if offset + index == 0 else 0)
                if char == "\n":
                    rows.append([])
                    column = 0
                    hanging = 0
                    continue
                for visible in (" " * (4 - column % 4) if char == "\t" else char):
                    # 控制符转为可见替代符，绝不向终端发送任意协议。
                    visible = visible if ord(visible) >= 32 and ord(visible) != 127 else "�"
                    size = max(0, get_cwidth(visible))
                    if not rows[-1]:
                        rows[-1].extend(Glyph(" ") for _ in range(margin))
                        column = margin
                    if column and column + size > edge:
                        rows.append([])
                        column = min(margin + hanging, max(0, edge - 2))
                        rows[-1].extend(Glyph(" ") for _ in range(column))
                    rows[-1].append(Glyph(visible, style, anchor, action))
                    column += size

        if message.role.startswith("Lynx"):
            emit("● " + message.role.removeprefix("Lynx"), "class:message.assistant")
            source = clean_text(message.content)
            offset = 0
            for style, text in render_markdown(source):
                if style == "class:md.code-start":
                    hanging = 0
                    continue
                visible = text[:max(0, message.visible_chars - offset)]
                emit(visible, style, source=True, offset=offset)
                offset += len(text)
            while rows and not rows[-1]:
                rows.pop()
            if not rows:
                rows = [[]]
            rows.append([])
            column = 0
            emit("⧉ ", "class:action", action=("copy", message.id, "plain"))
        elif message.role == "工具":
            summary = clean_text(message.content)
            emit(" • ", "class:marker.tool")
            emit(summary[:1], "class:message.error" if summary.startswith("!") else "class:message.success", source=True)
            name, separator, timing = summary[1:].partition(" · ")
            first = message.tools[0].call.name if message.tools else ""
            emit(name, tool_style(first), source=True, offset=1)
            emit(separator + timing, "class:hint", source=True, offset=1 + len(name))
            if message.tools:
                emit(" [收起]" if message.expanded else " [查看详情]", "class:action", action=("tool", message.id))
                if message.expanded:
                    try:
                        details, pages = detail_text(message, self.runner)
                    except (ToolExecutionError, ValueError):
                        emit("\n当前权限下无法验证该快照，未展示参数或正文。", "class:message.warning")
                    else:
                        # 逐行上色：工具结果里如果带统一 diff，增删行直接可读。
                        offset = len(summary)
                        for line in details.split("\n"):
                            emit("\n" + line, diff_style(line), source=True, offset=offset)
                            offset += len(line) + 1
                        emit(f"\n第 {message.detail_page + 1}/{pages} 页 ", "class:message.notice")
                        if message.detail_page:
                            emit("[上一页] ", "class:action", action=("tool_page", message.id, -1))
                        if message.detail_page + 1 < pages:
                            emit("[下一页] ", "class:action", action=("tool_page", message.id, 1))
                        emit("[复制本页详情]", "class:action", action=("tool_copy", message.id))
                        emit(" [复制结果快照]", "class:action", action=("tool_copy_all", message.id))
        elif message.role == "问候":
            # 开场问候：以助手身份显示，但不是模型回答，不参与“首条 Lynx 消息”的语义。
            emit("● ", "class:message.assistant")
            emit(clean_text(message.content)[:message.visible_chars], "class:welcome", source=True)
        else:
            user = message.role.startswith("你")
            warning = any(word in message.content for word in ("失败", "未核实", "未保存", "已取消"))
            style = "class:message.user" if user else "class:message.warning" if warning else "class:message.notice"
            marker_style = "class:message.marker" if user else "class:message.error" if "失败" in message.content else style
            emit("› " + message.role.removeprefix("你") if user else "· ", marker_style)
            emit(clean_text(message.content)[:message.visible_chars], "" if warning and not user else style, source=True)
        if len(copy_source(message)) > message.visible_chars:
            emit("\n[展开更多正文]", "class:action", action=("more", message.id))
        if message.role.startswith("Lynx"):
            linkify_rows(rows, copy_source(message))
        rows.append([])
        chars = sum(len(row) for row in rows)
        self.cache[key] = (rows, chars)
        self.cache_chars += chars
        while self.cache and (len(self.cache) > 96 or self.cache_chars > 160000):
            _, (_, removed) = self.cache.popitem(last=False)
            self.cache_chars -= removed
        return rows

    def update(self, width):
        self.width = max(2, width)
        messages = self.state.messages
        if self.state.history_end_id is None:
            end = len(messages)
        elif self._end_anchor and self._end_anchor[0] == self.state.history_end_id and self._end_anchor[1] < len(messages) and messages[self._end_anchor[1]].id == self.state.history_end_id:
            end = self._end_anchor[1]
        else:
            end = next((i for i, message in enumerate(messages) if message.id == self.state.history_end_id), len(messages))
            self._end_anchor = (self.state.history_end_id, end)
        start = max(0, end - self.state.visible_messages)
        budget = 0
        for index in range(end - 1, start - 1, -1):
            budget += min(len(messages[index].content), messages[index].visible_chars)
            if budget > 120000 and index < end - 1:
                start = index + 1
                break
        policy = None
        if self.runner.file_access:
            service = self.runner.file_access.policy
            policy = (str(service.root), service.mode, tuple(sorted(service.sensitive_grants)), tuple(sorted(str(path) for path in service.private_roots)))
        # /clear、会话切换后不能继续显示失效工具快照的缓存。
        validity = []
        for message in messages[start:end]:
            if message.tools and message.expanded:
                try:
                    for trace in message.tools:
                        inspect_trace(self.runner, trace)
                except (ToolExecutionError, ValueError):
                    validity.append((message.id, False))
                else:
                    validity.append((message.id, True))
        policy = (policy, tuple(validity))
        signature = (self.width, start, end, tuple((m.id, m.role, m.content, m.expanded, m.detail_page, m.visible_chars, m.tools) for m in messages[start:end]), policy)
        if signature == self._signature:
            return
        self._signature = signature
        self.order = {message.id: i for i, message in enumerate(messages[start:end])}
        self.start_id = messages[start].id if start < end else None
        rows = []
        if start:
            rows.append([Glyph(char, "class:action", action=("older",)) for char in f"[加载更早记录 · 前面还有 {start} 条]"])
        if self.state.history_end_id:
            rows.append([Glyph(char, "class:action", action=("latest",)) for char in "[回到最新记录]"])
        for message in messages[start:end]:
            if message.content:
                rows.extend(self._wrap(message, self.width, policy if message.tools else None))
        self.rows = rows or [[]]

    def get_line(self, index):
        if not 0 <= index < len(self.rows):
            return []
        order = self.order
        ranges = self.selection.bounds(self.state.messages, order=order)
        fragments = []
        for glyph in self.rows[index]:
            style = glyph.style
            if ranges and glyph.anchor and glyph.anchor.offset >= 0:
                key = (order.get(glyph.anchor.message_id, -1), glyph.anchor.offset)
                if (order[ranges[0].message_id], ranges[0].offset) <= key < (order[ranges[1].message_id], ranges[1].offset):
                    style += " class:selection"
            if fragments and fragments[-1][0] == style:
                fragments[-1] = (style, fragments[-1][1] + glyph.char)
            else:
                fragments.append((style, glyph.char))
        return fragments

    def hit(self, x, y):
        if not 0 <= y < len(self.rows) or not self.rows[y]:
            return None
        # Window 已将终端宽字符坐标转换为 UIContent 字符索引。
        return self.rows[y][max(0, min(x, len(self.rows[y]) - 1))]
