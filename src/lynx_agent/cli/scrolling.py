"""对话滚轮和可拖动滚动条，不改变输入焦点。"""

from prompt_toolkit.layout.controls import FormattedTextControl, UIContent
from prompt_toolkit.mouse_events import MouseButton, MouseEventType


class TranscriptControl(FormattedTextControl):
    def __init__(self, text, scroll, *, projection=None, interaction=None, **kwargs):
        super().__init__(text, **kwargs)
        self.scroll = scroll
        self.projection = projection
        self.interaction = interaction

    def create_content(self, width, height):
        if self.projection is None:
            return super().create_content(width, height)
        self.projection.update(width)
        return UIContent(get_line=self.projection.get_line, line_count=len(self.projection.rows),
                         cursor_position=self.get_cursor_position(), show_cursor=False)

    def mouse_handler(self, mouse_event):
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll(-3)
        elif mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll(3)
        else:
            return self.interaction(mouse_event) if self.interaction else NotImplemented
        return None


class ScrollbarControl(FormattedTextControl):
    def __init__(self, state, window, scroll_to, scroll_by):
        self.state = state
        self.window = window
        self.scroll_to = scroll_to
        self.scroll_by = scroll_by
        self.height = 1
        self.maximum = 0
        self.thumb_top = 0
        self.thumb_size = 1
        self.drag_offset = None
        super().__init__(self.fragments)

    def create_content(self, width, height):
        self.height = max(1, height or self.height)
        return super().create_content(width, height)

    def preferred_height(self, width, max_available_height, wrap_lines, get_line_prefix):
        return 1

    def fragments(self):
        info = self.window.render_info
        total = info.content_height if info else 1
        self.maximum = max(0, total - self.height)
        self.thumb_size = max(1, min(self.height, self.height * self.height // max(1, total)))
        start = self.maximum if self.state.follow_tail else self.state.scroll_line
        self.thumb_top = round((self.height - self.thumb_size) * min(start, self.maximum) / max(1, self.maximum))
        return self._rows()

    def _rows(self):
        result = []
        for row in range(self.height):
            thumb = self.thumb_top <= row < self.thumb_top + self.thumb_size
            result.append(("class:scrollbar.thumb" if thumb else "class:scrollbar.track", "█" if thumb else "│"))
            if row < self.height - 1:
                result.append(("", "\n"))
        return result

    def mouse_handler(self, mouse_event):
        kind = mouse_event.event_type
        row = mouse_event.position.y
        if kind in {MouseEventType.SCROLL_UP, MouseEventType.SCROLL_DOWN}:
            self.scroll_by(-3 if kind == MouseEventType.SCROLL_UP else 3)
        elif kind == MouseEventType.MOUSE_DOWN and mouse_event.button == MouseButton.LEFT:
            self.drag_offset = row - self.thumb_top if self.thumb_top <= row < self.thumb_top + self.thumb_size else self.thumb_size // 2
            self._drag(row)
        elif kind == MouseEventType.MOUSE_MOVE and mouse_event.button == MouseButton.LEFT and self.drag_offset is not None:
            self._drag(row)
        elif kind == MouseEventType.MOUSE_UP:
            if self.drag_offset is not None:
                self._drag(row)
            self.drag_offset = None

    def _drag(self, row):
        travel = max(1, self.height - self.thumb_size)
        fraction = max(0, min(travel, row - (self.drag_offset or 0))) / travel
        self.scroll_to(round(fraction * self.maximum))
