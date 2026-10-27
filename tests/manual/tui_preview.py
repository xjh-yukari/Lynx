"""打印底栏与工具行的真实渲染文本（不启动 App，不读密钥）。

直接用新代码里的纯函数拼出字符串，并按终端列宽校验不会溢出，
方便在没有真实控制台的环境里核对效果。
"""

from __future__ import annotations

import difflib
from pathlib import Path

from prompt_toolkit.formatted_text import fragment_list_to_text
from prompt_toolkit.utils import get_cwidth

from lynx_agent.cli import statusbar
from lynx_agent.cli.activities import tool_summary
from lynx_agent.cli.diff_style import diff_style
from lynx_agent.cli.layout import header_fragments
from lynx_agent.cli.state import UIState
from lynx_agent.cli.tool_style import tool_style


class Trace:
    def __init__(self, name, failed=False):
        self.call = type("Call", (), {"name": name})()
        self.is_error = failed
        self.execution_state = "error" if failed else "returned"


def status_line(width, *, model="deepseek-flash", effort="max", permissions="工作区修改",
                cache=96.7, cost="≈¥0.0642"):
    suffix = f" · {permissions}" + " ▾"
    name = f"{model} · {effort} ▾"
    right = statusbar.metrics(cache, cost, width=width)
    right_text = "".join(text for _, text in right)
    space = statusbar.gap(get_cwidth(name + suffix), get_cwidth(right_text), width)
    if not right or space < 1:
        return name + suffix
    return name + suffix + " " * space + right_text


def tool_line(traces, seconds, width):
    summary = tool_summary(traces, seconds=seconds)
    line = " • " + summary
    return line + " " * max(0, width - get_cwidth(line)) if get_cwidth(line) <= width else line[:width]


def show_header() -> None:
    state = UIState(project=Path(r"E:\github\nanobot"), model="deepseek-flash")
    state.session_title = "排查代理劫持 502"
    for width in (80, 40):
        print(f"[{width:>3} 列] " + fragment_list_to_text(header_fragments(state, width)).replace(chr(10), " / "))
    print()


def show_diff() -> None:
    before = "def search(kb, top_k=5):", "    return store.query(kb, top_k)"
    after = "def search(kb, top_k=5, ratio=0.5):", "    hits = store.query(kb, top_k)", "    return [h for h in hits if h.score >= hits[0].score * ratio]"
    diff = list(difflib.unified_diff(list(before), list(after), "a/storage.py", "b/storage.py", lineterm=""))
    marks = {"class:diff.add": "+", "class:diff.del": "-", "class:diff.hunk": "@", "class:diff.header": "!"}
    for line in diff:
        style = diff_style(line)
        print(f"  {marks.get(style, ' ')} {line}")


def show_screen() -> None:
    """按新布局拼一屏：单行顶栏 + 对话 + 单行底栏。"""
    from lynx_agent.cli.metrics import ContextReading, context_pill

    state = UIState(project=Path(r"E:\github\nanobot"), model="deepseek-flash")
    state.session_title = "排查代理劫持 502"
    state.permissions = "工作区修改"
    state.effort = "max"
    width = 96
    print(fragment_list_to_text(header_fragments(state, width)))
    print("─" * width)
    for line in ("› 帮我看看这个报错", "", "● 是系统代理劫持了 127.0.0.1：Clash 把本地请求转发出去，回了 502。",
                 "  加 NO_PROXY 就好。", "", " • ✓ 读取文件 · 2 项已返回 · 本批耗时 0.2s", "                 └ 140 lines · 84-125"):
        print(line)
    print("─" * width)
    reading = ContextReading(system=12000, messages=8000, tools=2500, draft=0, window=1_000_000, trigger=900_000)
    pill = "".join(text for _, text in context_pill(reading, 44))
    left = "deepseek-flash · max ▾ · 工作区修改 ▾ · 缓存 96.7% · ≈¥0.0642"
    print(left + " " * max(1, width - get_cwidth(left) - get_cwidth(pill)) + pill)
    print()


def main() -> None:
    show_screen()
    show_header()
    print("展开工具详情时的 diff（+ 绿底 / - 红底 / @ hunk / ! 文件头）")
    show_diff()
    print()
    for width in (96, 64, 43):
        line = status_line(width)
        print(f"[{width:>3} 列] {line}")
        assert get_cwidth(line) <= width, (width, get_cwidth(line))
    print()
    for width in (96,):
        for traces, seconds in (((Trace("read_file"), Trace("read_file")), 0.2), ((Trace("exec_command"),), 3.4), ((Trace("edit_file", failed=True),), 1.1)):
            print(f"[{width:>3} 列] {tool_line(traces, seconds, width)}")
    print()
    print("工具配色：", ", ".join(
        f"{name}→{tool_style(name).removeprefix('class:tool.name.')}"
        for name in ("read_file", "edit_file", "exec_command", "web_fetch", "memory_search", "calculator")))


if __name__ == "__main__":
    main()
