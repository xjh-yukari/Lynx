"""语义化终端主题与可选 UI 偏好；不修改 Agent 配置或模型密钥。"""

import json
import os
import tempfile
from pathlib import Path

from prompt_toolkit.styles import Style

BASE = {
    "title": "bold ansicyan", "path": "ansibrightblack", "separator": "ansibrightblack",
    "status": "", "hint": "ansibrightblack", "selected": "reverse",
    "permission-danger": "bold ansiyellow", "message.user": "#D77757",
    "message.marker": "bold #D77757", "input": "#D77757",
    "input.border": "ansibrightblack",
    "message.assistant": "bold ansicyan", "message.notice": "ansidefault",
    "message.tool": "ansicyan", "message.warning": "ansiyellow",
    "message.success": "ansigreen", "message.error": "ansired",
    "md.heading": "bold", "md.code": "ansigreen", "md.link": "ansicyan",
    "md.codeblock": "", "md.syntax.keyword": "ansimagenta",
    "md.syntax.string": "ansigreen", "md.syntax.comment": "ansibrightblack italic",
    "md.syntax.number": "ansicyan", "md.syntax.name": "",
    "scrollbar.thumb": "ansicyan", "scrollbar.track": "ansibrightblack",
    "selection": "reverse", "action": "ansicyan",
    "completion-menu.completion.current": "reverse", "completion-menu.meta.completion.current": "reverse",
    "context.0": "ansibrightblue", "context.1": "ansicyan",
    "context.2": "ansiblue", "context.3": "ansibrightcyan",
    "context.free": "ansiblue",
    "context.label": "bold ansibrightwhite",
    "context.warning": "bold ansiyellow", "context.danger": "bold ansired",
    # 语义令牌：颜色只表达角色，不表达组件（工具按类型上色，不按工具名）。
    "marker.tool": "ansibrightblack",
    "tool.name.read": "ansicyan", "tool.name.write": "ansigreen",
    "tool.name.exec": "ansiyellow", "tool.name.web": "ansiblue",
    "tool.name.memory": "ansimagenta", "tool.name.other": "",
    "diff.add.bg": "bg:ansigreen", "diff.add.fg": "ansigreen",
    "diff.del.bg": "bg:ansired", "diff.del.fg": "ansired",
    "diff.add": "bg:ansigreen ansiwhite", "diff.del": "bg:ansired ansiwhite",
    "diff.hunk": "ansiblue", "diff.header": "bold ansiwhite",
    "status.model": "ansicyan", "status.metric": "ansibrightblack",
    "status.value": "", "status.warn": "bold ansiyellow",
    "usage.tokens": "bold #7dd3fc", "usage.cost": "bold #fde68a",
    "shimmer": "bold ansibrightcyan",
    "welcome": "bold ansicyan", "hyperlink": "underline ansicyan",
}


# 主题已合并成一个：统一使用默认深色真彩调色板，不再区分 ansi/dark/light。
# 参数 name 只为兼容旧调用保留（/theme、偏好文件、外部脚本）。
SINGLE_THEME = "dark"


def theme_style(name="ansi"):
    if name not in {"ansi", "dark", "light"}:
        raise ValueError("主题已合并为默认主题")
    name = SINGLE_THEME
    values = dict(BASE)
    if name != "ansi":
        light = name == "light"
        accent, subtle, warning = ("#005f87", "#666666", "#875f00") if light else ("#87cfff", "#aaaaaa", "#ffd787")
        values[""] = "bg:#f5f3ef #242424" if light else "bg:#20242a #eeeeee"
        for key in ("message.tool", "md.link", "scrollbar.thumb"):
            values[key] = accent
        for key in ("title", "message.assistant"):
            values[key] = "bold " + accent
        for key in ("path", "hint", "message.notice", "separator", "scrollbar.track"):
            values[key] = subtle
        values["message.warning"] = warning
        values["message.success"] = "#286428" if light else "#afd7af"
        values["message.error"] = "#a12632" if light else "#ff9b9b"
        values["permission-danger"] = "bold " + warning
        values["action"] = accent
        values["md.code"] = "#005f00" if light else "#afffaf"
        surface = "#e9edf1" if light else "#2b3038"
        values["md.codeblock"] = "bg:" + surface
        values["md.syntax.keyword"] = "#7030a0" if light else "#d7afff"
        values["md.syntax.string"] = "#286428" if light else "#afd7af"
        values["md.syntax.comment"] = subtle + " italic"
        values["md.syntax.number"] = accent
        values["selection"] = "bg:#005f87 #ffffff" if light else "bg:#afd7ff #16202a"
        orange = "#a94425" if light else "#D77757"
        values["message.user"] = values["input"] = orange
        values["message.marker"] = "bold " + orange
        values["input.border"] = "#6d7886" if light else "#687384"
        fills = ("#2563eb", "#0f766e", "#7c3aed", "#a16207") if light else ("#60a5fa", "#5eead4", "#c4b5fd", "#fcd34d")
        for index, fill in enumerate(fills):
            values[f"context.{index}"] = "bold " + fill
        values["context.free"] = "#64748b" if light else "#718096"
        values["context.label"] = "bold " + ("#172033" if light else "#edf4ff")
        values["context.warning"] = "bold " + ("#6a3300" if light else "#ffd787")
        values["context.danger"] = "bold " + ("#800000" if light else "#ffbaba")
        # 语义令牌：同一角色在状态栏、工具行、diff 里共用一个颜色。
        values["marker.tool"] = "#5b6472" if light else "#6a7280"
        for key, color in (("tool.name.read", "#3d6ea8" if light else "#8fb3e0"),
                           ("tool.name.write", "#8a6a00" if light else "#e5c07b"),
                           ("tool.name.exec", "#0f7a8a" if light else "#56b6c2"),
                           ("tool.name.web", "#4a6fa5" if light else "#7da1de"),
                           ("tool.name.memory", "#7a4b96" if light else "#c79be0")):
            values[key] = color
        values["tool.name.other"] = ""
        values["diff.add.bg"] = "bg:" + ("#d7f0cf" if light else "#2c4030")
        values["diff.del.bg"] = "bg:" + ("#ffd7d9" if light else "#452c30")
        values["diff.add.fg"] = "#286428" if light else "#9bd07c"
        values["diff.del.fg"] = "#a12632" if light else "#e58a90"
        values["diff.add"] = "bg:" + ("#d7f0cf" if light else "#2c4030") + " " + ("#286428" if light else "#9bd07c")
        values["diff.del"] = "bg:" + ("#ffd7d9" if light else "#452c30") + " " + ("#a12632" if light else "#e58a90")
        values["diff.hunk"] = "#005f87" if light else "#7da1de"
        values["diff.header"] = "bold " + ("#242424" if light else "#c9d3e0")
        values["status.model"] = "#005f87" if light else "#87cfff"
        values["status.metric"] = subtle
        values["status.value"] = "#242424" if light else "#e6e9ef"
        values["status.warn"] = "bold " + warning
        values["shimmer"] = "bold " + ("#005f87" if light else "#cfe4ff")
        values["welcome"] = "#005f87" if light else "#8fb3e0"
        values["hyperlink"] = "underline " + ("#005f87" if light else "#5aa8e0")
    # 根布局必须显式给所有空白格上色；仅设置默认样式会留下透明格，
    # 增量 renderer 会把它们视为未变，露出终端清屏后的默认黑底。
    values["app"] = values.get("", "")
    # prompt_toolkit 默认的 scrollbar.arrow 使用 noinherit，需显式补背景。
    values["scrollbar.arrow"] = values["app"] + " bold"
    return Style.from_dict(values)


def _load_preferences(path):
    try:
        path = Path(path)
        if path.is_symlink() or path.stat().st_size > 65536:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
        return {key: value[key] for key in ("theme", "effort", "effort_model") if key in value} if isinstance(value, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def load_theme(path):
    name = _load_preferences(path).get("theme")
    return name if isinstance(name, str) and name in {"ansi", "dark", "light"} else "ansi"


def load_effort(path, model):
    value = _load_preferences(path)
    effort = value.get("effort")
    return effort if value.get("effort_model") == model and isinstance(effort, str) and effort in {"off", "low", "high", "max"} else None


def save_theme(path, name):
    theme_style(name)
    _save_preferences(path, {"theme": name})


def save_effort(path, model, effort):
    if effort not in {"off", "low", "high", "max"} or not model or len(model) > 256:
        raise ValueError("推理档位偏好不合法")
    _save_preferences(path, {"effort": effort, "effort_model": model})


def _save_preferences(path, updates):
    path = Path(path)
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise ValueError("UI 偏好路径不能是符号链接")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = _load_preferences(path)
    values.update(updates)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(values, stream)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
