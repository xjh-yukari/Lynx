"""自写 SVG 图表：不依赖外部网络与前端图表库。"""

from __future__ import annotations

from html import escape
from math import cos, pi, sin

PALETTE = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")
# 已经是完整属性，调用处直接插值，避免属性值没有引号导致 SVG 不是合法 XML。
FONT = ('style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,'
        'Noto Sans CJK SC,sans-serif"')


def _color(index: int) -> str:
    return PALETTE[index % len(PALETTE)]


def radar(axes: list[str], series: list[tuple[str, list[float | None]]],
          *, size: int = 320) -> str:
    """多 agent 维度雷达图；缺失维度按 0 处理并在图例中标注。"""
    if not axes:
        return ""
    center = size / 2
    radius = size * 0.34
    step = 2 * pi / len(axes)
    parts = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" role="img">']
    for ring in (0.25, 0.5, 0.75, 1.0):
        points = []
        for index in range(len(axes)):
            angle = -pi / 2 + index * step
            points.append(f"{center + radius * ring * cos(angle):.1f},"
                          f"{center + radius * ring * sin(angle):.1f}")
        parts.append(f'<polygon points="{" ".join(points)}" fill="none" '
                     f'stroke="#e2e8f0" stroke-width="1"/>')
    for index, axis in enumerate(axes):
        angle = -pi / 2 + index * step
        x = center + radius * cos(angle)
        y = center + radius * sin(angle)
        parts.append(f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" '
                     f'stroke="#e2e8f0" stroke-width="1"/>')
        label_x = center + radius * 1.18 * cos(angle)
        label_y = center + radius * 1.18 * sin(angle)
        anchor = "middle" if abs(cos(angle)) < 0.3 else ("start" if cos(angle) > 0 else "end")
        parts.append(f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="{anchor}" '
                     f'dominant-baseline="middle" font-size="11" fill="#334155" '
                     f'{FONT}>{escape(axis)}</text>')
    for index, (name, values) in enumerate(series):
        points = []
        for axis_index in range(len(axes)):
            value = values[axis_index] if axis_index < len(values) else None
            ratio = 0.0 if value is None else max(0.0, min(1.0, value / 100))
            angle = -pi / 2 + axis_index * step
            points.append(f"{center + radius * ratio * cos(angle):.1f},"
                          f"{center + radius * ratio * sin(angle):.1f}")
        parts.append(f'<polygon points="{" ".join(points)}" fill="{_color(index)}22" '
                     f'stroke="{_color(index)}" stroke-width="2"/>')
    legend_y = size - 8
    for index, (name, _values) in enumerate(series):
        x = 8 + index * 96
        parts.append(f'<rect x="{x}" y="{legend_y - 9}" width="9" height="9" '
                     f'fill="{_color(index)}"/>')
        parts.append(f'<text x="{x + 13}" y="{legend_y}" font-size="10" fill="#334155" '
                     f'{FONT}>{escape(name)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def heatmap(rows: list[str], columns: list[str],
            values: list[list[float | None]], *, cell: tuple[int, int] = (92, 34)) -> str:
    """行列热力图；value 为 0~1 的得分，None 显示为“—”。"""
    if not rows or not columns:
        return ""
    width_cell, height_cell = cell
    left = 92
    top = 38
    width = left + width_cell * len(columns)
    height = top + height_cell * len(rows) + 12
    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">']
    for column_index, column in enumerate(columns):
        x = left + column_index * width_cell + width_cell / 2
        parts.append(f'<text x="{x}" y="22" text-anchor="middle" font-size="11" '
                     f'fill="#334155" {FONT}>{escape(column)}</text>')
    for row_index, row in enumerate(rows):
        y = top + row_index * height_cell
        parts.append(f'<text x="{left - 8}" y="{y + height_cell / 2 + 4}" text-anchor="end" '
                     f'font-size="11" fill="#334155" {FONT}>{escape(row)}</text>')
        for column_index in range(len(columns)):
            value = values[row_index][column_index] if row_index < len(values) and \
                column_index < len(values[row_index]) else None
            x = left + column_index * width_cell
            fill = "#f1f5f9" if value is None else _scale(value)
            parts.append(f'<rect x="{x}" y="{y}" width="{width_cell - 2}" '
                         f'height="{height_cell - 2}" rx="3" fill="{fill}"/>')
            text = "—" if value is None else f"{value * 100:.0f}"
            ink = "#0f172a" if value is None or value > 0.55 else "#ffffff"
            parts.append(f'<text x="{x + (width_cell - 2) / 2}" y="{y + height_cell / 2 + 4}" '
                         f'text-anchor="middle" font-size="11" fill="{ink}" {FONT}>{text}</text>')
    parts.append("</svg>")
    return "".join(parts)


def lines(series: list[tuple[str, list[tuple[float, float]]]], *,
          x_label: str = "", y_label: str = "", width: int = 640,
          height: int = 240, y_max: float | None = None,
          markers: list[tuple[float, float, str]] | None = None) -> str:
    """折线图：series 里每个元素是 (名称, [(x, y), ...])，markers 画成红色三角。"""
    if not series:
        return ""
    left, right, top, bottom = 52, 16, 22, 34
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [x for _, points in series for x, _ in points]
    ys = [y for _, points in series for _, y in points]
    if not xs or not ys:
        return ""
    x_min, x_max = min(xs), max(xs)
    y_top = y_max if y_max is not None else max(ys + [1.0])
    x_span = (x_max - x_min) or 1
    y_span = y_top or 1
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">']
    for ratio in (0, 0.25, 0.5, 0.75, 1.0):
        y = top + plot_h * (1 - ratio)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
                     f'fill="#64748b" {FONT}>{y_span * ratio:.0f}</text>')
    for index, (name, points) in enumerate(series):
        ordered = sorted(points)
        path_points = " ".join(
            f"{left + plot_w * ((x - x_min) / x_span):.1f},"
            f"{top + plot_h * (1 - min(1.0, y / y_span)):.1f}" for x, y in ordered)
        parts.append(f'<polyline points="{path_points}" fill="none" stroke="{_color(index)}" '
                     f'stroke-width="2"/>')
        for x, y in ordered:
            cx = left + plot_w * ((x - x_min) / x_span)
            cy = top + plot_h * (1 - min(1.0, y / y_span))
            parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="2.5" fill="{_color(index)}"/>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 8}" text-anchor="middle" '
                 f'font-size="10" fill="#64748b" {FONT}>{escape(x_label)}</text>')
    parts.append(f'<text x="12" y="{top + plot_h / 2}" text-anchor="middle" font-size="10" '
                 f'fill="#64748b" transform="rotate(-90 12 {top + plot_h / 2})" '
                 f'{FONT}>{escape(y_label)}</text>')
    for index, (name, _points) in enumerate(series):
        x = left + index * 110
        parts.append(f'<rect x="{x}" y="{height - 8}" width="8" height="8" fill="{_color(index)}"/>')
        parts.append(f'<text x="{x + 12}" y="{height - 1}" font-size="10" fill="#334155" '
                     f'{FONT}>{escape(name)}</text>')
    for x, y, label in markers or ():
        cx = left + plot_w * ((x - x_min) / x_span)
        cy = top + plot_h * (1 - min(1.0, y / y_span))
        parts.append(f'<polygon points="{cx:.1f},{cy - 5:.1f} {cx - 4:.1f},{cy + 3:.1f} '
                     f'{cx + 4:.1f},{cy + 3:.1f}" fill="#dc2626">'
                     f"<title>{escape(label)}</title></polygon>")
    parts.append("</svg>")
    return "".join(parts)


def _scale(value: float) -> str:
    ratio = max(0.0, min(1.0, value))
    red = int(239 - 120 * ratio)
    green = int(148 + 80 * ratio)
    blue = int(120 + 40 * ratio)
    return f"rgb({red},{green},{blue})"
