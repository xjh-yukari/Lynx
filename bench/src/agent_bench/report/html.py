"""单文件 HTML 报告：数据内联、图表为自写 SVG、不依赖外部网络。"""

from __future__ import annotations

import itertools
import json
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from agent_bench.metrics import DIMENSION_LABELS, DIMENSIONS
from agent_bench.models import TurnTrace
from agent_bench.report import svg
from agent_bench.scoring import ScoreReport

STYLE = """
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:#f8fafc;color:#0f172a;
     font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans CJK SC',sans-serif;
     font-size:14px;line-height:1.55}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid #e2e8f0}
.meta{color:#64748b;font-size:12px;margin-bottom:18px;white-space:pre-wrap}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #eef2f7;padding:6px 8px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:#475569;font-weight:600;background:#f8fafc}
.muted{color:#94a3b8}
.badge{display:inline-block;padding:1px 7px;border-radius:999px;font-size:12px;
       background:#fee2e2;color:#991b1b;margin-left:6px}
.pass{color:#047857;font-weight:600}
.fail{color:#b91c1c;font-weight:600}
details{margin:8px 0;border:1px solid #e2e8f0;border-radius:8px;background:#fff}
details>summary{cursor:pointer;padding:8px 12px;font-weight:600}
details[open]>summary{border-bottom:1px solid #eef2f7}
.detail{padding:10px 12px}
.kv{margin:2px 0}
.kv b{color:#475569;font-weight:600}
pre{background:#0f172a;color:#e2e8f0;padding:10px;border-radius:8px;overflow:auto;
    font-size:12px;max-height:320px;white-space:pre-wrap}
code{background:#f1f5f9;padding:1px 4px;border-radius:4px}
"""


def render_html(*, report: ScoreReport, traces: list[TurnTrace], run_meta: dict[str, Any],
                output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        _header(report, run_meta),
        _radar_card(report),
        _dimension_card(report),
        _metric_card(report),
        _heatmap_card(traces),
        _curve_card(traces, report),
        _efficiency_card(report),
        _unsupported_card(report),
        _failures_card(traces, report),
        _data_card(report, traces),
    ]
    html = (
        "<!doctype html><html lang=\"zh-Hans\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>agent-bench · {escape(report.run_id)}</title>"
        f"<style>{STYLE}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    output.write_text(html, encoding="utf-8")
    return output


def _header(report: ScoreReport, run_meta: dict[str, Any]) -> str:
    model = run_meta.get("model", {})
    alignment = run_meta.get("context_alignment", {})
    judge = run_meta.get("llm_judge", {})
    judge_text = f"开启（模型 {judge.get('model')}）" if judge.get("enabled") else "关闭"
    meta = [
        f"run {report.run_id}",
        (f"模型 {model.get('name', '—')} · 温度 {model.get('temperature', '—')} · "
         f"输出上限 {model.get('max_tokens', '—')} · "
         f"推理强度 {model.get('reasoning_effort') or '默认'}"),
        (f"对齐窗口 {alignment.get('align_context_window', '—')} Tokens · "
         f"种子 {run_meta.get('seed', '—')}"),
        (f"预算上限 {model.get('budget_max_cost_cny', '—')} CNY · "
         f"bootstrap {report.bootstrap_iterations} 次 / {report.confidence:.0%} 置信区间"),
        f"LLM 评分：{judge_text}",
    ]
    names = "".join(
        f"<tr><td>{escape(name)}</td><td>{_score_text(score.composite)}</td>"
        f"<td>{_score_text(None if score.pass_k is None else score.pass_k * 100)}</td>"
        f"<td>{score.efficiency.calls}</td><td>{score.efficiency.cost_cny:.3f}</td>"
        f"<td class=\"muted\">{escape(', '.join(score.capabilities)) or '—'}</td></tr>"
        for name, score in report.agents.items())
    shared = ", ".join(DIMENSION_LABELS.get(item, item) for item in report.shared_dimensions) or "无"
    return (
        f"<h1>agent-bench 成绩单</h1><div class=\"meta\">{escape(chr(10).join(meta))}</div>"
        "<div class=\"card\"><table><thead><tr><th>agent</th><th>综合分</th><th>pass^k</th>"
        "<th>调用次数</th><th>费用 CNY</th><th>能力</th></tr></thead><tbody>"
        f"{names}</tbody></table><p class=\"muted\">综合分只使用双方都支持的维度：{escape(shared)}</p></div>"
    )


def _score_text(value: float | None) -> str:
    return "<span class=\"muted\">不支持</span>" if value is None else f"{value:.1f}"


def _radar_card(report: ScoreReport) -> str:
    names = list(report.agents)
    axes = [DIMENSION_LABELS[dimension] for dimension in DIMENSIONS
            if any(report.agents[name].dimensions.get(dimension) is not None
                   and report.agents[name].dimensions[dimension].score is not None
                   for name in names)]
    if len(axes) < 3:
        return ('<h2>维度雷达</h2><div class="card muted">'
                "至少需要 3 个已测维度才能画雷达图。</div>")
    dimensions = [dimension for dimension in DIMENSIONS
                  if any(report.agents[name].dimensions.get(dimension) is not None
                         and report.agents[name].dimensions[dimension].score is not None
                         for name in names)]
    series = []
    for name in names:
        values = []
        for dimension in dimensions:
            item = report.agents[name].dimensions.get(dimension)
            values.append(None if item is None or item.score is None else item.score)
        series.append((name, values))
    return f"<h2>维度雷达</h2><div class=\"card\">{svg.radar(axes, series)}</div>"


def _dimension_card(report: ScoreReport) -> str:
    names = list(report.agents)
    header = "".join(f"<th>{escape(name)}</th>" for name in names)
    rows = []
    for dimension in DIMENSIONS:
        cells = [report.agents[name].dimensions.get(dimension) for name in names]
        if not any(cell is not None and cell.score is not None for cell in cells):
            continue
        row = "".join(f"<td>{_score_text(cell.score if cell else None)}</td>" for cell in cells)
        rows.append(f"<tr><td>{DIMENSION_LABELS[dimension]}（{dimension}）</td>{row}</tr>")
    composite = "".join(f"<td><b>{_score_text(report.agents[name].composite)}</b></td>"
                        for name in names)
    rows.append(f"<tr><td><b>综合分</b></td>{composite}</tr>")
    return (f"<h2>维度分</h2><div class=\"card\"><table><thead><tr><th>维度</th>{header}</tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table></div>")


def _metric_card(report: ScoreReport) -> str:
    names = list(report.agents)
    metrics = sorted({metric for score in report.agents.values() for metric in score.metrics})
    if not metrics:
        return ""
    header = "".join(f"<th>{escape(name)}</th>" for name in names)
    rows = []
    for metric in metrics:
        cells = []
        for name in names:
            stat = report.agents[name].metrics.get(metric)
            cells.append("<td class=\"muted\">不支持</td>" if stat is None else
                         f"<td>{stat.mean:.2f} ± {stat.std:.2f}"
                         f"<span class=\"muted\"> [{stat.ci_low:.2f},{stat.ci_high:.2f}] n={stat.n}</span></td>")
        rows.append(f"<tr><td>{escape(metric)}</td>{''.join(cells)}</tr>")
    return (f"<h2>指标明细</h2><div class=\"card\"><table><thead><tr><th>指标</th>{header}</tr>"
            f"</thead><tbody>{''.join(rows)}</tbody></table>"
            "<p class=\"muted\">括号内为 bootstrap 置信区间与样本数。</p></div>")


def _tag_value(trace: TurnTrace, prefix: str) -> str | None:
    for tag in trace.tags:
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return None


def _heatmap_card(traces: list[TurnTrace]) -> str:
    tagged = [trace for trace in traces if _tag_value(trace, "position:")
              and _tag_value(trace, "window:")]
    if not tagged:
        return ""
    positions = sorted({_tag_value(trace, "position:") for trace in tagged},
                       key=lambda item: int(item) if item and item.isdigit() else 0)
    windows = sorted({_tag_value(trace, "window:") for trace in tagged},
                     key=lambda item: int(item) if item and item.isdigit() else 0)
    agents = sorted({trace.agent for trace in tagged})
    blocks = []
    for agent in agents:
        matrix = []
        for position in positions:
            row = []
            for window in windows:
                values = [mean(list(trace.metrics.values()))
                          for trace in tagged
                          if trace.agent == agent
                          and _tag_value(trace, "position:") == position
                          and _tag_value(trace, "window:") == window
                          and trace.metrics]
                row.append(mean(values) if values else None)
            matrix.append(row)
        blocks.append(f"<div class=\"card\"><b>{escape(agent)}</b>"
                      f"{svg.heatmap([f'位置 {item}%' for item in positions], [f'{item}' for item in windows], matrix)}"
                      "<p class=\"muted\">行=事实在上下文中的位置（百分比），列=对齐窗口 Tokens，"
                      "格内数值=该组用例的平均得分×100。</p></div>")
    return f"<h2>位置与深度召回热力图</h2><div class=\"grid\">{''.join(blocks)}</div>"


def _curve_card(traces: list[TurnTrace], report: ScoreReport) -> str:
    agents = list(report.agents)
    growth_series = []
    markers: list[tuple[float, float, str]] = []
    for agent in agents:
        by_case: dict[tuple[str, int], list[TurnTrace]] = {}
        for trace in traces:
            if trace.agent != agent or trace.step_kind == "case":
                continue
            by_case.setdefault((trace.case_id, trace.repeat), []).append(trace)
        if not by_case:
            continue
        best = max(by_case.values(), key=len)
        points = []
        real_events = False
        for index, trace in enumerate(best, start=1):
            tokens = sum(call.prompt_tokens or 0 for call in trace.calls)
            points.append((float(index), float(tokens)))
            for event in trace.diagnostics.get("events", []) or []:
                if event.get("type") == "compaction":
                    real_events = True
                    label = (f"{agent}: 压缩 {event.get('before_tokens')}"
                             f"→{event.get('after_tokens')}")
                    markers.append((float(index), float(tokens), label))
                elif event.get("type") == "prune":
                    markers.append((float(index), float(tokens),
                                    f"{agent}: 回收 {event.get('count')} 条工具结果"))
        if not real_events:
            # 黑盒 agent 没有内部事件：用相邻轮次输入明显下降推断压缩点，并标注为推断值。
            for previous, current in itertools.pairwise(points):
                if previous[1] > 0 and current[1] < previous[1] * 0.8:
                    label = (f"{agent}: 推断压缩（输入 {previous[1]:.0f}"
                             f"→{current[1]:.0f}，非 agent 自报）")
                    markers.append((current[0], current[1], label))
        growth_series.append((agent, points))
    growth = (svg.lines(growth_series, x_label="轮次（取该 agent 最长用例）",
                        y_label="本轮输入 Token", markers=markers)
              if growth_series else "")

    decay_series = []
    for agent in agents:
        by_point: dict[int, list[float]] = {}
        for trace in traces:
            if trace.agent != agent:
                continue
            compactions = _tag_value(trace, "compactions:")
            if not compactions or not compactions.isdigit():
                continue
            by_point.setdefault(int(compactions), []).extend(trace.metrics.values())
        if by_point:
            decay_series.append((agent, [(float(key), mean(values))
                                         for key, values in sorted(by_point.items())]))
    decay = (svg.lines(decay_series, x_label="累计压缩次数", y_label="事实保留率", y_max=1.0)
             if decay_series else "")
    if not growth and not decay:
        return ""
    growth_card = growth or '<span class="muted">没有可绘制的轮次数据</span>'
    decay_card = decay or '<span class="muted">没有 retention_drift 用例</span>'
    return (
        '<h2>压缩与上下文增长</h2><div class="grid">'
        f'<div class="card">{growth_card}'
        '<p class="muted">红色三角：压缩（Lynx 用真实诊断事件，黑盒 agent 由输入 Token 突降推断）。'
        "</p></div>"
        f'<div class="card">{decay_card}'
        '<p class="muted">横轴为填充到触发约 N 次压缩后的提问，纵轴为该组事实保留率。</p></div>'
        "</div>"
    )


def _efficiency_card(report: ScoreReport) -> str:
    rows = []
    for name, score in report.agents.items():
        item = score.efficiency
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{item.calls}</td>"
            f"<td>{item.prompt_tokens:,}</td><td>{item.completion_tokens:,}</td>"
            f"<td>{'—' if item.cached_ratio is None else f'{item.cached_ratio:.0%}'}</td>"
            f"<td>{'—' if item.latency_p50_ms is None else f'{item.latency_p50_ms:.0f} ms'}</td>"
            f"<td>{item.cost_cny:.3f}</td><td>{item.cost_per_case_cny:.4f}</td>"
            f"<td>{item.retries}</td></tr>")
    return (
        "<h2>效率与成本</h2><div class=\"card\"><table><thead><tr><th>agent</th><th>调用</th>"
        "<th>输入 Token</th><th>输出 Token</th><th>缓存命中</th><th>P50 延迟</th>"
        "<th>费用 CNY</th><th>每用例 CNY</th><th>重试</th></tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table></div>")


def _unsupported_card(report: ScoreReport) -> str:
    rows = []
    for name, score in report.agents.items():
        if not score.unsupported_cases:
            continue
        rows.append(f"<tr><td>{escape(name)}</td><td class=\"muted\">"
                    f"{escape(', '.join(score.unsupported_cases))}</td></tr>")
    if not rows:
        return ""
    return ("<h2>不支持的用例</h2><div class=\"card\"><table><thead><tr><th>agent</th>"
            f"<th>用例</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
            "<p class=\"muted\">能力不足的用例不计 0 分，也不进入综合分。</p></div>")


def _failures_card(traces: list[TurnTrace], report: ScoreReport) -> str:
    failures: dict[tuple[str, str, int], list[TurnTrace]] = {}
    for trace in traces:
        failed = any(not item.passed for item in trace.grades) or bool(trace.error)
        if failed:
            failures.setdefault((trace.agent, trace.case_id, trace.repeat), []).append(trace)
    if not failures:
        return "<h2>失败用例</h2><div class=\"card pass\">本次没有失败用例。</div>"
    blocks = []
    for (agent, case_id, repeat), items in sorted(failures.items()):
        first = items[0]
        facts = "".join(f"<div class=\"kv\"><b>{escape(key)}</b> = <code>{escape(value)}</code></div>"
                        for key, value in first.facts.items())
        turns = []
        for trace in items:
            grades = "".join(
                f"<li>{'<span class=pass>通过</span>' if item.passed else '<span class=fail>未通过</span>'} "
                f"{escape(item.grader)}：{escape(item.detail)}"
                f"{'' if item.expected is None else '（期望 ' + escape(str(item.expected)) + '）'}</li>"
                for item in trace.grades)
            calls = "".join(
                f"<div class=\"kv\"><b>调用</b> 状态 {call.status} · 输入 "
                f"{call.prompt_tokens} · 输出 {call.completion_tokens} · "
                f"缓存 {call.cached_prompt_tokens} · {call.latency_ms:.0f} ms"
                f"{' · 补注 usage' if call.usage_injected else ''}</div>"
                for call in trace.calls)
            extra = _diagnostics_block(trace)
            turns.append(
                f"<div class=\"detail\"><div class=\"kv\"><b>第 {trace.turn} 轮"
                f"（步骤 {trace.step}，{'填充' if trace.step_kind == 'filler' else '提问'}）</b></div>"
                f"<div class=\"kv\"><b>用户</b></div><pre>{escape(trace.user[:4000])}</pre>"
                f"<div class=\"kv\"><b>回答</b></div><pre>{escape(trace.answer[:4000])}</pre>"
                f"{'<div class=kv><b>错误</b> ' + escape(str(trace.error)) + '</div>' if trace.error else ''}"
                f"<ul>{grades or '<li class=muted>本轮没有判分</li>'}</ul>{calls}{extra}</div>")
        blocks.append(
            f"<details><summary>{escape(agent)} · {escape(case_id)} · 重复 {repeat}"
            f"<span class=\"badge\">{len(items)} 轮失败</span></summary>"
            f"<div class=\"detail\">{facts}{''.join(turns)}</div></details>")
    return f"<h2>失败用例</h2><div class=\"card\">{''.join(blocks)}</div>"


def _diagnostics_block(trace: TurnTrace) -> str:
    parts = []
    compaction = trace.diagnostics.get("compaction") or []
    if compaction:
        summary = compaction[-1].get("summary", "")
        parts.append("<div class=\"kv\"><b>Lynx 压缩摘要（诊断，不计分）</b></div>"
                     f"<pre>{escape(str(summary)[:4000])}</pre>")
    prune = trace.diagnostics.get("prune") or []
    if prune:
        total = sum(int(item.get("count", 0)) for item in prune)
        parts.append(f"<div class=\"kv\"><b>Lynx 工具结果回收</b> {total} 条</div>")
    recall = trace.diagnostics.get("recalled_text")
    if recall:
        parts.append("<div class=\"kv\"><b>Lynx 记忆召回（诊断，不计分）</b></div>"
                     f"<pre>{escape(str(recall)[:2000])}</pre>")
    return "".join(parts)


def _data_card(report: ScoreReport, traces: list[TurnTrace]) -> str:
    payload = {
        "report": json.loads(report.model_dump_json()),
        "traces": [json.loads(trace.model_dump_json()) for trace in traces[:2000]],
    }
    data = escape(json.dumps(payload, ensure_ascii=False), quote=False).replace("</", "<\\/")
    return ("<h2>内联数据</h2><div class=\"card\"><p class=\"muted\">"
            "scores.json 与 traces.jsonl 的关键内容已内联，离线也能查看。</p>"
            "<details><summary>展开原始 JSON</summary>"
            f"<pre id=\"bench-data\">{data}</pre></details></div>")
