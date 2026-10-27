"""确定性判分器：字符串、数值、工具调用与最终状态。

只有 llm_judge 需要模型，且默认关闭；关闭时该条判分不参与结论，
`bench validate` 会拒绝“只靠 llm_judge 判分”的用例。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from agent_bench.graders.llm_judge import LlmJudge
from agent_bench.models import GradeResult, GradeSpec, ToolCallRecord

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class GradeContext:
    answer: str
    tools: tuple[ToolCallRecord, ...] = ()
    case_tools: tuple[ToolCallRecord, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)
    judge: LlmJudge | None = None
    question: str = ""


def normalize(text: str) -> str:
    """全角转半角（NFKC）、大小写折叠、去除全部空白后比较。"""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"\s+", "", folded)


async def grade(specs: tuple[GradeSpec, ...], context: GradeContext) -> list[GradeResult]:
    results = []
    for spec in specs:
        results.append(await _grade_one(spec, context))
    return results


async def _grade_one(spec: GradeSpec, context: GradeContext) -> GradeResult:
    kind = spec.type
    if kind == "llm_judge":
        return await _judge(spec, context)
    if kind == "expect_tool":
        return _expect_tool(spec, context)
    if kind == "expect_state":
        return _expect_state(spec, context)
    if kind == "numeric":
        return _numeric(spec, context.answer)
    if kind == "length_limit":
        return _length(spec, context.answer)
    if kind == "regex":
        return _regex(spec, context.answer)
    expected = str(spec.value if spec.value is not None else "")
    answer = context.answer or ""
    if kind == "not_contains":
        passed = normalize(expected) not in normalize(answer)
        return GradeResult(grader=kind, passed=passed, expected=expected,
                           actual=_clip(answer),
                           detail="未出现期望排除的内容" if passed else "回答里仍出现了应被排除的内容")
    if kind == "exact":
        passed = normalize(expected) == normalize(answer)
        return GradeResult(grader=kind, passed=passed, expected=expected,
                           actual=_clip(answer),
                           detail="规范化后完全一致" if passed else "规范化后与期望值不同")
    passed = normalize(expected) in normalize(answer)
    return GradeResult(grader=kind, passed=passed, expected=expected, actual=_clip(answer),
                       detail="包含期望内容" if passed else "回答中找不到期望内容")


def _regex(spec: GradeSpec, answer: str) -> GradeResult:
    pattern = str(spec.value or "")
    try:
        matched = re.search(pattern, answer or "", 0 if spec.case_sensitive else re.IGNORECASE) is not None
        error = ""
    except re.error:
        matched, error = False, "正则表达式不合法"
    return GradeResult(grader="regex", passed=matched, expected=pattern, actual=_clip(answer),
                       detail=error or ("匹配成功" if matched else "未匹配"))


def _numeric(spec: GradeSpec, answer: str) -> GradeResult:
    expected = float(spec.value)  # type: ignore[arg-type]
    tolerance = abs(float(spec.tolerance or 0))
    found = NUMBER.search(answer or "")
    if found is None:
        return GradeResult(grader="numeric", passed=False, expected=str(expected),
                           actual=_clip(answer), detail="回答中没有可解析的数字")
    value = float(found.group(0))
    passed = abs(value - expected) <= tolerance
    return GradeResult(grader="numeric", passed=passed, expected=str(expected),
                       actual=found.group(0),
                       detail=f"差值 {abs(value - expected):g}，容差 {tolerance:g}")


def _length(spec: GradeSpec, answer: str) -> GradeResult:
    size = len(answer or "")
    passed = True
    notes = []
    if spec.max is not None and size > spec.max:
        passed = False
        notes.append(f"超过上限 {spec.max}")
    if spec.min is not None and size < spec.min:
        passed = False
        notes.append(f"低于下限 {spec.min}")
    return GradeResult(grader="length_limit", passed=passed, expected=f"{spec.min or 0}..{spec.max or '∞'}",
                       actual=str(size), detail="；".join(notes) or "长度符合要求")


def _expect_tool(spec: GradeSpec, context: GradeContext) -> GradeResult:
    records = context.tools or context.case_tools
    if spec.name == "*":
        # 通配：检查本轮是否调用了任何工具。
        passed = (not records) if not spec.called else bool(records)
        return GradeResult(
            grader="expect_tool", passed=passed, expected="任意工具" if spec.called else "不调用任何工具",
            actual=", ".join(sorted({record.tool for record in records})) or "无工具调用",
            detail="符合预期" if passed else ("不应调用工具却调用了" if not spec.called else "没有调用任何工具"),
        )
    wanted = [record for record in records if record.tool == spec.name]
    matched = [record for record in wanted if _args_match(spec.args, record.arguments)]
    if spec.called:
        passed = bool(matched)
        detail = ("已调用且参数匹配" if passed else
                  ("调用了但参数不匹配" if wanted else "没有调用该工具"))
    else:
        passed = not matched
        detail = "确认未调用" if passed else "不应调用却调用了"
    return GradeResult(grader="expect_tool", passed=passed, expected=spec.name or "",
                       actual=", ".join(sorted({record.tool for record in records})) or "无工具调用",
                       detail=detail)


def _args_match(expected: dict[str, Any] | None, actual: dict[str, Any]) -> bool:
    if not expected:
        return True
    for key, value in expected.items():
        if key not in actual:
            return False
        found = actual[key]
        if isinstance(value, (int, float)) and isinstance(found, (int, float)):
            if abs(float(value) - float(found)) > 1e-9:
                return False
        elif normalize(str(value)) != normalize(str(found)):
            return False
    return True


def _expect_state(spec: GradeSpec, context: GradeContext) -> GradeResult:
    passed, path, detail = subset_match(spec.state or {}, context.state)
    return GradeResult(grader="expect_state", passed=passed, expected=path, actual=detail,
                       detail="最终状态符合预期" if passed else f"状态不匹配：{path}（{detail}）")


def subset_match(expected: Any, actual: Any, path: str = "") -> tuple[bool, str, str]:
    """期望值是实际值的递归子集；字符串按规范化比较。"""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, path or "<root>", f"期望对象，实际是 {type(actual).__name__}"
        for key, value in expected.items():
            if key not in actual:
                return False, f"{path}.{key}".lstrip("."), "缺少该键"
            ok, sub_path, detail = subset_match(value, actual[key], f"{path}.{key}".lstrip("."))
            if not ok:
                return ok, sub_path, detail
        return True, path, "ok"
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) > len(actual):
            return False, path, f"期望至少 {len(expected)} 个元素"
        for index, value in enumerate(expected):
            ok, sub_path, detail = subset_match(value, actual[index], f"{path}[{index}]")
            if not ok:
                return ok, sub_path, detail
        return True, path, "ok"
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual, path, f"{expected!r} vs {actual!r}"
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return float(expected) == float(actual), path, f"{expected} vs {actual}"
    if normalize(str(expected)) == normalize(str(actual)):
        return True, path, "ok"
    return False, path, f"{_clip(str(actual))}"


async def _judge(spec: GradeSpec, context: GradeContext) -> GradeResult:
    if context.judge is None:
        # 明确标成“未参与判分”，避免默认关闭时把用例判成通过。
        return GradeResult(grader="llm_judge_skipped", passed=True,
                           detail="llm_judge 已关闭，本条不参与判分")
    passed, detail = await context.judge.evaluate(
        rubric=spec.rubric or "", question=context.question, answer=context.answer,
    )
    return GradeResult(grader="llm_judge", passed=passed, expected="PASS", actual=detail,
                       detail=detail)


def _clip(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"
