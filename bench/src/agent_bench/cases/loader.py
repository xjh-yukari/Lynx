"""用例加载、占位符展开与校验。

`bench validate` 依赖 validate_cases 做静态体检：结构、占位符、指标名、
所需能力、以及判分规则与套件维度是否自洽。
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import yaml

from agent_bench.cases.facts import FactGenerator
from agent_bench.cases.filler import generate_document
from agent_bench.cases.models import (
    BenchCase,
    ResolvedCase,
    ResolvedStep,
    StepSpec,
)
from agent_bench.env_mcp.server import TOOL_NAMES
from agent_bench.metrics import METRICS, is_known
from agent_bench.models import CAPABILITIES

PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
SUITE_NAMES = ("context", "memory", "tools", "robustness", "injection")
STATE_KEYS = {"kv", "counter", "files", "orders", "documents", "pages"}


class CaseError(Exception):
    """用例不合法；错误信息带文件与字段位置。"""


def default_suites_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "suites"


def load_cases(directory: str | Path | None = None,
               suites: tuple[str, ...] | None = None) -> list[BenchCase]:
    """读取套件目录下的全部 YAML；按 id 排序保证输出稳定。"""
    root = Path(directory) if directory is not None else default_suites_dir()
    if not root.is_dir():
        raise CaseError(f"找不到套件目录：{root}")
    cases: list[BenchCase] = []
    for path in sorted(item for item in root.rglob("*") if item.suffix in {".yaml", ".yml"}):
        if suites and path.parent.name not in suites:
            continue
        cases.extend(_load_file(path))
    ids = [case.id for case in cases]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise CaseError(f"用例 id 重复：{', '.join(duplicates)}")
    return sorted(cases, key=lambda case: case.id)


def _load_file(path: Path) -> list[BenchCase]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise CaseError(f"{path}: 无法读取或不是合法 YAML") from error
    if isinstance(raw, dict) and "cases" in raw:
        items = raw["cases"]
    elif isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    cases = []
    for item in items:
        if not isinstance(item, dict):
            raise CaseError(f"{path}: 每个用例必须是映射")
        try:
            case = BenchCase.model_validate(item)
        except Exception as error:
            raise CaseError(f"{path}: {error}") from error
        if case.suite != path.parent.name:
            raise CaseError(f"{path}: suite 字段必须与目录名一致（{path.parent.name}）")
        if not case.id.startswith(case.suite + "."):
            raise CaseError(f"{path}: 用例 id 必须以 {case.suite}. 开头")
        cases.append(case)
    if not cases:
        raise CaseError(f"{path}: 没有用例")
    return cases


def _substitute(value: Any, facts: dict[str, str], where: str,
                rng: random.Random | None = None) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in facts:
                raise CaseError(f"{where}: 未定义占位符 {{{key}}}")
            return facts[key]

        return PLACEHOLDER.sub(replace, value)
    if isinstance(value, dict):
        if "$text" in value:
            return _generated_text(value["$text"], facts, where, rng)
        return {key: _substitute(item, facts, where, rng) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_substitute(item, facts, where, rng) for item in value]
    return value


def _generated_text(spec: Any, facts: dict[str, str], where: str,
                    rng: random.Random | None) -> str:
    """把 `{$text: {chars, topic, facts}}` 展开成按种子生成的正文。"""
    if not isinstance(spec, dict):
        raise CaseError(f"{where}: $text 需要对象参数")
    payload = _substitute(spec, facts, where, rng)
    try:
        return generate_document(chars=int(payload.get("chars", 5000)),
                                 topic=str(payload.get("topic", "资料")),
                                 facts=dict(payload.get("facts") or {}),
                                 rng=rng or random.Random(0))
    except (TypeError, ValueError) as error:
        raise CaseError(f"{where}: $text 参数不合法：{error}") from error


def _resolve_step(index: int, step: StepSpec, facts: dict[str, str], where: str,
                  rng: random.Random | None = None) -> ResolvedStep:
    grades = []
    for spec in step.grade:
        payload = spec.model_dump()
        payload["value"] = _substitute(payload.get("value"), facts, where, rng)
        payload["args"] = _substitute(payload.get("args"), facts, where, rng)
        payload["state"] = _substitute(payload.get("state"), facts, where, rng)
        from agent_bench.models import GradeSpec

        grades.append(GradeSpec.model_validate(payload))
    return ResolvedStep(
        index=index,
        session=step.session,
        user=_substitute(step.user, facts, where, rng),
        filler=step.filler,
        env=_substitute(step.env, facts, where, rng),
        expect_tool=None if step.expect_tool is None else step.expect_tool.model_copy(
            update={"args": _substitute(step.expect_tool.args, facts, where, rng)}),
        expect_state=_substitute(step.expect_state, facts, where, rng),
        grade=tuple(grades),
        metric=step.metric,
        max_tokens=step.max_tokens,
    )


def resolve_case(case: BenchCase, generator: FactGenerator, repeat: int = 0) -> ResolvedCase:
    facts = generator.generate(case.facts, case_id=case.id, repeat=repeat)
    seed = generator.case_seed(case.id, repeat)
    steps = tuple(
        _resolve_step(index, step, facts, f"{case.id} 步骤 {index + 1}",
                      random.Random(seed + index))
        for index, step in enumerate(case.steps)
    )
    return ResolvedCase(case=case, facts=facts, steps=steps, seed=seed)


def validate_cases(cases: list[BenchCase], generator: FactGenerator | None = None) -> list[str]:
    """返回问题列表；空列表表示全部通过。"""
    problems: list[str] = []
    factory = generator or FactGenerator(seed=1)
    for case in cases:
        unknown = [name for name in case.metrics if not is_known(name)]
        if unknown:
            problems.append(f"{case.id}: 未登记的指标 {', '.join(unknown)}")
        for requires in case.requires:
            if requires not in CAPABILITIES:
                problems.append(f"{case.id}: 未知能力 {requires}")
        if not any(step.grade or step.expect_tool or step.expect_state for step in case.steps):
            problems.append(f"{case.id}: 没有任何判分步骤（grade/expect_tool/expect_state）")
        for step_index, step in enumerate(case.steps):
            where = f"{case.id} 步骤 {step_index + 1}"
            if step.expect_tool is not None and step.expect_tool.name != "*" \
                    and step.expect_tool.name not in TOOL_NAMES:
                problems.append(f"{where}: expect_tool 使用了未暴露的工具 {step.expect_tool.name}")
            if step.expect_state is not None:
                unknown_state = set(step.expect_state) - STATE_KEYS
                if unknown_state:
                    problems.append(f"{where}: expect_state 含未知分区 {', '.join(sorted(unknown_state))}")
            if step.filler is not None and step.user:
                problems.append(f"{where}: 同一步骤不能既有 user 又有 filler")
        for metric in case.metrics:
            definition = METRICS.get(metric)
            if definition is None:
                continue
            if definition.dimension not in {case.suite, "efficiency", "stability"}:
                problems.append(
                    f"{case.id}: 指标 {metric} 属于 {definition.dimension} 维度，与套件 {case.suite} 不一致")
        try:
            resolved = resolve_case(case, factory, 0)
        except CaseError as error:
            problems.append(str(error))
            continue
        for step in resolved.steps:
            where = f"{case.id} 步骤 {step.index + 1}"
            metric = step.metric
            if metric is not None and metric not in case.metrics:
                problems.append(f"{case.id}: 步骤声明的指标 {metric} 不在 metrics 列表里")
            for spec in step.grade:
                problem = _check_grade(spec, where)
                if problem:
                    problems.append(problem)
        if all(spec.type == "llm_judge" for step in case.steps for spec in step.grade) \
                and any(step.grade for step in case.steps):
            problems.append(f"{case.id}: 只使用 llm_judge 判分；该功能默认关闭，无法离线复现")
    return problems


def _check_grade(spec, where: str) -> str | None:
    kind = spec.type
    if kind in {"contains", "not_contains", "exact", "regex"} and not isinstance(spec.value, str):
        return f"{where}: {kind} 需要字符串 value"
    if kind == "numeric":
        if spec.value is None:
            return f"{where}: numeric 需要数值 value"
        try:
            float(spec.value)
        except (TypeError, ValueError):
            return f"{where}: numeric 的 value 不是数字"
        if spec.tolerance is None:
            return f"{where}: numeric 需要 tolerance"
    if kind == "length_limit" and spec.max is None and spec.min is None:
        return f"{where}: length_limit 需要 max 或 min"
    if kind == "expect_tool" and not spec.name:
        return f"{where}: expect_tool 需要 name"
    if kind == "expect_tool" and spec.name and spec.name != "*" and spec.name not in TOOL_NAMES:
        return f"{where}: expect_tool 使用了未暴露的工具 {spec.name}"
    if kind == "expect_state" and not spec.state:
        return f"{where}: expect_state 需要 state 断言"
    if kind == "llm_judge" and not spec.rubric:
        return f"{where}: llm_judge 需要 rubric"
    return None
