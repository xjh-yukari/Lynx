"""用例加载、随机事实、填充文本与校验。"""

import pytest
from agent_bench.cases import FactGenerator, load_cases, resolve_case, validate_cases
from agent_bench.cases.filler import estimate_tokens, generate_document, generate_filler
from agent_bench.cases.loader import CaseError
from agent_bench.cases.models import FillerSpec


def test_shipped_suites_are_valid():
    cases = load_cases()
    assert len(cases) >= 50
    suites = {case.suite for case in cases}
    assert {"context", "memory", "tools", "robustness", "injection"} <= suites
    assert validate_cases(cases) == []
    assert len([case for case in cases if case.suite == "context"]) >= 12
    assert len([case for case in cases if case.suite == "memory"]) >= 12


def test_facts_are_deterministic_and_reproducible():
    cases = [case for case in load_cases() if case.facts]
    assert cases
    case = cases[0]
    first = FactGenerator(seed=7).generate(case.facts, case_id=case.id, repeat=0)
    second = FactGenerator(seed=7).generate(case.facts, case_id=case.id, repeat=0)
    other = FactGenerator(seed=8).generate(case.facts, case_id=case.id, repeat=0)
    assert first == second
    assert first != other
    for key, spec in case.facts.items():
        if spec.distinct_from:
            assert first[key] != first[spec.distinct_from]


def test_placeholder_expansion_and_distinct_choice():
    cases = {case.id: case for case in load_cases()}
    case = cases["memory.update.package_manager"]
    resolved = resolve_case(case, FactGenerator(seed=11), repeat=2)
    assert resolved.facts["old_pm"] != resolved.facts["new_pm"]
    text = " ".join(step.user or "" for step in resolved.steps)
    assert "{" not in text
    assert resolved.facts["new_pm"] in text


def test_filler_is_seeded_and_roughly_sized():
    spec = FillerSpec(tokens=3000, turns=2, topic="测试")
    import random

    first = generate_filler(spec, random.Random(1), turn=0)
    second = generate_filler(spec, random.Random(1), turn=0)
    assert first == second
    assert abs(estimate_tokens(first) - 1500) < 600
    assert "测试" in first


def test_generated_document_contains_facts_at_position():
    import random

    text = generate_document(chars=4000, topic="手册",
                             facts={"50": "口令是 ABC-123"}, rng=random.Random(3))
    assert len(text) > 4000
    assert "口令是 ABC-123" in text


def test_unknown_placeholder_is_reported(tmp_path):
    path = tmp_path / "suites" / "context" / "bad.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "id: context.bad\nsuite: context\nmetrics: [context.retention]\n"
        "steps:\n  - user: '代号是 {missing}'\n    grade: [{type: contains, value: x}]\n",
        encoding="utf-8",
    )
    cases = load_cases(tmp_path / "suites")
    problems = validate_cases(cases)
    assert any("missing" in problem for problem in problems)


def test_loader_rejects_wrong_suite_directory(tmp_path):
    path = tmp_path / "suites" / "memory" / "bad.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("id: context.bad\nsuite: context\nmetrics: [context.retention]\n"
                    "steps:\n  - user: hi\n", encoding="utf-8")
    with pytest.raises(CaseError):
        load_cases(tmp_path / "suites")


def test_validate_flags_llm_judge_only_case(tmp_path):
    path = tmp_path / "suites" / "injection" / "judge.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "id: injection.only_judge\nsuite: injection\nmetrics: [injection.resistance]\n"
        "steps:\n  - user: hi\n    grade: [{type: llm_judge, rubric: ok}]\n",
        encoding="utf-8",
    )
    problems = validate_cases(load_cases(tmp_path / "suites"))
    assert any("llm_judge" in problem for problem in problems)


def test_document_directive_expands_to_generated_text():
    cases = {case.id: case for case in load_cases()}
    case = cases["context.tool_result.recall"]
    resolved = resolve_case(case, FactGenerator(seed=5))
    document = resolved.steps[0].env["documents"]["ops-manual"]
    assert isinstance(document, str) and len(document) > 10000
    assert resolved.facts["token"] in document
