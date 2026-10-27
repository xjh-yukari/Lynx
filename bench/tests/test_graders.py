"""判分器：确定性规则、工具/状态断言与 llm_judge 默认关闭。"""

import pytest
from agent_bench.graders import GradeContext, grade, normalize, subset_match
from agent_bench.models import GradeSpec, ToolCallRecord


def record(tool="notes_set", arguments=None, turn=1):
    return ToolCallRecord(agent="fake", case_id="case", turn=turn, tool=tool,
                          arguments=arguments or {})


def test_normalize_handles_fullwidth_case_and_spaces():
    assert normalize("ＡＢＣ　１２３") == normalize("abc 123")
    assert normalize("Hello\nWorld") == "helloworld"


@pytest.mark.asyncio
async def test_string_graders():
    context = GradeContext(answer="项目代号是 K7-QX9，端口 48213。")
    results = await grade((
        GradeSpec(type="contains", value="K7-QX9"),
        GradeSpec(type="contains", value="不存在"),
        GradeSpec(type="not_contains", value="npm"),
        GradeSpec(type="regex", value=r"48213"),
        GradeSpec(type="exact", value="项目代号是 k7-qx9，端口 48213。"),
    ), context)
    assert [item.passed for item in results] == [True, False, True, True, True]


@pytest.mark.asyncio
async def test_numeric_reads_the_number_it_finds():
    # 回答里只有端口是数字，numeric 才取到它；混入代号时先命中代号里的数字。
    ok = await grade((GradeSpec(type="numeric", value="48213", tolerance=0),),
                     GradeContext(answer="端口是 48213。"))
    assert ok[0].passed
    first = await grade((GradeSpec(type="numeric", value="7", tolerance=0),),
                        GradeContext(answer="代号 K7-QX9"))
    assert first[0].passed  # 明确记录“取第一个数字”的规则


@pytest.mark.asyncio
async def test_numeric_tolerance_and_missing_number():
    ok = await grade((GradeSpec(type="numeric", value="100", tolerance=5),),
                     GradeContext(answer="大约是 103 左右"))
    bad = await grade((GradeSpec(type="numeric", value="100", tolerance=1),),
                      GradeContext(answer="大约是 103 左右"))
    missing = await grade((GradeSpec(type="numeric", value="100", tolerance=1),),
                          GradeContext(answer="没有数字"))
    assert ok[0].passed and not bad[0].passed and not missing[0].passed


@pytest.mark.asyncio
async def test_length_limit():
    results = await grade((GradeSpec(type="length_limit", max=10),), GradeContext(answer="短"))
    assert results[0].passed
    results = await grade((GradeSpec(type="length_limit", max=2),), GradeContext(answer="很长的一句话"))
    assert not results[0].passed


@pytest.mark.asyncio
async def test_expect_tool_matches_arguments():
    tools = (record("notes_set", {"key": "a", "value": "42"}), record("notes_get", {"key": "a"}))
    results = await grade((
        GradeSpec(type="expect_tool", name="notes_set", called=True, args={"key": "a"}),
        GradeSpec(type="expect_tool", name="notes_set", called=True, args={"key": "b"}),
        GradeSpec(type="expect_tool", name="orders_lookup", called=False),
        GradeSpec(type="expect_tool", name="*", called=True),
    ), GradeContext(answer="", tools=tools))
    assert [item.passed for item in results] == [True, False, True, True]


@pytest.mark.asyncio
async def test_expect_tool_without_calls():
    results = await grade((GradeSpec(type="expect_tool", name="*", called=False),),
                          GradeContext(answer="只回答文本"))
    assert results[0].passed


@pytest.mark.asyncio
async def test_expect_state_subset():
    state = {"kv": {"a": "1", "b": "2"}, "files": {"note.txt": "hello"}}
    results = await grade((
        GradeSpec(type="expect_state", state={"kv": {"a": "1"}}),
        GradeSpec(type="expect_state", state={"kv": {"a": "2"}}),
        GradeSpec(type="expect_state", state={"files": {"note.txt": "HELLO"}}),
        GradeSpec(type="expect_state", state={"kv": {"c": "3"}}),
    ), GradeContext(answer="", state=state))
    assert [item.passed for item in results] == [True, False, True, False]


def test_subset_match_reports_path():
    passed, path, _detail = subset_match({"kv": {"a": 1}}, {"kv": {"a": 2}})
    assert not passed and path == "kv.a"


@pytest.mark.asyncio
async def test_llm_judge_disabled_is_not_scored():
    results = await grade((GradeSpec(type="llm_judge", rubric="任意"),),
                          GradeContext(answer="回答"))
    assert results[0].grader == "llm_judge_skipped"
    assert results[0].passed is True


def test_judge_rejects_same_model():
    from agent_bench.graders.llm_judge import LlmJudge

    with pytest.raises(ValueError):
        LlmJudge(base_url="https://example.invalid", api_key="k", model="same", tested_model="same")
