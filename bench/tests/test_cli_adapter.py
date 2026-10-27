"""通用 CLI 适配器：命令模板展开、会话续接参数位置与环境注入。

只验证参数拼装与输出解析，不真正启动子进程。
"""

import pytest
from agent_bench.adapters.base import BenchEnv, RunContext
from agent_bench.adapters.cli import CliAdapter
from agent_bench.config import AgentSettings, ModelSettings


class StubProxy:
    def scope_url(self, scope: str) -> str:
        return f"http://127.0.0.1:9/p/{scope}"


def make_adapter(tmp_path, **settings):
    payload = {
        "kind": "cli",
        "command": ["dsh", "--profile", "headless", "--json", "{task}"],
        "resume_args": ["--session-id", "{session}"],
        "extra_args": ["--patch", "{patch}"],
        "patch": "- id: session-log-deepseek\n  disabled: true\n",
        "stdout": "ndjson",
        "answer_event_type": "final",
        "text_field": "text",
        "session_event_type": "session",
        "session_field": "sessionId",
        "tool_event_type": "tool_call",
    }
    payload.update(settings)
    context = RunContext(run_id="run", model=ModelSettings(), proxy=StubProxy(),  # type: ignore[arg-type]
                         recorder=None,  # type: ignore[arg-type]
                         api_key="test-key", turn_timeout_seconds=30)
    adapter = CliAdapter("dsh", AgentSettings.model_validate(payload), context)
    adapter.workspace = tmp_path
    adapter.env = BenchEnv(workspace=tmp_path, state_dir=tmp_path / "env",
                           scope="dsh~case-r0")
    adapter.env.reset()
    return adapter


def test_three_option_positions(tmp_path):
    adapter = make_adapter(tmp_path)
    adapter._patch_path = tmp_path / "bench-agent.patch.yml"
    adapter._session = "session-abc"
    argv = adapter._argv("请回答问题")
    # 启动器选项紧跟可执行文件，命令模板居中，续接选项紧贴任务。
    assert argv[:3] == ["dsh", "--patch", str(tmp_path / "bench-agent.patch.yml")]
    assert argv[3:6] == ["--profile", "headless", "--json"]
    task_index = argv.index("请回答问题")
    assert argv[task_index - 2: task_index] == ["--session-id", "session-abc"]
    assert len(argv) == task_index + 1


def test_stdin_mode_replaces_task_with_marker(tmp_path):
    adapter = make_adapter(tmp_path, stdin=True)
    adapter._session = "session-abc"
    argv = adapter._argv("任务里有 空格 与 % & 等字符")
    assert argv[-1] == "-"
    assert "任务里有 空格 与 % & 等字符" not in argv
    assert "--session-id" in argv


def test_cmd_shim_is_launched_through_cmd_exe(tmp_path, monkeypatch):
    import shutil as _shutil

    adapter = make_adapter(tmp_path)
    monkeypatch.setattr(_shutil, "which", lambda name: r"C:\npm\dsh.cmd")
    argv = adapter._launch_argv(["dsh", "--profile", "headless", "-"])
    assert argv[0] == "cmd.exe"
    assert argv[1:4] == ["/d", "/s", "/c"]
    assert "dsh.cmd" in argv[4]


def test_exe_is_launched_directly(tmp_path, monkeypatch):
    import shutil as _shutil

    adapter = make_adapter(tmp_path)
    monkeypatch.setattr(_shutil, "which", lambda name: r"C:\npm\myagent.exe")
    assert adapter._launch_argv(["myagent", "--json"]) == [r"C:\npm\myagent.exe", "--json"]


def test_new_session_omits_resume_args(tmp_path):
    adapter = make_adapter(tmp_path)
    argv = adapter._argv("任务")
    assert "--session-id" not in argv
    assert argv[-1] == "任务"


def test_command_without_task_placeholder_appends_it(tmp_path):
    adapter = make_adapter(tmp_path, command=["myagent", "--run"], extra_args=[])
    adapter._session = "s1"
    assert adapter._argv("任务") == ["myagent", "--run", "--session-id", "s1", "任务"]


def test_environment_carries_proxy_url_and_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP_ME", "1")
    adapter = make_adapter(tmp_path)
    env = adapter._environment()
    assert env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:9/p/dsh~case-r0"
    assert env["DEEPSEEK_API_KEY"] == "test-key"
    assert env["BENCH_WORKSPACE"] == str(tmp_path)
    assert env["BENCH_ENV_DIR"] == str(tmp_path / "env")
    assert env["KEEP_ME"] == "1"  # 继承父进程环境


def test_ndjson_parsing_takes_final_and_session(tmp_path):
    adapter = make_adapter(tmp_path)
    output = (
        '{"type": "session", "sessionId": "session-9", "cwd": "."}' "\n"
        '{"type": "text", "text": "中间文本"}' "\n"
        '{"type": "tool_call", "tool": "notes_set", "input": {"key": "k"}}' "\n"
        '{"type": "final", "text": "最终答案"}'
    )
    text, diagnostics, finish = adapter._parse(output)
    assert text == "最终答案"
    assert diagnostics["session"] == "session-9"
    assert diagnostics["tool_events"][0]["tool"] == "notes_set"
    assert finish is None
    assert adapter._session == "session-9"


def test_ndjson_error_event_marks_finish(tmp_path):
    adapter = make_adapter(tmp_path)
    text, diagnostics, finish = adapter._parse(
        '{"type": "error", "message": "boom"}')
    assert text == "" and finish == "error"
    assert diagnostics["error_events"] == ["boom"]


def test_plain_text_output(tmp_path):
    adapter = make_adapter(tmp_path, stdout="text")
    text, diagnostics, finish = adapter._parse("就是这段回答\n")
    assert text == "就是这段回答" and diagnostics == {} and finish is None


def test_secret_is_redacted_from_error_text(tmp_path):
    adapter = make_adapter(tmp_path)
    assert "***" in adapter._redact("key=test-key")


def test_patch_without_content_is_rejected():
    with pytest.raises(ValueError):
        AgentSettings.model_validate({
            "kind": "cli", "command": ["dsh", "{task}"], "extra_args": ["--patch", "{patch}"],
        })
