"""配置解析、命令行退出码与报告渲染（假数据，不联网）。"""

import json
import sys
from pathlib import Path

import pytest
from agent_bench.cli import main
from agent_bench.config import (
    BenchConfig,
    ConfigError,
    config_fingerprint,
    load_bench_config,
    model_parameters,
    read_env_file,
    resolve_api_key,
)

VALID = """
[budget]
max_cost_cny = 5.0

[model]
name = "deepseek-flash"

[agents.fake]
kind = "cli"
command = ["python", "-c", "pass"]
"""


def write_config(tmp_path, text=VALID):
    path = tmp_path / "bench.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_config_and_defaults(tmp_path):
    config = load_bench_config(write_config(tmp_path))
    assert config.budget.max_cost_cny == 5.0
    assert config.model.name == "deepseek-flash"
    assert config.run.align_context_window == 131072
    assert config.agents["fake"].default_capabilities() == {"multi_turn"}


def test_budget_is_required(tmp_path):
    path = write_config(tmp_path, "[model]\nname = 'x'\n[agents.fake]\nkind='cli'\ncommand=['x']\n")
    with pytest.raises(ConfigError) as error:
        load_bench_config(path)
    assert "budget" in str(error.value)


def test_unknown_field_is_rejected(tmp_path):
    path = write_config(tmp_path, VALID + "\nnonsense = 1\n")
    with pytest.raises(ConfigError):
        load_bench_config(path)


def test_missing_cli_command_is_rejected(tmp_path):
    path = write_config(tmp_path, "[budget]\nmax_cost_cny = 1\n[agents.fake]\nkind = 'cli'\n")
    with pytest.raises(ConfigError):
        load_bench_config(path)


def test_fingerprint_is_stable_and_secret_free(tmp_path):
    config = load_bench_config(write_config(tmp_path))
    assert config_fingerprint(config) == config_fingerprint(config)
    assert "sk-" not in config_fingerprint(config)
    assert model_parameters(config)["name"] == "deepseek-flash"


def test_env_file_and_api_key_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=from-file\nother=1\n", encoding="utf-8")
    config = load_bench_config(write_config(tmp_path))
    assert read_env_file(tmp_path / ".env")["DEEPSEEK_API_KEY"] == "from-file"
    assert resolve_api_key(config, root=tmp_path) == "from-file"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    assert resolve_api_key(config, root=tmp_path) == "from-env"


def test_missing_api_key_reports_name_not_value(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = load_bench_config(write_config(tmp_path))
    with pytest.raises(ConfigError) as error:
        resolve_api_key(config, root=tmp_path)
    assert "DEEPSEEK_API_KEY" in str(error.value)


def test_cli_list_and_validate(monkeypatch, tmp_path, capsys):
    path = write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bench", "--config", str(path), "validate"])
    assert main() == 0
    monkeypatch.setattr(sys, "argv", ["bench", "--config", str(path), "list"])
    assert main() == 0


def test_cli_estimate_does_not_call_api(monkeypatch, tmp_path, capsys):
    path = write_config(tmp_path)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")
    monkeypatch.setattr(sys, "argv", ["bench", "--config", str(path), "run",
                                      "--agents", "fake", "--estimate"])
    assert main() == 0
    output = capsys.readouterr().out
    assert "预计费用" in output


def test_cli_trace_reports_missing_run(monkeypatch, tmp_path):
    path = write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bench", "--config", str(path), "trace", "nope", "case"])
    assert main() == 2


def test_cli_usage_error(monkeypatch, tmp_path):
    path = write_config(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bench", "--config", str(path)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_html_report_is_self_contained(tmp_path):
    from agent_bench.models import TurnTrace
    from agent_bench.report import render_html
    from agent_bench.scoring import AgentRunData, build_report

    data = AgentRunData(
        name="fake",
        samples=[],
        traces=[TurnTrace(run_id="r", agent="fake", case_id="context.x", suite="context",
                          track="shared", turn=1, user="问题", answer="回答",
                          metrics={"context.retention": 0.0},
                          tags=("position:10", "window:32768"))],
    )
    report = build_report("run-1", [data],
                          settings=BenchConfig.model_validate(
                              {"budget": {"max_cost_cny": 1},
                               "agents": {"fake": {"kind": "cli", "command": ["x"]}}}).scoring,
                          price=BenchConfig.model_validate(
                              {"budget": {"max_cost_cny": 1},
                               "agents": {"fake": {"kind": "cli", "command": ["x"]}}}).price)
    output = render_html(report=report, traces=data.traces,
                         run_meta={"model": {"name": "deepseek-flash"},
                                   "context_alignment": {"align_context_window": 32768},
                                   "seed": 1},
                         output=tmp_path / "report.html")
    html = output.read_text(encoding="utf-8")
    assert "<svg" in html
    assert "http://" not in html.split("<script")[0].replace("http://127.0.0.1", "")
    assert "位置与深度召回热力图" in html
    assert "失败用例" in html
    # 数据内联：离线打开仍能看到原始 JSON。
    assert '"run_id": "run-1"' in html or '"run_id":"run-1"' in html
    assert not list(Path(tmp_path).glob("*.js"))


def test_scores_json_roundtrip(tmp_path):
    from agent_bench.scoring import ScoreReport

    report = ScoreReport(run_id="r")
    path = tmp_path / "scores.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    loaded = ScoreReport.model_validate_json(path.read_text(encoding="utf-8"))
    assert loaded.run_id == "r"
    assert json.loads(path.read_text(encoding="utf-8"))["agents"] == {}
