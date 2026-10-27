import importlib.util
from pathlib import Path


def test_launcher_keeps_key_local_and_reuses_it(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "scripts" / "rag_runtime.py"
    spec = importlib.util.spec_from_file_location("rag_runtime_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "RUNTIME", tmp_path)
    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "mcp_token.key")
    first = module.environment()
    second = module.environment()
    assert first["RAG_MCP_TOKEN"] == second["RAG_MCP_TOKEN"]
    assert len(first["RAG_MCP_TOKEN"]) >= 32
    assert first["POWERSHELL_UPDATECHECK"] == "Off"
    assert first["TOOLS_TIMEOUT_SECONDS"] == "120"
