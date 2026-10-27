"""启动已建好的本机RAG和Agent；不自动导入资料，不把密钥放进进程参数。"""
import argparse
import asyncio
import csv
import os
import secrets
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "services" / "document_rag"
RUNTIME = ROOT.parent / "lynx_agent_data" / "rag"
TOKEN_PATH = RUNTIME / "mcp_token.key"
SETTINGS = RUNTIME / "settings.toml"


def environment():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    if not TOKEN_PATH.exists():
        descriptor = os.open(TOKEN_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(secrets.token_urlsafe(48))
    if os.name == "nt":
        identity = subprocess.check_output(["whoami", "/user", "/fo", "csv", "/nh"], text=True)
        sid = next(csv.reader(identity.strip().splitlines()))[1]
        subprocess.run(["icacls", str(TOKEN_PATH), "/inheritance:r", "/grant:r", f"*{sid}:(F)", "*S-1-5-18:(F)"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    token = TOKEN_PATH.read_text(encoding="ascii").strip()
    if len(token) < 32:
        raise ValueError("本地MCP密钥文件无效；未覆盖，请检查文件权限和内容")
    return os.environ | {"RAG_MCP_TOKEN": token, "TOOLS_TIMEOUT_SECONDS": "120",
                         "POWERSHELL_UPDATECHECK": "Off", "PYTHONUTF8": "1"}


async def probe(search=False):
    from lynx_agent.config import load_config
    from lynx_agent.knowledge.access import KnowledgeAccess
    from lynx_agent.mcp.client import MCPConnection
    from lynx_agent.tools import ToolRegistry

    cfg = load_config()
    registry = ToolRegistry()
    connection = MCPConnection("documents", cfg.mcp.servers["documents"], registry)
    connection.knowledge = KnowledgeAccess(cfg.knowledge, cfg.memory.project_id)
    try:
        await connection.connect()
        if connection.state != "connected" or "mcp__documents__search_knowledge" not in registry.names:
            raise RuntimeError("文档MCP连接或工具发现失败，请检查RAG日志和配置")
        print("MCP已连接：mcp__documents__search_knowledge")
        if search:
            import json
            policy = json.loads(Path(cfg.knowledge.policy_file).read_text(encoding="utf-8"))
            enabled = [kb for kb, entry in policy["grants"].items() if entry.get("enabled") is True]
            if len(enabled) != 1:
                raise RuntimeError("探测搜索要求恰好一个已启用知识库；请在Agent中明确指定kb_id查询")
            text = await registry.get("mcp__documents__search_knowledge").execute(query="会话采用什么格式保存，关闭后如何恢复？", kb_id=enabled[0])
            data = json.loads(text)
            print(json.dumps({"status": data["status"], "index_version": data["index_version"], "hits": data["hits"]}, ensure_ascii=False, indent=2))
    finally:
        await connection.close()


def start_service(env):
    subprocess.run(["docker", "start", "lynx-rag-validation-etcd-1", "lynx-rag-validation-minio-1", "lynx-rag-validation-milvus-1"],
                   check=True, stdout=subprocess.DEVNULL)
    with socket.socket() as sock:
        running = sock.connect_ex(("127.0.0.1", 8766)) == 0
    if running:
        asyncio.run(probe())  # 不能把其他进程占用端口当成本服务已就绪。
        return
    executable = SERVICE / ".venv" / "Scripts" / "python.exe"
    if not executable.is_file() or not SETTINGS.is_file():
        raise RuntimeError("RAG环境或配置尚未准备好")
    with (RUNTIME / "service.stdout.log").open("ab") as output, (RUNTIME / "service.stderr.log").open("ab") as errors:
        process = subprocess.Popen([str(executable), "-m", "document_rag.server", "--config", str(SETTINGS)],
            cwd=SERVICE, env=env, stdout=output, stderr=errors,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    import httpx2
    deadline = time.monotonic() + 90
    with httpx2.Client(trust_env=False, timeout=1) as http:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("RAG服务启动失败；查看运行目录中的service.stderr.log，不要发送密钥")
            try:
                if http.post("http://127.0.0.1:8766/mcp", json={}).status_code == 401:
                    asyncio.run(probe())
                    print(f"RAG已启动，PID={process.pid}")
                    return
            except httpx2.HTTPError:
                pass
            time.sleep(0.5)
    raise RuntimeError("RAG启动等待超时，服务可能仍在初始化；请检查日志，不重复启动")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init-key", "start", "probe", "agent"])
    args = parser.parse_args()
    env = environment()
    os.environ.update({key: env[key] for key in ("RAG_MCP_TOKEN", "TOOLS_TIMEOUT_SECONDS", "POWERSHELL_UPDATECHECK", "PYTHONUTF8")})
    os.chdir(ROOT)
    if args.action == "init-key":
        print("本地MCP密钥已就绪，未输出密钥")
    elif args.action == "probe":
        asyncio.run(probe(search=True))
    else:
        start_service(env)
        if args.action == "agent":
            from lynx_agent.cli.commands import main as launch
            sys.argv = ["lynx-agent"]
            launch()


if __name__ == "__main__":
    main()
