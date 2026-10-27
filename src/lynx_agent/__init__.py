"""Lynx Agent 主包。"""


def main() -> None:
    """启动 Lynx Agent。"""

    from lynx_agent.cli import main as cli_main

    cli_main()
