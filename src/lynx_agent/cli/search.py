"""普通终端和 TUI 共用的运行时搜索开关。"""


def handle_search_command(agent, argument: str = "") -> str:
    action = argument.strip().lower()
    if action not in {"", "on", "off"}:
        raise ValueError("用法：/search 查看状态；/search on 开启；/search off 关闭")
    if action:
        agent.set_search_enabled(action == "on")
    status = f"已开启（{agent.search_backend}）" if agent.search_enabled else "已关闭"
    return (f"联网搜索：{status}。\n"
            "仅控制关键词搜索 web_search；网页读取 web_fetch 和普通聊天不变。\n"
            "开关仅在本次程序运行中生效，切换会话后保持；重启按配置恢复。")
