"""单行命令和受当前文件权限约束的路径补全；不读取文件正文。"""

import os
import re
from pathlib import Path

from prompt_toolkit.completion import Completer, Completion

from lynx_agent.tools.errors import ToolExecutionError

COMMANDS = {
    "/training": "训练助手M1（只读监测/日志/指标/实验台账，不提交训练）",
    "/shell": "受控本机命令（非沙箱，逐次审批）",
    "/help": "帮助", "/sessions": "搜索和选择会话", "/new": "准备新会话", "/rename": "修改标题",
    "/memory": "长期记忆", "/profile": "档案", "/tools": "工具定义", "/diagnostics": "诊断",
    "/permissions": "文件权限", "/approval": "审批档位（每次/自动/暂停）", "/search": "联网搜索", "/tokens": "用量与本地估算", "/cost": "模型金额估算与峰谷时段",
    "/compact": "压缩历史（调用模型）", "/retry": "恢复未完成回复", "/clear": "清空上下文",
    "/save": "保存会话", "/workspace": "工作区", "/mcp": "连接诊断", "/speed": "显示速度",
    "/theme": "主题", "/effort": "选择推理强度", "/editor": "展开/收起草稿", "/copy": "复制最近回答或选区", "/exit": "退出",
    "/approve": "批准已预览操作", "/reject": "拒绝候选", "/approvals": "查看审批",
}
SUBCOMMANDS = {
    "/training": ("status", "on confirm", "off", "environment", "projects", "register", "inspect", "plan", "records", "record", "compare", "progress", "logs", "metrics", "files"),
    "/shell": ("status", "on", "on confirm", "off"),
    "/memory": ("list", "rules", "show", "search", "changes", "pending", "autosave on", "autosave off", "recall on", "recall off", "reconcile"),
    "/permissions": ("read", "workspace", "full"), "/search": ("on", "off"),
    "/approval": ("ask", "auto", "paused", "status", "revoke", "stop"),
    "/theme": ("ansi", "dark", "light"), "/mcp": ("tools", "connect", "disconnect"),
    "/effort": ("status", "off", "low", "high", "max"),
    "/profile": ("show soul", "show user", "show agents", "reload", "changes"),
}


class AgentCompleter(Completer):
    def __init__(self, runner):
        self.runner = runner

    def get_completions(self, document, complete_event):
        before = document.text_before_cursor
        if "\n" in document.text or "\r" in document.text:
            return
        if before.startswith("/"):
            root, _, rest = before.partition(" ")
            if " " not in before:
                for name, description in COMMANDS.items():
                    if name.startswith(root):
                        yield Completion(name, start_position=-len(root), display_meta=description)
            elif root in SUBCOMMANDS:
                for value in SUBCOMMANDS[root]:
                    if value.startswith(rest):
                        yield Completion(value, start_position=-len(rest))
            return
        match = re.search(r'(?:^|\s)(@(?:"[^"\n]*|[^\s]*))$', before)
        if match is None or self.runner.file_access is None:
            return
        token = match[1]
        raw = token[1:].lstrip('"').replace("\\", "/")
        if raw.endswith('"'):
            return
        parent, _, prefix = raw.rpartition("/")
        supplied = (parent + "/") if parent else "."
        policy = self.runner.file_access.policy
        try:
            directory = policy.resolve(supplied)
            candidates = []
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= 1000:
                        break
                    if not entry.name.casefold().startswith(prefix.casefold()):
                        continue
                    try:
                        path = policy.resolve(str(Path(entry.path)))
                    except ToolExecutionError:
                        continue
                    if policy.sensitive(path):
                        continue  # 补全不暴露凭据路径，即使曾为单次工具读取授权。
                    relative = str(path.relative_to(policy.root)).replace("\\", "/") if path.is_relative_to(policy.root) else str(path).replace("\\", "/")
                    value = relative + ("/" if entry.is_dir(follow_symlinks=False) else "")
                    candidates.append(value)
            for value in sorted(candidates, key=str.casefold)[:100]:
                # 在产生候选时再次核对，模式切换不会复用旧授权。
                policy.resolve(value.rstrip("/"))
                text = '@"' + value + ('"' if not value.endswith("/") else "") if " " in value else "@" + value
                yield Completion(text, start_position=-len(token), display=value, display_meta="路径引用；不自动附加正文")
        except (OSError, ToolExecutionError, ValueError):
            return
