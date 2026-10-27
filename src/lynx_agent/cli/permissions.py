"""只能由终端用户调用的权限和审批指令，不注册为模型工具。"""

from lynx_agent.approval.models import MODES as APPROVAL_MODES
from lynx_agent.approval.models import SHORT_LABELS
from lynx_agent.filesystem.policy import MODES

PERMISSION_OPTIONS = tuple(MODES.items())
APPROVAL_OPTIONS = tuple(APPROVAL_MODES.items())
APPROVAL_SHORT = {label: SHORT_LABELS[key] for key, label in APPROVAL_OPTIONS}


def handle_permissions_command(agent, argument=""):
    action = argument.strip().lower()
    if agent.file_access is None:
        raise ValueError("当前 Runner 未装配本地文件工具")
    if action == "full":
        return ("完全权限将允许读写工作区外、当前系统用户可访问的本地文件。敏感文件仍需单独授权，覆盖仍需预览审批。\n"
                "会话存储目录仍为内部保护范围，不能通过通用文件工具读取或修改。\n"
                "完全权限只改变文件访问范围，不取消命令审批或批量审批。\n"
                "若确认，请输入 /permissions full confirm；当前权限尚未改变。")
    if action:
        if action not in {"read", "workspace", "full confirm"}:
            raise ValueError("用法：/permissions [read|workspace|full]；完全权限需 full confirm")
        agent.set_file_mode("full" if action == "full confirm" else action)
    policy = agent.file_access.policy
    return (f"文件权限：{policy.label}\n固定工作区：{policy.root}\n"
            "仅查看和工作区修改均不允许读取工作区外；完全权限不提升系统账户权限。\n"
            "当前会话存储目录始终禁止通用文件工具直接访问。\n"
            "读取内容可能发往模型 API。权限切换会清除敏感授权、待审批操作和批量授权；重启默认工作区修改。")


def handle_approval_mode_command(agent, argument=""):
    """独立的审批档位：不改变文件权限，也不提供永久放行任意程序。"""
    if agent.approval is None:
        raise ValueError("当前 Runner 未装配审批策略")
    action = argument.strip().lower()
    if action in {"ask", "auto", "paused"}:
        agent.set_approval_mode(action)
    elif action == "revoke":
        agent.revoke_approval()
    elif action == "stop":
        return agent.request_stop()
    elif action not in {"", "status"}:
        raise ValueError("用法：/approval [ask|auto|paused|revoke|stop]")
    manager = agent.approval
    return (f"审批档位：{manager.label}\n" + manager.status_text() +
            f"\n待审批批量任务：{'1' if manager.pending else '0'}")


def handle_approval_command(agent, verb, argument=""):
    parts = argument.split()
    identifier = parts[0] if parts else ""
    scope = parts[1].lower() if len(parts) > 1 else "once"
    manager = getattr(agent, "approval", None)
    if verb == "/approvals" and not identifier:
        text = agent.file_access.approvals() if agent.file_access is not None else "当前 Runner 未装配本地文件工具。"
        if manager is not None:
            text += "\n\n" + manager.approvals()
        return text
    if not identifier:
        raise ValueError(f"请指定审批编号，例如 {verb} <编号>")
    if manager is not None and manager.pending is not None and manager.pending.identifier == identifier:
        if verb == "/approve":
            decision = "batch" if scope == "batch" else "once"
            if not manager.decide(identifier, decision):
                return "该审批不允许本批次授权；未批准。请只用 /approve <编号> 允许一次，或按拒绝处理。"
            return ("已允许本批次；授权仅限当前任务与当前会话，切换、重启或权限变化后失效。"
                    if decision == "batch" else "已允许一次；操作将在重新检查路径后继续。")
        if verb == "/reject":
            manager.decide(identifier, "deny")
            return "已拒绝该操作，未修改文件、未下载、未执行命令。本回合不能换工具重试同一操作。"
        raise ValueError("用法：/approvals、/approve <编号> [batch]、/reject <编号>")
    if agent.file_access is None:
        raise ValueError("当前 Runner 未装配本地文件工具")
    if agent._operation_running:
        raise ValueError("请先等待或取消当前请求，再处理审批")
    if verb == "/approve":
        return agent.approve_file(identifier)
    if verb == "/reject":
        return agent.file_access.reject(identifier)
    raise ValueError("用法：/approvals、/approve <编号>、/reject <编号或 all>")
