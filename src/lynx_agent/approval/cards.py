"""审批卡片文本：纯函数，便于在没有 TUI 的情况下测试。"""

from lynx_agent.approval.models import BatchPlan, GateRequest, PendingApproval


def human_size(value):
    size = max(0, int(value))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size} B"


def _yes(value):
    return "是" if value else "否"


def page_count(plan: BatchPlan, page_size: int) -> int:
    return max(1, (len(plan.files) + page_size - 1) // page_size)


def _file_lines(plan: BatchPlan, page: int, page_size: int) -> list[str]:
    total = page_count(plan, page_size)
    start = min(page, total - 1) * page_size
    selected = plan.files[start:start + page_size]
    lines = [f"文件清单（共 {len(plan.files)} 个，第 {min(page, total - 1) + 1}/{total} 页）："]
    for number, item in enumerate(selected, start + 1):
        lines.append(f"  {number}. {item.path}（{human_size(item.size)}）")
    if total > 1:
        lines.append("  翻页：PgUp/PgDn（文件较多时只分页展示，执行仍按完整清单复核）")
    return lines


def batch_text(pending: PendingApproval, page_size: int = 10) -> str:
    plan = pending.payload
    assert isinstance(plan, BatchPlan)
    lines = [
        "批量任务授权（应用层审批，不是操作系统沙箱）",
        f"任务说明（模型填写，仅供说明）：{plan.purpose or '未填写'}",
        f"操作类型：{plan.operation}（工具 {plan.tool}）",
        "读取目录：",
        *[f"  - {path}" for path in plan.read_roots],
        "写入目录：",
        *[f"  - {path}" for path in plan.write_roots],
        *_file_lines(plan, pending.page, page_size),
        (f"预计：{len(plan.files)} 个文件，共 {human_size(plan.total_bytes)}；"
         f"执行上限：{plan.limits.max_files} 个文件 / {human_size(plan.limits.max_bytes)} / "
         f"{plan.limits.max_operations} 次操作"),
        (f"覆盖原文件：{_yes(plan.allow_overwrite)}；移动/改名：{_yes(plan.allow_move)}；"
         f"永久删除：{_yes(plan.allow_delete)}"),
        (f"下载来源：{', '.join(plan.network_sources) if plan.network_sources else '无'}；"
         f"上传到外部：{_yes(plan.upload)}"),
        f"备份位置：{plan.backup_dir or '无（本任务不覆盖原文件）'}",
        *[f"说明：{label}：{value}" for label, value in plan.details],
        ("授权有效范围：当前会话、当前任务回合、当前文件权限和当前审批档位；"
         "任务完成、撤销、切换会话、重启、权限或审批档位变化后失效。"),
    ]
    if pending.supports_batch:
        lines.append("允许一次：Y · 允许本批次：B · 拒绝：N · 翻页：PgUp/PgDn")
    else:
        lines.append("允许一次：Y · 拒绝：N · 翻页：PgUp/PgDn（当前档位不提供批量授权）")
    return "\n".join(lines)


def gate_text(pending: PendingApproval, page_size: int = 10) -> str:
    request = pending.payload
    assert isinstance(request, GateRequest)
    lines = [
        "操作审批（应用层审批，不是操作系统沙箱）",
        f"动作：{request.action}；工具：{request.tool}",
        "目标：",
        *[f"  - {path}" for path in request.targets[:page_size]],
    ]
    if len(request.targets) > page_size:
        lines.append(f"  ……其余 {len(request.targets) - page_size} 个目标未在本页显示")
    if request.detail:
        lines.append(f"说明：{request.detail}")
    lines.append("允许一次：Y · 拒绝：N")
    return "\n".join(lines)


def pending_text(pending: PendingApproval, page_size: int = 10) -> str:
    return batch_text(pending, page_size) if pending.kind == "batch" else gate_text(pending, page_size)


def approval_hint(pending: PendingApproval) -> str:
    base = "Y 允许一次 · N 拒绝 · Esc 拒绝并取消本轮"
    return ("B 允许本批次 · " + base) if pending.supports_batch else base
