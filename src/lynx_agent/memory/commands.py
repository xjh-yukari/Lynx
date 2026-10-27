"""只供 CLI/TUI 调用的人工确认入口，不注册为模型工具。"""

from lynx_agent.memory.errors import MemoryError

DEFAULT_SOUL = "我是 Lynx，一个协助用户处理任务的 Agent。\n使用清晰、直接的表达；区分事实与推测，不宣称未完成的操作已经完成。"


async def handle_memory_command(service, verb, argument):
    if not service.enabled:
        return "长期记忆未启用；请先配置 [memory]。"
    action, _, rest = argument.strip().partition(" ")
    if verb == "/profile":
        if not action:
            return (
                service.status()
                + "\n档案："
                + ", ".join(
                    f"{name} v{record.revision}"
                    for name, record in service.profiles.items()
                )
            )
        if action == "show" and rest in {"soul", "user", "agents"}:
            record = service.profiles.get(rest)
            return record.body if record else "该档案尚未创建；不会根据旧聊天自动推断。"
        if action == "reload":
            await service.refresh()
            return service.status()
        if action == "init" and not rest:
            identifier = await service.propose("soul", DEFAULT_SOUL)
            return await service.preview(identifier)
        if action == "set":
            kind, _, body = rest.partition(" ")
            if kind not in {"soul", "user", "agents"} or not body.strip():
                raise MemoryError(
                    "用法：/profile set soul|user|agents 完整的新内容（将展示差异）"
                )
            identifier = await service.propose(kind, body)
            return await service.preview(identifier)
        if action == "changes":
            return await changes(service, profiles=True)
        raise MemoryError(
            "用法：/profile [show soul|user|agents | init | set 类别 内容 | reload | changes]"
        )
    if not action:
        return service.status()
    if action == "rules" and not rest:
        return (
            "当前保存规则：\n"
            f"自动任务结论：{'开启' if service.auto_save_enabled else '关闭'}，完整答复及会话保存成功后才执行，不逐条确认。\n"
            "筛选：有复用价值的命名、明确决策、完成结果和解决办法；最多2条，每条600字符，须有本轮答复原文依据。\n"
            "不保存整轮聊天、工具/网页正文、瞬时行情、猜测或未完成承诺；明显敏感内容及‘不要保存’请求跳过。\n"
            "SOUL/USER/AGENTS 不自动改写；人工候选、覆盖和删除仍需预览确认。\n"
            "只追加当前项目，精确去重，不复活已删除的同正文记录；只读或服务不可写时跳过。\n"
            "过滤为启发式，并非完整隐私检测或事实核验；每次筛选可能额外调用模型。\n"
            "查看：/memory list；/memory show 完整编号；/memory autosave on|off 切换。"
        )
    if action == "list":
        if rest and not rest.isdigit():
            raise MemoryError("用法：/memory list [offset]，每页10条")
        page = await service.list_records(offset=int(rest or 0), include_body=True)
        lines = [f"当前有效记录：{page['total']} 条（按存储库和记录编号分页，并非完整变更日志）"]
        for item in page["records"]:
            if not item["available"]:
                lines.append(f"{item['key']} · {item['error']}")
                continue
            origin = "自动任务结论" if item["origin"] == "auto_task_conclusion" else "人工确认"
            lines.append(f"{item['key']} · {item['title']} · {origin}\n"
                         f"来源会话：{item['source_session']}；时间：{item['updated_at']}\n{item['body']}")
        if page["next_offset"] is not None:
            lines.append(f"下一页：/memory list {page['next_offset']}")
        return "\n\n".join(lines)
    if action == "search" and rest:
        found = await service.search(rest)
        return (
            "\n".join(
                f"{key} · {record.title}\n{record.body[:300]}" for key, record in found
            )
            or "没有检索到相关的已管理记忆；关键词检索不保证召回。"
        )
    if action == "show":
        record = await service.get(rest)
        origin = "自动任务结论（未经人工确认）" if record.topic.startswith("auto/task/") else "人工确认"
        sources = "\n文档来源（历史版本）：" + "; ".join(f"{item.kb_id}@{item.index_version} {item.source} 页{item.page or '未知'} #{item.chunk_id}" for item in record.sources) if record.sources else ""
        return f"{rest} · {record.kind} · v{record.revision} · {origin}\n来源会话：{record.source_session}；保存时间：{record.updated_at}\n{record.body}{sources}"
    if action == "remember" and rest:
        identifier = await service.propose("fact", rest)
        return await service.preview(identifier)
    if action == "forget":
        identifier = await service.propose_delete(rest)
        return await service.preview(identifier)
    if action == "preview":
        return await service.preview(rest)
    if action == "approve":
        return await service.approve(rest)
    if action == "reject":
        service.pending.pop(rest, None)
        return "已拒绝候选，没有修改长期记忆。"
    if action == "pending":
        return (
            "\n".join(
                f"{key} · {item.action} · {item.record.kind}"
                for key, item in service.pending.items()
            )
            or "没有待确认记忆。"
        )
    if action == "recall" and rest in {"on", "off"}:
        service.recall_enabled = rest == "on"
        service.recalled = ""
        return "自动召回已" + (
            "开启。" if service.recall_enabled else "关闭；固定档案仍加载。"
        )
    if action == "autosave" and rest in {"on", "off"}:
        service.auto_save_enabled = rest == "on"
        return "任务结论自动保留已" + ("开启" if service.auto_save_enabled else "关闭") + "（仅本次运行；持久设置修改 [memory].auto_save）。"
    if action == "changes":
        return await changes(service)
    if action == "history":
        await service.get(rest)
        store, loc = service._lookup(rest)
        _, versions = await service.backends[store].get(loc.id, history=True)
        return (
            "\n\n".join(
                f"v{item.revision} · {item.updated_at}\n{item.body}"
                for item in versions
            )
            or "没有可恢复的历史正文。"
        )
    if action == "undo":
        key, _, version = rest.partition(" ")
        if not version.isdigit():
            raise MemoryError("用法：/memory undo 记录编号 旧修订号")
        identifier = await service.undo(key, int(version))
        return await service.preview(identifier)
    if action == "reconcile":
        return await service.reconcile()
    raise MemoryError(
        "用法：/memory [list [offset]|rules|search|show|remember|forget|preview|approve|reject|pending|recall|autosave on|off|changes|undo|reconcile]"
    )


async def changes(service, *, profiles=False):
    lines = []
    for index in service.indexes.values():
        for key, loc in index.items():
            if profiles and loc.kind not in {"soul", "user", "agents"}:
                continue
            lines.append(
                f"{key} · {loc.kind} · v{loc.revision} · {'已删除' if loc.deleted else '有效'} · {'自动任务结论' if loc.topic.startswith('auto/task/') else '人工确认'}"
            )
    return "\n".join(lines) or "暂无受管理的记忆变更。"
