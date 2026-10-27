"""统一生命周期、档案、召回和确认变更；普通工具不能直接写 Engram。"""

import asyncio
import os
from pathlib import Path
from time import monotonic
from uuid import uuid4

from lynx_agent.memory.cache import atomic_json, read_json
from lynx_agent.memory.changes import preview_change
from lynx_agent.memory.engram import EngramBackend
from lynx_agent.memory.errors import MemoryError, MemoryNotFound
from lynx_agent.memory.models import PROFILES, Change, Locator, MemoryRecord, digest
from lynx_agent.memory.policy import check_body, keywords, search_index
from lynx_agent.memory.profiles import edit_profile, render_profiles
from lynx_agent.memory.recall import bounded_recall


def estimate_text_tokens(text):
    # 延迟导入，避免 AgentRunner 和独立记忆服务导入形成环。
    from lynx_agent.agent.tokens import estimate_text_tokens as estimate

    return estimate(text)


class MemoryService:
    def __init__(self, config, root, workspace, mode, *, backend_factory=EngramBackend):
        self.config = config
        self.root = Path(config.data_dir or root)
        self.workspace = Path(workspace).resolve()
        self.mode = mode
        self.project_id = (
            config.project_id or "p" + digest(str(self.workspace).casefold())[:20]
        )
        identity = digest(config.agent_id + ":" + config.user_id)[:20]
        self.directories = {
            "profile": self.root / "profiles" / identity,
            "project": self.root / "projects" / self.project_id,
        }
        self.owners = {
            "profile": "lynx-profile-" + identity,
            "project": "lynx-project-" + digest(self.project_id)[:20],
        }
        self.backends = (
            {
                store: backend_factory(
                    config, directory / "engram", self.owners[store], "memory_" + store
                )
                for store, directory in self.directories.items()
            }
            if config.enabled
            else {}
        )
        self.indexes = {"profile": {}, "project": {}}
        self.profiles = {}
        self.pending = {}
        self.warnings = []
        self.recalled = ""
        self.started = False
        self.recall_enabled = config.auto_recall
        self.auto_save_enabled = config.auto_save
        self.retention_notice = ""
        self.uncertain = {}
        self._locks = []
        self._mutation = asyncio.Lock()
        self.rules_unavailable = False
        self.source_session = lambda: None
        self.source_references = lambda: ()
        self.check_references = lambda sources: None
        self.recalled_sources = ()
        # 自动批准留下的回执；工具层据此如实告诉模型是否已写入。
        self.last_bench_receipt = ""

    @property
    def bench_auto_approve(self):
        """配置显式开启且进程处于评测模式时返回真；缺任何一个条件都不生效。"""
        return self.config.auto_approve_for_bench and os.environ.get("LYNX_BENCH_MODE") == "1"

    def _validated_sources(self):
        try:
            sources = tuple(dict.fromkeys(self.source_references()))
            if len(sources) > 32:
                raise ValueError("来源数量超过预算")
            self.check_references(sources)
            return sources
        except Exception as error:
            raise MemoryError("文档来源不可验证，禁止去掉来源后保存") from error

    @property
    def enabled(self):
        return self.config.enabled

    @property
    def writable(self):
        return (
            self.enabled
            and self.started
            and all(
                item.connection.state == "connected" for item in self.backends.values()
            )
            and self.config.allow_write
            and self.mode() != "read"
            and not self.rules_unavailable
            and not self.uncertain
        )

    @property
    def label(self):
        if not self.enabled:
            return "关闭"
        connected = sum(
            item.connection.state == "connected" for item in self.backends.values()
        )
        return f"{connected}/2 · {'可写' if self.writable else '只读'}" + (
            " · 降级" if self.warnings or self.uncertain else ""
        )

    def status(self):
        return (
            f"长期记忆：{self.label}\n用户：{self.config.user_id}；项目：{self.project_id}\n自动召回：{'开启' if self.recall_enabled else '关闭'}；待确认：{len(self.pending)}\n"
            + f"任务结论自动保留：{'开启' if self.auto_save_enabled else '关闭'}\n"
            + (self.retention_notice + "\n" if self.retention_notice else "")
            + "\n".join(
                dict.fromkeys(
                    self.warnings
                    + (
                        ["存在未核实写入，使用 /memory reconcile 核实；不会自动重放"]
                        if self.uncertain
                        else []
                    )
                )
            )
        )

    def ensure_rules(self):
        if self.rules_unavailable:
            raise MemoryError(
                "当前项目规则没有可信版本，禁止写工具；请恢复记忆服务或检查档案"
            )

    def _path(self, store, name):
        return self.directories[store] / name

    def _key(self, store, identifier):
        return f"{store}:{self.owners[store]}:{identifier}"

    def _persist(self, store):
        atomic_json(
            self._path(store, "index.json"),
            {key: item.model_dump() for key, item in self.indexes[store].items()},
        )

    def _lease(self, path):
        # 一个存储只允许一个 Lynx 进程写；不会宣称支持无锁多客户端协作。
        import os

        stream = path.open("a+b")
        stream.seek(0)
        if not stream.read(1):
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise MemoryError("记忆目录正在被另一个 Lynx 实例使用") from None
        self._locks.append(stream)

    async def start(self):
        if not self.enabled or self.started:
            return
        try:
            for store, directory in self.directories.items():
                for parent in [*reversed(directory.parents), directory]:
                    if parent.exists() and (
                        parent.is_symlink()
                        or getattr(parent.lstat(), "st_file_attributes", 0) & 0x400
                    ):
                        raise MemoryError("记忆路径不能经过链接或目录联接")
                directory.mkdir(parents=True, exist_ok=True)
                self._lease(directory / ".lynx.lock")
                index = directory / "index.json"
                if not index.exists() and (directory / "engram" / "engram.db").exists():
                    raise MemoryError("已有数据库缺少定位索引，禁止静默接管")
                self.indexes[store] = {
                    key: Locator.model_validate(item)
                    for key, item in read_json(index, {}).items()
                }
                if not index.exists():
                    self._persist(store)
                (directory / "engram").mkdir(exist_ok=True)
                intent = read_json(directory / "intent.json", None)
                if intent:
                    self.uncertain[store] = intent
            self.started = True
            for backend in self.backends.values():
                try:
                    await backend.start()
                except MemoryError:
                    self.warnings.append("Engram 连接失败；仅可使用已确认档案缓存")
            await self.refresh()
        except asyncio.CancelledError:
            await self.close()
            raise
        except Exception:  # noqa: BLE001 - 不泄漏本地索引或配置内容
            self.warnings.append("记忆初始化失败，未自动修复或覆盖已有数据")
            self.rules_unavailable = True
            await self.close()

    async def close(self):
        errors = []
        try:
            for backend in self.backends.values():
                try:
                    await backend.close()
                except Exception as error:  # noqa: BLE001 - 清理其余连接后统一报告
                    errors.append(error)
        finally:
            if not errors:
                for stream in self._locks:
                    stream.close()
                self._locks.clear()
            self.started = False
        if errors:
            raise MemoryError("部分记忆连接未完成清理，请检查本地服务")

    def _lookup(self, key):
        store = key.split(":")[0]
        if store not in self.indexes or key not in self.indexes[store]:
            raise MemoryError("编号不属于当前用户或项目")
        locator = self.indexes[store][key]
        if (
            locator.deleted
            or key != self._key(store, locator.id)
            or locator.store != store
        ):
            raise MemoryError("记录已删除或定位信息不合法")
        return store, locator

    async def get(self, key):
        store, locator = self._lookup(key)
        record, _ = await self.backends[store].get(locator.id)
        if (
            record.topic != locator.topic
            or record.kind != locator.kind
            or digest(record.encode()) != locator.content_hash
        ):
            raise MemoryError("记录内容或版本与已确认索引不同，拒绝使用")
        if record.sources:
            try:
                self.check_references(record.sources)
            except Exception as error:
                raise MemoryError("关联知识库已撤权或无法核验，禁止重新读取记忆") from error
        return record

    async def refresh(self):
        if not self.enabled or not self.started:
            return
        self.profiles = {}
        self.rules_unavailable = any(
            loc.kind == "agents" and not loc.deleted
            for loc in self.indexes["project"].values()
        )
        self.warnings = []
        if any(item.connection.state != "connected" for item in self.backends.values()):
            self.warnings.append("部分记忆服务未连接；无法确认后端最新状态")
        for store, index in self.indexes.items():
            try:
                cache = read_json(self._path(store, "profiles.cache.json"), {})
            except MemoryError:
                cache = {}
                self.warnings.append("档案缓存损坏，只接受后端重新验证的内容")
            changed = False
            for key in tuple(cache):
                if key not in index or index[key].deleted:
                    cache.pop(key)
                    changed = True
            for key, locator in index.items():
                if locator.kind not in PROFILES or locator.deleted:
                    continue
                try:
                    record = await self.get(key)
                    cache[key] = record.model_dump()
                    changed = True
                except MemoryError:
                    cached = cache.get(key)
                    try:
                        record = MemoryRecord.model_validate(cached) if cached else None
                    except ValueError:
                        record = None
                    if (
                        record is None
                        or digest(record.encode()) != locator.content_hash
                    ):
                        self.warnings.append(
                            f"{locator.kind} 档案读取失败且没有可信缓存"
                        )
                        self.rules_unavailable = (
                            self.rules_unavailable or locator.kind == "agents"
                        )
                        continue
                    self.warnings.append(
                        f"{locator.kind} 使用已确认档案缓存；后端未核实最新状态"
                    )
                self.profiles[record.kind] = record
            if changed:
                atomic_json(self._path(store, "profiles.cache.json"), cache)
        self.rules_unavailable = (
            self.rules_unavailable and "agents" not in self.profiles
        )
        if estimate_text_tokens(self.profile_text) > self.config.profile_tokens:
            self.profiles = {}
            self.rules_unavailable = True
            self.warnings.append("固定档案超过预算，未静默截断；请缩短档案或提高预算")

    @property
    def profile_text(self):
        return render_profiles(self.profiles)

    async def search(self, query):
        if not self.started:
            raise MemoryError("记忆服务未就绪")
        check_body(query)
        query = keywords(query)
        if not query:
            return []
        records = []
        for store, backend in self.backends.items():
            if not any(
                not loc.deleted and loc.kind not in PROFILES
                for loc in self.indexes[store].values()
            ):
                continue
            for identifier in await backend.search(query, self.config.max_results):
                key = self._key(store, identifier)
                if (
                    key not in self.indexes[store]
                    or self.indexes[store][key].deleted
                    or self.indexes[store][key].kind in PROFILES
                ):
                    continue
                try:
                    record = await self.get(key)
                except MemoryError:
                    continue
                records.append((key, record))
        return records[: self.config.max_results]

    async def prepare(self, query):
        self.recalled = ""
        self.recalled_sources = ()
        if not self.enabled or not self.started:
            return
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                await self.refresh()
                if self.recall_enabled:
                    records = await self.search(query)
                    self.recalled = bounded_recall(
                        records,
                        self.config.recall_tokens,
                        estimate_text_tokens,
                    )
                    self.recalled_sources = tuple(dict.fromkeys(source for _, record in records for source in record.sources))
        except (MemoryError, TimeoutError, OSError, ValueError):
            self.warnings.append("记忆检索未完成；没有把失败当作无相关记忆")

    def _write_check(self, *, repair_profile=False):
        repair_allowed = (
            repair_profile
            and self.enabled
            and self.started
            and self.config.allow_write
            and self.mode() != "read"
            and not self.uncertain
            and all(
                item.connection.state == "connected" for item in self.backends.values()
            )
        )
        if not self.writable and not repair_allowed:
            raise MemoryError(
                "记忆当前不可写：检查只读模式、记忆授权、档案或未核实操作"
            )

    def _target(self, kind, title, scope):
        store = "profile" if kind in {"soul", "user"} or scope == "user" else "project"
        if kind == "agents":
            store = "project"
        topic = (
            f"profile/{kind}/{self.config.agent_id if kind == 'soul' else self.config.user_id if kind == 'user' else self.project_id}"
            if kind in PROFILES
            else f"memory/{kind}/{digest(title)[:20]}"
        )
        return store, topic

    async def propose(
        self,
        kind,
        body,
        *,
        title="",
        scope="project",
        key=None,
        source_session=None,
        edit="replace_all",
        find=None,
    ):
        self._write_check(repair_profile=kind in PROFILES)
        check_body(body)
        title = title.strip() or (
            kind.upper() if kind in PROFILES else " ".join(body.split())[:60]
        )
        check_body(title)
        store, topic = self._target(kind, title, scope)
        expected = None
        revision = 1
        if key is None:
            key = next(
                (
                    key
                    for key, loc in self.indexes[store].items()
                    if loc.topic == topic and not loc.deleted
                ),
                None,
            )
        if key is not None:
            existing_store, loc = self._lookup(key)
            if existing_store != store or loc.kind != kind:
                raise MemoryError("不能用普通记忆更新覆盖其他类别、归属或主题")
            topic = loc.topic
            previous = await self.get(key)
            if kind in PROFILES:
                body = edit_profile(previous.body, body, edit, find)
                check_body(body)
            expected, revision = loc.content_hash, previous.revision + 1
        record = MemoryRecord(
            kind=kind,
            owner=self.owners[store],
            topic=topic,
            title=title,
            body=body,
            revision=revision,
            source_session=source_session or self.source_session(),
            index_terms=search_index(body),
            sources=(tuple(dict.fromkeys((*previous.sources, *self._validated_sources()))) if key is not None
                     else self._validated_sources()) if kind not in PROFILES else (),
        )
        if kind in PROFILES:
            bodies = {k: value.body for k, value in self.profiles.items()}
            bodies[kind] = body
            if (
                estimate_text_tokens("\n".join(bodies.values())) + 150
                > self.config.profile_tokens
            ):
                raise MemoryError("档案变更超过固定档案预算，未创建确认请求")
        identifier = uuid4().hex[:12]
        self.pending = {
            key: item
            for key, item in self.pending.items()
            if monotonic() - item.created < 600
        }
        if len(self.pending) >= 20:
            raise MemoryError("待确认记忆过多，请先处理或拒绝")
        self.pending[identifier] = Change(
            id=identifier,
            action="save",
            key=key or f"{store}:new",
            record=record,
            expected_hash=expected,
            mode=self.mode(),
            created=monotonic(),
        )
        if self.bench_auto_approve:
            await self._bench_approve(identifier)
        return identifier

    async def _bench_approve(self, identifier):
        """评测模式：跳过人工预览，但仍执行完整批准、读回校验与失败保护。"""
        item = self.pending.get(identifier)
        if item is None:
            raise MemoryError("确认编号不存在")
        item.viewed = True
        result = await self.approve(identifier)
        self.last_bench_receipt = (
            f"{result}（bench 模式已自动批准：LYNX_BENCH_MODE=1 且 auto_approve_for_bench=true；"
            "仅用于隔离评测环境）"
        )

    async def propose_delete(self, key):
        self._write_check()
        record = await self.get(key)
        _, loc = self._lookup(key)
        identifier = uuid4().hex[:12]
        self.pending[identifier] = Change(
            id=identifier,
            action="delete",
            key=key,
            record=record,
            expected_hash=loc.content_hash,
            mode=self.mode(),
            created=monotonic(),
        )
        return identifier

    async def retain_completed(self, provider, tracker, user_input, answer, on_status=None):
        """仅由宿主成功提交回合后调用；失败不撤销已经完成的答复。"""
        from lynx_agent.memory.retention import eligible, extract

        self.retention_notice = ""
        if not self.auto_save_enabled or not self.enabled:
            return
        saved = 0
        try:
            if not self.writable:
                self.retention_notice = "本轮未自动保留结论：当前记忆不可写。"
                return
            if self.pending:
                return  # 不抢先保存正在等待人工批准的内容。
            if not eligible(user_input, answer):
                return
            if on_status:
                on_status("memorizing")
            async with asyncio.timeout(self.config.auto_save_timeout_seconds):
                items = await extract(provider, tracker, user_input, answer)
                for item in items:
                    # 只追加项目普通记忆；独立命名空间，不覆盖人工条目和档案。
                    topic = "auto/task/" + digest(" ".join(item.body.split()).casefold())
                    if any(loc.topic == topic for loc in self.indexes["project"].values()):
                        continue  # 已删除的条目也不自动复活。
                    identifier = await self.propose("decision", item.body, title=item.title)
                    candidate = self.pending.get(identifier)
                    if candidate is None:
                        # 评测模式已在 propose 内自动批准；不再重复操作。
                        saved += 1
                        continue
                    candidate.key = "project:new"
                    candidate.expected_hash = None
                    candidate.record = candidate.record.model_copy(update={
                        "topic": topic, "revision": 1,
                    })
                    candidate.viewed = True
                    try:
                        await self.approve(identifier)
                    finally:
                        self.pending.pop(identifier, None)
                    saved += 1
            if saved:
                self.retention_notice = f"已自动保留 {saved} 条任务结论（历史资料，非已核实事实）；/memory list 查看内容。"
            elif items:
                self.retention_notice = "任务结论已存在或曾被删除，本轮没有重复保存。"
        except asyncio.CancelledError:
            self.retention_notice = f"回答已完成；结论自动保留已中止（已验证保存 {saved} 条）；未核实写入请 /memory reconcile。"
        except Exception:  # noqa: BLE001 -- 已提交答复后的隔离边界，未知异常也不能撤销成功回合。
            # 不输出模型原文、密钥或底层响应；不重试可能已经生效的写入。
            self.retention_notice = f"回答已完成；本轮结论自动保留未完成（已验证保存 {saved} 条），不会自动重试；/memory 查看状态。"

    async def list_records(self, *, offset=0, limit=10, include_body=False):
        """只读、分页、当前身份/项目内的有效记录；逐条回读校验，不是全量变更审计。"""
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 50:
            raise MemoryError("offset 必须为非负整数，limit 必须为1至50")
        if not self.enabled or not self.started:
            raise MemoryError("记忆服务未启动")
        entries = sorted(
            ((key, loc) for index in self.indexes.values() for key, loc in index.items() if not loc.deleted),
            key=lambda pair: (pair[1].store, -pair[1].id, pair[0]),
        )
        records = []
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                for key, _ in entries[offset:offset + limit]:
                    try:
                        record = await self.get(key)
                    except MemoryError:
                        records.append({"key": key, "available": False, "error": "记录暂不可验证；未返回正文"})
                        continue
                    item = {
                        "key": key, "available": True, "title": record.title,
                        "kind": record.kind, "revision": record.revision,
                        "origin": "auto_task_conclusion" if record.topic.startswith("auto/task/") else "user_confirmed",
                        "source_session": record.source_session, "updated_at": record.updated_at,
                        "sources": [source.model_dump() for source in record.sources],
                    }
                    if include_body:
                        item["body"] = record.body
                    records.append(item)
        except TimeoutError:
            raise MemoryError("读取记录列表超时；未修改长期记忆，请稍后重试") from None
        return {"records": records, "total": len(entries), "offset": offset,
                "next_offset": offset + limit if offset + limit < len(entries) else None}

    async def preview(self, identifier):
        item = self.pending.get(identifier)
        if item is None:
            raise MemoryError("确认编号不存在")
        previous = (await self.get(item.key)).body if item.expected_hash else ""
        item.viewed = True
        return preview_change(item, previous)

    async def approve(self, identifier):
        async with self._mutation:
            item = self.pending.get(identifier)
            self._write_check(
                repair_profile=item is not None and item.record.kind in PROFILES
            )
            if (
                item is None
                or monotonic() - item.created >= 600
                or item.mode != self.mode()
            ):
                raise MemoryError("确认已过期或权限已改变，请重新提出修改")
            if not item.viewed:
                return await self.preview(identifier)
            try:
                self.check_references(item.record.sources)
            except Exception as error:
                raise MemoryError("文档来源已撤权或不可核验，禁止提交") from error
            store = item.key.split(":")[0]
            if item.expected_hash:
                old = await self.get(item.key)
                if digest(old.encode()) != item.expected_hash:
                    raise MemoryError("内容已经改变，旧确认失效")
            elif any(
                loc.topic == item.record.topic and not loc.deleted
                for loc in self.indexes[store].values()
            ):
                raise MemoryError("同主题已有其他变更生效，请重新预览")
            intent = {
                "action": item.action,
                "key": item.key,
                "operation_id": item.record.operation_id,
                "hash": digest(item.record.encode()),
                "topic": item.record.topic,
                "kind": item.record.kind,
            }
            self.uncertain[store] = intent
            atomic_json(self._path(store, "intent.json"), intent)
            try:
                if item.action == "save":
                    native_id = await self.backends[store].save(item.record)
                    actual, _ = await self.backends[store].get(native_id)
                    if actual.encode() != item.record.encode():
                        raise MemoryError("保存后的内容未完整匹配，未启用")
                    key = self._key(store, native_id)
                    self.indexes[store][key] = Locator(
                        store=store,
                        id=native_id,
                        kind=actual.kind,
                        topic=actual.topic,
                        content_hash=digest(actual.encode()),
                        revision=actual.revision,
                    )
                else:
                    _, loc = self._lookup(item.key)
                    await self.backends[store].delete(loc.id)
                    try:
                        await self.backends[store].get(loc.id)
                    except MemoryNotFound:
                        pass
                    else:
                        raise MemoryError("删除后记录仍可读取，未确认删除成功")
                    loc.deleted = True
                self._persist(store)
                atomic_json(self._path(store, "intent.json"), None)
                self.uncertain.pop(store)
                self.pending.pop(identifier)
                await self.refresh()
                return (
                    "记忆已保存并验证。"
                    if item.action == "save"
                    else "记忆已删除并验证；历史聊天和备份未删除。"
                )
            except BaseException:
                self.pending.pop(identifier, None)
                self.warnings.append("记忆变更结果尚未核实；已停止新写入，不自动重试")
                raise

    async def reconcile(self):
        for store, intent in tuple(self.uncertain.items()):
            backend = self.backends[store]
            if backend.connection.state != "connected":
                await backend.start()
            if intent["action"] == "save":
                ids = await backend.search(intent["operation_id"], 2)
                if len(ids) != 1:
                    raise MemoryError("尚不能唯一核实写入；未重新执行或解除保护")
                record, _ = await backend.get(ids[0])
                if digest(record.encode()) != intent["hash"]:
                    raise MemoryError("核实内容不匹配，保持写入保护")
                key = self._key(store, ids[0])
                self.indexes[store][key] = Locator(
                    store=store,
                    id=ids[0],
                    kind=record.kind,
                    topic=record.topic,
                    content_hash=intent["hash"],
                    revision=record.revision,
                )
            else:
                loc = self.indexes[store].get(intent["key"])
                if loc is None or intent["key"] != self._key(store, loc.id):
                    raise MemoryError("未完成删除缺少有效定位，保持保护")
                try:
                    await backend.get(loc.id)
                except MemoryNotFound:
                    loc.deleted = True
                else:
                    raise MemoryError("记录仍存在，原删除未确认；未自动重做")
            self._persist(store)
            atomic_json(self._path(store, "intent.json"), None)
            self.uncertain.pop(store)
        await self.refresh()
        return "已核实未完成操作，没有重放写入。"

    async def undo(self, key, version):
        record = await self.get(key)
        store, loc = self._lookup(key)
        _, history = await self.backends[store].get(loc.id, history=True)
        previous = next((item for item in history if item.revision == version), None)
        if (
            previous is None
            or previous.kind != record.kind
            or previous.topic != record.topic
        ):
            raise MemoryError("未找到匹配的旧版本；不会猜测恢复内容")
        return await self.propose(
            record.kind,
            previous.body,
            title=record.title,
            scope="user" if store == "profile" else "project",
            key=key,
        )

    def invalidate(self):
        self.pending.clear()
        self.recalled = ""
        self.recalled_sources = ()
