"""业务工具实现：笔记库、订单、超大文档、网页注入、沙箱文件与故障注入。

这里不依赖 MCP 运行时，方便单独测试；stdio 服务器只负责把它们暴露成工具。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

from agent_bench.env_mcp.store import EnvStore, deep_merge


class ToolFault(Exception):
    """工具失败：由服务器转换成 isError 结果，并记录到调用日志。"""

    def __init__(self, message: str, kind: str = "tool_error") -> None:
        super().__init__(message)
        self.kind = kind


def _ok(**payload: Any) -> dict[str, Any]:
    return payload


class EnvTools:
    """所有工具都走同一条记录路径，保证各 agent 口径一致。"""

    def __init__(self, store: EnvStore, workspace: str | Path) -> None:
        self.store = store
        self.workspace = Path(workspace).resolve()
        self.files_root = self.workspace / "files"

    # ---- 记录与故障 ----
    async def _run(self, tool: str, arguments: dict[str, Any], body) -> dict[str, Any]:
        started = time.monotonic()
        payload: dict[str, Any] = {}
        error: str | None = None
        try:
            await self._fault(tool)
            payload = await body()
            return payload
        except ToolFault as fault:
            error = fault.kind
            raise
        finally:
            self.store.append_record({
                "seq": self.store.next_seq(),
                "tool": tool,
                "arguments": arguments,
                "result_chars": len(str(payload)),
                "elapsed_ms": round((time.monotonic() - started) * 1000, 3),
                "is_error": error is not None,
                "error_kind": error,
                "at": time.time(),
            })

    async def _fault(self, tool: str) -> None:
        fault = self.store.consume_fault(tool)
        if not fault:
            return
        if fault.get("mode") == "timeout":
            await asyncio.sleep(float(fault.get("seconds", 2)))
            raise ToolFault(fault.get("message") or "工具调用超时，未返回结果", "timeout")
        raise ToolFault(fault.get("message") or "工具暂时不可用，请稍后重试", "tool_error")

    # ---- 键值笔记库 ----
    async def notes_set(self, key: str, value: str) -> dict[str, Any]:
        arguments = {"key": key, "value": value}
        return await self._run("notes_set", arguments, lambda: self._notes_set(key, value))

    async def _notes_set(self, key: str, value: str) -> dict[str, Any]:
        if not key.strip():
            raise ToolFault("key 不能为空", "invalid_argument")
        if len(value) > 4000:
            raise ToolFault("单条笔记超过 4000 字符，未保存", "invalid_argument")

        def operation(state, control):
            state.setdefault("kv", {})[key] = value
            return state["kv"]

        self.store.mutate(operation)
        return _ok(saved=True, key=key, value_chars=len(value))

    async def notes_get(self, key: str) -> dict[str, Any]:
        return await self._run("notes_get", {"key": key}, lambda: self._notes_get(key))

    async def _notes_get(self, key: str) -> dict[str, Any]:
        stored = self.store.read(lambda state, control: state.get("kv", {}).get(key))
        if stored is None:
            raise ToolFault(f"没有找到笔记 {key!r}", "not_found")
        return _ok(found=True, key=key, value=stored)

    async def notes_search(self, query: str) -> dict[str, Any]:
        return await self._run("notes_search", {"query": query}, lambda: self._notes_search(query))

    async def _notes_search(self, query: str) -> dict[str, Any]:
        def operation(state, control):
            items = state.get("kv", {})
            needle = query.strip().lower()
            matched = {key: value for key, value in items.items()
                       if not needle or needle in key.lower() or needle in str(value).lower()}
            return matched

        matched = self.store.read(operation)
        return _ok(query=query, count=len(matched), items=matched)

    # ---- 订单查询 ----
    async def orders_lookup(self, order_id: str) -> dict[str, Any]:
        arguments = {"order_id": order_id}
        return await self._run("orders_lookup", arguments, lambda: self._lookup(order_id))

    async def _lookup(self, order_id: str) -> dict[str, Any]:
        order = self.store.read(lambda state, control: state.get("orders", {}).get(order_id))
        if not isinstance(order, dict):
            raise ToolFault(f"订单 {order_id} 不存在", "not_found")
        return _ok(order_id=order_id, **order)

    # ---- 超大文档 ----
    async def documents_read(self, document_id: str, offset: int = 0, limit: int = 12000) -> dict[str, Any]:
        arguments = {"document_id": document_id, "offset": offset, "limit": limit}
        return await self._run("documents_read", arguments,
                               lambda: self._read_document(document_id, offset, limit))

    async def _read_document(self, document_id: str, offset: int, limit: int) -> dict[str, Any]:
        text = self.store.read(lambda state, control: state.get("documents", {}).get(document_id))
        if not isinstance(text, str):
            raise ToolFault(f"文档 {document_id} 不存在", "not_found")
        start = max(0, int(offset))
        size = max(200, min(int(limit), 120_000))
        stop = min(len(text), start + size)
        return _ok(document_id=document_id, offset=start, total_chars=len(text),
                   next_offset=stop if stop < len(text) else None,
                   content=text[start:stop])

    # ---- 网页（可能含注入指令）----
    async def web_page(self, url: str) -> dict[str, Any]:
        return await self._run("web_page", {"url": url}, lambda: self._page(url))

    async def _page(self, url: str) -> dict[str, Any]:
        page = self.store.read(lambda state, control: state.get("pages", {}).get(url))
        if not isinstance(page, dict):
            raise ToolFault(f"无法访问 {url}（404）", "not_found")
        return _ok(url=url, title=page.get("title", ""), content=page.get("content", ""),
                   updated_at=page.get("updated_at", ""))

    # ---- 沙箱文件系统 ----
    async def sandbox_write_file(self, path: str, content: str) -> dict[str, Any]:
        arguments = {"path": path, "content": content}
        return await self._run("sandbox_write_file", arguments, lambda: self._write(path, content))

    async def _write(self, path: str, content: str) -> dict[str, Any]:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _ok(path=self._relative(target), chars=len(content))

    async def sandbox_read_file(self, path: str) -> dict[str, Any]:
        return await self._run("sandbox_read_file", {"path": path}, lambda: self._read(path))

    async def _read(self, path: str) -> dict[str, Any]:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolFault(f"文件 {path} 不存在", "not_found")
        text = target.read_text(encoding="utf-8", errors="replace")
        return _ok(path=self._relative(target), chars=len(text), content=text)

    async def sandbox_list_files(self, prefix: str = "") -> dict[str, Any]:
        return await self._run("sandbox_list_files", {"prefix": prefix}, lambda: self._list(prefix))

    async def _list(self, prefix: str) -> dict[str, Any]:
        root = self.files_root
        if not root.is_dir():
            return _ok(prefix=prefix, paths=[])
        paths = sorted(self._relative(path) for path in root.rglob("*") if path.is_file())
        return _ok(prefix=prefix, paths=[item for item in paths if item.startswith(prefix)])

    # ---- 故障注入工具 ----
    async def flaky_task(self, name: str = "sync") -> dict[str, Any]:
        # 故意先消耗故障配额，再正常返回；故障规则由用例通过 env 通道设置。
        return await self._run("flaky_task", {"name": name}, lambda: self._flaky(name))

    async def _flaky(self, name: str) -> dict[str, Any]:
        return _ok(task=name, status="done")

    # ---- 路径与状态 ----
    def _resolve(self, relative: str) -> Path:
        root = self.files_root.resolve()
        candidate = (root / str(relative)).resolve()
        try:
            common = Path(os.path.commonpath([str(root), str(candidate)]))
        except ValueError:
            common = root.parent
        if common != root:
            raise ToolFault("路径越界，未执行", "invalid_argument")
        return candidate

    def _relative(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.files_root.resolve()).as_posix()
        except ValueError:
            return path.name

    def state_view(self) -> dict[str, Any]:
        """给 expect_state 用的可核对视图：笔记、计数与沙箱文件正文。"""
        state = self.store.snapshot()
        files: dict[str, str] = {}
        root = self.files_root
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    files[self._relative(path)] = path.read_text(encoding="utf-8", errors="replace")[:20000]
                except OSError:
                    continue
        return {"kv": state.get("kv", {}), "counter": state.get("counter", {}), "files": files}

    def apply_env(self, payload: dict[str, Any]) -> None:
        """用例的 env 步骤：深合并业务状态，可同时设置故障注入。"""
        state = {key: value for key, value in payload.items()
                 if key in {"kv", "orders", "files", "documents", "pages", "counter"}}
        control = {}
        if "faults" in payload:
            control["faults"] = payload["faults"]
        if state:
            def operation(current, _control):
                merged = deep_merge(current, state)
                current.clear()
                current.update(merged)

            self.store.mutate(operation)
        if control:
            self.store.set_control(control)
