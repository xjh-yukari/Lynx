import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

from document_rag.config import grant


class Retriever:
    def __init__(self, settings, encoder, store):
        self.settings, self.encoder, self.store = settings, encoder, store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-inference")
        self.slots = 0

    def search(self, query, kb_id="default", top_k=5):
        if not isinstance(query, str) or not query.strip() or len(query) > 2000 or not 1 <= top_k <= 10:
            raise ValueError("invalid_argument")
        grant(self.settings, kb_id)
        dense, sparse = self.encoder.encode([query], query=True)[0]
        data = self.store.search(kb_id, dense, sparse, self.settings.mode, top_k)
        data["query"] = query
        # 总模型token预算和Agent MCP接入字符预算同时满足；不截断半个JSON。
        while data["hits"] and (self.encoder.tokens(json.dumps(data, ensure_ascii=False)) > self.settings.output_tokens
                                or len(json.dumps(data, ensure_ascii=False)) > 15000):
            data["hits"].pop()
            data["truncated"] = True
        if self.encoder.tokens(json.dumps(data, ensure_ascii=False)) > self.settings.output_tokens:
            raise ValueError("invalid_argument: query exceeds output envelope budget")
        grant(self.settings, kb_id)
        return data

    async def call(self, query, kb_id="default", top_k=5):
        if self.slots >= self.settings.queue_size:
            raise RuntimeError("unavailable: queue full")
        self.slots += 1
        future = asyncio.get_running_loop().run_in_executor(self.pool, self.search, query, kb_id, top_k)
        # shield使超时/取消不释放实际仍在推理的槽位；完成回调才释放。
        future.add_done_callback(lambda completed: self._finished(completed))
        async with asyncio.timeout(self.settings.timeout):
            return await asyncio.shield(future)

    def _finished(self, future):
        self.slots -= 1
        if not future.cancelled():
            future.exception()  # 取消调用后也消费后台异常，避免泄露日志。

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
