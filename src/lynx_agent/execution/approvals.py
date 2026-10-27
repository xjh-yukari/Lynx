"""通用回合内审批；只有宿主界面可回填，模型不持有回填接口。"""

import asyncio


class ApprovalBroker:
    def __init__(self):
        self.presenter = None
        self.pending = None
        self._future = None
        self.generation = 0

    def invalidate(self):
        self.generation += 1
        if self._future is not None and not self._future.done():
            self._future.set_result(False)

    def decide(self, identifier, approved):
        if self.pending is None or self.pending.identifier != identifier or self._future.done():
            return False
        self._future.set_result(approved is True)
        return True

    async def request(self, item, timeout):
        if self.presenter is None or self.pending is not None:
            return False
        self.pending = item
        self._future = asyncio.get_running_loop().create_future()
        try:
            self.presenter(item)
            async with asyncio.timeout(timeout):
                return await self._future
        except TimeoutError:
            return False
        finally:
            self.pending = None
            self._future = None
            self.presenter(None)
