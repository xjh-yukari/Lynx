"""整轮共享的自动恢复预算，不隐藏扩大模型输出上限。"""

from time import monotonic


class RecoveryBudget:
    def __init__(self, config, used=0):
        self.config = config
        self.used = used
        self.started = None

    @property
    def remaining_seconds(self):
        return self.config.max_seconds if self.started is None else max(0, self.config.max_seconds - (monotonic() - self.started))

    def consume(self):
        if self.used >= self.config.max_extra_requests or self.remaining_seconds <= 0:
            return False
        if self.started is None:
            self.started = monotonic()
        self.used += 1
        return True
