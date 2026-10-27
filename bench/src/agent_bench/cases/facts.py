"""按种子编造虚构事实：模型预训练里不存在的代号、端口、人名与日期。"""

from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta

from agent_bench.cases.models import FactSpec

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
DIGITS = "23456789"
SURNAMES = ("林", "沈", "顾", "陆", "程", "苏", "邵", "容", "闻", "闵", "柏", "阮")
GIVEN = ("泽", "岚", "屿", "珩", "惟", "澈", "岑", "砚", "昭", "栖", "沐", "珂")
WORDS = ("Nimbus", "Quartz", "Lattice", "Harbor", "Cinder", "Vellum", "Basalt", "Meridian")
SUFFIX = ("作坊", "小组", "维护队", "交付线", "观测站")


class FactGenerator:
    """同一个 run 种子 + 用例 + 重复序号 → 同一组事实，可复现。"""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def case_seed(self, case_id: str, repeat: int) -> int:
        digest = hashlib.sha256(f"{self.seed}:{case_id}".encode()).hexdigest()[:8]
        return self.seed ^ int(digest, 16) ^ (repeat * 1_000_003)

    def generate(self, facts: dict[str, FactSpec], *, case_id: str, repeat: int = 0) -> dict[str, str]:
        rng = random.Random(self.case_seed(case_id, repeat))
        values: dict[str, str] = {}
        for name, spec in facts.items():
            blocked = {values[spec.distinct_from]} if spec.distinct_from in values else set()
            values[name] = self._one(name, spec, rng, blocked)
        return values

    def _one(self, name: str, spec: FactSpec, rng: random.Random, blocked: set[str]) -> str:
        if spec.kind == "choice":
            pool = [item for item in spec.choice if item not in blocked]
            return rng.choice(pool or list(spec.choice))
        if spec.kind == "code":
            return self._code(rng, spec.prefix)
        if spec.kind == "port":
            return str(rng.randint(10_000, 64_999))
        if spec.kind == "person":
            return rng.choice(SURNAMES) + "".join(rng.sample(GIVEN, rng.choice((1, 2))))
        if spec.kind == "date":
            start = date(2031, 1, 1)
            return (start + timedelta(days=rng.randint(0, 5_000))).isoformat()
        if spec.kind == "integer":
            return str(rng.randint(spec.min or 1, spec.max or 100))
        token = self._token(rng)
        return spec.prefix + token

    @staticmethod
    def _code(rng: random.Random, prefix: str) -> str:
        head = rng.choice(LETTERS) + rng.choice(DIGITS)
        tail = "".join(rng.choice(LETTERS) for _ in range(2)) + rng.choice(DIGITS)
        return f"{prefix}{head}-{tail}"

    @staticmethod
    def _token(rng: random.Random) -> str:
        core = "".join(rng.choice(LETTERS + DIGITS) for _ in range(rng.choice((5, 6))))
        return f"{core[:3]}-{core[3:]}"

    @staticmethod
    def fictional_name(rng: random.Random) -> str:
        return rng.choice(WORDS) + rng.choice(SUFFIX)
