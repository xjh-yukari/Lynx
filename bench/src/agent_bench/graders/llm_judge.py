"""LLM 评分：默认关闭；开启时必须使用与被测模型不同的模型。

只在没有确定性判据时补充使用，并在报告中明确标注哪些结论来自模型评分。
"""

from __future__ import annotations

import json
import re

from aiohttp import ClientError, ClientSession, ClientTimeout

PROMPT = (
    "你是评测裁判。只根据问题与候选回答判断是否满足要求，不要评价文风。"
    "严格输出一行 JSON：{\"pass\": true|false, \"reason\": \"不超过40字\"}。"
)

PASS = re.compile(r'"pass"\s*:\s*(true|false)', re.IGNORECASE)


class LlmJudge:
    def __init__(self, *, base_url: str, api_key: str, model: str, tested_model: str,
                 max_tokens: int = 512, timeout: float = 60) -> None:
        if model.strip() == tested_model.strip():
            raise ValueError("评分模型不能与被测模型相同")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def evaluate(self, *, rubric: str, question: str, answer: str) -> tuple[bool, str]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": PROMPT},
                {"role": "user", "content": json.dumps(
                    {"rubric": rubric, "question": question, "answer": answer},
                    ensure_ascii=False)},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        try:
            async with ClientSession(timeout=ClientTimeout(total=self.timeout)) as session, \
                    session.post(f"{self.base_url}/chat/completions",
                                 json=payload, headers=headers) as response:
                if response.status >= 400:
                    return False, f"评分请求失败：HTTP {response.status}"
                data = await response.json()
        except (ClientError, TimeoutError, ValueError) as error:
            return False, f"评分请求失败：{type(error).__name__}"
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return False, "评分响应缺少内容"
        matched = PASS.search(content or "")
        if matched is None:
            return False, "评分结果无法解析"
        reason = " ".join((content or "").split())[:120]
        return matched.group(1).lower() == "true", reason


class DisabledJudge:
    """占位：显式表达“没有启用模型评分”。"""

    enabled = False
