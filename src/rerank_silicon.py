import os, json, requests
from typing import List, Tuple

class SiliconFlowReranker:
    """调用 SiliconFlow /rerank。返回按得分降序的 (index, score) 列表。"""
    def __init__(self, model: str, timeout: int = 60, base: str | None = None, key: str | None = None):
        self.base = base or os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        self.key  = key  or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.timeout = timeout
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def rerank(self, query: str, docs: List[str], top_n: int) -> List[Tuple[int, float]]:
        url = f"{self.base}/rerank"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "query": query, "documents": docs, "top_n": min(top_n, len(docs))}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if r.status_code != 200:
            raise requests.HTTPError(f"{r.status_code} {url}: {r.text[:400]}")
        data = r.json()
        # results: [{index:int, score:float}, ...] 已为降序
        return [(it["index"], float(it.get("relevance_score", it.get("score", 0.0)))) for it in data.get("results", [])]