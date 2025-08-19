import os, json, requests, asyncio
from typing import List, Optional
from llama_index.core.embeddings import BaseEmbedding

def _safe_truncate(text: str, max_tokens: int = 350) -> str:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text
        return enc.decode(ids[:max_tokens])
    except Exception:
        return text[:2000]

class SiliconFlowEmbedding(BaseEmbedding):
    model: str
    timeout: int = 60
    base: Optional[str] = None
    key: Optional[str] = None
    max_tokens_per_input: int = 350

    def model_post_init(self, __context) -> None:
        if self.base is None:
            self.base = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        if self.key is None:
            self.key = os.getenv("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def _embed(self, inputs: List[str]) -> List[List[float]]:
        url = f"{self.base}/embeddings"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        safe_inputs = [_safe_truncate(x, self.max_tokens_per_input) for x in inputs]
        payload = {"model": self.model, "input": safe_inputs}
        for i in range(3):
            try:
                r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
                r.raise_for_status()
                data = r.json()
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                return [it["embedding"] for it in items]
            except requests.RequestException:
                if i == 2:
                    raise

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed([query])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._embed(texts)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_query_embedding, query)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_text_embeddings, texts)