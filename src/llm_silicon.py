# src/llm_silicon.py
import os, json, requests, asyncio
from typing import List, Iterator, AsyncIterator, Optional
from llama_index.core.llms import (
    LLM, ChatMessage, MessageRole,
    CompletionResponse, CompletionResponseGen, LLMMetadata
)

class SiliconFlowChat(LLM):
    model: str
    timeout: int = 60
    base: Optional[str] = None
    key:  Optional[str] = None

    def model_post_init(self, __context) -> None:  # pydantic v2
        # 运行时读取，而不是在类定义时
        if self.base is None:
            self.base = os.getenv("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
        if self.key is None:
            self.key = os.getenv("OPENAI_API_KEY")
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set")

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model)

    def _call(self, messages: List[dict]) -> str:
        url = f"{self.base}/chat/completions"
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "messages": messages}
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout)
        if r.status_code != 200:
            # 打印服务端原因，便于区分“模型权限不足”与其它问题
            raise requests.HTTPError(f"{r.status_code} {url}: {r.text[:400]}")
        return r.json()["choices"][0]["message"]["content"]

    # ===== complete =====
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        text = self._call([{"role": "user", "content": prompt}])
        return CompletionResponse(text=text)

    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._call, [{"role": "user", "content": prompt}])
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        resp = self.complete(prompt, **kwargs)
        def gen() -> Iterator[CompletionResponse]:
            yield resp
        return CompletionResponseGen(gen())

    async def astream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        resp = await self.acomplete(prompt, **kwargs)
        async def agen() -> AsyncIterator[CompletionResponse]:
            yield resp
        return CompletionResponseGen(agen())

    # ===== chat =====
    def chat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponse:
        msgs = [{"role": m.role.value, "content": m.content} for m in messages]
        text = self._call(msgs)
        return CompletionResponse(text=text)

    async def achat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponse:
        msgs = [{"role": m.role.value, "content": m.content} for m in messages]
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, self._call, msgs)
        return CompletionResponse(text=text)

    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponseGen:
        resp = self.chat(messages, **kwargs)
        def gen() -> Iterator[CompletionResponse]:
            yield resp
        return CompletionResponseGen(gen())

    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> CompletionResponseGen:
        resp = await self.achat(messages, **kwargs)
        async def agen() -> AsyncIterator[CompletionResponse]:
            yield resp
        return CompletionResponseGen(agen())