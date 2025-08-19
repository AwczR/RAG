import json as pyjson
from typing import Tuple, List

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config_loader import load_cfg
from src.emb_silicon import SiliconFlowEmbedding
from src.llm_silicon import SiliconFlowChat
from src.rerank_silicon import SiliconFlowReranker


def _set_llm(cfg: dict):
    Settings.llm = SiliconFlowChat(
        model=cfg["llm"]["model"],
        timeout=cfg["runtime"]["timeout_sec"],
    )

def _set_embed(cfg: dict):
    Settings.embed_model = SiliconFlowEmbedding(
        model=cfg["embedding"]["model"],
        timeout=cfg["runtime"]["timeout_sec"],
    )
    Settings.embed_model.embed_batch_size = 8
    return Settings.embed_model

def _load_vector_store(cfg: dict):
    vs = cfg["vector_store"]
    client = chromadb.PersistentClient(path=vs["persist_dir"])
    collection = client.get_or_create_collection(vs["collection"])
    return ChromaVectorStore(chroma_collection=collection)

def build_or_load_index(cfg_path: str) -> Tuple[VectorStoreIndex, dict]:
    cfg = load_cfg(cfg_path)
    _set_llm(cfg)
    _set_embed(cfg)
    vs = _load_vector_store(cfg)
    storage = StorageContext.from_defaults(vector_store=vs)
    index = VectorStoreIndex.from_vector_store(vector_store=vs, storage_context=storage)
    return index, cfg

def _format_nodes(nodes, limit_chars=1200):
    lines: List[str] = []
    for i, n in enumerate(nodes, 1):
        meta = n.metadata or {}
        src = meta.get("file_path") or meta.get("source") or meta.get("filename") or ""
        score = getattr(n, "score", None)
        head = n.get_content()[:limit_chars].replace("\n", " ")
        lines.append(f"[{i}] score={score} src={src}\n{head}\n")
    return "\n".join(lines)

def ask(cfg_path: str, question: str, show_context: bool = False, as_json: bool = False) -> str:
    index, cfg = build_or_load_index(cfg_path)

    # 初次召回
    top_k = int(cfg["query"].get("top_k", 12))
    nodes = index.as_retriever(similarity_top_k=top_k).retrieve(question)

    # 可选重排
    rr_cfg = cfg.get("rerank", {}) or {}
    if rr_cfg.get("enabled") and nodes:
        model = rr_cfg.get("model", "BAAI/bge-reranker-v2-m3")
        top_n = int(rr_cfg.get("top_n", min(5, len(nodes))))
        reranker = SiliconFlowReranker(model=model, timeout=cfg["runtime"]["timeout_sec"])
        texts = [n.get_content() for n in nodes]
        ranking = reranker.rerank(question, texts, top_n=top_n)  # [(idx, score), ...] 降序
        keep_idx = [idx for idx, _ in ranking]
        # 将得分写入节点，便于展示
        score_map = {idx: sc for idx, sc in ranking}
        nodes = [nodes[i] for i in keep_idx]
        for i, n in zip(keep_idx, nodes):
            setattr(n, "score", score_map.get(i))

    # 生成答案；用重排后的节点数作为 top_k
    qe = index.as_query_engine(similarity_top_k=len(nodes) if nodes else top_k)
    answer = str(qe.query(question))

    if not show_context:
        return answer

    if as_json:
        payload = {
            "answer": answer,
            "contexts": [
                {
                    "rank": i + 1,
                    "score": getattr(n, "score", None),
                    "metadata": n.metadata,
                    "text": n.get_content(),
                }
                for i, n in enumerate(nodes)
            ],
        }
        return pyjson.dumps(payload, ensure_ascii=False, indent=2)

    ctx = _format_nodes(nodes)
    return f"{answer}\n\n--- CONTEXT ---\n{ctx}"