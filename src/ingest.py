import os, argparse, fnmatch
from typing import List, Iterable

import chromadb
from llama_index.core import Settings, VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config_loader import load_cfg
from src.emb_silicon import SiliconFlowEmbedding

def _load_vector_store(cfg: dict) -> ChromaVectorStore:
    vs = cfg["vector_store"]
    client = chromadb.PersistentClient(path=vs["persist_dir"])
    col = client.get_or_create_collection(vs["collection"])
    return ChromaVectorStore(chroma_collection=col)

def _iter_files(root: str, recursive: bool) -> Iterable[str]:
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                yield os.path.join(dirpath, fn)
    else:
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p):
                yield p

def _filter_files(all_files: Iterable[str], includes: List[str] | None, excludes: List[str] | None) -> List[str]:
    files = list(all_files)
    if includes:
        keep = []
        for pat in includes:
            keep.extend([f for f in files if fnmatch.fnmatch(f, pat)])
        files = sorted(set(keep))
    if excludes:
        files = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in excludes)]
    return files

def run_ingest(cfg_path: str, rebuild: bool):
    cfg = load_cfg(cfg_path)

    # LLM 可不设；主要设嵌入与分块
    Settings.embed_model = SiliconFlowEmbedding(
        model=cfg["embedding"]["model"],
        timeout=cfg["runtime"]["timeout_sec"],
    )
    ingest = cfg.get("ingest", {})
    chunk_size = int(ingest.get("chunk_size", 300))      # 避免 512 上限
    chunk_overlap = int(ingest.get("chunk_overlap", 60))
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.embed_model.embed_batch_size = 8

    # 向量库
    vs_cfg = cfg["vector_store"]
    client = chromadb.PersistentClient(path=vs_cfg["persist_dir"])
    if rebuild:
        try:
            client.delete_collection(vs_cfg["collection"])
        except Exception:
            pass
    collection = client.get_or_create_collection(vs_cfg["collection"])
    vs = ChromaVectorStore(chroma_collection=collection)
    storage = StorageContext.from_defaults(vector_store=vs)

    # 读取文档
    data_dir = cfg["paths"]["data_dir"]
    recursive = bool(cfg["paths"].get("recursive", True))
    includes = ingest.get("include_globs")
    excludes = ingest.get("exclude_globs")
    all_files = list(_iter_files(data_dir, recursive))
    sel_files = _filter_files(all_files, includes, excludes) if (includes or excludes) else None
    reader = SimpleDirectoryReader(input_files=sel_files) if sel_files else SimpleDirectoryReader(data_dir, recursive=recursive)
    docs = reader.load_data()

    VectorStoreIndex.from_documents(docs, storage_context=storage, embed_model=Settings.embed_model)
    print(f"[ingest] ok | store=chroma dir={vs_cfg['persist_dir']} collection={vs_cfg['collection']} | docs={len(docs)}")

def main():
    ap = argparse.ArgumentParser("ingest")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    run_ingest(args.config, args.rebuild)

if __name__ == "__main__":
    main()