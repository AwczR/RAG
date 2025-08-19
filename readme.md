
# RAG
### v0.0 baseline

这是一个基于 **LlamaIndex** 和 **SiliconFlow API** 的简单 RAG 项目。  
 

---

## 📂 项目结构

```
src/
 ├── config_loader.py     # 统一读取 config.yaml 或 .env
 ├── emb_silicon.py       # SiliconFlow 的 embedding 封装
 ├── llm_silicon.py       # SiliconFlow 的大模型封装
 ├── rerank_silicon.py    # SiliconFlow 的重排模型封装
 ├── ingest.py            # 文档加载、分块、向量化、写入数据库
 ├── query.py             # 查询逻辑：检索 → 重排 → 拼接上下文 → 调用 LLM
 └── main.py              # 命令行入口 (ingest / ask)
config.yaml               # 配置文件（模型、参数、路径等）
requirements.txt          # 依赖列表
```

---

## 🔧 工作流程

1. **ingest**：加载文档 → 分块 → 生成向量 → 存入向量库（Chroma）  
2. **ask**：输入问题 → 从向量库检索相关文档 → 重排 → 提供上下文给大模型 → 输出答案  

---