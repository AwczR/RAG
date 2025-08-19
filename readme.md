
# RAG
### v0.0 baseline

这是一个基于 **LlamaIndex** 和 **SiliconFlow API** 的简单 RAG 项目。  
主要功能是：把文档切分、向量化存入数据库，然后通过检索 + 重排 + 大模型回答问题。  

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
## 🛠️ 用法（索引 / 提问）

### 索引
```bash
# 全量重建（首次或想重置）
python -m src.ingest -c config.yaml --rebuild

# 增量更新（data/ 新增或改动文件）
python -m src.ingest -c config.yaml
```

### 提问
```bash
# 只看答案
python -m src.main -c config.yaml ask "这批文档主要讲了什么？"

# 答案 + 检索到的上下文
python -m src.main -c config.yaml ask --show-context "这批文档主要讲了什么？"

# 上下文用 JSON 输出
python -m src.main -c config.yaml ask --show-context --json-context "这批文档主要讲了什么？"
```