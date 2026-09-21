# RAG 技术细节

## 为什么不是直接调用云 Embedding

这批数据有 26 万余条消费者叙事。项目目标是本地可审计、低成本和隐私边界清晰，因此没有把全量原文上传到 Embedding API，而是选择 TF-IDF + Truncated SVD：前者擅长实体、法规和产品名的精确匹配，后者从词项共现中学习低维语义空间。

代价是语义泛化能力弱于现代神经 Embedding，尤其对跨语言查询敏感。因此 Agent 会把中文检索意图改写成英文核心词；项目采用的是本地混合向量索引，不使用 OpenAI Embeddings 或专用向量数据库。

## 索引构建

1. 按 `complaint_id` 去重，确保主键唯一。
2. 按 `narrative_sha256` 去重，防止模板原文重复进入训练索引。
3. 按产品分层、以 SHA-256 稳定键抽样 20,000 条；相同种子可复现同一子集。
4. 长文本使用 320 词窗口和 48 词重叠；短文本保持投诉级单块。
5. 拟合带英文停用词、1–2 gram、sublinear TF 的 TF-IDF。
6. 用 128 维 Truncated SVD 得到稠密向量，再做 L2 标准化。
7. 落盘 metadata Parquet、稠密 `.npy`、稀疏 `.npz`、两类 joblib 模型和 manifest。
8. manifest 记录源文件哈希、参数、文档/chunk 数、解释方差和每个产物哈希。

本次 SVD 128 维累计解释方差为 0.2211。这个值不是检索准确率，只说明低维空间保留的 TF-IDF 方差信息比例。

## 查询算法

给定查询 `q`：

```text
q_sparse = TFIDF.transform(q)
q_dense  = L2_normalize(SVD.transform(q_sparse))

lexical_i = document_tfidf_i · q_sparse
dense_i   = document_svd_i · q_dense
score_i   = 0.35 × lexical_i + 0.65 × dense_i
```

随后应用产品、问题、公司、日期和主题掩码，使用 `argpartition` 取 Top-K，再按混合分数精排。由于矩阵已经 L2 标准化，点积等价于余弦相似度。

`min_score` 不是概率。当前默认 0.05 只是召回门槛；不同语料和特征参数下不能横向比较。首次在线评测曾让模型使用 0.70，导致有效案例被全部过滤，因此工具描述与系统规则明确要求从 0.05 开始，零结果时以不高于 0.20 重试。

## 自定义工具契约

`search_complaint_narratives` 输入：

- `query`：英文或中文问题，Agent 会优先转成英文核心概念；
- 元数据过滤：`product`、`issue`、`company`、`date_from`、`date_to`、`topic_id`；
- `top_k`：1–20；
- `min_score`：0–1。

输出：候选数、结果数、rank、混合/稠密/词法分数、Complaint ID、chunk、原文片段、业务字段、主题和来源快照。

`get_complaint_narrative` 按 Complaint ID 返回该投诉的所有有序 chunks，用于核对引用上下文。

## Prompt Injection 与数据泄漏防护

Narrative 被明确视为不可信数据。系统提示禁止执行原文中的指令；工具只能返回数据，不能触发网络、文件或写入操作。Agent 没有通用 shell 或文件读取工具。

公开 Git 不包含 ZIP、处理后 Narrative、向量、模型、DuckDB 或运行日志。数据回执和主题摘要不含消费者原文，可用于复现来源与参数。二次正则扫描只是一层补充防护，不是完整匿名化保证。

## 如何升级为真正的向量数据库

当数据量、并发或增量更新需求上升时，可保持 `search_complaint_narratives` 工具契约不变，把底层替换为 Qdrant、Milvus、Weaviate 或 pgvector：

- 主键使用 `chunk_id`；
- payload 保留当前元数据字段；
- dense vector 替换为神经 Embedding；
- lexical 通道可使用 BM25；
- 用 Reciprocal Rank Fusion 或学习排序融合；
- 增加索引版本、Embedding 模型版本和回滚策略。

这样 Agent 与 UI 无需大改，体现了工具接口和检索实现解耦。
