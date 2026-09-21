# 实现与验证指南

本文按系统依赖顺序说明各模块的输入、处理逻辑、输出和验证方法，用于复现实现与定位故障。

## 1. 数据口径

- 区分投诉量、及时响应率和救济率，比例统一按汇总后的分子与分母计算。
- CFPB 数据不是消费者经历的统计代表性样本，投诉量不能直接解释为伤害率或公司质量排名。
- 来源回执保存查询区间、更新时间、记录数量与文件哈希。

验证：使用 `data/raw/*_receipt.json` 对账结构化聚合与 Narrative 处理数量。

## 2. 数据处理

- 结构化数据使用 pandas、Parquet 与 DuckDB，统一字段类型和月度粒度。
- Narrative ZIP 采用分块读取，执行非空门禁、日期标准化、保守二次脱敏、Complaint ID 去重和 SHA-256 哈希。
- 完全重复文本在总体主题计数中保留，在索引和主题训练池中按文本哈希去重。

验证：运行 `uv run project-agent build-db`，并检查来源回执、主键唯一性和空值约束。

## 3. SQL 分析层

- DuckDB 读取小型 Parquet 和本地生成的主题聚合表。
- 区间比例使用 `SUM(numerator) / SUM(denominator)`，避免直接平均月度百分比。
- Agent SQL 经 SQLGlot AST 白名单、只读连接和行数限制后执行。

验证：运行 `tests/test_database.py`、`tests/test_analytics.py` 与 `tests/test_sql_guard.py`。

## 4. 本地检索层

- 唯一文本按产品平衡抽样，以固定种子生成 20,000 条索引文档。
- 长文本按 320 词窗口、48 词重叠切块。
- 检索分数融合 TF-IDF 词法相似度与 128 维 Truncated SVD 稠密相似度，并支持产品、问题、公司、日期和主题过滤。

验证：运行 `uv run project-agent search` 检查混合分数、过滤条件、Complaint ID 和 chunk 顺序，并执行 `tests/test_retrieval.py`。

## 5. 主题建模

- 从唯一文本池中产品平衡抽取 30,000 条，清理脱敏占位符和纯数字 token。
- 使用 TF-IDF + NMF 训练 12 个描述性主题，再分批为全部处理后 Narrative 分配主主题。
- 主题是无监督描述性聚类，不是人工分类标签，也不代表投诉原因或伤害率。

验证：检查 `reports/narrative_theme_summary.json` 的参数、主题词、训练规模与全量分配规模。

## 6. Agent 工具编排

- 路由器将问题分为 SQL、RAG 或 Hybrid。
- LangGraph 管理状态与工具循环；OpenAI Responses API Function Calling 只负责选择受控工具和组织答案。
- Narrative 被视为不可信数据，工具不能触发网络、文件或写入操作。

验证：运行固定的 SQL、RAG 与 Hybrid 问题集，检查实际工具轨迹、引用完整性和工具错误。

## 7. 评测与可观测性

- 单元测试覆盖数据、数据库、分析、检索、路由、SQL 安全与 Dashboard。
- 检索评测使用固定问题和弱标签，只作为冒烟检查，不表述为通用准确率。
- 在线 Agent 评测记录模型、工具轨迹、token 和延迟；运行日志不提交 Git。

验证：运行 `uv run pytest -q`、`uv run python scripts/evaluate_rag.py` 和按需执行的在线编排评测。

## 8. 重建顺序

1. 下载或放置官方 Narrative ZIP。
2. 执行 `./rag_pipeline.command` 完成清洗、索引、主题、数据库和测试。
3. 使用 `./run.command` 启动 Streamlit。
4. 使用固定问题集复核 SQL、RAG 和 Hybrid 输出。

各阶段产物、输入哈希和参数必须与来源回执及 manifest 对账后，才视为重建完成。
