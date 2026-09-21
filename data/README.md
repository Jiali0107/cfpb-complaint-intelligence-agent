# Data package

仓库只保留运行 Dashboard 和测试所需的三份小型 Parquet：

- `cfpb_monthly_kpis.parquet`：20 个月的总体 KPI；
- `cfpb_monthly_dimensions.parquet`：产品、问题、渠道、州等月度聚合；
- `cfpb_complaints_sample.parquet`：1,000 条非随机明细质量样本。

`raw/` 中只保存来源回执，不提交约 6 MB 的原始 API 响应。`warehouse/` 中的 DuckDB 由 `setup.command` 或 `uv run project-agent build-db` 本地生成。

文本 RAG 扩展使用 CFPB 官方 Narrative Archive。本地运行时会额外生成：

- `raw/narratives/*.zip`：官方归档原文件；
- `processed/cfpb_narratives.parquet`：通过二次敏感信息扫描的 narrative 数据；
- `processed/cfpb_narrative_topics.parquet`：全量文本主题标签；
- `processed/cfpb_narrative_theme_monthly.parquet`：月度主题聚合；
- `vector_store/`：TF-IDF + Truncated SVD 本地混合向量索引。

以上文件可能较大且包含消费者叙述，不进入公开 Git。仓库只保留代码、来源说明、汇总报告和不含原文的评测摘要。

数据来自 [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)。CFPB 明确提示该数据库不是消费者经历的统计代表性样本；不得将投诉数量直接解释为伤害率或公司质量排名。
