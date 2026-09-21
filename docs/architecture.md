# 端到端架构与数据流

## 一句话架构

这是一个“双证据通道”投诉分析系统：DuckDB 提供可聚合、可复算的统计证据，本地混合向量索引提供带 Complaint ID 的文本案例证据，LangGraph 只负责选择和编排工具，模型不直接接触数据库或文件系统。

```text
CFPB 官方来源
  ├─ Aggregation API ─> 月度 KPI / 维度 Parquet ─────────────┐
  ├─ 1,000 条明细样本 ─> 数据质量 Parquet ────────────────┤
  └─ Narrative ZIP ─> 清洗/哈希/去重 ─┬─> TF-IDF + SVD 索引 │
                                      └─> NMF 主题聚合 ──────┤
                                                             v
用户问题 ─> 确定性路由(sql/rag/hybrid) ─> LangGraph 工具循环
                                  ├─> SQLGlot 网关 ─> DuckDB
                                  └─> 自定义检索器 ─> 本地索引
                                                             v
                              带口径、SQL 数字、原文引用的回答
```

## 数据层

### 结构化总体数据

`monthly_kpis` 和 `monthly_dimensions` 来自 CFPB 月度聚合接口，用来回答总量、趋势、比例和排名。每个比率保留分子与分母，跨月计算时先 `SUM` 再相除。

`complaint_sample` 是每月定额抽取的 1,000 条明细，只用于字段、主键、日期和流程长尾诊断。它不是随机样本，不能代替总体数据。

### Narrative 数据

`prepare_cfpb_narratives.py` 以 chunk 方式读取官方 ZIP，不需要把 491 MB CSV 一次性载入内存。处理包括字段重命名、非空与最短长度门禁、日期标准化、保守二次脱敏、Complaint ID 去重、Narrative SHA-256 和来源回执。

本次快照的对账链：

- 原始投诉 811,019；
- 非空 Narrative 265,405；
- 排除少于 80 字符的 3,958 条；
- 写入 261,447 条；
- 观察到 120,680 条完全重复文本，因此唯一文本训练池为 140,767 条。

重复文本保留在全量主题计数中，因为它们仍对应真实发布的不同投诉记录；但在检索索引与主题训练池中去重，避免模板文本支配特征学习。

## 检索层

`build_narrative_index.py` 用固定种子 42 对唯一文本做产品平衡抽样。这样既控制本地资源，又避免 64% 左右的信用报告类文本完全淹没小产品类别。

每条文本按 320 词窗口、48 词重叠切块。TF-IDF 使用 1–2 gram、最多 30,000 特征；Truncated SVD 把稀疏矩阵投影到 128 维并做 L2 标准化。检索分数为：

```text
hybrid_score = 0.35 × TF-IDF cosine + 0.65 × SVD cosine
```

元数据过滤先生成候选掩码，再在候选内排序。输出包含两类分数、文本片段、产品、问题、公司、日期、主题和 `Complaint ID — source snapshot` 引用。

## 主题层

`build_narrative_themes.py` 从唯一文本池中产品平衡抽取 30,000 条，清除 `XX/XXXX` 占位符和纯数字 token，训练 12 个 TF-IDF + NMF 主题。随后分批为全部 261,447 条 Narrative 分配主主题。

主题结果聚合成 `month × product × topic`，进入 DuckDB 的 `narrative_theme_monthly`。这一表负责总体计数；向量 Top-K 只负责案例检索，两者不能互换。

## Agent 运行时

`router.py` 用确定性规则给出三种路由：

- `sql`：数量、比例、趋势、排名；
- `rag`：案例、原文、消费者如何描述；
- `hybrid`：同时要求总体数字和案例。

路由还限制模型能看到的工具集合。SQL 问题只开放 Schema、指标字典和安全 SQL；RAG 问题只开放检索与单投诉读取；Hybrid 同时开放两类工具。

模型通过 Responses API 返回 function call。Python 执行工具后，把 JSON 作为 function call output 回传。`store=False`，但本地 `logs/agent_runs.jsonl` 保留问题、实际模型、工具轨迹和 token 用量；该文件不进入 Git。

## SQL 安全边界

SQLGlot 把 SQL 解析为 AST，网关执行以下检查：

1. 只允许单语句 `SELECT`、CTE 或集合查询；
2. 禁止 DDL、DML、事务、命令和附加数据库；
3. 表必须来自白名单；
4. 阻断 `read_csv*`、`read_parquet`、`parquet_scan`、`read_text`、`read_blob` 等外部读取；
5. 特别拒绝 SQLGlot 中表现为 `Table(Anonymous(...))` 的 DuckDB 表函数；
6. 自动施加结果行数上限；
7. 最终以 DuckDB `read_only=True` 连接执行。

AST 白名单和只读连接分别控制“能读什么”和“能不能写”，必须同时存在。

## 评测边界

- 单元/集成测试覆盖切块、路由、检索过滤、SQL 注入/外部读取、数据库口径和 Streamlit 基础结构。
- 5 个 RAG 弱标签问题只检查预期词、过滤和引用是否出现，不是人工相关性标注。
- 3 个在线问题检查 SQL/RAG/Hybrid 路由、工具、目标表、引用和工具错误；它们是冒烟测试，不是通用准确率。
- 更严格的下一步应建立人工标注的 Recall@K、MRR/nDCG、回答忠实度和拒答测试集。
