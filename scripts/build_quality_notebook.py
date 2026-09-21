"""Build a compact, reviewable CFPB data-quality notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "01_cfpb_data_quality.ipynb"

notebook = nbf.v4.new_notebook()
notebook["metadata"]["kernelspec"] = {
    "display_name": "Python 3 (Project_agent)",
    "language": "python",
    "name": "python3",
}
notebook["metadata"]["language_info"] = {"name": "python", "version": "3.12"}
notebook["cells"] = [
    nbf.v4.new_markdown_cell(
        """# CFPB Consumer Complaint Database — 数据质量检查

## tl;dr

本 notebook 对一个有界、可复现的 CFPB API 样本执行字段、主键、缺失、日期、枚举与覆盖检查。结论只用于判断数据是否适合进入本地 DuckDB / SQL Agent 原型，不用于推断全体消费者体验或给公司排名。"""
    ),
    nbf.v4.new_markdown_cell(
        """## Context & Methods

### Key Assumptions

- 一行代表一条已公开投诉，候选主键为 `complaint_id`。
- API 样本按月等额、月内按 `created_date_desc` 截取，属于工程验证样本而非概率随机样本。
- CFPB 官方说明该数据库不是消费者经历的统计代表性样本；公司投诉量必须结合公司规模或市场份额解释。
- 来源回执：`../data/raw/cfpb_source_receipt.json`。"""
    ),
    nbf.v4.new_code_cell(
        """from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
DATA_PATH = ROOT / "data" / "processed" / "cfpb_complaints_sample.parquet"
RECEIPT_PATH = ROOT / "data" / "raw" / "cfpb_source_receipt.json"

df = pd.read_parquet(DATA_PATH)
receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
sns.set_theme(style="whitegrid")
pd.set_option("display.max_colwidth", 80)
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
print("API last_updated:", receipt["api_meta"].get("last_updated"))
print("API stale/data issue:", receipt["api_meta"].get("is_data_stale"), receipt["api_meta"].get("has_data_issue"))"""
    ),
    nbf.v4.new_markdown_cell("## Data\n\n### 1. 字段与样例"),
    nbf.v4.new_code_cell(
        """display(pd.DataFrame({"column": df.columns, "dtype": [str(df[c].dtype) for c in df.columns]}))
display(df.head(5))"""
    ),
    nbf.v4.new_markdown_cell("### 2. 主键、重复与缺失"),
    nbf.v4.new_code_cell(
        """key_checks = pd.Series({
    "rows": len(df),
    "distinct_complaint_id": df["complaint_id"].nunique(dropna=True),
    "missing_complaint_id": df["complaint_id"].isna().sum(),
    "duplicate_complaint_id_rows": df["complaint_id"].duplicated(keep=False).sum(),
    "exact_duplicate_rows": df.duplicated().sum(),
})
display(key_checks.to_frame("value"))

nulls = pd.DataFrame({
    "null_count": df.isna().sum(),
    "null_rate_pct": (df.isna().mean() * 100).round(2),
    "distinct_non_null": df.nunique(dropna=True),
}).sort_values("null_rate_pct", ascending=False)
display(nulls)"""
    ),
    nbf.v4.new_markdown_cell("## Results\n\n### 3. 日期范围和发送时效"),
    nbf.v4.new_code_cell(
        """received = pd.to_datetime(df["date_received"], utc=True, errors="coerce")
sent = pd.to_datetime(df["date_sent_to_company"], utc=True, errors="coerce")
lag_days = (sent - received).dt.total_seconds() / 86400

date_checks = pd.Series({
    "date_received_min": received.min(),
    "date_received_max": received.max(),
    "invalid_date_received": received.isna().sum(),
    "invalid_date_sent": sent.isna().sum(),
    "negative_send_lag_rows": (lag_days < 0).sum(),
    "send_lag_over_15_days_rows": (lag_days > 15).sum(),
    "send_lag_over_30_days_rows": (lag_days > 30).sum(),
    "send_lag_median_days": lag_days.median(),
    "send_lag_p95_days": lag_days.quantile(0.95),
    "send_lag_max_days": lag_days.max(),
})
display(date_checks.to_frame("value"))"""
    ),
    nbf.v4.new_markdown_cell("### 4. 样本覆盖（不是总体分布估计）"),
    nbf.v4.new_code_cell(
        """monthly = received.dt.tz_convert(None).dt.to_period("M").astype("string").value_counts().sort_index()
top_products = df["product"].fillna("<MISSING>").value_counts().head(10).sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top_products.plot.barh(ax=axes[0], color="#2F6B8A")
axes[0].set(title="Top 10 products in sample", xlabel="Complaint rows", ylabel="")
monthly.plot(ax=axes[1], marker="o", color="#D97706")
axes[1].set(title="Rows by complaint received month", xlabel="Month", ylabel="Complaint rows")
axes[1].tick_params(axis="x", rotation=60)
fig.suptitle("CFPB bounded sample coverage checks")
plt.tight_layout()
plt.show()"""
    ),
    nbf.v4.new_markdown_cell("### 5. 枚举和格式"),
    nbf.v4.new_code_cell(
        """timely_values = df["timely"].fillna("<MISSING>").value_counts(dropna=False)
state_valid = df["state"].isna() | df["state"].astype(str).str.fullmatch(r"[A-Z]{2}")
zip_text = df["zip_code"].astype("string")
privacy_masked_zip = zip_text.str.fullmatch(r"(?:[0-9]{3}XX|XXXXX)", na=False)
zip_valid = (
    df["zip_code"].isna()
    | zip_text.str.fullmatch(r"[0-9]{3}(?:[0-9]{2})?", na=False)
    | privacy_masked_zip
)

display(timely_values.to_frame("rows"))
display(pd.Series({
    "invalid_state_format_rows": (~state_valid).sum(),
    "privacy_masked_zip_rows": privacy_masked_zip.sum(),
    "invalid_zip_format_rows": (~zip_valid).sum(),
}).to_frame("value"))"""
    ),
    nbf.v4.new_markdown_cell(
        """## Takeaways

- 主键、重复、日期与字段结构的实际结果见上方已执行输出。
- 结构化字段足以支持 DuckDB 指标查询、产品/问题/渠道拆解和响应时效分析。
- 当前官方 API 不含投诉叙述文本，不能凭空宣称完成 narrative RAG。
- 高缺失字段需要按业务定义解释；可选公开回应、标签、子产品和子问题的空值不应自动当作数据错误。
- 此样本按月等额、月内按创建时间降序截取，只适合工程验证；公司比较需要公司规模/市场份额等外部分母。"""
    ),
]

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, NOTEBOOK_PATH)
print(f"Built {NOTEBOOK_PATH}")
