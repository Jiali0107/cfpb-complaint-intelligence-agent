"""Profile the downloaded CFPB sample and write inspectable quality artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cfpb_complaints_sample.parquet"
RECEIPT_PATH = PROJECT_ROOT / "data" / "raw" / "cfpb_source_receipt.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPECTED_COLUMNS = {
    "company",
    "company_public_response",
    "company_response",
    "complaint_id",
    "date_received",
    "date_sent_to_company",
    "issue",
    "product",
    "state",
    "sub_issue",
    "sub_product",
    "submitted_via",
    "tags",
    "timely",
    "zip_code",
}


def json_safe(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def build_profile(frame: pd.DataFrame, receipt: dict) -> dict:
    profiled = frame.copy()
    for column in ("date_received", "date_sent_to_company"):
        profiled[column] = pd.to_datetime(profiled[column], utc=True, errors="coerce")

    lag_days = (
        profiled["date_sent_to_company"] - profiled["date_received"]
    ).dt.total_seconds() / 86_400
    id_counts = profiled["complaint_id"].astype("string").value_counts(dropna=False)
    unexpected_timely = sorted(
        set(profiled["timely"].dropna().astype(str)) - {"Yes", "No"}
    )
    state_valid = profiled["state"].isna() | profiled["state"].astype(str).str.fullmatch(
        r"[A-Z]{2}"
    )
    zip_text = profiled["zip_code"].astype("string")
    privacy_masked_zip = zip_text.str.fullmatch(r"(?:[0-9]{3}XX|XXXXX)", na=False)
    zip_valid = (
        profiled["zip_code"].isna()
        | zip_text.str.fullmatch(r"[0-9]{3}(?:[0-9]{2})?", na=False)
        | privacy_masked_zip
    )

    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source": receipt,
        "shape": {"rows": len(profiled), "columns": len(profiled.columns)},
        "schema": {
            "observed_columns": list(profiled.columns),
            "missing_expected_columns": sorted(EXPECTED_COLUMNS - set(profiled.columns)),
            "unexpected_columns": sorted(set(profiled.columns) - EXPECTED_COLUMNS),
            "dtypes": {column: str(dtype) for column, dtype in profiled.dtypes.items()},
        },
        "grain_and_keys": {
            "intended_grain": "one published complaint per complaint_id",
            "missing_complaint_id": int(profiled["complaint_id"].isna().sum()),
            "distinct_complaint_id": int(profiled["complaint_id"].nunique(dropna=True)),
            "duplicate_complaint_id_rows": int(id_counts[id_counts > 1].sum()),
            "exact_duplicate_rows": int(profiled.duplicated().sum()),
        },
        "date_checks": {
            "date_received_min": json_safe(profiled["date_received"].min()),
            "date_received_max": json_safe(profiled["date_received"].max()),
            "invalid_date_received": int(profiled["date_received"].isna().sum()),
            "invalid_date_sent_to_company": int(
                profiled["date_sent_to_company"].isna().sum()
            ),
            "negative_send_lag_rows": int((lag_days < 0).sum()),
            "send_lag_over_15_days_rows": int((lag_days > 15).sum()),
            "send_lag_over_30_days_rows": int((lag_days > 30).sum()),
            "send_lag_days_median": json_safe(lag_days.median()),
            "send_lag_days_p95": json_safe(lag_days.quantile(0.95)),
            "send_lag_days_max": json_safe(lag_days.max()),
        },
        "validity": {
            "unexpected_timely_values": unexpected_timely,
            "invalid_state_format_rows": int((~state_valid).sum()),
            "privacy_masked_zip_rows": int(privacy_masked_zip.sum()),
            "invalid_zip_format_rows": int((~zip_valid).sum()),
        },
    }


def write_artifacts(frame: pd.DataFrame, profile: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    null_profile = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(frame[column].dtype) for column in frame.columns],
            "null_count": [int(frame[column].isna().sum()) for column in frame.columns],
            "null_rate_pct": [round(frame[column].isna().mean() * 100, 2) for column in frame.columns],
            "distinct_non_null": [int(frame[column].nunique(dropna=True)) for column in frame.columns],
        }
    ).sort_values("null_rate_pct", ascending=False)
    null_profile.to_csv(REPORTS_DIR / "field_profile.csv", index=False)

    category_rows = []
    for column in (
        "product",
        "issue",
        "company",
        "submitted_via",
        "company_response",
        "timely",
        "state",
    ):
        counts = frame[column].fillna("<MISSING>").astype(str).value_counts().head(20)
        category_rows.extend(
            {
                "column": column,
                "value": value,
                "count": int(count),
                "share_pct": round(count / len(frame) * 100, 2),
            }
            for value, count in counts.items()
        )
    category_profile = pd.DataFrame(category_rows)
    category_profile.to_csv(REPORTS_DIR / "category_counts_top20.csv", index=False)

    dates = pd.to_datetime(frame["date_received"], utc=True, errors="coerce")
    monthly = (
        dates.dt.tz_convert(None).dt.to_period("M")
        .astype("string")
        .value_counts()
        .rename_axis("month")
        .reset_index(name="row_count")
        .sort_values("month")
    )
    monthly.to_csv(REPORTS_DIR / "monthly_counts.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top_products = frame["product"].fillna("<MISSING>").value_counts().head(10).sort_values()
    top_products.plot.barh(ax=axes[0], color="#2F6B8A")
    axes[0].set_title("Top 10 products in bounded API sample")
    axes[0].set_xlabel("Complaint rows")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="y", labelsize=8)
    monthly.plot(x="month", y="row_count", ax=axes[1], marker="o", color="#D97706", legend=False)
    axes[1].set_title("Rows by complaint received month")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Complaint rows")
    axes[1].tick_params(axis="x", rotation=60, labelsize=8)
    fig.suptitle("CFPB sample coverage checks (not market-share estimates)", fontsize=13)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "coverage_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (REPORTS_DIR / "profile_summary.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    key = profile["grain_and_keys"]
    dates_check = profile["date_checks"]
    validity = profile["validity"]
    source_meta = profile["source"].get("api_meta", {})
    highest_nulls = null_profile.head(8)
    null_lines = "\n".join(
        f"- `{row.column}`: {row.null_rate_pct:.2f}% ({int(row.null_count):,}/{len(frame):,})"
        for row in highest_nulls.itertuples()
    )
    report = f"""# CFPB 小样本数据质量报告

生成时间：{profile['generated_at_utc']}

## 结论

- 本次获得 **{len(frame):,} 行 × {len(frame.columns)} 列**，目标粒度为“一条 `complaint_id` 对应一条已公开投诉”。
- `complaint_id` 缺失 {key['missing_complaint_id']:,} 行、重复 ID 涉及 {key['duplicate_complaint_id_rows']:,} 行、完全重复 {key['exact_duplicate_rows']:,} 行。
- 投诉接收日期覆盖 **{dates_check['date_received_min']} 至 {dates_check['date_received_max']}**。
- 发送给公司的日期早于接收日期 {dates_check['negative_send_lag_rows']:,} 行；发送延迟中位数 {dates_check['send_lag_days_median']:.3f} 天，95 分位 {dates_check['send_lag_days_p95']:.3f} 天，最大 {dates_check['send_lag_days_max']:.3f} 天。
- 发送延迟超过 15 天 {dates_check['send_lag_over_15_days_rows']:,} 行，其中超过 30 天 {dates_check['send_lag_over_30_days_rows']:,} 行；这些是需要保留并进一步解释的真实流程长尾，不应静默删除。
- API 元数据：`last_updated={source_meta.get('last_updated')}`，`is_data_stale={source_meta.get('is_data_stale')}`，`has_data_issue={source_meta.get('has_data_issue')}`。
- 当前官方 API 返回 **15 个结构化字段且不含消费者投诉叙述文本**。本数据适合 SQL 指标、分类与流程时效分析，不足以单独支撑 narrative RAG。

## 高缺失字段

{null_lines}

这些缺失不应被自动填成“无”：`company_public_response` 为公司可选公开回应，`tags`、`sub_product`、`sub_issue` 也可能按业务定义合法为空。

## 格式与枚举检查

- 非 `Yes/No` 的 `timely` 值：{validity['unexpected_timely_values']}
- 不符合两位大写州代码的非空记录：{validity['invalid_state_format_rows']:,}
- CFPB 隐私掩码 ZIP（如 `123XX` / `XXXXX`）：{validity['privacy_masked_zip_rows']:,}
- 不属于 3/5 位数字或官方掩码格式的 ZIP：{validity['invalid_zip_format_rows']:,}

## 主要风险与严重度

1. **高：代表性偏差。** CFPB 明确说明数据库不是消费者经历的统计代表性样本；投诉量也受公司规模、市场份额和投诉倾向影响。不得直接生成“最差公司排行榜”。
2. **高：缺少叙述文本。** 当前 API 结构不支持直接训练/验证投诉文本 RAG；后续必须另行确认合规文本来源或将 Agent 定位为受控 SQL 分析 Agent。
3. **中：抽样偏差。** 这是按月等额、月内按 `created_date_desc` 截取的工程样本，不是概率随机样本；适合跨月份字段与流程验证，不适合总体推断或估计真实月份占比。
4. **中：日期窗口边界。** API 查询参数与实际记录时间戳应保留回执；业务分析前需明确上界的包含/排除规则。
5. **中：发送流程长尾。** 少量投诉从接收到发送给公司间隔较长；这可能源于转交或处理流程，当前字段不足以判断原因，应作为 Agent 的异常下钻项而不是直接判错。
6. **低：合法缺失。** 多个维度按官方字段定义允许为空，分析时应保留 `<MISSING>` 类别而不是静默删除。

## 可用性判断

在当前检查范围内，样本可以进入下一阶段的 **DuckDB 建模、指标定义、受控 SQL Agent 和可视化原型**。任何涉及公司优劣、市场份额或消费者总体体验的结论，都需要外部规模分母和更严格的抽样设计。
"""
    (REPORTS_DIR / "data_quality_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    if not DATA_PATH.exists() or not RECEIPT_PATH.exists():
        raise FileNotFoundError("Run scripts/download_cfpb_sample.py first")
    frame = pd.read_parquet(DATA_PATH)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    missing = EXPECTED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CFPB columns: {sorted(missing)}")
    profile = build_profile(frame, receipt)
    write_artifacts(frame, profile)
    print(json.dumps(profile["shape"], ensure_ascii=False))
    print(f"Report: {REPORTS_DIR / 'data_quality_report.md'}")


if __name__ == "__main__":
    main()
