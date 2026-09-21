"""Streamlit decision dashboard and governed SQL Agent interface."""

from __future__ import annotations

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from project_agent.agent import ask
from project_agent.analytics import overview
from project_agent.database import DEFAULT_DB_PATH
from project_agent.retrieval import search_narratives


st.set_page_config(page_title="CFPB Complaint Intelligence Agent", page_icon="📊", layout="wide")
st.title("CFPB Complaint Intelligence Agent")
st.caption("官方月度聚合 KPI + 本地 Narrative 混合检索 + 可审计的 SQL/RAG Agent")


@st.cache_data
def query(sql: str, params=None) -> pd.DataFrame:
    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        return con.execute(sql, params or []).df()


months = query("SELECT month FROM monthly_kpis ORDER BY month")["month"].tolist()
start_month, end_month = st.select_slider(
    "分析区间（完整自然月）", options=months, value=(months[0], months[-1])
)

tab_overview, tab_agent, tab_rag, tab_themes, tab_quality = st.tabs(
    ["经营概览", "SQL / RAG Agent", "投诉文本检索", "文本主题", "数据质量与口径"]
)

with tab_overview:
    metrics = overview(start_month, end_month)
    a, b, c, d = st.columns(4)
    a.metric("投诉量", f"{metrics['complaint_count']:,}")
    b.metric("及时响应率", f"{metrics['timely_response_rate']:.2%}")
    c.metric("救济占比", f"{metrics['relief_rate']:.2%}")
    d.metric("处理中占比", f"{metrics['in_progress_rate']:.2%}")
    trend = query(
        "SELECT month, complaint_count, timely_rate, relief_rate, in_progress_rate FROM monthly_kpis WHERE month BETWEEN ? AND ? ORDER BY month",
        [start_month, end_month],
    )
    st.plotly_chart(
        px.line(trend, x="month", y="complaint_count", markers=True, title="月度投诉量（官方完整聚合）"),
        width="stretch",
    )
    product = query(
        """SELECT value AS product, SUM(complaint_count) AS complaint_count
        FROM monthly_dimensions WHERE dimension='product' AND month BETWEEN ? AND ?
        GROUP BY value ORDER BY complaint_count DESC LIMIT 10""",
        [start_month, end_month],
    )
    left, right = st.columns(2)
    left.plotly_chart(px.bar(product, x="complaint_count", y="product", orientation="h", title="Top 10 产品类别"), width="stretch")
    rate_long = trend.melt(id_vars="month", value_vars=["timely_rate", "relief_rate", "in_progress_rate"], var_name="metric", value_name="rate")
    right.plotly_chart(px.line(rate_long, x="month", y="rate", color="metric", markers=True, title="月度响应结构"), width="stretch")
    st.info("投诉量不等于投诉率。缺少客户规模/市场份额分母，因此不能据此评价公司质量或消费者伤害率。")

with tab_agent:
    question = st.text_area("用自然语言提问", "2025年1月至2026年8月，投诉量最高的三个月是哪几个？")
    if st.button("运行 Agent", type="primary"):
        with st.spinner("Agent 正在查看口径、生成并执行只读 SQL…"):
            try:
                result = ask(question)
                st.markdown(result["answer"])
                with st.expander("审计轨迹"):
                    st.json(result["trace"])
            except Exception as exc:
                st.error(f"运行失败：{exc}")

with tab_rag:
    st.subheader("本地投诉叙述混合检索")
    st.caption("TF-IDF关键词相似度 + Truncated SVD稠密向量；结果仅作案例证据，不代表总体频率")
    rag_query = st.text_input("检索问题", "消费者如何描述信用报告中的身份盗用问题？")
    col1, col2, col3 = st.columns(3)
    product_filter = col1.text_input("产品精确过滤（可留空）")
    issue_filter = col2.text_input("Issue精确过滤（可留空）")
    top_k = col3.slider("返回案例数", 1, 20, 8)
    if st.button("检索本地 Narrative", type="primary"):
        try:
            payload = search_narratives(
                query=rag_query,
                product=product_filter or None,
                issue=issue_filter or None,
                company=None,
                date_from=None,
                date_to=None,
                top_k=top_k,
                min_score=0.05,
            )
            st.caption(
                f"过滤后候选 {payload['candidate_count']:,} 个 chunk；返回 {payload['result_count']} 条"
            )
            for item in payload["results"]:
                with st.container(border=True):
                    st.markdown(
                        f"**#{item['rank']} · {item['citation']} · score={item['score']:.3f}**"
                    )
                    st.caption(
                        f"{item['date_received']}｜{item['product']}｜{item['issue']}｜{item['company']}"
                    )
                    st.write(item["excerpt"])
        except Exception as exc:
            st.error(f"本地检索不可用：{exc}")

with tab_themes:
    theme_available = bool(
        query(
            """SELECT COUNT(*) AS n FROM information_schema.tables
            WHERE table_schema='main' AND table_name='narrative_theme_monthly'"""
        ).iloc[0, 0]
    )
    if not theme_available:
        st.info("尚未生成文本主题表。运行 scripts/build_narrative_themes.py 后重新构建数据库。")
    else:
        st.subheader("NMF 描述性文本主题")
        st.warning("主题名称来自高权重词，不是人工真值、消费者伤害率或因果解释。")
        theme_totals = query(
            """SELECT topic_id, topic_label, SUM(narrative_count) AS narrative_count
            FROM narrative_theme_monthly GROUP BY topic_id, topic_label
            ORDER BY narrative_count DESC"""
        )
        st.plotly_chart(
            px.bar(
                theme_totals.head(12),
                x="narrative_count",
                y="topic_label",
                orientation="h",
                title="公开 Narrative 的描述性主题规模",
            ),
            width="stretch",
        )

with tab_quality:
    quality = query(
        """SELECT COUNT(*) AS sample_rows, COUNT(DISTINCT complaint_id) AS unique_ids,
        SUM(CASE WHEN routing_lag_hours > 15*24 THEN 1 ELSE 0 END) AS over_15_days,
        SUM(CASE WHEN routing_lag_hours > 30*24 THEN 1 ELSE 0 END) AS over_30_days,
        MAX(routing_lag_hours) AS max_hours FROM v_sample_complaints"""
    )
    st.subheader("1,000 条明细质量样本")
    st.dataframe(quality, width="stretch", hide_index=True)
    st.warning("该样本按月定额获取、不是随机样本，只用于字段检查、异常发现与 Agent 工具演示。")
    st.markdown("详见 `docs/kpi_dictionary.md`、`reports/data_quality_report.md` 与 `docs/error_log.md`。")
