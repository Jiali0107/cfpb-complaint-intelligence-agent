"""Governed metric definitions for CFPB complaint analytics."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    label_cn: str
    formula: str
    grain: str
    source: str
    interpretation: str
    caveat: str


METRICS = (
    MetricDefinition(
        name="complaint_count",
        label_cn="投诉量",
        formula="COUNT(published complaints)",
        grain="calendar month",
        source="CFPB API hits.total.value",
        interpretation="CFPB在对应月份收到并纳入查询结果的公开投诉数量。",
        caveat="并非市场伤害率；公司规模、市场份额和投诉倾向会影响数量。",
    ),
    MetricDefinition(
        name="timely_response_rate",
        label_cn="及时响应率",
        formula="SUM(timely_yes_count) / SUM(timely_known_count)",
        grain="selected complete months",
        source="CFPB API timely aggregation",
        interpretation="已知及时性记录中，公司及时响应的比例。",
        caveat="这是加权比例，不能简单平均各月百分比。",
    ),
    MetricDefinition(
        name="relief_rate",
        label_cn="获得救济的投诉占比",
        formula="SUM(monetary_relief_count + non_monetary_relief_count) / SUM(complaint_count)",
        grain="selected complete months",
        source="CFPB API company_response aggregation",
        interpretation="公司回复结果为金钱或非金钱救济的投诉占全部投诉比例。",
        caveat="公开响应类别不等于消费者满意度，也不构成因果效果。",
    ),
    MetricDefinition(
        name="in_progress_rate",
        label_cn="处理中占比",
        formula="SUM(in_progress_count) / SUM(complaint_count)",
        grain="selected complete months",
        source="CFPB API company_response aggregation",
        interpretation="查询快照中仍为In progress的投诉占比。",
        caveat="受数据更新时间和近期月份成熟度影响。",
    ),
    MetricDefinition(
        name="sample_routing_lag_hours",
        label_cn="样本转交时长（小时）",
        formula="date_sent_to_company - date_received",
        grain="record-level quality sample",
        source="1,000-row bounded CFPB record sample",
        interpretation="CFPB收到投诉至发送给公司的时间差，用于发现流程长尾。",
        caveat="来自工程抽样，不应当作总体时效分布估计。",
    ),
    MetricDefinition(
        name="narrative_theme_share",
        label_cn="文本主题占比",
        formula="narrative_count / product_month_narratives",
        grain="product-month within published narrative corpus",
        source="CFPB opted-in published narratives + local NMF topic model",
        interpretation="同一产品和月份中，被NMF分配至某描述性主题的公开叙述占比。",
        caveat="只覆盖公开且通过清理的叙述；主题为无监督聚类，不代表因果或人工真值。",
    ),
)


def metric_catalog() -> list[dict[str, str]]:
    return [asdict(metric) for metric in METRICS]
