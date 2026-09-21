from project_agent.analytics import overview, run_safe_sql


def test_governed_overview_uses_weighted_rates():
    result = overview("2025-01", "2026-08")
    assert result["complaint_count"] == 10_287_130
    assert 0.99 < result["timely_response_rate"] < 1
    assert 0.3 < result["relief_rate"] < 0.4


def test_safe_query_returns_traceable_shape():
    result = run_safe_sql(
        "SELECT month, complaint_count FROM monthly_kpis ORDER BY complaint_count DESC LIMIT 3"
    )
    assert result["row_count"] == 3
    assert result["rows"][0] == ["2026-07", 671_212]
    assert result["tables"] == ["monthly_kpis"]
