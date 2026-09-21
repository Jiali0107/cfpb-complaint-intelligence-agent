import duckdb

from project_agent.database import DEFAULT_DB_PATH, database_summary


def test_expected_database_coverage():
    summary = database_summary()
    expected = {
        "complaint_sample_rows": 1000,
        "monthly_kpi_rows": 20,
        "monthly_dimension_rows": 3353,
        "month_min": "2025-01",
        "month_max": "2026-08",
    }
    assert {key: summary[key] for key in expected} == expected
    assert summary["narrative_rows"] >= 0
    assert summary["narrative_topic_rows"] >= 0
    assert summary["narrative_theme_monthly_rows"] >= 0


def test_monthly_denominators_reconcile():
    with duckdb.connect(str(DEFAULT_DB_PATH), read_only=True) as con:
        mismatches = con.execute(
            "SELECT COUNT(*) FROM monthly_kpis WHERE timely_known_count <> complaint_count"
        ).fetchone()[0]
    assert mismatches == 0
