import pytest

from project_agent.sql_guard import UnsafeSQLError, validate_sql


def test_select_and_cte_are_allowed():
    result = validate_sql(
        "WITH x AS (SELECT * FROM monthly_kpis) SELECT month FROM x", max_rows=25
    )
    assert result.limit == 25
    assert result.referenced_tables == ("monthly_kpis",)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM monthly_kpis",
        "DROP TABLE monthly_kpis",
        "SELECT * FROM monthly_kpis; SELECT 1",
        "SELECT * FROM unknown_table",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "SELECT * FROM read_parquet('/tmp/secret.parquet')",
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM read_blob('/etc/passwd')",
        "SELECT * FROM parquet_scan('/tmp/secret.parquet')",
        "SELECT * FROM read_csv('README.md')",
    ],
)
def test_unsafe_sql_is_blocked(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


def test_limit_is_capped():
    result = validate_sql("SELECT * FROM monthly_kpis LIMIT 99999", max_rows=50)
    assert result.limit == 50
    assert "LIMIT 50" in result.normalized_sql
