"""Read-only governed access to the local DuckDB warehouse."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from .database import DEFAULT_DB_PATH
from .kpis import metric_catalog
from .sql_guard import ALLOWED_TABLES, validate_sql


def _json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def database_schema(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    with duckdb.connect(str(db_path), read_only=True) as con:
        rows = con.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name IN (SELECT UNNEST(?))
            ORDER BY table_name, ordinal_position
            """,
            [sorted(ALLOWED_TABLES)],
        ).fetchall()
    grouped: dict[str, list[dict]] = {}
    for table, column, dtype in rows:
        grouped.setdefault(table, []).append({"name": column, "type": dtype})
    return [{"table": table, "columns": columns} for table, columns in grouped.items()]


def run_safe_sql(sql: str, db_path: Path = DEFAULT_DB_PATH, max_rows: int = 200) -> dict:
    validation = validate_sql(sql, max_rows=max_rows)
    with duckdb.connect(str(db_path), read_only=True) as con:
        cursor = con.execute(validation.normalized_sql)
        columns = [item[0] for item in cursor.description]
        rows = [[_json_value(value) for value in row] for row in cursor.fetchall()]
    return {
        "sql": validation.normalized_sql,
        "tables": list(validation.referenced_tables),
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated_at": validation.limit,
    }


def tool_schema_payload() -> str:
    return json.dumps(database_schema(), ensure_ascii=False)


def tool_metric_payload() -> str:
    return json.dumps(metric_catalog(), ensure_ascii=False)


def overview(start_month: str, end_month: str, db_path: Path = DEFAULT_DB_PATH) -> dict:
    with duckdb.connect(str(db_path), read_only=True) as con:
        row = con.execute(
            """
            SELECT
                SUM(complaint_count),
                SUM(timely_yes_count) / NULLIF(SUM(timely_known_count), 0),
                SUM(relief_count) / NULLIF(SUM(complaint_count), 0),
                SUM(in_progress_count) / NULLIF(SUM(complaint_count), 0)
            FROM monthly_kpis
            WHERE month BETWEEN ? AND ?
            """,
            [start_month, end_month],
        ).fetchone()
    return {
        "complaint_count": int(row[0] or 0),
        "timely_response_rate": float(row[1] or 0),
        "relief_rate": float(row[2] or 0),
        "in_progress_rate": float(row[3] or 0),
    }
