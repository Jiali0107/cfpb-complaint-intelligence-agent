"""AST-based read-only SQL policy for the analytics Agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from sqlglot import exp, parse


ALLOWED_TABLES = {
    "complaint_sample",
    "v_sample_complaints",
    "monthly_kpis",
    "monthly_dimensions",
    "narrative_theme_monthly",
    "metric_dictionary",
}
BLOCKED_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Copy,
    exp.Transaction,
    exp.Use,
)
BLOCKED_FUNCTIONS = {
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_json_auto",
    "read_parquet",
    "parquet_scan",
    "read_text",
    "read_blob",
    "glob",
    "httpfs",
    "sqlite_scan",
    "postgres_scan",
}


class UnsafeSQLError(ValueError):
    """Raised when SQL violates the local read-only policy."""


@dataclass(frozen=True)
class SQLValidation:
    normalized_sql: str
    referenced_tables: tuple[str, ...]
    limit: int

    def to_dict(self) -> dict:
        return asdict(self)


def validate_sql(sql: str, max_rows: int = 200) -> SQLValidation:
    if not sql or not sql.strip():
        raise UnsafeSQLError("SQL is empty")
    try:
        statements = parse(sql, read="duckdb")
    except Exception as exc:  # pragma: no cover - sqlglot error types vary
        raise UnsafeSQLError(f"SQL parse failed: {exc}") from exc
    if len(statements) != 1:
        raise UnsafeSQLError("Exactly one SQL statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise UnsafeSQLError("Only SELECT queries are allowed")
    if any(tree.find(node_type) is not None for node_type in BLOCKED_NODES):
        raise UnsafeSQLError("Mutation, DDL, command, and transaction SQL are blocked")

    for function in tree.find_all(exp.Func):
        function_name = (getattr(function, "name", "") or function.sql_name()).lower()
        if function_name in BLOCKED_FUNCTIONS:
            raise UnsafeSQLError(f"Blocked function: {function_name}")

    # DuckDB external table functions are represented as Table(Anonymous(...))
    # by SQLGlot and have no ordinary table name. Reject every table source that
    # is not a plain identifier; this closes read_text/read_blob/*_scan bypasses.
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise UnsafeSQLError(f"Table functions are not allowed: {table.sql()}")

    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    tables = {
        table.name.lower()
        for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in cte_names
    }
    unknown = tables - ALLOWED_TABLES
    if unknown:
        raise UnsafeSQLError(f"Unknown or disallowed table(s): {sorted(unknown)}")

    requested_limit = max_rows
    limit_node = tree.args.get("limit")
    if limit_node is not None and limit_node.expression is not None:
        try:
            requested_limit = int(limit_node.expression.name)
        except (TypeError, ValueError):
            requested_limit = max_rows
    enforced_limit = min(max(requested_limit, 1), max_rows)
    tree.set("limit", exp.Limit(expression=exp.Literal.number(enforced_limit)))
    return SQLValidation(
        normalized_sql=tree.sql(dialect="duckdb"),
        referenced_tables=tuple(sorted(tables)),
        limit=enforced_limit,
    )
