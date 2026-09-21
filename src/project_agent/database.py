"""Build the local DuckDB analytics layer from reviewed CFPB artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WAREHOUSE_DIR = PROJECT_ROOT / "data" / "warehouse"
DEFAULT_DB_PATH = WAREHOUSE_DIR / "cfpb.duckdb"


def build_database(db_path: Path = DEFAULT_DB_PATH) -> Path:
    required = {
        "complaint_sample": PROCESSED_DIR / "cfpb_complaints_sample.parquet",
        "monthly_kpis": PROCESSED_DIR / "cfpb_monthly_kpis.parquet",
        "monthly_dimensions": PROCESSED_DIR / "cfpb_monthly_dimensions.parquet",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing processed inputs: {missing}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute("DROP VIEW IF EXISTS v_sample_complaints")
        for table, path in required.items():
            connection.execute(
                f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?)",
                [str(path)],
            )
        optional = {
            "complaint_narratives": PROCESSED_DIR / "cfpb_narratives.parquet",
            "narrative_topics": PROCESSED_DIR / "cfpb_narrative_topics.parquet",
            "narrative_theme_monthly": PROCESSED_DIR
            / "cfpb_narrative_theme_monthly.parquet",
        }
        for table, path in optional.items():
            if path.exists():
                connection.execute(
                    f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM read_parquet(?)",
                    [str(path)],
                )
        connection.execute(
            """
            CREATE VIEW v_sample_complaints AS
            SELECT
                *,
                date_trunc('month', date_received) AS received_month,
                date_diff('second', date_received, date_sent_to_company) / 3600.0
                    AS routing_lag_hours
            FROM complaint_sample
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE metric_dictionary (
                metric_name VARCHAR,
                source_table VARCHAR,
                definition VARCHAR,
                caveat VARCHAR
            )
            """
        )
        connection.executemany(
            "INSERT INTO metric_dictionary VALUES (?, ?, ?, ?)",
            [
                (
                    "complaint_count",
                    "monthly_kpis",
                    "Published complaints returned by CFPB for a calendar month.",
                    "Not a population harm rate or company quality ranking.",
                ),
                (
                    "timely_response_rate",
                    "monthly_kpis",
                    "SUM(timely_yes_count) / SUM(timely_known_count).",
                    "Compute from counts; do not average monthly percentages.",
                ),
                (
                    "relief_rate",
                    "monthly_kpis",
                    "SUM(relief_count) / SUM(complaint_count).",
                    "Relief category is not consumer satisfaction.",
                ),
                (
                    "routing_lag_hours",
                    "v_sample_complaints",
                    "Hours from CFPB receipt to sending the complaint to a company.",
                    "Record sample is non-random and for quality diagnostics only.",
                ),
                (
                    "narrative_theme_share",
                    "narrative_theme_monthly",
                    "narrative_count / product_month_narratives.",
                    "NMF topics are descriptive clusters of published narratives, not causes or population rates.",
                ),
            ],
        )
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    return db_path


def database_summary(db_path: Path = DEFAULT_DB_PATH) -> dict:
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        summary = {
            "complaint_sample_rows": connection.execute(
                "SELECT COUNT(*) FROM complaint_sample"
            ).fetchone()[0],
            "monthly_kpi_rows": connection.execute(
                "SELECT COUNT(*) FROM monthly_kpis"
            ).fetchone()[0],
            "monthly_dimension_rows": connection.execute(
                "SELECT COUNT(*) FROM monthly_dimensions"
            ).fetchone()[0],
            "month_min": str(
                connection.execute("SELECT MIN(month) FROM monthly_kpis").fetchone()[0]
            ),
            "month_max": str(
                connection.execute("SELECT MAX(month) FROM monthly_kpis").fetchone()[0]
            ),
        }
        available = {
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        for table, key in (
            ("complaint_narratives", "narrative_rows"),
            ("narrative_topics", "narrative_topic_rows"),
            ("narrative_theme_monthly", "narrative_theme_monthly_rows"),
        ):
            summary[key] = (
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if table in available
                else 0
            )
        return summary
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()
    path = build_database(args.db_path)
    print(f"Built {path}")
    print(database_summary(path))


if __name__ == "__main__":
    main()
