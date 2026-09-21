"""Download exact monthly CFPB aggregations for dashboard KPI denominators."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd


API_URL = (
    "https://www.consumerfinance.gov/data-research/consumer-complaints/"
    "search/api/v1/"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DIMENSIONS = (
    "product",
    "issue",
    "company_response",
    "submitted_via",
    "timely",
    "state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-08-31")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def buckets(aggregation: dict, dimension: str) -> tuple[list[dict], int, int]:
    body = aggregation.get(dimension, {})
    return (
        body.get("buckets", []),
        int(body.get("sum_other_doc_count", 0)),
        int(body.get("doc_count_error_upper_bound", 0)),
    )


def bucket_count(aggregation: dict, dimension: str, value: str) -> int:
    rows, _, _ = buckets(aggregation, dimension)
    return int(next((row["doc_count"] for row in rows if row["key"] == value), 0))


def main() -> None:
    args = parse_args()
    months = pd.period_range(args.start_date, args.end_date, freq="M")
    if len(months) == 0:
        raise ValueError("The requested date range contains no calendar month")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_responses: list[dict] = []
    monthly_rows: list[dict] = []
    dimension_rows: list[dict] = []
    headers = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}

    with httpx.Client(timeout=90.0, follow_redirects=True, headers=headers) as client:
        for month in months:
            params = {
                "date_received_min": month.start_time.date().isoformat(),
                "date_received_max": month.end_time.date().isoformat(),
                "size": 1,
                "frm": 0,
                "sort": "created_date_desc",
                "no_aggs": "false",
                "no_highlight": "true",
            }
            response = client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            total = int(payload["hits"]["total"]["value"])
            aggregations = payload.get("aggregations", {})
            meta = payload.get("_meta", {})

            timely_yes = bucket_count(aggregations["timely"], "timely", "Yes")
            timely_no = bucket_count(aggregations["timely"], "timely", "No")
            monetary = bucket_count(
                aggregations["company_response"],
                "company_response",
                "Closed with monetary relief",
            )
            non_monetary = bucket_count(
                aggregations["company_response"],
                "company_response",
                "Closed with non-monetary relief",
            )
            explanation = bucket_count(
                aggregations["company_response"],
                "company_response",
                "Closed with explanation",
            )
            in_progress = bucket_count(
                aggregations["company_response"], "company_response", "In progress"
            )
            untimely_response = bucket_count(
                aggregations["company_response"],
                "company_response",
                "Untimely response",
            )

            monthly_rows.append(
                {
                    "month": str(month),
                    "month_start": month.start_time.date(),
                    "month_end": month.end_time.date(),
                    "complaint_count": total,
                    "timely_yes_count": timely_yes,
                    "timely_no_count": timely_no,
                    "timely_known_count": timely_yes + timely_no,
                    "timely_rate": timely_yes / (timely_yes + timely_no)
                    if timely_yes + timely_no
                    else None,
                    "monetary_relief_count": monetary,
                    "non_monetary_relief_count": non_monetary,
                    "relief_count": monetary + non_monetary,
                    "relief_rate": (monetary + non_monetary) / total if total else None,
                    "closed_explanation_count": explanation,
                    "in_progress_count": in_progress,
                    "in_progress_rate": in_progress / total if total else None,
                    "untimely_response_count": untimely_response,
                    "api_last_updated": meta.get("last_updated"),
                    "api_is_data_stale": meta.get("is_data_stale"),
                    "api_has_data_issue": meta.get("has_data_issue"),
                }
            )

            for dimension in DIMENSIONS:
                rows, other_count, error_bound = buckets(
                    aggregations[dimension], dimension
                )
                for row in rows:
                    dimension_rows.append(
                        {
                            "month": str(month),
                            "dimension": dimension,
                            "value": row["key"],
                            "complaint_count": int(row["doc_count"]),
                            "share_within_month": row["doc_count"] / total
                            if total
                            else None,
                            "sum_other_doc_count": other_count,
                            "doc_count_error_upper_bound": error_bound,
                        }
                    )

            raw_responses.append(
                {
                    "month": str(month),
                    "request": params,
                    "hits_total": payload["hits"]["total"],
                    "aggregations": aggregations,
                    "meta": meta,
                }
            )
            print(f"Downloaded official aggregations for {month}: {total:,}")
            time.sleep(0.05)

    raw_path = RAW_DIR / "cfpb_monthly_aggregates.json"
    raw_path.write_text(
        json.dumps(raw_responses, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    monthly_path = PROCESSED_DIR / "cfpb_monthly_kpis.parquet"
    dimensions_path = PROCESSED_DIR / "cfpb_monthly_dimensions.parquet"
    pd.DataFrame(monthly_rows).to_parquet(monthly_path, index=False)
    pd.DataFrame(dimension_rows).to_parquet(dimensions_path, index=False)

    receipt = {
        "source_name": "CFPB Consumer Complaint Database API aggregations",
        "source_url": API_URL,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "date_received_min": args.start_date,
        "date_received_max": args.end_date,
        "months": len(months),
        "dimensions": list(DIMENSIONS),
        "method": "One exact aggregation query per complete calendar month",
        "files": {
            str(raw_path.relative_to(PROJECT_ROOT)): sha256(raw_path),
            str(monthly_path.relative_to(PROJECT_ROOT)): sha256(monthly_path),
            str(dimensions_path.relative_to(PROJECT_ROOT)): sha256(dimensions_path),
        },
    }
    receipt_path = RAW_DIR / "cfpb_aggregates_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(monthly_rows)} monthly KPI rows")
    print(f"Saved {len(dimension_rows)} monthly dimension rows")


if __name__ == "__main__":
    main()
