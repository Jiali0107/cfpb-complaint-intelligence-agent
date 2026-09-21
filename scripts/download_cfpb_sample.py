"""Download a bounded, reproducible sample from the official CFPB API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default="2026-08-31")
    parser.add_argument("--limit", type=int, default=1_000)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--sampling-strategy",
        choices=("monthly", "latest"),
        default="monthly",
        help="monthly spans the period; latest stress-tests CFPB deep pagination",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_latest_records(
    start_date: str, end_date: str, limit: int, page_size: int
) -> tuple[list[dict], dict, dict]:
    if not 1 <= page_size <= 100:
        raise ValueError("CFPB API page size must be between 1 and 100")
    if not 1 <= limit <= 100_000:
        raise ValueError("limit must be between 1 and 100,000")

    records: list[dict] = []
    api_meta: dict = {}
    pagination_breakpoints: dict = {}
    page_number = 1
    search_after: str | None = None
    # CFPB's edge layer currently rejects a custom Python-style user agent with
    # HTTP 403 while the same public endpoint accepts a standard curl client.
    # Keep the request read-only and identify a compatible generic HTTP client.
    headers = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}

    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        while len(records) < limit:
            current_size = min(page_size, limit - len(records))
            params: dict[str, str | int] = {
                "date_received_min": start_date,
                "date_received_max": end_date,
                "size": current_size,
                # CFPB uses `frm` on the first call to pre-compute the
                # break_points needed for deep pagination; it does not behave
                # like a conventional row offset by itself.
                "frm": max(limit - current_size, 0) if page_number == 1 else current_size,
                "sort": "created_date_desc",
                "no_aggs": "true",
                "no_highlight": "true",
            }
            if page_number > 1 and search_after:
                params["page"] = page_number
                params["search_after"] = search_after
            response = client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break

            current_meta = payload.get("_meta", {})
            if page_number == 1:
                api_meta = current_meta
                pagination_breakpoints = current_meta.get("break_points", {})
            records.extend(hit["_source"] for hit in hits)
            print(f"Downloaded {len(records):,}/{limit:,} records", flush=True)
            if len(hits) < current_size:
                break
            page_number += 1
            next_breakpoint = pagination_breakpoints.get(str(page_number))
            if not next_breakpoint or len(next_breakpoint) != 2:
                raise RuntimeError(
                    f"CFPB API did not return a breakpoint for page {page_number}"
                )
            search_after = f"{next_breakpoint[0]}_{next_breakpoint[1]}"
            time.sleep(0.05)

    return records, api_meta, {"strategy": "latest", "pages": page_number}


def fetch_monthly_records(
    start_date: str, end_date: str, limit: int
) -> tuple[list[dict], dict, dict]:
    """Take an equal-sized newest-record slice from each calendar month."""
    months = pd.period_range(start=start_date, end=end_date, freq="M")
    if len(months) == 0:
        raise ValueError("The requested date range contains no calendar month")
    per_month = math.ceil(limit / len(months))
    if per_month > 100:
        raise ValueError(
            "Monthly strategy supports at most 100 records per month; "
            "reduce --limit or use --sampling-strategy latest"
        )

    records: list[dict] = []
    api_meta: dict = {}
    monthly_counts: dict[str, int] = {}
    headers = {"User-Agent": "curl/8.7.1", "Accept": "*/*"}
    with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
        for month in months:
            if len(records) >= limit:
                break
            current_size = min(per_month, limit - len(records))
            params = {
                "date_received_min": month.start_time.date().isoformat(),
                "date_received_max": month.end_time.date().isoformat(),
                "size": current_size,
                "frm": 0,
                "sort": "created_date_desc",
                "no_aggs": "true",
                "no_highlight": "true",
            }
            response = client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()
            hits = payload.get("hits", {}).get("hits", [])
            if not api_meta:
                api_meta = payload.get("_meta", {})
            month_records = [hit["_source"] for hit in hits]
            records.extend(month_records)
            monthly_counts[str(month)] = len(month_records)
            print(
                f"Downloaded {len(records):,}/{limit:,} records "
                f"through {month}",
                flush=True,
            )
            time.sleep(0.05)

    return records, api_meta, {
        "strategy": "monthly",
        "months": [str(month) for month in months],
        "target_rows_per_month": per_month,
        "actual_rows_per_month": monthly_counts,
        "within_month_order": "created_date_desc",
    }


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if args.sampling_strategy == "monthly":
        records, api_meta, sampling_details = fetch_monthly_records(
            args.start_date, args.end_date, args.limit
        )
    else:
        records, api_meta, sampling_details = fetch_latest_records(
            args.start_date, args.end_date, args.limit, args.page_size
        )
    if not records:
        raise RuntimeError("CFPB API returned no records for the requested window")

    raw_path = RAW_DIR / "cfpb_complaints_sample.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    frame = pd.DataFrame.from_records(records)
    if "complaint_id" in frame:
        frame["complaint_id"] = frame["complaint_id"].astype("string")
    for column in ("date_received", "date_sent_to_company"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")

    duplicate_ids = int(frame["complaint_id"].duplicated(keep=False).sum())
    if duplicate_ids:
        raise RuntimeError(
            "Pagination integrity check failed: "
            f"{duplicate_ids} rows have duplicated complaint_id values"
        )

    parquet_path = PROCESSED_DIR / "cfpb_complaints_sample.parquet"
    preview_path = PROCESSED_DIR / "cfpb_complaints_preview.csv"
    frame.to_parquet(parquet_path, index=False)
    frame.head(200).to_csv(preview_path, index=False)

    receipt = {
        "source_name": "CFPB Consumer Complaint Database API",
        "source_url": API_URL,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "query": {
            "date_received_min": args.start_date,
            "date_received_max": args.end_date,
            "sort": "created_date_desc",
            "limit": args.limit,
            "page_size": args.page_size,
            "sampling_strategy": args.sampling_strategy,
        },
        "rows_downloaded": len(frame),
        "sampling_details": sampling_details,
        "duplicate_complaint_id_rows": duplicate_ids,
        "columns_downloaded": list(frame.columns),
        "api_meta": api_meta,
        "files": {
            str(raw_path.relative_to(PROJECT_ROOT)): sha256(raw_path),
            str(parquet_path.relative_to(PROJECT_ROOT)): sha256(parquet_path),
            str(preview_path.relative_to(PROJECT_ROOT)): sha256(preview_path),
        },
    }
    receipt_path = RAW_DIR / "cfpb_source_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(frame):,} rows and {len(frame.columns)} columns")
    print(f"Receipt: {receipt_path}")


if __name__ == "__main__":
    main()
