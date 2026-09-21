"""Prepare a privacy-aware CFPB narrative corpus from an official ZIP export."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "cfpb_narratives.parquet"
DEFAULT_RECEIPT = ROOT / "data" / "raw" / "cfpb_narratives_receipt.json"

COLUMN_MAP = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Complaint ID": "complaint_id",
}

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}(?!\d)")
SSN_RE = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
LONG_NUMBER_RE = re.compile(r"(?<!\d)\d{9,}(?!\d)")
WHITESPACE_RE = re.compile(r"\s+")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_narrative(text: str) -> tuple[str, bool]:
    """Apply a conservative second pass for obvious residual identifiers."""
    normalized = WHITESPACE_RE.sub(" ", str(text or "")).strip()
    sanitized = EMAIL_RE.sub("[REDACTED_EMAIL]", normalized)
    sanitized = PHONE_RE.sub("[REDACTED_PHONE]", sanitized)
    sanitized = SSN_RE.sub("[REDACTED_SSN]", sanitized)
    sanitized = LONG_NUMBER_RE.sub("[REDACTED_LONG_NUMBER]", sanitized)
    return sanitized, sanitized != normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-snapshot", required=True)
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--chunksize", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.archive) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in archive, found {csv_names}")
        csv_name = csv_names[0]

    writer: pq.ParquetWriter | None = None
    seen_ids: set[str] = set()
    source_rows = narrative_rows = written_rows = duplicate_ids = 0
    short_rows = residual_pii_rows = exact_duplicate_texts = 0
    narrative_hashes: set[str] = set()
    date_min: str | None = None
    date_max: str | None = None
    product_counts: dict[str, int] = {}

    try:
        for chunk in pd.read_csv(
            args.archive,
            compression="zip",
            chunksize=args.chunksize,
            dtype=str,
            keep_default_na=False,
        ):
            source_rows += len(chunk)
            missing = set(COLUMN_MAP) - set(chunk.columns)
            if missing:
                raise ValueError(f"Archive is missing expected columns: {sorted(missing)}")
            frame = chunk[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()
            frame["narrative"] = frame["narrative"].str.strip()
            frame = frame[frame["narrative"] != ""].copy()
            narrative_rows += len(frame)

            sanitized = frame["narrative"].map(sanitize_narrative)
            frame["narrative"] = sanitized.map(lambda item: item[0])
            frame["secondary_redaction_applied"] = sanitized.map(lambda item: item[1])
            residual_pii_rows += int(frame["secondary_redaction_applied"].sum())
            frame["narrative_char_count"] = frame["narrative"].str.len()
            frame["narrative_word_count"] = frame["narrative"].str.split().str.len()
            short_rows += int((frame["narrative_char_count"] < args.min_chars).sum())
            frame = frame[frame["narrative_char_count"] >= args.min_chars].copy()

            frame["complaint_id"] = frame["complaint_id"].astype(str).str.strip()
            valid_id = frame["complaint_id"] != ""
            frame = frame[valid_id].copy()
            duplicate_mask = frame["complaint_id"].isin(seen_ids) | frame["complaint_id"].duplicated()
            duplicate_ids += int(duplicate_mask.sum())
            frame = frame[~duplicate_mask].copy()
            seen_ids.update(frame["complaint_id"].tolist())

            frame["narrative_sha256"] = frame["narrative"].map(
                lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
            )
            for text_hash in frame["narrative_sha256"]:
                if text_hash in narrative_hashes:
                    exact_duplicate_texts += 1
                narrative_hashes.add(text_hash)

            frame["date_received"] = pd.to_datetime(
                frame["date_received"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            frame["date_sent_to_company"] = pd.to_datetime(
                frame["date_sent_to_company"], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            frame["month"] = frame["date_received"].str.slice(0, 7)
            frame["source_snapshot"] = args.source_snapshot
            frame["source_url"] = args.source_url

            observed_dates = frame["date_received"].dropna()
            if not observed_dates.empty:
                local_min = observed_dates.min()
                local_max = observed_dates.max()
                date_min = local_min if date_min is None else min(date_min, local_min)
                date_max = local_max if date_max is None else max(date_max, local_max)
            for product, count in frame["product"].value_counts().items():
                product_counts[product] = product_counts.get(product, 0) + int(count)

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema, compression="zstd")
            writer.write_table(table)
            written_rows += len(frame)
            print(f"processed source rows={source_rows:,}; kept narratives={written_rows:,}")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError("No usable narratives were found")

    receipt = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_url": args.source_url,
        "source_snapshot": args.source_snapshot,
        "source_archive": args.archive.name,
        "source_csv": csv_name,
        "source_archive_sha256": sha256(args.archive),
        "source_rows": source_rows,
        "non_empty_narrative_rows": narrative_rows,
        "written_rows": written_rows,
        "minimum_narrative_characters": args.min_chars,
        "short_narratives_excluded": short_rows,
        "duplicate_complaint_ids_excluded": duplicate_ids,
        "exact_duplicate_narrative_rows_observed": exact_duplicate_texts,
        "secondary_redaction_rows": residual_pii_rows,
        "date_received_min": date_min,
        "date_received_max": date_max,
        "top_products": sorted(
            product_counts.items(), key=lambda item: item[1], reverse=True
        )[:20],
        "output": str(args.output.relative_to(ROOT)),
        "output_sha256": sha256(args.output),
        "privacy_note": (
            "CFPB publishes opted-in, scrubbed narratives. This pipeline applies only "
            "a conservative second scan for obvious residual identifiers; it is not a "
            "guarantee of complete de-identification."
        ),
    }
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
