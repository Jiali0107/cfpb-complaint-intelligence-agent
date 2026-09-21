import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "cfpb_complaints_sample.parquet"
RECEIPT_PATH = ROOT / "data" / "raw" / "cfpb_source_receipt.json"


@pytest.mark.skipif(not DATA_PATH.exists(), reason="CFPB sample has not been downloaded")
def test_downloaded_sample_integrity() -> None:
    frame = pd.read_parquet(DATA_PATH)
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    assert len(frame) == receipt["rows_downloaded"] == 1_000
    assert len(frame.columns) == 15
    assert frame["complaint_id"].notna().all()
    assert frame["complaint_id"].is_unique
    assert frame.duplicated().sum() == 0
    assert receipt["query"]["sampling_strategy"] == "monthly"
    assert len(receipt["sampling_details"]["actual_rows_per_month"]) == 20
    assert set(receipt["sampling_details"]["actual_rows_per_month"].values()) == {50}


@pytest.mark.skipif(not DATA_PATH.exists(), reason="CFPB sample has not been downloaded")
def test_core_field_validity() -> None:
    frame = pd.read_parquet(DATA_PATH)
    received = pd.to_datetime(frame["date_received"], utc=True, errors="coerce")
    sent = pd.to_datetime(frame["date_sent_to_company"], utc=True, errors="coerce")

    assert received.notna().all()
    assert sent.notna().all()
    assert (sent >= received).all()
    assert set(frame["timely"].dropna().unique()) <= {"Yes", "No"}
