"""Fit interpretable NMF topics and tag the full CFPB narrative corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "cfpb_narratives.parquet"
DEFAULT_TOPICS = ROOT / "data" / "processed" / "cfpb_narrative_topics.parquet"
DEFAULT_MONTHLY = ROOT / "data" / "processed" / "cfpb_narrative_theme_monthly.parquet"
DEFAULT_MODEL_DIR = ROOT / "data" / "vector_store" / "themes"
DEFAULT_REPORT = ROOT / "reports" / "narrative_theme_summary.json"


def stable_key(complaint_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{complaint_id}".encode()).hexdigest()


def clean_topic_text(text: str) -> str:
    """Remove CFPB masking placeholders and numeric-only tokens before topic fitting."""
    text = str(text).lower()
    text = re.sub(r"\b[x]{2,}\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def select_balanced_training(
    frame: pd.DataFrame, training_documents: int, seed: int
) -> pd.DataFrame:
    """Create a deterministic product-balanced sample of unique narratives."""
    if len(frame) <= training_documents:
        return frame.copy()
    groups = list(frame.groupby("product", dropna=False))
    per_product = max(training_documents // max(len(groups), 1), 1)
    selected_parts = []
    selected_ids: set[str] = set()
    for _, group in groups:
        ranked = group.assign(
            _stable=group["complaint_id"].astype(str).map(
                lambda value: stable_key(value, seed)
            )
        ).sort_values("_stable")
        chosen = ranked.head(min(per_product, len(ranked))).drop(columns="_stable")
        selected_parts.append(chosen)
        selected_ids.update(chosen["complaint_id"].astype(str))
    selected = pd.concat(selected_parts, ignore_index=True)
    remaining_slots = training_documents - len(selected)
    if remaining_slots > 0:
        remaining = frame[~frame["complaint_id"].astype(str).isin(selected_ids)].copy()
        remaining["_stable"] = remaining["complaint_id"].astype(str).map(
            lambda value: stable_key(value, seed)
        )
        selected = pd.concat(
            [selected, remaining.sort_values("_stable").head(remaining_slots).drop(columns="_stable")],
            ignore_index=True,
        )
    return selected.head(training_documents).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--topics-output", type=Path, default=DEFAULT_TOPICS)
    parser.add_argument("--monthly-output", type=Path, default=DEFAULT_MONTHLY)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--training-documents", type=int, default=30_000)
    parser.add_argument("--topics", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def topic_labels(model: NMF, feature_names: np.ndarray, top_terms: int = 6) -> list[str]:
    labels = []
    for component in model.components_:
        indexes = component.argsort()[-top_terms:][::-1]
        labels.append(" / ".join(feature_names[indexes]))
    return labels


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    for path in (args.topics_output, args.monthly_output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(
        args.input,
        columns=[
            "complaint_id",
            "narrative",
            "narrative_sha256",
            "month",
            "product",
            "issue",
        ],
    )
    frame = frame.dropna(subset=["narrative"]).drop_duplicates("complaint_id").reset_index(drop=True)
    training_pool = frame.drop_duplicates("narrative_sha256").copy()
    training = select_balanced_training(
        training_pool,
        min(args.training_documents, len(training_pool)),
        args.seed,
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        preprocessor=clean_topic_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    training_matrix = vectorizer.fit_transform(training["narrative"])
    topic_count = min(args.topics, training_matrix.shape[0] - 1, training_matrix.shape[1] - 1)
    model = NMF(
        n_components=topic_count,
        init="nndsvda",
        random_state=args.seed,
        max_iter=250,
        l1_ratio=0.1,
    )
    model.fit(training_matrix)
    labels = topic_labels(model, vectorizer.get_feature_names_out())

    assignments = []
    for start in range(0, len(frame), args.batch_size):
        batch = frame.iloc[start : start + args.batch_size]
        weights = model.transform(vectorizer.transform(batch["narrative"]))
        ids = weights.argmax(axis=1)
        scores = weights.max(axis=1)
        assignments.append(
            pd.DataFrame(
                {
                    "complaint_id": batch["complaint_id"].astype(str).to_numpy(),
                    "month": batch["month"].astype(str).to_numpy(),
                    "product": batch["product"].astype(str).to_numpy(),
                    "issue": batch["issue"].astype(str).to_numpy(),
                    "topic_id": ids.astype(int),
                    "topic_label": [labels[index] for index in ids],
                    "topic_score": scores.astype(float),
                }
            )
        )
        print(f"assigned topics to {min(start + args.batch_size, len(frame)):,}/{len(frame):,}")
    topics = pd.concat(assignments, ignore_index=True)
    topics.to_parquet(args.topics_output, index=False)

    monthly = (
        topics.groupby(["month", "product", "topic_id", "topic_label"], dropna=False)
        .size()
        .reset_index(name="narrative_count")
    )
    totals = (
        topics.groupby(["month", "product"], dropna=False)
        .size()
        .reset_index(name="product_month_narratives")
    )
    monthly = monthly.merge(totals, on=["month", "product"], how="left")
    monthly["share_within_product_month"] = (
        monthly["narrative_count"] / monthly["product_month_narratives"]
    )
    monthly.to_parquet(args.monthly_output, index=False)

    joblib.dump(vectorizer, args.model_dir / "vectorizer.joblib")
    joblib.dump(model, args.model_dir / "nmf.joblib")
    topic_totals = (
        topics.groupby(["topic_id", "topic_label"])
        .size()
        .reset_index(name="narrative_count")
        .sort_values("narrative_count", ascending=False)
    )
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": "TF-IDF + non-negative matrix factorization",
        "scope": "Topics are descriptive clusters, not ground-truth complaint causes.",
        "source_narratives": len(frame),
        "unique_narrative_training_pool": len(training_pool),
        "training_documents": len(training),
        "training_selection": "deterministic product-balanced sample of exact-text-unique narratives",
        "topic_count": topic_count,
        "seed": args.seed,
        "max_features": args.max_features,
        "topics": topic_totals.to_dict(orient="records"),
        "monthly_rows": len(monthly),
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
