"""Build a deterministic local TF-IDF + SVD hybrid narrative index."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from project_agent.retrieval import chunk_narrative


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "processed" / "cfpb_narratives.parquet"
DEFAULT_TOPICS = ROOT / "data" / "processed" / "cfpb_narrative_topics.parquet"
DEFAULT_OUTPUT = ROOT / "data" / "vector_store" / "narratives"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(complaint_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{complaint_id}".encode()).hexdigest()


def select_documents(frame: pd.DataFrame, max_documents: int, seed: int) -> pd.DataFrame:
    """Select a reproducible, product-balanced retrieval corpus."""
    if len(frame) <= max_documents:
        return frame.copy()
    groups = list(frame.groupby("product", dropna=False))
    base = max(max_documents // max(len(groups), 1), 1)
    selected_parts = []
    selected_ids: set[str] = set()
    for _, group in groups:
        ranked = group.assign(
            _stable=group["complaint_id"].astype(str).map(lambda value: stable_key(value, seed))
        ).sort_values("_stable")
        chosen = ranked.head(min(base, len(ranked))).drop(columns="_stable")
        selected_parts.append(chosen)
        selected_ids.update(chosen["complaint_id"].astype(str))

    selected = pd.concat(selected_parts, ignore_index=True)
    remaining_slots = max_documents - len(selected)
    if remaining_slots > 0:
        remaining = frame[~frame["complaint_id"].astype(str).isin(selected_ids)].copy()
        remaining["_stable"] = remaining["complaint_id"].astype(str).map(
            lambda value: stable_key(value, seed)
        )
        selected = pd.concat(
            [selected, remaining.sort_values("_stable").head(remaining_slots).drop(columns="_stable")],
            ignore_index=True,
        )
    if len(selected) > max_documents:
        selected["_stable"] = selected["complaint_id"].astype(str).map(
            lambda value: stable_key(value, seed)
        )
        selected = selected.sort_values("_stable").head(max_documents).drop(columns="_stable")
    return selected.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-documents", type=int, default=20_000)
    parser.add_argument("--max-words", type=int, default=320)
    parser.add_argument("--overlap-words", type=int, default=48)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    columns = [
        "complaint_id",
        "narrative",
        "date_received",
        "month",
        "product",
        "issue",
        "sub_issue",
        "company",
        "company_response",
        "source_snapshot",
        "narrative_sha256",
    ]
    source = pd.read_parquet(args.input, columns=columns)
    source = source.drop_duplicates("complaint_id")
    source = source.drop_duplicates("narrative_sha256")
    topic_columns: list[str] = []
    if DEFAULT_TOPICS.exists():
        topics = pd.read_parquet(
            DEFAULT_TOPICS,
            columns=["complaint_id", "topic_id", "topic_label", "topic_score"],
        ).drop_duplicates("complaint_id")
        source["complaint_id"] = source["complaint_id"].astype(str)
        topics["complaint_id"] = topics["complaint_id"].astype(str)
        source = source.merge(topics, on="complaint_id", how="left", validate="one_to_one")
        topic_columns = ["topic_id", "topic_label", "topic_score"]
    selected = select_documents(source, args.max_documents, args.seed)

    chunks = []
    for row in selected.itertuples(index=False):
        pieces = chunk_narrative(
            row.narrative,
            max_words=args.max_words,
            overlap_words=args.overlap_words,
        )
        for index, text in enumerate(pieces):
            record = {
                    "chunk_id": f"{row.complaint_id}_{index}",
                    "complaint_id": str(row.complaint_id),
                    "chunk_index": index,
                    "text": text,
                    "date_received": str(row.date_received),
                    "month": str(row.month),
                    "product": str(row.product),
                    "issue": str(row.issue),
                    "sub_issue": str(row.sub_issue),
                    "company": str(row.company),
                    "company_response": str(row.company_response),
                    "source_snapshot": str(row.source_snapshot),
                }
            for column in topic_columns:
                record[column] = getattr(row, column)
            chunks.append(record)
    metadata = pd.DataFrame(chunks)
    if metadata.empty:
        raise RuntimeError("No chunks were produced")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=args.max_features,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf = vectorizer.fit_transform(metadata["text"])
    dimensions = min(args.dimensions, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if dimensions < 2:
        raise RuntimeError(f"Corpus is too small for SVD: {tfidf.shape}")
    svd = TruncatedSVD(n_components=dimensions, random_state=args.seed, n_iter=7)
    dense = normalize(svd.fit_transform(tfidf), norm="l2").astype("float32")

    metadata_path = args.output_dir / "metadata.parquet"
    dense_path = args.output_dir / "dense.npy"
    tfidf_path = args.output_dir / "tfidf.npz"
    vectorizer_path = args.output_dir / "vectorizer.joblib"
    svd_path = args.output_dir / "svd.joblib"
    metadata.to_parquet(metadata_path, index=False)
    np.save(dense_path, dense)
    sparse.save_npz(tfidf_path, tfidf, compressed=True)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(svd, svd_path)

    product_counts = (
        selected["product"].value_counts().rename_axis("product").reset_index(name="documents")
    )
    manifest = {
        "built_at_utc": datetime.now(UTC).isoformat(),
        "method": "hybrid TF-IDF cosine + Truncated SVD cosine",
        "source": str(args.input.relative_to(ROOT)),
        "source_sha256": sha256(args.input),
        "source_narratives": len(source),
        "selection": "deterministic product-balanced sample; exact text duplicates removed",
        "seed": args.seed,
        "indexed_documents": int(metadata["complaint_id"].nunique()),
        "chunks": len(metadata),
        "max_words": args.max_words,
        "overlap_words": args.overlap_words,
        "tfidf_features": int(tfidf.shape[1]),
        "svd_dimensions": int(dimensions),
        "svd_explained_variance_ratio": float(svd.explained_variance_ratio_.sum()),
        "default_dense_weight": 0.65,
        "topic_metadata_included": bool(topic_columns),
        "product_document_counts": product_counts.to_dict(orient="records"),
        "artifacts": {
            path.name: sha256(path)
            for path in (metadata_path, dense_path, tfidf_path, vectorizer_path, svd_path)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
