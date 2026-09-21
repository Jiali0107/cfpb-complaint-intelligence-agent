"""Local, auditable hybrid retrieval for CFPB complaint narratives.

The index deliberately stays on disk. TF-IDF provides lexical matching while
Truncated SVD supplies dense latent-semantic vectors. Retrieval blends both
scores and applies metadata filters before ranking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import sparse

from .config import PROJECT_ROOT


DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "vector_store" / "narratives"
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace without altering the meaning of a narrative."""
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def chunk_narrative(
    text: str, max_words: int = 320, overlap_words: int = 48
) -> list[str]:
    """Split a narrative into deterministic word windows.

    CFPB narratives are usually short, so texts within the limit remain one
    complaint-level document. Long narratives retain a small overlap so a
    sentence near a boundary remains retrievable.
    """
    if max_words < 50:
        raise ValueError("max_words must be at least 50")
    if not 0 <= overlap_words < max_words:
        raise ValueError("overlap_words must be between 0 and max_words - 1")
    words = normalize_text(text).split()
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    step = max_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start : start + max_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        if start + max_words >= len(words):
            break
    return chunks


@dataclass(frozen=True)
class NarrativeSearchFilters:
    product: str | None = None
    issue: str | None = None
    company: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    topic_id: int | None = None


class LocalNarrativeIndex:
    """Load and query a persisted local hybrid vector index."""

    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR):
        self.index_dir = Path(index_dir)
        required = {
            "manifest": self.index_dir / "manifest.json",
            "metadata": self.index_dir / "metadata.parquet",
            "dense": self.index_dir / "dense.npy",
            "tfidf": self.index_dir / "tfidf.npz",
            "vectorizer": self.index_dir / "vectorizer.joblib",
            "svd": self.index_dir / "svd.joblib",
        }
        missing = [str(path) for path in required.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Narrative index is incomplete. Run scripts/build_narrative_index.py. "
                f"Missing: {missing}"
            )
        self.manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
        self.metadata = pd.read_parquet(required["metadata"])
        self.dense = np.load(required["dense"], mmap_mode="r")
        self.tfidf = sparse.load_npz(required["tfidf"])
        self.vectorizer = joblib.load(required["vectorizer"])
        self.svd = joblib.load(required["svd"])
        expected = len(self.metadata)
        if self.dense.shape[0] != expected or self.tfidf.shape[0] != expected:
            raise ValueError("Narrative index row counts do not reconcile")

    def _mask(self, filters: NarrativeSearchFilters) -> np.ndarray:
        mask = np.ones(len(self.metadata), dtype=bool)
        for column, value in (
            ("product", filters.product),
            ("issue", filters.issue),
            ("company", filters.company),
        ):
            if value:
                mask &= self.metadata[column].fillna("").str.casefold().to_numpy() == value.casefold()
        dates = self.metadata["date_received"].fillna("").astype(str).to_numpy()
        if filters.date_from:
            mask &= dates >= filters.date_from
        if filters.date_to:
            mask &= dates <= filters.date_to
        if filters.topic_id is not None:
            if "topic_id" not in self.metadata:
                return np.zeros(len(self.metadata), dtype=bool)
            mask &= self.metadata["topic_id"].fillna(-1).astype(int).to_numpy() == filters.topic_id
        return mask

    def search(
        self,
        query: str,
        *,
        filters: NarrativeSearchFilters | None = None,
        top_k: int = 8,
        min_score: float = 0.05,
        dense_weight: float = 0.65,
    ) -> dict[str, Any]:
        query = normalize_text(query)
        if not query:
            raise ValueError("Retrieval query is empty")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not 0 <= dense_weight <= 1:
            raise ValueError("dense_weight must be between 0 and 1")

        query_sparse = self.vectorizer.transform([query])
        query_dense = self.svd.transform(query_sparse).astype("float32", copy=False)
        norm = np.linalg.norm(query_dense, axis=1, keepdims=True)
        query_dense = query_dense / np.maximum(norm, 1e-12)

        dense_scores = np.asarray(self.dense @ query_dense[0]).reshape(-1)
        dense_scores = np.clip(dense_scores, 0.0, 1.0)
        sparse_scores = (self.tfidf @ query_sparse.T).toarray().reshape(-1)
        hybrid_scores = dense_weight * dense_scores + (1 - dense_weight) * sparse_scores

        active_filters = filters or NarrativeSearchFilters()
        mask = self._mask(active_filters)
        hybrid_scores = np.where(mask, hybrid_scores, -1.0)
        candidate_count = int(mask.sum())
        if not candidate_count:
            return {
                "query": query,
                "candidate_count": 0,
                "result_count": 0,
                "results": [],
                "index_manifest": self.manifest,
            }

        k = min(top_k, candidate_count)
        indexes = np.argpartition(-hybrid_scores, k - 1)[:k]
        indexes = indexes[np.argsort(-hybrid_scores[indexes])]
        results = []
        for rank, index in enumerate(indexes, start=1):
            score = float(hybrid_scores[index])
            if score < min_score:
                continue
            row = self.metadata.iloc[int(index)]
            results.append(
                {
                    "rank": rank,
                    "score": round(score, 6),
                    "dense_score": round(float(dense_scores[index]), 6),
                    "lexical_score": round(float(sparse_scores[index]), 6),
                    "complaint_id": str(row["complaint_id"]),
                    "chunk_id": str(row["chunk_id"]),
                    "excerpt": str(row["text"])[:1200],
                    "date_received": str(row.get("date_received", "")),
                    "product": str(row.get("product", "")),
                    "issue": str(row.get("issue", "")),
                    "company": str(row.get("company", "")),
                    "company_response": str(row.get("company_response", "")),
                    "topic_id": (
                        int(row["topic_id"])
                        if "topic_id" in row and pd.notna(row["topic_id"])
                        else None
                    ),
                    "topic_label": str(row.get("topic_label", "")),
                    "source_snapshot": str(row.get("source_snapshot", "")),
                    "citation": (
                        f"Complaint ID {row['complaint_id']} — "
                        f"{row.get('source_snapshot', '')}"
                    ),
                }
            )
        return {
            "query": query,
            "candidate_count": candidate_count,
            "result_count": len(results),
            "results": results,
            "index": {
                "documents": self.manifest.get("indexed_documents"),
                "chunks": self.manifest.get("chunks"),
                "method": self.manifest.get("method"),
            },
        }

    def get_complaint(self, complaint_id: str) -> dict[str, Any]:
        rows = self.metadata[
            self.metadata["complaint_id"].astype(str) == str(complaint_id)
        ].sort_values("chunk_index")
        if rows.empty:
            return {"complaint_id": str(complaint_id), "found": False, "chunks": []}
        first = rows.iloc[0]
        return {
            "complaint_id": str(complaint_id),
            "found": True,
            "date_received": str(first.get("date_received", "")),
            "product": str(first.get("product", "")),
            "issue": str(first.get("issue", "")),
            "company": str(first.get("company", "")),
            "source_snapshot": str(first.get("source_snapshot", "")),
            "chunks": rows["text"].astype(str).tolist(),
        }


_INDEX_CACHE: LocalNarrativeIndex | None = None


def local_index() -> LocalNarrativeIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = LocalNarrativeIndex()
    return _INDEX_CACHE


def search_narratives(**kwargs) -> dict[str, Any]:
    filter_names = {"product", "issue", "company", "date_from", "date_to", "topic_id"}
    filter_values = {key: kwargs.pop(key, None) for key in filter_names}
    return local_index().search(
        filters=NarrativeSearchFilters(**filter_values),
        **kwargs,
    )


def get_narrative(complaint_id: str) -> dict[str, Any]:
    return local_index().get_complaint(complaint_id)
