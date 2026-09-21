import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from project_agent.retrieval import (
    LocalNarrativeIndex,
    NarrativeSearchFilters,
    chunk_narrative,
)
from project_agent.router import route_question


def test_chunk_narrative_is_deterministic_and_overlapping():
    text = " ".join(f"word{i}" for i in range(120))
    chunks = chunk_narrative(text, max_words=50, overlap_words=10)
    assert len(chunks) == 3
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_question_router():
    assert route_question("2026年投诉量趋势是多少？") == "sql"
    assert route_question("给我几个消费者如何描述身份盗用的案例") == "rag"
    assert route_question("投诉量是否增长，消费者怎么描述这个问题？") == "hybrid"
    assert route_question("叙事投诉中数量最多的三个文本主题是什么？") == "sql"
    assert route_question("给两个案例，不要推断总体频率") == "rag"


def test_local_hybrid_retrieval_and_filter(tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "chunk_id": "1_0",
                "complaint_id": "1",
                "chunk_index": 0,
                "text": "identity theft created fraudulent accounts on my credit report",
                "date_received": "2025-01-01",
                "product": "Credit reporting",
                "issue": "Incorrect information",
                "company": "A",
                "company_response": "Closed with explanation",
                "source_snapshot": "test",
            },
            {
                "chunk_id": "2_0",
                "complaint_id": "2",
                "chunk_index": 0,
                "text": "bank charged an unexpected overdraft fee",
                "date_received": "2025-01-02",
                "product": "Checking account",
                "issue": "Fees",
                "company": "B",
                "company_response": "Closed with relief",
                "source_snapshot": "test",
            },
            {
                "chunk_id": "3_0",
                "complaint_id": "3",
                "chunk_index": 0,
                "text": "debt collector contacted me about a debt I do not owe",
                "date_received": "2025-01-03",
                "product": "Debt collection",
                "issue": "Debt not owed",
                "company": "C",
                "company_response": "Closed with explanation",
                "source_snapshot": "test",
            },
        ]
    )
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(metadata["text"])
    svd = TruncatedSVD(n_components=2, random_state=42).fit(tfidf)
    dense = normalize(svd.transform(tfidf)).astype("float32")
    metadata.to_parquet(tmp_path / "metadata.parquet", index=False)
    np.save(tmp_path / "dense.npy", dense)
    sparse.save_npz(tmp_path / "tfidf.npz", tfidf)
    joblib.dump(vectorizer, tmp_path / "vectorizer.joblib")
    joblib.dump(svd, tmp_path / "svd.joblib")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"indexed_documents": 3, "chunks": 3, "method": "test"})
    )

    index = LocalNarrativeIndex(tmp_path)
    result = index.search("fraudulent credit identity", top_k=2, min_score=0)
    assert result["results"][0]["complaint_id"] == "1"
    assert "Complaint ID 1" in result["results"][0]["citation"]

    filtered = index.search(
        "fee",
        filters=NarrativeSearchFilters(product="Checking account"),
        top_k=3,
        min_score=0,
    )
    assert filtered["candidate_count"] == 1
    assert filtered["results"][0]["complaint_id"] == "2"

    unavailable_topic = index.search(
        "identity theft",
        filters=NarrativeSearchFilters(topic_id=10),
        top_k=3,
        min_score=0,
    )
    assert unavailable_topic["candidate_count"] == 0
    assert unavailable_topic["results"] == []
