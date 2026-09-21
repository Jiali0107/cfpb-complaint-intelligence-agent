"""Run a deterministic weak-label smoke evaluation of local narrative retrieval."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from project_agent.retrieval import NarrativeSearchFilters, local_index


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation" / "rag_questions.json").read_text())
    index = local_index()
    results = []
    for case in cases:
        result = index.search(
            case["query"],
            filters=NarrativeSearchFilters(product=case["product"]),
            top_k=args.top_k,
            min_score=0,
        )
        texts = " ".join(item["excerpt"].casefold() for item in result["results"])
        terms = [term.casefold() for term in case["expected_terms"]]
        term_hits = {term: term in texts for term in terms}
        filter_ok = bool(result["results"]) and all(
            item["product"] == case["product"] for item in result["results"]
        )
        citations_ok = bool(result["results"]) and all(
            item["complaint_id"] and item["source_snapshot"] and item["citation"]
            for item in result["results"]
        )
        results.append(
            {
                "id": case["id"],
                "result_count": result["result_count"],
                "filter_ok": filter_ok,
                "citations_ok": citations_ok,
                "expected_term_hits": term_hits,
                "term_coverage": sum(term_hits.values()) / len(term_hits),
                "top_score": result["results"][0]["score"] if result["results"] else None,
            }
        )

    impossible = index.search(
        "complaint",
        filters=NarrativeSearchFilters(product="__NO_SUCH_PRODUCT__"),
        top_k=3,
        min_score=0,
    )
    summary = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "Five weak-label retrieval smoke cases plus one impossible-filter case; "
            "not a general RAG accuracy claim."
        ),
        "cases": len(results),
        "filter_accuracy": sum(item["filter_ok"] for item in results) / len(results),
        "citation_completeness": sum(item["citations_ok"] for item in results) / len(results),
        "mean_expected_term_coverage": sum(item["term_coverage"] for item in results)
        / len(results),
        "impossible_filter_returns_zero": impossible["result_count"] == 0,
        "results": results,
    }
    out = ROOT / "evaluation" / "latest_rag_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
