"""Run a small live SQL/RAG/hybrid orchestration evaluation."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from project_agent.agent import ask


ROOT = Path(__file__).resolve().parents[1]
CITATION_RE = re.compile(r"Complaint\s*ID\s*[:：#]?\s*\d+", re.I)


def main() -> None:
    cases = json.loads(
        (ROOT / "evaluation" / "hybrid_questions.json").read_text(encoding="utf-8")
    )
    results = []
    for case in cases:
        result = ask(case["question"])
        tool_events = [event for event in result["trace"] if event["event"] == "tool"]
        called_tools = {event["name"] for event in tool_events if event["status"] == "ok"}
        sql_text = " ".join(
            event.get("arguments", {}).get("sql", "")
            for event in tool_events
            if event["name"] == "run_safe_sql"
        )
        citations = CITATION_RE.findall(result["answer"])
        results.append(
            {
                "id": case["id"],
                "route": result["route"],
                "route_ok": result["route"] == case["expected_route"],
                "tools_ok": set(case["expected_tools"]).issubset(called_tools),
                "tables_ok": all(table in sql_text for table in case["expected_tables"]),
                "citation_count": len(citations),
                "citations_ok": len(citations) >= case["minimum_citations"],
                "tool_errors": [
                    event for event in tool_events if event["status"] != "ok"
                ],
                "answer": result["answer"],
                "trace": result["trace"],
            }
        )
        print(f"evaluated {case['id']}")
    summary = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "Three live orchestration cases; a smoke evaluation, not general model accuracy.",
        "cases": len(results),
        "route_accuracy": sum(item["route_ok"] for item in results) / len(results),
        "tool_coverage": sum(item["tools_ok"] for item in results) / len(results),
        "table_coverage": sum(item["tables_ok"] for item in results) / len(results),
        "citation_coverage": sum(item["citations_ok"] for item in results) / len(results),
        "tool_error_count": sum(len(item["tool_errors"]) for item in results),
        "results": results,
    }
    output = ROOT / "evaluation" / "latest_hybrid_results.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
