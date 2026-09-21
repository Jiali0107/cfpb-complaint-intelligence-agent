"""Run a small auditable live evaluation of the SQL Agent."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from project_agent.agent import ask


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    cases = json.loads((ROOT / "evaluation" / "questions.json").read_text())[: args.limit]
    results = []
    for case in cases:
        result = ask(case["question"])
        tool_events = [e for e in result["trace"] if e["event"] == "tool"]
        sql_events = [e for e in tool_events if e["name"] == "run_safe_sql"]
        used_tables = {
            table
            for event in sql_events
            for table in case.get("expected_tables", [])
            if table in event.get("arguments", {}).get("sql", "")
        }
        expected_numbers = case.get("expected_numbers", [])
        called_tools = {event["name"] for event in tool_events}
        sql_success = all(e["status"] == "ok" for e in sql_events) and (
            bool(sql_events) or not case.get("require_sql", True)
        )
        required_tools_ok = set(case.get("expected_tools", [])).issubset(called_tools)
        results.append(
            {
                "id": case["id"],
                "answer": result["answer"],
                "tool_use": bool(tool_events),
                "sql_success": sql_success,
                "expected_table_coverage": len(used_tables) == len(case.get("expected_tables", [])),
                "expected_tool_coverage": required_tools_ok,
                "expected_number_coverage": all(n in result["answer"] for n in expected_numbers),
                "trace": result["trace"],
            }
        )
        print(f"evaluated {case['id']}")
    summary = {
        "evaluated_at_utc": datetime.now(UTC).isoformat(),
        "cases": len(results),
        "tool_use_rate": sum(r["tool_use"] for r in results) / len(results),
        "sql_success_rate": sum(r["sql_success"] for r in results) / len(results),
        "table_coverage_rate": sum(r["expected_table_coverage"] for r in results) / len(results),
        "tool_coverage_rate": sum(r["expected_tool_coverage"] for r in results) / len(results),
        "number_coverage_rate": sum(r["expected_number_coverage"] for r in results) / len(results),
        "results": results,
    }
    out = ROOT / "evaluation" / "latest_results.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
