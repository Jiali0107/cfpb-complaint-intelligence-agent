"""Command-line entry point for the local analytics Agent."""

from __future__ import annotations

import argparse
import json

from .agent import ask
from .analytics import database_schema, run_safe_sql
from .database import build_database, database_summary
from .retrieval import search_narratives


def main() -> None:
    parser = argparse.ArgumentParser(prog="project-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-db")
    sub.add_parser("schema")
    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("question")
    sql_parser = sub.add_parser("sql")
    sql_parser.add_argument("query")
    search_parser = sub.add_parser("search")
    search_parser.add_argument("query")
    search_parser.add_argument("--product")
    search_parser.add_argument("--issue")
    search_parser.add_argument("--company")
    search_parser.add_argument("--date-from")
    search_parser.add_argument("--date-to")
    search_parser.add_argument("--topic-id", type=int)
    search_parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    if args.command == "build-db":
        path = build_database()
        payload = {"path": str(path), **database_summary(path)}
    elif args.command == "schema":
        payload = database_schema()
    elif args.command == "sql":
        payload = run_safe_sql(args.query)
    elif args.command == "search":
        payload = search_narratives(
            query=args.query,
            product=args.product,
            issue=args.issue,
            company=args.company,
            date_from=args.date_from,
            date_to=args.date_to,
            topic_id=args.topic_id,
            top_k=args.top_k,
        )
    else:
        payload = ask(args.question)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
