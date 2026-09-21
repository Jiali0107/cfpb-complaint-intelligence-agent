"""Deterministic first-pass routing for SQL, RAG, and hybrid questions."""

from __future__ import annotations

import re


SQL_PATTERNS = (
    r"多少|数量|占比|比例|趋势|最高|最低|排名|同比|环比|增长|下降|响应率|救济率|及时率|投诉率|按月|top\s*\d+",
    r"how many|count|rate|percentage|trend|highest|lowest|increase|decrease|by month",
)
RAG_PATTERNS = (
    r"案例|原文|类似投诉|如何描述|怎么描述|抱怨什么|经历|消费者怎么说|代表性投诉",
    r"example|narrative|similar complaint|describe|what consumers say|experience|excerpt",
)


def route_question(question: str) -> str:
    text = str(question or "").casefold()
    wants_sql = any(re.search(pattern, text, re.I) for pattern in SQL_PATTERNS)
    wants_rag = any(re.search(pattern, text, re.I) for pattern in RAG_PATTERNS)
    if wants_sql and wants_rag:
        return "hybrid"
    if wants_rag:
        return "rag"
    return "sql"
