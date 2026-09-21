"""Project configuration with secret-safe local environment discovery."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_local_environment() -> Path | None:
    """Load the first local env file without overriding process variables."""
    for candidate in (
        PROJECT_ROOT / ".env.local",
        PROJECT_ROOT / ".env",
        PROJECT_ROOT.parent / ".env.local",
        PROJECT_ROOT.parent / ".env",
    ):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return candidate
    return None


def openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-5-mini")


def fallback_openai_model() -> str:
    return os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4.1-mini")


def require_openai_key() -> None:
    load_local_environment()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY was not found. Put it in the parent .env.local or export it."
        )
