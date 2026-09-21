#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
export UV_CACHE_DIR="$PWD/.uv-cache"

if [[ "$(uname -s)" == "Darwin" && -d .venv ]]; then
  chflags -R nohidden .venv 2>/dev/null || true
fi

if [[ ! -f data/warehouse/cfpb.duckdb ]]; then
  uv run project-agent build-db
fi

uv run streamlit run app.py
