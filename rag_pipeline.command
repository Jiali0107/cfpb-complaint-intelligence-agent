#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
export UV_CACHE_DIR="$PWD/.uv-cache"

if [[ $# -ne 1 ]]; then
  print "用法：./rag_pipeline.command path/to/cfpb-narratives.zip"
  exit 1
fi

archive_path="$1"
if [[ ! -f "$archive_path" ]]; then
  print "找不到 Narrative ZIP：$archive_path"
  exit 1
fi

source_url="https://files.consumerfinance.gov/f/documents/CCDB_Export_9_January_2025_through_February_2025.zip"
source_snapshot="CFPB CCDB Export: 9 January 2025 through February 2025"

uv sync --locked
uv run python scripts/prepare_cfpb_narratives.py \
  --archive "$archive_path" \
  --source-url "$source_url" \
  --source-snapshot "$source_snapshot"
uv run python scripts/build_narrative_index.py \
  --max-documents 20000 --max-features 30000 --dimensions 128 --seed 42
uv run python scripts/build_narrative_themes.py \
  --training-documents 30000 --topics 12 --max-features 20000 --batch-size 10000 --seed 42
uv run python scripts/build_narrative_index.py \
  --max-documents 20000 --max-features 30000 --dimensions 128 --seed 42
uv run project-agent build-db
uv run pytest -q
uv run python scripts/evaluate_rag.py

print "三阶段 RAG 构建完成。在线编排评测需显式运行：uv run python scripts/evaluate_hybrid_agent.py"
