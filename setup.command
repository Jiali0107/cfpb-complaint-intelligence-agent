#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
export UV_CACHE_DIR="$PWD/.uv-cache"

if ! command -v uv >/dev/null 2>&1; then
  print "未找到 uv。请先安装：https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

uv sync

# Some macOS setups mark files inside .venv as hidden; Python then skips editable .pth files.
if [[ "$(uname -s)" == "Darwin" ]]; then
  chflags -R nohidden .venv 2>/dev/null || true
fi

uv run project-agent build-db
uv run pytest -q

print "初始化完成。双击 run.command 或运行：uv run streamlit run app.py"
