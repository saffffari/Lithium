#!/usr/bin/env bash
# 3Photon launcher (Linux/macOS). Uses this repo's .venv python and runs
# from the repo root so `src` resolves; works for worktrees too (each
# worktree's own run.sh resolves its own root).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

PY="$HERE/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[run.sh] no .venv at $PY — create it with:" >&2
  echo "         uv venv .venv --python 3.12 && uv pip install -r requirements.txt" >&2
  echo "         (falling back to system python3)" >&2
  PY="$(command -v python3)"
fi

exec "$PY" -m src.main "$@"
