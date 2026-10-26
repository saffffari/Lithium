#!/usr/bin/env bash
# One-command Lithium setup for annotators (Linux / macOS).
#
#   ./setup.sh                      install + launch
#   ./setup.sh --kit path/to/x.li   also import a project bundle first
#   ./setup.sh --no-launch          install only
#
# Installs uv if missing, creates .venv on Python 3.12, installs the
# annotator requirements, optionally imports a .li, then launches.
# Re-running is safe; it only updates what changed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

KIT=""
LAUNCH=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --kit) KIT="$2"; shift 2 ;;
    --no-launch) LAUNCH=0; shift ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;33m[lithium]\033[0m %s\n' "$*"; }

if ! command -v uv >/dev/null 2>&1; then
  say "installing uv (Python package manager)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
command -v uv >/dev/null || { echo "uv not on PATH after install; open a new shell and re-run." >&2; exit 1; }

say "python 3.12 venv"
uv venv .venv --python 3.12 --quiet
say "packages"
uv pip install --python .venv/bin/python -r requirements-annotator.txt --quiet

if ! .venv/bin/python -c "import tkinter" 2>/dev/null; then
  say "note: tkinter missing (file dialogs). Debian/Ubuntu: sudo apt install python3-tk; Fedora: sudo dnf install python3-tkinter"
fi

if [[ -n "$KIT" ]]; then
  [[ -f "$KIT" ]] || { echo "no such file: $KIT" >&2; exit 1; }
  say "importing $KIT"
  .venv/bin/python tools/bundle_cli.py import "$KIT"
fi

say "ready. Launch any time with ./run.sh"
if [[ $LAUNCH -eq 1 ]]; then
  exec ./run.sh
fi
