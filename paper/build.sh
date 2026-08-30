#!/bin/sh
# Build paper/main.pdf with a TeX Live container (no local TeX needed).
#   sh paper/build.sh
# Regenerate numbers/figures first if any experiment changed:
#   .venv/bin/python scripts/paper_numbers.py && .venv/bin/python scripts/paper_figs.py
# Private vfs storage root so it works when the host's overlay backend is broken
# (overlay-over-btrfs on rolling-release). The root is shared with the xtracer
# paper so the cached texlive image is reused.
set -e
cd "$(dirname "$0")"
STORE="${PODMAN_VFS_ROOT:-$HOME/.local/share/containers/xtracer-vfs}"
mkdir -p "$STORE"
podman --root "$STORE" --storage-driver vfs run --rm -v "$PWD":/work -w /work \
  docker.io/texlive/texlive:latest-small sh -c '
  pdflatex -interaction=nonstopmode main >/dev/null
  bibtex main >/dev/null
  pdflatex -interaction=nonstopmode main >/dev/null
  pdflatex -interaction=nonstopmode main | grep -E "^!|Warning: (Citation|Reference)|Output written"'
