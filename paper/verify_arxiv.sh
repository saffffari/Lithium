#!/bin/sh
# Verify the arXiv tarball builds the way arXiv builds it: extract fresh, run
# pdflatex TWICE using the bundled main.bbl, no bibtex. Exits non-zero on error.
set -e
cd "$(dirname "$0")"
rm -rf arxiv_verify; mkdir arxiv_verify
tar xzf lithium_arxiv.tar.gz -C arxiv_verify
STORE="${PODMAN_VFS_ROOT:-$HOME/.local/share/containers/xtracer-vfs}"
podman --root "$STORE" --storage-driver vfs run --rm -v "$PWD/arxiv_verify":/work -w /work \
  docker.io/texlive/texlive:latest-small sh -c '
  pdflatex -interaction=nonstopmode -halt-on-error main >/dev/null &&
  pdflatex -interaction=nonstopmode -halt-on-error main >l.log 2>&1
  grep -E "Output written|Warning: (Citation|Reference).*undefined" l.log | head'
echo "arXiv-style build OK: arxiv_verify/main.pdf"
