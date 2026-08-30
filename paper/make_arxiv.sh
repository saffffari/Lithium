#!/bin/sh
# Assemble a self-contained arXiv submission tarball from paper/.
#   sh paper/make_arxiv.sh        (run scripts/paper_numbers.py + build.sh first)
# arXiv compiles main.tex with the bundled main.bbl (no bibtex run), so build.sh
# (which runs bibtex) must have succeeded beforehand.
set -e
cd "$(dirname "$0")"
OUT=arxiv
rm -rf "$OUT"; mkdir -p "$OUT/figures" "$OUT/tables"
# strip the local preview stamp — arXiv adds its own left-margin stamp
sed '/>>> arxiv-preview-stamp/,/<<< arxiv-preview-stamp/d' main.tex > "$OUT/main.tex"
cp refs.bib main.bbl "$OUT/"
cp tables/*.tex "$OUT/tables/"
# resolve each referenced figure to its real file (prefer .pdf, else .png)
for ref in $(grep -oE 'includegraphics\[[^]]*\]\{[^}]*\}' main.tex | sed 's/.*{//;s/}//'); do
  base=$(basename "$ref")
  if   [ -f "figures/$base.pdf" ]; then cp "figures/$base.pdf" "$OUT/figures/";
  elif [ -f "figures/$base.png" ]; then cp "figures/$base.png" "$OUT/figures/";
  else echo "MISSING figure: $ref" >&2; exit 1; fi
done
( cd "$OUT" && tar czf ../lithium_arxiv.tar.gz . )
echo "wrote paper/lithium_arxiv.tar.gz"
tar tzf lithium_arxiv.tar.gz | sort
