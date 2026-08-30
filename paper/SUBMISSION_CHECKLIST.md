# Submission checklist — Lithium manuscript

Human-only items (never filled in by the agent):
- [ ] Affiliation(s) under the author line in `main.tex` (none stated now).
- [ ] Acknowledgements / funding / data-provider statements (placeholder present).
- [ ] Decide whether the "Formerly known as 3Photon" history matters for the paper (not mentioned).
- [ ] arXiv: create submission, category **cs.CV** primary, cross-list **eess.IV** (and **cs.HC** if desired); paste `ABSTRACT.txt`; upload `lithium_arxiv.tar.gz` (built by `make_arxiv.sh`, verified by `verify_arxiv.sh`); replace the `\arxivid` preview stamp is auto-stripped.
- [ ] Optional: Zenodo DOI for release v1.1 and cite it in §Availability.
- [ ] Journal target after arXiv: JOSS (short software paper + repository review), SoftwareX, or a MICCAI open-source-software track.
- [ ] **Run the timing study** (`fig_timing_design`): two annotators, 30 bones, minutes + agreement. Then: write `paper/tables/timing_study.json`, add macros in `scripts/paper_numbers.py`, replace the "DATA PENDING" panels in `scripts/paper_figs.py::fig_timing_design`, and rewrite §Timing.

Agent-verified before hand-off:
- numbers.tex regenerated from tables/*.json; all figures rebuilt from scripts;
  PDF compiled in the TeX Live container; every page inspected; arXiv-style
  tarball extracted and compiled twice without bibtex.
