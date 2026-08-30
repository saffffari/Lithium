# Lithium manuscript — readiness (2026-08-29)

**Paper:** "Lithium: a point-cloud annotation and in-loop training environment for
anatomical surface labelling" — software/tools paper (targets: arXiv cs.CV /
eess.IV cross-list; JOSS / SoftwareX / MICCAI open-source track as venue).

## What exists (all numbers traceable to files in `paper/tables/*.json`)
- **Model results on public data**: Gold247 (247 VerSe bones, 12 subjects,
  frozen subject split 170/38/39). EP4 evaluation on the 38 val bones for two
  models trained *inside* Lithium's training path — Yamato (PT-v3m1 from
  scratch) and Sonata (PT-v3m2 fine-tune): per-class precision/recall/IoU,
  confidence-gated precision, per-bone precision, per-bone val dumps
  (`training/runs/{sonata_full6_gold247_v1,baselines}`), and local-axis
  errors derived from labels (`axis_val.json`). → `paper_model_metrics.py`.
- **Dataset composition**: manifest + class point counts. → `paper_dataset_stats.py`.
- **Interaction latency** at 1M points (pick/box/lasso/brush/apply/undo):
  re-measured on HAL by `paper_bench_ops.py` (same code the app runs).
- **Inference latency**: `infer_single.py` on val bones, RTX 4090 → `paper_bench_infer.py`.
- **Screenshots**: the real app on public VerSe bones (`paper_screenshots.py`,
  needs the new `--project/--cloud/--tab` startup flags).

## What does NOT exist yet (flagged, never fabricated)
- **Annotation timing study** (the number reviewers want): two annotators,
  ~30 bones, minutes/bone + agreement, surface (Lithium) vs volume baseline.
  The manuscript ships the *protocol* as a figure with an explicit
  "data pending" results slot. The 3–5 min/bone vs 25–40 min claim is stated
  only as the study's hypothesis.
- **Rendering fps** at 1M+ points: not instrumented; described qualitatively.
- No patient data (EPIC cohort) appears anywhere; VerSe/Gold247 only.

## Human-only decisions
- Affiliation, funding, acknowledgements (placeholders left).
- License of the manuscript figures (code is MIT).
- arXiv account/submission; Zenodo DOI for the v1.1 release (optional).
