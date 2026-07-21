# 1.1 bug ledger

Status of the 1.0 audit backlog (`_audit/REPORT.md` in the 1.0 repo,
wave-3 synthesis 2026-05-20) as of the 1.1 rebuild. Most findings were
fixed in 1.0's 2026-05-21 fix session and carried over with the import;
this ledger records what 1.1 changed on top.

## Fixed in 1.1

| ID | Finding | 1.1 resolution |
|----|---------|----------------|
| CP-3 | Cross-project label-id drift on view switch | **Obsoleted by design**: per-project label namespaces (v2 layout) — there is no shared label file to drift. `migrate_cloud_labels_to_project` now copies+translates into the destination namespace; sources are never rewritten. |
| LS-1 | 4D sequence frames had zero on-disk persistence | Frames registered with the catalog; per-frame file_keys; persist + reload through `save_cloud_labels`/`load_cloud_labels`; shutdown flush covers cached frames. (branch `bugfix-ls1-ls6`) |
| LS-6 | Paint during preview→full-res window orphaned labels | Cached preview→full nearest-neighbor projection built off-thread when full-res arrives; painted preview labels merge into the full array and persist. (branch `bugfix-ls1-ls6`) |
| CP-5 | No re-attach UX for moved/renamed sources | `catalog.reattach_entry_source` + gallery context-menu "Re-attach source..." (file_key preserved, all sidecars survive). |
| — | Inference class-map resolution was cwd-fragile | `TrainedModel.class_map` frozen at launch + `classes.json` mirrored into the run's work_dir; registry entry consulted first. |
| — | `_predict_cloud` ran under whatever ontology was active | Batch runner freezes the label namespace at submit; a mid-batch project switch can no longer misroute worker writes. |
| — | Ontology picker defined but never drawn (orphaned in the SpineLab cut) | Wired into SHEETS under PROJECTS. |
| — | Project duplicate assigned `app.active_view` directly | Goes through `set_active_view` (entries + namespace actually switch). |
| — | Fine-tune names accumulated " ft ft ft" | Deduped: "base ft", "base ft2", ... |
| — | Stale index-based selections survived view switches | `_train_selected` / label filter / gallery multi-select cleared on view change. |
| — | STAGED DATASET collapse state broken | Stable section title; open/closed return honored; starts collapsed. |
| — | No model delete UI; no lineage display | Right-click delete (registry-only or +run-dir with size + shared-work_dir guard); "↳ parent" lineage in the model list. |
| — | `tools/generate_test_ply.py` wrote to a literal `D:/` dir on Linux | Takes an output path argument. |

## Carried over already-fixed (1.0 fix session)

LS-2..LS-5, LS-7..LS-11, CP-1, CP-2, CP-4, CP-6, CP-8, CP-9, ST-3,
ST-5, TR-1, AR-1, AR-2, AR-5 — verified still in place during the 1.1
catalog (2026-07-20). ST-1/2/4, CP-7, AN-*, IM-* died with the SpineLab
cut and do not apply.

## Known-open / deferred

- ARCH-017/019/020/024, ARCH-005/021 — large extraction refactors
  (train-tab split, measure overlay, project snapshot). Deliberate:
  refactors of that size deserve their own change, not a rebuild rider.
- Phase-13 token drift (~22 sites of raw theme tuples vs `tokens.py`
  semantics) — cosmetic, migrate opportunistically when touching those
  panels.
- GP-2 (`PrimitiveRenderer` VAO leak) — module left with SpineLab;
  verify N/A if primitives ever return.
