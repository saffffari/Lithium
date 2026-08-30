# Intake: prelabeled cohorts (64k-point Sonata set, ~15k clouds)

Landing zone on HAL: `/board_rack/Lithium/datasets/incoming/<cohort>/`
(board_rack, 690 GB free on 2026-08-29; the catalog itself lives on `/home`
under `~/.lithium/library`, 317 GB free — a 15k × 64k cohort is ~30 GB of
catalog npz + ~4 GB of labels).

1. Copy the cohort in (F-35 stick or rsync). Any of these layouts work,
   mixed is fine:
   - `*.ply` with a per-vertex `label` (or `segment`/`class`/`pred`) field
   - `*.npz` with `coord`, `color`[, `normal`] and `label`/`segment`/`pred`
   - Pointcept scene dirs: `<scene>/coord.npy`, `color.npy`, `segment.npy`|`pred.npy`
2. Close Lithium (single-instance catalog lock).
3. Dry run, then import:

   ```bash
   cd ~/Lithium
   .venv/bin/python tools/import_prelabeled.py \
       --src /board_rack/Lithium/datasets/incoming/<cohort> \
       --project "Deepfield · 64k Sonata · <cohort>" --region spine --dry-run
   .venv/bin/python tools/import_prelabeled.py --src ... --project "..." --region spine
   ```

   Defaults: classes + palette are copied from the Sonata Validation
   project (`--ontology-from`), and the Sonata checkpoint
   `training/runs/sonata_full6_gold247_v1/model/model_best.pth` is attached
   as the project's active model so INFER works immediately
   (`--checkpoint ''` to skip, `--classes sonata6|classes.json|A,B,C` for a
   different ontology). Label ids in the files must index that class list
   (0 = Unlabeled).

   The importer content-hashes every file (re-runs dedupe), writes catalog
   data + labels per cloud, and saves `index.json`/`projects.json` once at
   the end — 15k clouds do not rewrite the index 15k times. Expect roughly
   20–40 clouds/s for 64k-point clouds (≈10 min for 15k).
4. Open Lithium → the project shows the Sonata labels as editable starting
   labels. Gallery previews build lazily as cells scroll into view.

Sanity checks after import: `python -m src.main --help` (CLI) is unchanged;
in the app, the project row count equals the number imported; pick a
cloud → LIGHT TABLE → labels visible; INFER re-runs Sonata on it.
