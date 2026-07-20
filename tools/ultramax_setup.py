#!/usr/bin/env python
"""End-to-end setup for fight_club_ultramax: re-extract all labeled
subjects at 32k density, build a dense training project with propagated
labels, export as a PT-v3 dataset ready to train on.

After this runs, you can launch training via the TRAIN tab in 3Photon
or via the existing launch helpers. The dataset path will be printed
at the end.

Steps:
  1. Walk all spinelab_training_g* projects, identify unique subjects.
  2. For each subject not already in E:/data/verse/exports_32k/, run
     convert_verse.py with --target-points 32000.
  3. Create the destination project spinelab_training_dense in the catalog.
  4. Import the 32k PLYs into it.
  5. Propagate labels from spinelab_training_g1 (the most complete
     labeled set) via KD-tree nearest-neighbor.
  6. Export the dense project as a PT-v3 training dataset at
     D:/3Photon/dataset_32k.
"""
from __future__ import annotations

import os
import shutil
import subprocess as sp
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


VERSE_ROOT = Path(r"E:/data/verse")
EXPORTS_32K = VERSE_ROOT / "exports_32k"
DENSE_PROJECT_NAME = "spinelab_training_dense"
SOURCE_PROJECT_NAME = "spinelab_training_g1"  # most-complete labels live here
DATASET_OUT_DIR = Path(r"D:/3Photon/dataset_32k")
TARGET_POINTS = 32000

SPLIT_DIRS = {
    "train": "01_training",
    "val":   "02_validation",
    "test":  "03_test",
}


def _find_subject_split(subj: str) -> str | None:
    for split, sd in SPLIT_DIRS.items():
        if (VERSE_ROOT / sd / "rawdata" / subj).is_dir():
            return split
    return None


def _convert_subject(subj: str, split: str) -> bool:
    """Run convert_verse.py for a single subject at 32k points."""
    cmd = [
        sys.executable, str(_REPO_ROOT / "tools" / "stacker" / "convert_verse.py"),
        str(VERSE_ROOT),
        "--subject", subj,
        "--splits", split,
        "--target-points", str(TARGET_POINTS),
        "--output", str(EXPORTS_32K),
        "--force",
        "--workers", "1",
    ]
    print(f"  extracting {subj} ({split}) ...")
    t0 = time.time()
    res = sp.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    wall = time.time() - t0
    if res.returncode != 0:
        print(f"  FAILED ({wall:.1f}s): {res.stderr[-500:]}")
        return False
    print(f"  OK ({wall:.1f}s)")
    return True


def main() -> int:
    from src.data.library_catalog import LibraryCatalog
    from src.data.cloud_store import (
        load_cloud_data, load_cloud_labels, save_cloud_data, save_cloud_labels,
    )
    from src.data.ply_loader import load_ply

    catalog = LibraryCatalog()

    # ---- Step 1: collect unique subjects across labeled projects ----
    subjects: set[str] = set()
    for p in catalog.projects.values():
        if "spinelab_training_g" not in p.name:
            continue
        for fk in p.file_keys:
            e = catalog.entries.get(fk)
            if e is None:
                continue
            name = Path(e.file_path).name
            subj = name.split("_")[0]
            subjects.add(subj)
    subjects = sorted(subjects)
    print(f"Unique subjects: {len(subjects)}")
    for s in subjects:
        print(f"  {s}")
    print()

    # ---- Step 2: re-extract missing subjects at 32k ----
    print("Step 2: extracting missing subjects at 32k...")
    extracted_paths: dict[str, list[Path]] = {}  # subj -> list of dense PLYs
    for subj in subjects:
        # Check whether we already have dense plys for this subject
        existing = list(EXPORTS_32K.glob(f"**/{subj}_*.ply"))
        if existing:
            print(f"  {subj}: {len(existing)} already extracted, skipping")
            extracted_paths[subj] = existing
            continue
        split = _find_subject_split(subj)
        if split is None:
            print(f"  {subj}: NOT FOUND in any VerSe split, SKIPPING")
            continue
        ok = _convert_subject(subj, split)
        if ok:
            extracted_paths[subj] = list(EXPORTS_32K.glob(f"**/{subj}_*.ply"))
    total_dense = sum(len(v) for v in extracted_paths.values())
    print(f"Total dense PLYs: {total_dense}")
    print()

    # ---- Step 3: find or create the dense project ----
    print(f"Step 3: setting up project {DENSE_PROJECT_NAME!r}...")
    src_proj = next(
        (p for p in catalog.projects.values() if p.name == SOURCE_PROJECT_NAME),
        None,
    )
    if src_proj is None:
        print(f"  ERROR: source project {SOURCE_PROJECT_NAME!r} not found")
        return 2

    dense_proj = next(
        (p for p in catalog.projects.values() if p.name == DENSE_PROJECT_NAME),
        None,
    )
    if dense_proj is None:
        dense_proj = catalog.create_project(DENSE_PROJECT_NAME)
        print(f"  created project {dense_proj.name} ({dense_proj.id})")
    else:
        print(f"  reusing project {dense_proj.name} ({dense_proj.id})")

    # Copy ontology
    dense_proj.ontology_data = src_proj.ontology_data
    catalog._save_projects()
    print(f"  copied ontology ({len(dense_proj.ontology_data.get('labels', []))} labels)")
    print()

    # v2 label layout: sparse labels are read from the SOURCE project's
    # namespace; the propagated dense labels are written to the dense
    # project's namespace.
    from src.data import cloud_store
    cloud_store.set_active_label_namespace(dense_proj.id)

    # ---- Step 4: build sparse lookup from source project labels ----
    print(f"Step 4: indexing labeled clouds from {SOURCE_PROJECT_NAME}...")
    sparse_by_name: dict[str, tuple] = {}
    for fk in src_proj.file_keys:
        e = catalog.entries.get(fk)
        if e is None:
            continue
        name = Path(e.file_path).name
        cached = load_cloud_data(fk)
        if cached is None:
            continue
        sc, _ = cached
        lbl = load_cloud_labels(fk, namespace=src_proj.id)
        if lbl is not None and len(lbl) == sc.point_count and (lbl != 0).any():
            sparse_by_name[name] = (sc.positions, lbl.astype(np.int32))
    print(f"  {len(sparse_by_name)} labeled sparse clouds available")
    print()

    # ---- Step 5: import each dense PLY, propagate labels ----
    print("Step 5: importing dense clouds + propagating labels...")
    dense_keys: list[str] = []
    propagated = 0
    no_match = 0
    for subj in subjects:
        for ply_path in extracted_paths.get(subj, []):
            name = ply_path.name
            # Avoid duplicate registration if this ply is already in catalog
            already = any(
                Path(e.file_path).resolve() == ply_path.resolve()
                for e in catalog.entries.values()
            )
            if already:
                # find its file_key
                existing_fk = next(
                    (e.file_key for e in catalog.entries.values()
                     if Path(e.file_path).resolve() == ply_path.resolve()),
                    None,
                )
                if existing_fk:
                    dense_keys.append(existing_fk)
                continue

            dense_cloud = load_ply(str(ply_path))
            entry = catalog.register_file(str(ply_path))
            if entry is None:
                continue
            catalog.update_metrics(
                entry.file_key,
                dense_cloud.point_count,
                dense_cloud.bounds_min,
                dense_cloud.bounds_max,
            )
            save_cloud_data(entry.file_key, dense_cloud, source_path=str(ply_path))
            entry.region = "spine"
            dense_keys.append(entry.file_key)

            # Propagate labels from matching sparse cloud
            if name in sparse_by_name:
                sp_pos, sp_lbl = sparse_by_name[name]
                tree = cKDTree(sp_pos)
                _, idx = tree.query(dense_cloud.positions, k=1)
                save_cloud_labels(entry.file_key, sp_lbl[idx].astype(np.int32))
                propagated += 1
            else:
                no_match += 1
                print(f"    {name}: no sparse match for label propagation")
    catalog.add_to_project(dense_proj.id, dense_keys)
    catalog._save_index()
    print(f"  staged {len(dense_keys)} dense clouds, "
          f"propagated {propagated} label sets, {no_match} no-match")
    print()

    # ---- Step 6: export as PT-v3 dataset ----
    print(f"Step 6: exporting dataset to {DATASET_OUT_DIR}...")
    DATASET_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the cloud list + per-cloud filters in the same way the
    # GUI export path does.
    from src.export.dataset_export import export_dataset
    from src.data.labels import LabelRegistry

    reg = LabelRegistry.from_json(dense_proj.ontology_data)
    clouds = []
    for fk in dense_keys:
        cached = load_cloud_data(fk)
        if cached is None:
            continue
        c, _ = cached
        lbl = load_cloud_labels(fk)
        if lbl is None or len(lbl) != c.point_count:
            continue
        c.labels = lbl.astype(np.int32)
        clouds.append(c)
    print(f"  loaded {len(clouds)} clouds for export")
    summary = export_dataset(
        clouds, reg,
        output_dir=str(DATASET_OUT_DIR),
        format="ptv3",
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        seed=42,
    )
    print(f"  export complete")
    print(f"  splits: train={summary.get('n_train', '?')}, "
          f"val={summary.get('n_val', '?')}, test={summary.get('n_test', '?')}")
    print()

    # ---- Step 7: print the training launch instructions ----
    print("=" * 70)
    print("READY TO TRAIN — fight_club_ultramax")
    print("=" * 70)
    print()
    print(f"Dataset: {DATASET_OUT_DIR}")
    print(f"Active project: switch 3Photon to '{DENSE_PROJECT_NAME}' before launching")
    print()
    print("In TRAIN tab, set:")
    print("  Model name:     fight_club_ultramax")
    print("  Epochs:         250")
    print("  Batch:          2  (drop to 1 if OOM)")
    print("  base_lr:        0.003")
    print("  weight_decay:   0.025")
    print("  warmup_epochs:  7")
    print("  drop_path:      0.25")
    print("  minority_boost: 1.0")
    print("  random_rotate_so3: OFF")
    print("  Fine-tune from: (none — train from scratch)")
    print()
    print("Or pre-set via app state:")
    print("  app._train_epochs = 250")
    print("  app._train_batch = 2")
    print("  app._train_base_lr = 0.003")
    print("  app._train_weight_decay = 0.025")
    print("  app._train_warmup_epochs = 7")
    print("  app._train_drop_path = 0.25")
    print("  app._train_minority_boost = 1.0")
    print("  app._train_random_rotate_so3 = False")
    return 0


if __name__ == "__main__":
    sys.exit(main())
