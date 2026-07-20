#!/usr/bin/env python
"""Repair the spinelab_training_dense project: re-propagate manual
labels from spinelab_training_g1 onto every dense cloud, overwriting
any stale labels (including the ones that leaked through from the
density test, which contained fight_club_heavy predictions).

After this runs:
  - every dense cloud's labels are KD-tree-propagated from the
    matching sparse g1 cloud's manual labels
  - no model-prediction contamination
  - dataset_32k can be re-exported and used for training

Caveat: the spinelab_density_test_g1 project shares its 89 cloud
entries with spinelab_training_dense (same file_keys). Overwriting
the labels here will ALSO overwrite the density-test predictions —
acceptable because the density-test results are already captured
quantitatively elsewhere.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SOURCE_PROJECT = "spinelab_training_g1"
DENSE_PROJECT = "spinelab_training_dense"


def main() -> int:
    from src.data.library_catalog import LibraryCatalog
    from src.data.cloud_store import (
        load_cloud_data, load_cloud_labels, save_cloud_labels,
    )

    catalog = LibraryCatalog()
    src_proj = next(
        (p for p in catalog.projects.values() if p.name == SOURCE_PROJECT),
        None,
    )
    dense_proj = next(
        (p for p in catalog.projects.values() if p.name == DENSE_PROJECT),
        None,
    )
    if src_proj is None or dense_proj is None:
        print(f"ERROR: required project missing "
              f"({SOURCE_PROJECT!r}={src_proj is not None}, "
              f"{DENSE_PROJECT!r}={dense_proj is not None})")
        return 2

    # Index labeled sparse clouds from g1 by basename.
    print(f"Indexing labeled clouds from {SOURCE_PROJECT}...")
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
        lbl = load_cloud_labels(fk)
        if lbl is not None and len(lbl) == sc.point_count and (lbl != 0).any():
            sparse_by_name[name] = (sc.positions, lbl.astype(np.int32))
    print(f"  {len(sparse_by_name)} labeled sparse clouds available")
    print()

    # Walk every dense cloud and re-propagate.
    print(f"Re-propagating labels onto {len(dense_proj.file_keys)} dense clouds...")
    n_propagated = 0
    n_no_match = 0
    n_unchanged = 0
    n_overwritten = 0
    t0 = time.time()
    for i, fk in enumerate(dense_proj.file_keys, start=1):
        e = catalog.entries.get(fk)
        if e is None:
            continue
        name = Path(e.file_path).name
        if name not in sparse_by_name:
            n_no_match += 1
            continue
        cached = load_cloud_data(fk)
        if cached is None:
            continue
        dc, _ = cached
        sp_pos, sp_lbl = sparse_by_name[name]
        tree = cKDTree(sp_pos)
        _, idx = tree.query(dc.positions, k=1)
        new_labels = sp_lbl[idx].astype(np.int32)

        # Compare to existing to flag overwrites
        existing = load_cloud_labels(fk)
        if existing is not None and len(existing) == len(new_labels):
            diff = int((existing != new_labels).sum())
            if diff > 0:
                n_overwritten += 1
            else:
                n_unchanged += 1
        save_cloud_labels(fk, new_labels)
        n_propagated += 1
        if i == 1 or i % 25 == 0 or i == len(dense_proj.file_keys):
            print(f"  [{i}/{len(dense_proj.file_keys)}] {name}")

    wall = time.time() - t0
    print()
    print(f"Done in {wall:.1f}s")
    print(f"  Propagated: {n_propagated}")
    print(f"    - overwrote stale labels: {n_overwritten}")
    print(f"    - unchanged (already matched): {n_unchanged}")
    print(f"  No-match (no sparse counterpart): {n_no_match}")
    print()
    print("Next: re-export the dataset so D:/3Photon/dataset_32k/ "
          "reflects the corrected labels.")
    print("  python tools/ultramax_setup.py  # re-exports as part of step 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
