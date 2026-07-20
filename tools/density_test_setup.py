#!/usr/bin/env python
"""Stage the 32k density-test project alongside the existing group_1.

What this does:

1. Creates a fresh project ``spinelab_density_test_g1`` in the catalog.
2. Imports the 98 dense (32,000 pt/vertebra) PLYs from
   ``E:/data/verse/exports_32k/`` into that project.
3. Copies the label ontology from ``spinelab_training_g1`` so the
   classes line up.
4. For each new cloud, propagates labels from the matching original
   sparse cloud via KD-tree nearest-neighbor.

After this runs, the new project is ready to be opened in 3Photon:
- LIGHT TABLE / CONTACT SHEETS can browse the 98 dense clouds.
- Existing label propagation gives every dense point a class so that
  per-class IoU can be computed on inference predictions.
- Run inference with ``fight_club_heavy`` from the SHEETS tab.
- Compare predictions against the propagated ground-truth labels.

The test is whether the model trained on ~10k-point clouds generalizes
to 32k-point clouds without retraining. If it does, density was a
meaningful underused lever. If it doesn't, we know the model needs to
be retrained at higher density.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


SOURCE_PROJECT_NAME = "spinelab_training_g1"
DEST_PROJECT_NAME = "spinelab_density_test_g1"
DENSE_DIR = Path(r"E:/data/verse/exports_32k")


def _find_dense_ply(name: str) -> Path | None:
    """Locate the dense version of a sparse cloud's PLY by basename.
    The dense extraction wrote into ``<split>/<basename>.ply`` per the
    convert_verse layout."""
    candidates = list(DENSE_DIR.glob(f"**/{name}"))
    return candidates[0] if candidates else None


def _propagate_labels_knn(sparse_positions, sparse_labels,
                          dense_positions):
    """Each dense point inherits the label of its nearest sparse point."""
    from scipy.spatial import cKDTree
    import numpy as np

    if sparse_labels is None or len(sparse_labels) == 0:
        return np.zeros(len(dense_positions), dtype=np.int32)
    tree = cKDTree(sparse_positions)
    _, idx = tree.query(dense_positions, k=1)
    return sparse_labels[idx].astype(np.int32)


def main() -> int:
    import numpy as np
    from src.data.library_catalog import LibraryCatalog
    from src.data.cloud_store import (
        load_cloud_data, save_cloud_data, save_cloud_labels,
    )
    from src.data.ply_loader import load_ply

    catalog = LibraryCatalog()

    # Find source project
    src_proj = None
    for p in catalog.projects.values():
        if p.name == SOURCE_PROJECT_NAME:
            src_proj = p
            break
    if src_proj is None:
        print(f"ERROR: source project {SOURCE_PROJECT_NAME!r} not found.")
        return 2
    print(f"Source: {src_proj.name} ({len(src_proj.file_keys)} clouds)")

    # Create / find destination project
    dest_proj = None
    for p in catalog.projects.values():
        if p.name == DEST_PROJECT_NAME:
            dest_proj = p
            print(f"Reusing existing dest project: {dest_proj.id}")
            break
    if dest_proj is None:
        dest_proj = catalog.create_project(DEST_PROJECT_NAME)
        print(f"Created dest project: {dest_proj.name} ({dest_proj.id})")

    # Copy ontology
    dest_proj.ontology_data = src_proj.ontology_data
    catalog._save_projects()
    print(f"Copied ontology ({len(dest_proj.ontology_data.get('labels', []))} labels)")

    # Walk every sparse cloud in source project, find matching dense PLY,
    # propagate labels, register in dest project.
    dest_keys: list[str] = []
    n = len(src_proj.file_keys)
    t0 = time.time()
    propagated = 0
    missing_dense = 0
    missing_labels = 0
    for i, sparse_fk in enumerate(src_proj.file_keys, start=1):
        sparse_entry = catalog.entries.get(sparse_fk)
        if sparse_entry is None:
            continue
        sparse_name = Path(sparse_entry.file_path).name

        dense_ply = _find_dense_ply(sparse_name)
        if dense_ply is None:
            print(f"  [{i}/{n}] {sparse_name}: NO MATCHING DENSE CLOUD")
            missing_dense += 1
            continue

        # Load sparse cloud + labels for source positions/labels
        sparse_data = load_cloud_data(sparse_fk)
        if sparse_data is None:
            print(f"  [{i}/{n}] {sparse_name}: sparse cloud cache miss")
            continue
        sparse_cloud, _ = sparse_data
        from src.data.cloud_store import load_cloud_labels
        sparse_labels = load_cloud_labels(sparse_fk)
        if sparse_labels is None or (sparse_labels == 0).all():
            missing_labels += 1

        # Load dense cloud
        dense_cloud = load_ply(str(dense_ply))

        # Register dense cloud in catalog under dest project
        dense_entry = catalog.register_file(str(dense_ply))
        if dense_entry is None:
            continue
        catalog.update_metrics(
            dense_entry.file_key,
            dense_cloud.point_count,
            dense_cloud.bounds_min,
            dense_cloud.bounds_max,
        )
        save_cloud_data(dense_entry.file_key, dense_cloud, source_path=str(dense_ply))
        dense_entry.region = "spine"

        # Propagate labels via KD-tree
        if sparse_labels is not None and len(sparse_labels) == sparse_cloud.point_count:
            propagated_labels = _propagate_labels_knn(
                sparse_cloud.positions, sparse_labels,
                dense_cloud.positions,
            )
            save_cloud_labels(dense_entry.file_key, propagated_labels)
            propagated += 1

        dest_keys.append(dense_entry.file_key)
        if i == 1 or i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] {sparse_name} -> {dense_ply.name} "
                  f"({dense_cloud.point_count} pts, "
                  f"labels propagated: {sparse_labels is not None})")

    if dest_keys:
        catalog.add_to_project(dest_proj.id, dest_keys)
    catalog._save_index()

    wall = time.time() - t0
    print()
    print(f"Done in {wall:.1f}s")
    print(f"  staged: {len(dest_keys)} dense clouds")
    print(f"  labels propagated: {propagated}")
    print(f"  missing dense ply: {missing_dense}")
    print(f"  sparse clouds with no labels (propagation skipped): {missing_labels}")
    print()
    print(f"Next: launch the app, open '{DEST_PROJECT_NAME}', run inference "
          f"with fight_club_heavy from the SHEETS tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
