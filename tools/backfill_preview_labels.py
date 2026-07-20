#!/usr/bin/env python
"""Rebuild ``preview_labels/<fk>.npy`` for every catalog cloud whose
full-resolution labels exist but whose preview labels are missing.

Use case: after running ``prune_catalog_to_project.py --training-ready``
the preview_labels subdir gets wiped (it's marked as "derived" since
it's regenerable). LIGHT TABLE still shows labels (it reads full
labels directly), but the SHEETS gallery thumbnails come up white
because the gallery renders previews — which can't paint colours
without preview labels.

This script repropagates labels from the full-resolution cloud onto
each preview's points via a KD-tree nearest-neighbour query. Voxel
downsampling produces previews whose points are a *subset* of the
full cloud's points, so the KD-tree returns a zero-distance match
for every preview point — the label transfer is exact, not lossy.

Idempotent: skips clouds that already have preview_labels on disk.
Safe to re-run.

Usage::

    python tools/backfill_preview_labels.py [--force]

    --force re-propagates even when preview_labels already exist
    (useful after a label re-import that left preview_labels stale).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import library_catalog as lc
from src.data.cloud_store import (
    cloud_data_path, cloud_labels_path, load_cloud_data, load_cloud_labels,
    save_preview_labels, preview_labels_path,
)


def _load_preview_positions(library_dir: Path, file_key: str) -> np.ndarray | None:
    """Load just the ``coord`` array from the preview NPZ. Skip the rest
    (colors, scalars) — we only need positions for the KD-tree query."""
    prev_path = library_dir / "previews" / f"{file_key}.npz"
    if not prev_path.is_file():
        return None
    try:
        with np.load(prev_path, allow_pickle=False) as npz:
            if "coord" in npz.files:
                return np.asarray(npz["coord"], dtype=np.float32)
            # Some older previews use a different key name; fall back.
            for k in ("positions", "xyz"):
                if k in npz.files:
                    return np.asarray(npz[k], dtype=np.float32)
    except (OSError, ValueError, EOFError) as e:
        print(f"  [{file_key}] failed to read preview: {e}")
    return None


def backfill(force: bool) -> int:
    library_dir = Path(lc.library_dir())
    if not library_dir.is_dir():
        print(f"ERROR: library not found at {library_dir}", file=sys.stderr)
        return 2

    # Walk the labels/ directory — those are the clouds with full
    # labels worth propagating from. Cheaper than loading the whole
    # index.json + filtering.
    labels_dir = library_dir / "labels"
    if not labels_dir.is_dir():
        print(f"ERROR: labels dir missing at {labels_dir}", file=sys.stderr)
        return 2

    try:
        from scipy.spatial import cKDTree
    except ImportError:
        print("ERROR: scipy not installed in this venv. Backfill needs it "
              "for KD-tree-based label propagation.", file=sys.stderr)
        return 4

    keys: list[str] = []
    for fp in sorted(labels_dir.iterdir()):
        if fp.is_file() and fp.suffix == ".npy" and not fp.name.startswith("_tmp_"):
            keys.append(fp.stem)

    if not keys:
        print("No clouds with full-resolution labels found.")
        return 0

    skipped = 0
    propagated = 0
    failed = 0
    t0 = time.time()
    for fk in keys:
        out_path = preview_labels_path(fk)
        if out_path.exists() and not force:
            skipped += 1
            continue

        # Load full positions + full labels.
        loaded = load_cloud_data(fk)
        if loaded is None:
            print(f"  [{fk}] no full data; skipping")
            failed += 1
            continue
        cloud, _meta = loaded
        full_positions = cloud.positions
        full_labels = load_cloud_labels(fk)
        if full_labels is None or full_labels.size == 0:
            failed += 1
            continue

        # Load preview positions.
        prev_positions = _load_preview_positions(library_dir, fk)
        if prev_positions is None or prev_positions.size == 0:
            print(f"  [{fk}] no preview; skipping")
            failed += 1
            continue

        # Build KD-tree on full + query each preview point. Since
        # previews are voxel-downsampled subsets of the full cloud,
        # the nearest-neighbour distance is zero and the index gives
        # us the *exact* corresponding full-cloud point per preview.
        try:
            tree = cKDTree(np.ascontiguousarray(full_positions, dtype=np.float32))
            _, idx = tree.query(prev_positions, k=1, workers=4)
            mapping = np.asarray(idx, dtype=np.int64)
            prev_labels = full_labels[mapping].astype(np.int32)
            save_preview_labels(fk, prev_labels)
            propagated += 1
            print(f"  [{fk}] {len(prev_labels)} preview points propagated "
                  f"({len(full_positions)} full)")
        except Exception as e:
            print(f"  [{fk}] propagation failed: {type(e).__name__}: {e}")
            failed += 1

    wall = time.time() - t0
    print()
    print(f"Done in {wall:.1f}s")
    print(f"  propagated: {propagated}")
    print(f"  skipped (already had preview labels): {skipped}")
    print(f"  failed: {failed}")
    return 0 if failed == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true",
                    help="Re-propagate even when preview_labels already "
                         "exist. Use after a label re-import.")
    ap.add_argument("--label-namespace", default=None, metavar="PROJECT_ID",
                    help="v2 label namespace to backfill (a project id). "
                         "Default: the shared _library namespace.")
    args = ap.parse_args()
    from src.data import cloud_store
    cloud_store.set_active_label_namespace(args.label_namespace)
    return backfill(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
