#!/usr/bin/env python
"""Map per-scene predictions back onto the user's full-resolution project clouds.

The training pipeline exports staged scenes at voxel-downsampled resolution
(~20K pts) and writes integer class IDs in ``predictions/<scene>_pred.npy``.
This tool:

  1. Reads each project cloud's positions from the catalog (``data/<file_key>.npz``).
  2. For each test scene, finds which catalog cloud it was exported from
     by checking that the scene's coords are a near-byte-exact subset of
     a single catalog cloud's coords (KD-tree nearest distance ≈ 0).
  3. Upsamples the scene's prediction to the matched cloud's full
     resolution via nearest-neighbour mapping (each full point inherits
     its closest staged point's class).
  4. Maps prediction class IDs to the project's label registry IDs by
     name match (e.g. dataset class 0 'superior endplate' → registry
     id 1).
  5. Writes the result to ``library/labels/<file_key>.npy`` (full-res)
     and ``library/preview_labels/<file_key>.npy`` (preview-res, via
     a second NN pass against the cached preview .npz).

After running, relaunch 3Photon: the matched test clouds will show the
model's predictions as labels in the same orange/blue palette your hand
labels use. Backups of the previous (hand-labelled) catalog labels can
be restored from ``library/backups/<timestamp>/`` if needed.

Usage:
    python tools/predictions_to_project.py \\
        --predictions <dataset>/training_runs/<run>/predictions \\
        --project-id proj:5a770138
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def library_dir() -> Path:
    return Path(os.path.expanduser("~")) / ".3photon" / "library"


def load_project(project_id: str) -> dict:
    p = library_dir() / "projects.json"
    with open(p) as f:
        d = json.load(f)
    if project_id not in d:
        raise KeyError(f"project {project_id!r} not in {p}")
    return d[project_id]


def project_class_to_registry(project: dict) -> dict[str, int]:
    """Map lowercased label name → registry id (skip 'unlabeled')."""
    out = {}
    for entry in project.get("ontology_data", {}).get("labels", []):
        if int(entry["id"]) == 0:
            continue
        out[entry["name"].lower()] = int(entry["id"])
    return out


def load_catalog_cloud_positions(file_key: str) -> np.ndarray | None:
    """Read the cached positions array from the catalog data .npz."""
    p = library_dir() / "data" / f"{file_key}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p)
        # cloud_store stores positions under 'coord' or 'positions';
        # try both.
        for key in ("coord", "positions"):
            if key in z.files:
                return z[key].astype(np.float32)
    except Exception as e:
        print(f"  [warn] failed to read {p}: {e}")
    return None


def load_catalog_preview_positions(file_key: str) -> np.ndarray | None:
    """Read positions from the catalog preview .npz if present."""
    p = library_dir() / "previews" / f"{file_key}.npz"
    if not p.exists():
        return None
    try:
        z = np.load(p)
        for key in ("coord", "positions"):
            if key in z.files:
                return z[key].astype(np.float32)
    except Exception as e:
        print(f"  [warn] failed to read preview {p}: {e}")
    return None


def find_source_cloud(
    scene_coord: np.ndarray,
    candidate_keys: list[str],
    candidate_bounds: dict[str, tuple[np.ndarray, np.ndarray]],
    candidate_point_counts: dict[str, int],
) -> str | None:
    """Identify which candidate cloud the staged scene was exported from.

    Strategy: shortlist by bbox containment (cloud must fully enclose
    scene). When there are multiple matches — common when a prediction
    PLY built from the staged scene has been re-imported into the same
    project — prefer the candidate with the largest point count, which
    is the original full-resolution source. The staged scene and the
    prediction PLY both have ~20K points; the source has 1M+.
    """
    s_min = scene_coord.min(0)
    s_max = scene_coord.max(0)
    eps = np.float32(1e-3)
    candidates = []
    for k in candidate_keys:
        if k not in candidate_bounds:
            continue
        c_min, c_max = candidate_bounds[k]
        # Loosen the cloud's bbox by eps (not the scene's) so float
        # round-trip noise doesn't reject a true source.
        if (c_min - eps <= s_min).all() and (s_max <= c_max + eps).all():
            candidates.append(k)
    if not candidates:
        return None
    # Prefer largest point count — that's the original full-res source.
    candidates.sort(key=lambda k: candidate_point_counts.get(k, 0), reverse=True)
    return candidates[0]


def upsample_predictions(
    scene_coord: np.ndarray,
    scene_pred: np.ndarray,
    target_coord: np.ndarray,
) -> np.ndarray:
    """For each target point, take its nearest staged scene point's class."""
    from scipy.spatial import cKDTree
    tree = cKDTree(scene_coord)
    _, nn = tree.query(target_coord, k=1, workers=-1)
    return scene_pred[nn].astype(np.int32)


def remap_to_registry(
    pred: np.ndarray,
    class_names: list[str],
    name_to_registry: dict[str, int],
) -> np.ndarray:
    """Translate dataset class IDs into the project's registry IDs.

    Out-of-range or unmapped classes become 0 (Unlabeled) so they don't
    invent stray labels in the registry.
    """
    table = np.zeros(len(class_names), dtype=np.int32)
    for cls_id, name in enumerate(class_names):
        rid = name_to_registry.get(name.lower(), 0)
        table[cls_id] = rid
    out = np.zeros(pred.shape, dtype=np.int32)
    valid = (pred >= 0) & (pred < len(class_names))
    out[valid] = table[pred[valid]]
    return out


def write_atomic_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"_tmp_{path.stem}"
    np.save(str(tmp), np.ascontiguousarray(arr, dtype=np.int32))
    os.replace(tmp.with_suffix(".npy"), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--data-root",
                    help="Root that contains classes.json (defaults to "
                         "predictions/.. dir)")
    args = ap.parse_args()

    pred_dir = Path(args.predictions)
    if not pred_dir.is_dir():
        raise FileNotFoundError(args.predictions)

    # Resolve data-root: predictions dir may have a copy of classes.json
    classes_json = pred_dir / "classes.json"
    if not classes_json.exists():
        if args.data_root:
            classes_json = Path(args.data_root) / "classes.json"
        else:
            # default to predictions/../../classes.json (work_dir → dataset)
            classes_json = pred_dir.parent.parent / "classes.json"
    with open(classes_json) as f:
        classes = json.load(f)
    class_names = classes["class_names"]
    print(f"Classes: {class_names}")

    project = load_project(args.project_id)
    name_to_registry = project_class_to_registry(project)
    print(f"Project '{project['name']}': {len(project['file_keys'])} clouds, "
          f"label map {name_to_registry}")

    # Build candidate bbox cache from the catalog data .npz files we
    # actually have. Small projects: cheap; large: still acceptable since
    # we only need bbox.
    print("Indexing project clouds...")
    candidate_bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_point_counts: dict[str, int] = {}
    cat_index_path = library_dir() / "index.json"
    catalog_index = {}
    if cat_index_path.exists():
        with open(cat_index_path) as f:
            catalog_index = json.load(f)
    for fk in project["file_keys"]:
        # Prefer catalog index bounds — much faster than loading .npz
        meta = catalog_index.get(fk)
        if meta and meta.get("bounds_min") and meta.get("bounds_max"):
            candidate_bounds[fk] = (
                np.asarray(meta["bounds_min"], dtype=np.float32),
                np.asarray(meta["bounds_max"], dtype=np.float32),
            )
            candidate_point_counts[fk] = int(meta.get("point_count", 0))

    pred_files = sorted(pred_dir.glob("scene_*_pred.npy"))
    print(f"Found {len(pred_files)} prediction files")

    written = 0
    for pred_path in pred_files:
        scene_name = pred_path.stem.replace("_pred", "")
        # Locate the matching staged scene's coord.npy. The predictions
        # dir typically lives at ``<dataset>/training_runs/<run>/predictions``,
        # so the dataset is three levels up. Walk upward and accept the
        # first ancestor that has a ``test/`` subdir.
        dataset_root = None
        anc = pred_dir
        for _ in range(5):
            anc = anc.parent
            if (anc / "test").is_dir() or (anc / "val").is_dir() or (anc / "train").is_dir():
                dataset_root = anc
                break
        if dataset_root is None:
            print(f"  {scene_name} - dataset root not found, skip")
            continue
        scene_coord_path = None
        for split in ("test", "val", "train"):
            candidate = dataset_root / split / scene_name / "coord.npy"
            if candidate.exists():
                scene_coord_path = candidate
                break
        if scene_coord_path is None:
            print(f"  {scene_name} - coord.npy not found under {dataset_root}, skip")
            continue

        scene_coord = np.load(scene_coord_path).astype(np.float32)
        scene_pred = np.load(pred_path).reshape(-1)
        if len(scene_coord) != len(scene_pred):
            print(f"  {scene_name} - length mismatch, skip")
            continue

        t0 = time.time()
        match_key = find_source_cloud(
            scene_coord, project["file_keys"], candidate_bounds,
            candidate_point_counts)
        if match_key is None:
            print(f"  {scene_name} - no source cloud match in project, skip")
            continue

        full_pos = load_catalog_cloud_positions(match_key)
        if full_pos is None:
            print(f"  {scene_name} - catalog data missing for {match_key}, skip")
            continue

        full_pred = upsample_predictions(scene_coord, scene_pred, full_pos)
        full_labels = remap_to_registry(full_pred, class_names, name_to_registry)
        write_atomic_npy(library_dir() / "labels" / f"{match_key}.npy", full_labels)

        # Preview labels too — keeps the Train tab filter / row swatches /
        # gallery thumbnails consistent without re-opening each cloud.
        preview_pos = load_catalog_preview_positions(match_key)
        if preview_pos is not None:
            preview_pred = upsample_predictions(scene_coord, scene_pred, preview_pos)
            preview_labels = remap_to_registry(preview_pred, class_names, name_to_registry)
            write_atomic_npy(
                library_dir() / "preview_labels" / f"{match_key}.npy",
                preview_labels,
            )
            prev_msg = f"+preview {len(preview_pos):,}"
        else:
            prev_msg = "(no preview)"

        elapsed = time.time() - t0
        unique, counts = np.unique(full_labels, return_counts=True)
        summary = ", ".join(
            f"{int(u)}={int(c)}" for u, c in zip(unique, counts))
        print(f"  {scene_name} -> {match_key} ({len(full_pos):,} pts {prev_msg}, "
              f"{elapsed:.1f}s) labels: {summary}")
        written += 1

    print(f"\nWrote labels for {written} cloud(s) into the catalog.")
    print("Reopen 3Photon to see them; existing hand labels are in "
          f"{library_dir() / 'backups'}.")


if __name__ == "__main__":
    main()
