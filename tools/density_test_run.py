#!/usr/bin/env python
"""Snapshot ground-truth labels → run fight_club_heavy inference on
the spinelab_density_test_g1 project → compute per-class IoU vs the
snapshots.

This is the actual density-test result. Does fight_club_heavy
(trained on ~10k-point clouds) generalize to 32k-point clouds without
retraining?

The comparison:
  - Ground truth = labels propagated from the original sparse clouds
    via KD-tree (set up by density_test_setup.py). Same anatomical
    labels, just regridded onto the dense geometry.
  - Predictions = fight_club_heavy's output on the dense clouds.

Per-class IoU between those two tells us whether density was the
underused lever.

Usage::

    python tools/density_test_run.py
    python tools/density_test_run.py --grid-size 0.25  # OOD test
    python tools/density_test_run.py --limit 10        # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess as sp
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


PROJECT_NAME = "spinelab_density_test_g1"
CHECKPOINT = Path(r"D:/3Photon/dataset/training_runs/fight_club_heavy_1778724923/model/model_best.pth")
CLASSES_JSON = Path(r"D:/3Photon/dataset/training_runs/fight_club_heavy_1778724923/classes.json")
# Inference subprocess needs PyTorch + Pointcept. The 3Photon .venv
# doesn't have torch installed; use the dedicated 3photon-ptv3 conda
# env that the app's TRAIN tab uses by default. Read from prefs so we
# match whatever the user has configured.
def _resolve_python_exe() -> str:
    from src.utils.prefs import load_prefs
    p = load_prefs()
    return p.get("train_python_exe", sys.executable)

PYTHON_EXE = _resolve_python_exe()


def _resolve_class_map(label_registry):
    """Build (class_names, cls_to_rid) from the checkpoint's classes.json
    + the project's label registry. Mirrors what panels._resolve_inference_class_map
    does but standalone."""
    if not CLASSES_JSON.is_file():
        raise FileNotFoundError(f"classes.json missing at {CLASSES_JSON}")
    meta = json.loads(CLASSES_JSON.read_text())
    class_names = list(meta.get("class_names", []))
    if not class_names:
        raise RuntimeError("classes.json had no class_names")

    name_to_rid: dict[str, int] = {}
    for info in label_registry.all_labels():
        if info.id != 0:
            name_to_rid[info.name.lower()] = int(info.id)
    cls_to_rid = np.zeros(len(class_names), dtype=np.int32)
    for i, name in enumerate(class_names):
        cls_to_rid[i] = name_to_rid.get(name.lower(), 0)
    return class_names, cls_to_rid


def _run_inference_one(cloud, class_names, cls_to_rid, grid_size: float):
    """Run the inference subprocess for one cloud and return mapped labels."""
    from src.export.dataset_export import compute_normals

    tmp_dir = tempfile.mkdtemp(prefix="3photon_dense_infer_")
    try:
        in_path = os.path.join(tmp_dir, "input.npz")
        out_path = os.path.join(tmp_dir, "pred.npy")
        coord = cloud.positions.astype("float32")
        color = (cloud.colors.astype("float32")
                 if cloud.colors is not None else np.full_like(coord, 0.5))
        normal = (cloud.scalars.get("normal")
                  if hasattr(cloud, "scalars") and cloud.scalars else None)
        if normal is None:
            normal = compute_normals(coord)
        np.savez(in_path, coord=coord, color=color,
                 normal=normal.astype("float32"))

        infer_script = _REPO_ROOT / "tools" / "infer_single.py"
        cmd = [
            PYTHON_EXE, "-u", str(infer_script),
            "--checkpoint", str(CHECKPOINT),
            "--input", in_path,
            "--output", out_path,
            "--num-classes", str(len(class_names)),
            "--grid-size", str(grid_size),
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = sp.run(cmd, capture_output=True, text=True,
                      encoding="utf-8", errors="replace",
                      env=env, timeout=300)
        if proc.returncode != 0:
            print(f"  [infer] returncode={proc.returncode}")
            print("STDOUT:", proc.stdout[-500:])
            print("STDERR:", proc.stderr[-500:])
            return None
        if not os.path.isfile(out_path):
            return None
        pred = np.load(out_path).reshape(-1)
        if pred.shape[0] != cloud.point_count:
            print(f"  [infer] length mismatch: pred={pred.shape[0]}, cloud={cloud.point_count}")
            return None
        valid = (pred >= 0) & (pred < len(class_names))
        new_labels = np.zeros(cloud.point_count, dtype=np.int32)
        new_labels[valid] = cls_to_rid[pred[valid]]
        return new_labels
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _per_class_iou(gt: np.ndarray, pred: np.ndarray, registry_ids: list[int]):
    """Per-class IoU on a single cloud. Returns dict {rid: iou}."""
    out = {}
    for rid in registry_ids:
        g = (gt == rid)
        p = (pred == rid)
        union = int((g | p).sum())
        if union == 0:
            out[rid] = float("nan")  # neither GT nor pred has this class
        else:
            inter = int((g & p).sum())
            out[rid] = inter / union
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--grid-size", type=float, default=0.5,
                    help="GridSample voxel size for inference (default "
                         "0.5 mm — matches training; try 0.25 to push "
                         "the model OOD on the density gain).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only the first N clouds (smoke test).")
    args = ap.parse_args()

    from src.data.library_catalog import LibraryCatalog
    from src.data.cloud_store import (
        load_cloud_data, load_cloud_labels, save_cloud_labels,
    )
    from src.data.labels import LabelRegistry

    catalog = LibraryCatalog()
    proj = None
    for p in catalog.projects.values():
        if p.name == PROJECT_NAME:
            proj = p
            break
    if proj is None:
        print(f"ERROR: project {PROJECT_NAME!r} not found")
        return 2

    label_registry = LabelRegistry.from_json(proj.ontology_data)
    registry_ids = sorted(info.id for info in label_registry.all_labels())
    id_to_name = {info.id: info.name for info in label_registry.all_labels()}

    class_names, cls_to_rid = _resolve_class_map(label_registry)
    print(f"Project: {proj.name}")
    print(f"Checkpoint: {CHECKPOINT.parent.parent.name}")
    print(f"Classes (model order): {class_names}")
    print(f"cls_to_rid mapping: {dict(enumerate(int(x) for x in cls_to_rid))}")
    print(f"Inference grid_size: {args.grid_size} mm")
    print()

    file_keys = list(proj.file_keys)
    if args.limit and args.limit > 0:
        file_keys = file_keys[:args.limit]
    print(f"Processing {len(file_keys)} clouds...")
    print()

    # Per-class running totals across all clouds (micro-average style)
    inter_total = {rid: 0 for rid in registry_ids}
    union_total = {rid: 0 for rid in registry_ids}
    cloud_count = 0
    t0 = time.time()

    for i, fk in enumerate(file_keys, start=1):
        entry = catalog.entries.get(fk)
        if entry is None:
            continue
        name = Path(entry.file_path).name

        # 1) Load cloud + the propagated GT labels currently on disk.
        cached = load_cloud_data(fk)
        if cached is None:
            print(f"  [{i}/{len(file_keys)}] {name}: cache miss")
            continue
        cloud, _ = cached
        gt = load_cloud_labels(fk)
        if gt is None or len(gt) != cloud.point_count:
            print(f"  [{i}/{len(file_keys)}] {name}: GT missing or length mismatch")
            continue
        gt = gt.astype(np.int32)

        # 2) Run inference (returns mapped registry-id labels).
        pred = _run_inference_one(cloud, class_names, cls_to_rid, args.grid_size)
        if pred is None:
            print(f"  [{i}/{len(file_keys)}] {name}: inference failed")
            continue

        # 3) Save predictions in place (overwrites the propagated GT —
        #    intentional, so the project reflects model output after
        #    this script runs).
        save_cloud_labels(fk, pred.astype(np.int32))

        # 4) Accumulate per-class confusion-set sizes for micro-IoU.
        for rid in registry_ids:
            g_mask = (gt == rid)
            p_mask = (pred == rid)
            inter_total[rid] += int((g_mask & p_mask).sum())
            union_total[rid] += int((g_mask | p_mask).sum())
        cloud_count += 1

        if i == 1 or i % 10 == 0 or i == len(file_keys):
            per_cloud = _per_class_iou(gt, pred, registry_ids)
            iou_str = " ".join(
                f"{id_to_name.get(rid, str(rid))[:6]}={per_cloud[rid]:.2f}"
                for rid in registry_ids if not np.isnan(per_cloud[rid]))
            print(f"  [{i}/{len(file_keys)}] {name}: {iou_str}")

    wall = time.time() - t0
    print()
    print(f"Done in {wall:.1f}s ({cloud_count} clouds processed)")
    print()

    # Micro-averaged per-class IoU across the whole project.
    print("=" * 70)
    print(f"Micro-averaged per-class IoU (across {cloud_count} dense clouds)")
    print("=" * 70)
    print(f"{'class':<24s} {'IoU':>8s} {'inter pts':>12s} {'union pts':>12s}")
    print("-" * 60)
    miou_terms = []
    for rid in registry_ids:
        if union_total[rid] == 0:
            iou = float("nan")
        else:
            iou = inter_total[rid] / union_total[rid]
            miou_terms.append(iou)
        name = id_to_name.get(rid, f"id={rid}")
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "  n/a "
        print(f"{name:<24s} {iou_str:>8s} {inter_total[rid]:>12,d} {union_total[rid]:>12,d}")
    print("-" * 60)
    if miou_terms:
        print(f"{'mean IoU':<24s} {np.mean(miou_terms):.4f}")
    print()
    print("Reference: fight_club_heavy's training val mIoU was ~0.79")
    print("If dense-test mIoU is similar -> model generalizes to 32k density")
    print("If higher -> density helped")
    print("If lower -> distribution shift / GT propagation noise dominates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
