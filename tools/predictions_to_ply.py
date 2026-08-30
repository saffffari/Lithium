#!/usr/bin/env python
"""Bake prediction .npy files back into PLY point clouds for visualisation.

After running ``tools/infer_lithium.py`` you have per-scene predictions at
the voxel-downsampled resolution of the staged dataset. Those predictions
can't be applied to the original full-resolution clouds without a spatial
mapping, but they CAN be visualised at the staged resolution by writing
each scene back out as a PLY with vertex colors taken from the predicted
class palette.

The resulting PLYs drop straight into Lithium via File → Import File and
display the model's segmentation as point colors — no label registry
import needed.

Usage:
    python tools/predictions_to_ply.py \\
        --dataset <path-to-exported-dataset> \\
        --predictions <dataset>/training_runs/<run>/predictions \\
        --split test \\
        --output <dataset>/training_runs/<run>/predictions_ply
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def write_ply(path: str, coord: np.ndarray, color_u8: np.ndarray) -> None:
    """Write a colored point cloud as a binary little-endian PLY."""
    n = coord.shape[0]
    verts = np.empty(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    verts['x'] = coord[:, 0]
    verts['y'] = coord[:, 1]
    verts['z'] = coord[:, 2]
    verts['red'] = color_u8[:, 0]
    verts['green'] = color_u8[:, 1]
    verts['blue'] = color_u8[:, 2]
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el], text=False, byte_order='<').write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Lithium dataset root")
    ap.add_argument("--predictions", required=True, help="predictions dir")
    ap.add_argument("--split", default="test", help="train/val/test")
    ap.add_argument("--output", required=True, help="output dir for PLYs")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load class colors from the dataset's classes.json (same colors that
    # were used for export, so the predicted PLYs read in the same palette
    # as the GT labelling did).
    with open(os.path.join(args.dataset, "classes.json")) as f:
        meta = json.load(f)
    class_colors = np.asarray(meta["class_colors"], dtype=np.uint8)
    num_classes = int(meta["num_classes"])
    print(f"Classes ({num_classes}): {meta.get('class_names', [])}")

    split_dir = os.path.join(args.dataset, args.split)
    scenes = sorted(
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
        and os.path.exists(os.path.join(split_dir, d, "coord.npy"))
    )

    written = 0
    for name in scenes:
        scene_dir = os.path.join(split_dir, name)
        pred_path = os.path.join(args.predictions, f"{name}_pred.npy")
        if not os.path.exists(pred_path):
            print(f"  {name} — no prediction, skip")
            continue

        coord = np.load(os.path.join(scene_dir, "coord.npy"))
        pred = np.load(pred_path).astype(np.int64).reshape(-1)

        if len(coord) != len(pred):
            print(f"  {name} — length mismatch coord={len(coord)} pred={len(pred)}, skip")
            continue

        # Out-of-range / 255 (ignore_index) → mid-grey so they read as
        # "no prediction" without dominating the view.
        col = np.full((len(pred), 3), 128, dtype=np.uint8)
        valid = (pred >= 0) & (pred < num_classes)
        col[valid] = class_colors[pred[valid]]

        out_path = os.path.join(args.output, f"{name}_pred.ply")
        write_ply(out_path, coord.astype(np.float32), col)
        unique, counts = np.unique(pred, return_counts=True)
        summary = ", ".join(
            f"{int(u)}={int(c)}" for u, c in zip(unique, counts))
        print(f"  {name} — {len(coord):,} pts, {summary}  ->{Path(out_path).name}")
        written += 1

    print(f"\nWrote {written} PLY(s) to {args.output}")


if __name__ == "__main__":
    main()
