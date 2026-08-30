#!/usr/bin/env python
"""Import pre-processed S3DIS scenes into Lithium-compatible PLY files.

Reads the Pointcept-format per-room directories (coord.npy, color.npy,
segment.npy) and writes PLY + companion _labels.npy for each room.
Lithium auto-detects the companion labels on import.

Setup
-----
1. Download pre-processed S3DIS from Hugging Face:
       git lfs install
       git clone https://huggingface.co/datasets/Pointcept/s3dis-compressed

2. Run this script:
       python tools/import_s3dis.py <path_to_s3dis> --output <output_dir>

3. Open the output folder in Lithium.

The 13 S3DIS classes:
    0 ceiling, 1 floor, 2 wall, 3 beam, 4 column, 5 window,
    6 door, 7 table, 8 chair, 9 sofa, 10 bookcase, 11 board, 12 clutter
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data.point_cloud import PointCloudData
from src.data.ply_writer import save_ply

S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
]


def find_scenes(root: str) -> list[dict]:
    """Find all Pointcept scene directories under root."""
    scenes = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "coord.npy" in filenames:
            name = os.path.basename(dirpath)
            # Try to extract area from path
            parts = Path(dirpath).parts
            area = ""
            for p in parts:
                if p.lower().startswith("area"):
                    area = p
                    break
            scenes.append({
                "name": name,
                "area": area,
                "path": dirpath,
            })
    scenes.sort(key=lambda s: (s["area"], s["name"]))
    return scenes


def convert_scene(info: dict, output_dir: str) -> dict:
    """Convert a single Pointcept scene to PLY + labels."""
    scene_dir = info["path"]
    area_out = os.path.join(output_dir, info["area"]) if info["area"] else output_dir
    os.makedirs(area_out, exist_ok=True)

    ply_path = os.path.join(area_out, f"{info['name']}.ply")
    npy_path = os.path.join(area_out, f"{info['name']}_labels.npy")

    t0 = time.time()

    coord = np.load(os.path.join(scene_dir, "coord.npy")).astype(np.float32)
    color_path = os.path.join(scene_dir, "color.npy")
    if os.path.exists(color_path):
        color = np.load(color_path).astype(np.float32)
        if color.max() > 1.0:
            color = color / 255.0
    else:
        color = np.ones((len(coord), 3), dtype=np.float32)

    seg_path = os.path.join(scene_dir, "segment.npy")
    if os.path.exists(seg_path):
        labels = np.load(seg_path).astype(np.int32).ravel()
        # S3DIS uses 0-indexed classes, shift to 1-indexed for Lithium
        # (0 = unlabeled in Lithium)
        labels = labels + 1
    else:
        labels = np.zeros(len(coord), dtype=np.int32)

    cloud = PointCloudData(
        positions=coord,
        colors=color,
        labels=labels,
        file_path=ply_path,
        metadata={"source_unit": "m"},
    )

    save_ply(cloud, ply_path, include_labels=True)
    np.save(npy_path, cloud.labels)

    elapsed = time.time() - t0
    unique = sorted(int(v) for v in np.unique(labels) if v > 0)

    return {
        "name": info["name"],
        "area": info["area"],
        "points": cloud.point_count,
        "classes": unique,
        "time": round(elapsed, 1),
    }


def write_ontology(output_dir: str):
    """Write an S3DIS class mapping JSON for reference."""
    os.makedirs(output_dir, exist_ok=True)
    mapping = {i + 1: name for i, name in enumerate(S3DIS_CLASSES)}
    path = os.path.join(output_dir, "s3dis_classes.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert pre-processed S3DIS to Lithium PLY files.")
    parser.add_argument("s3dis_root", type=str,
                        help="Path to pre-processed S3DIS (contains Area_*/room_*/coord.npy)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: <s3dis_root>/s3dis_ply)")
    parser.add_argument("--area", type=str, default=None,
                        help="Convert only this area (e.g. Area_5)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing files")
    args = parser.parse_args()

    root = args.s3dis_root
    if not os.path.isdir(root):
        print(f"Error: not found: {root}")
        sys.exit(1)

    output_dir = args.output or os.path.join(root, "s3dis_ply")

    scenes = find_scenes(root)
    if args.area:
        scenes = [s for s in scenes if s["area"].lower() == args.area.lower()]

    if not scenes:
        print("No scenes found. Check the directory structure.")
        sys.exit(1)

    print(f"S3DIS root:  {root}")
    print(f"Output:      {output_dir}")
    print(f"Scenes:      {len(scenes)}")
    print()

    write_ontology(output_dir)

    converted = 0
    skipped = 0
    errors = 0
    total_t0 = time.time()

    for i, info in enumerate(scenes, 1):
        area_out = os.path.join(output_dir, info["area"]) if info["area"] else output_dir
        ply_path = os.path.join(area_out, f"{info['name']}.ply")

        if os.path.exists(ply_path) and not args.force:
            skipped += 1
            continue

        try:
            print(f"  [{i}/{len(scenes)}] {info['area']}/{info['name']} ... ", end="", flush=True)
            meta = convert_scene(info, output_dir)
            print(f"{meta['points']:,} pts, {meta['time']}s")
            converted += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

    total = time.time() - total_t0
    print(f"\nDone in {total:.0f}s, {converted} converted, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
