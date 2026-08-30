#!/usr/bin/env python
"""Run single-pass inference on S3DIS Area_5 using a trained PT-v3 checkpoint.

Simpler than Pointcept's full TTA test pipeline — no augmentation ensemble,
just one forward pass per voxelized room. Saves per-room predictions as .npy
files that can be imported back into Lithium.

Usage:
    # From the lithium-ptv3 conda env:
    python tools/infer_s3dis.py \
        --checkpoint training/runs/s3dis_smoke_test/model/model_best.pth \
        --data-root data/s3dis-raw \
        --split Area_5 \
        --output training/runs/s3dis_smoke_test/predictions
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add Pointcept to path
POINTCEPT_DIR = str(Path(__file__).resolve().parent.parent / "training" / "pointcept")
sys.path.insert(0, POINTCEPT_DIR)

S3DIS_CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window",
    "door", "table", "chair", "sofa", "bookcase", "board", "clutter",
]

GRID_SIZE = 0.02  # 2cm voxel grid, same as training


def build_model(checkpoint_path: str, num_classes: int = 13):
    """Build PT-v3m1 model and load checkpoint."""
    from pointcept.models import build_model as _build_model

    model_cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=num_classes,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1",
            in_channels=6,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024, 1024, 1024, 1024, 1024),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024, 1024, 1024, 1024),
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            drop_path=0.3,
            shuffle_orders=True,
            pre_norm=True,
            enable_rpe=False,
            enable_flash=False,
            upcast_attention=False,
            upcast_softmax=False,
            enc_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
            pdnorm_decouple=True,
            pdnorm_adaptive=False,
            pdnorm_affine=True,
            pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        ),
    )

    model = _build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    # Strip "module." prefix if present (DDP artifact)
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=True)
    model = model.cuda().eval()
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")
    return model


def voxelize(coord, color, normal, grid_size=GRID_SIZE):
    """Grid-sample points. In test mode returns a list of fragment dicts,
    each with 'coord', 'color', 'normal', 'grid_coord', 'index'."""
    from pointcept.datasets.transform import GridSample
    data = dict(coord=coord, color=color, normal=normal)
    transform = GridSample(
        grid_size=grid_size, hash_type="fnv", mode="test",
        return_grid_coord=True,
    )
    result = transform(data)
    # In test mode, returns list of fragment dicts
    if isinstance(result, dict):
        return [result]
    return result


def find_rooms(data_root: str, split: str) -> list[dict]:
    """Find all rooms in a split."""
    split_dir = os.path.join(data_root, split)
    rooms = []
    for name in sorted(os.listdir(split_dir)):
        room_dir = os.path.join(split_dir, name)
        if os.path.isdir(room_dir) and os.path.exists(os.path.join(room_dir, "coord.npy")):
            rooms.append({"name": name, "path": room_dir})
    return rooms


@torch.no_grad()
def predict_room(model, room: dict, num_classes: int = 13) -> np.ndarray:
    """Run single-pass inference on one room. Returns per-point predictions.

    Uses only the first voxelized fragment (which covers all points via
    the index mapping) to keep memory bounded. Processes one fragment at
    a time and immediately frees GPU memory.
    """
    coord = np.load(os.path.join(room["path"], "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(room["path"], "color.npy")).astype(np.float32)
    if color.max() > 1.0:
        color = color / 255.0
    normal_path = os.path.join(room["path"], "normal.npy")
    normal = np.load(normal_path).astype(np.float32) if os.path.exists(normal_path) else np.zeros_like(coord)

    n_points = len(coord)

    # Center shift
    coord -= coord.mean(0)

    # Normalize color to [-1, 1]
    color_norm = (color - 0.5) / 0.5

    # Voxelize — returns list of fragment dicts in test mode
    fragments = voxelize(coord, color_norm, normal)

    # Use first fragment only — it covers all points via different
    # voxel hash seeds but a single pass is enough for a smoke test.
    pred = np.zeros(n_points, dtype=np.int32)

    frag = fragments[0]
    vox_coord = frag["coord"]
    vox_color = frag["color"]
    vox_normal = frag["normal"]
    grid_coord = frag["grid_coord"]
    index = frag["index"]  # maps voxel points → original point indices

    feat = np.concatenate([vox_color, vox_normal], axis=1)
    offset = np.array([len(vox_coord)], dtype=np.int64)

    input_dict = dict(
        coord=torch.from_numpy(vox_coord).float().cuda(),
        grid_coord=torch.from_numpy(grid_coord).int().cuda(),
        feat=torch.from_numpy(feat).float().cuda(),
        offset=torch.from_numpy(offset).long().cuda(),
    )

    output = model(input_dict)
    logits = output["seg_logits"]  # (M, num_classes)
    pred_vox = logits.argmax(dim=1).cpu().numpy().astype(np.int32)

    # Free GPU immediately
    del input_dict, output, logits
    torch.cuda.empty_cache()

    # Map voxel predictions back to original points
    pred[index] = pred_vox

    return pred


def main():
    parser = argparse.ArgumentParser(description="Run S3DIS inference with trained PT-v3")
    parser.add_argument("--checkpoint", required=True, help="Path to model_best.pth")
    parser.add_argument("--data-root", required=True, help="Path to S3DIS data root")
    parser.add_argument("--split", default="Area_5", help="Which split to test")
    parser.add_argument("--output", required=True, help="Output directory for predictions")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Write classes.json sidecar for import
    classes_json = os.path.join(args.output, "classes.json")
    if not os.path.exists(classes_json):
        meta = {
            "num_classes": len(S3DIS_CLASSES),
            "class_names": S3DIS_CLASSES,
            "class_colors": [
                [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
                [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
                [64, 64, 64],
            ],
            "ignore_index": 255,
        }
        with open(classes_json, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {classes_json}")

    print(f"Loading model from {args.checkpoint} ...")
    model = build_model(args.checkpoint)

    rooms = find_rooms(args.data_root, args.split)
    print(f"Found {len(rooms)} rooms in {args.split}")

    total_t0 = time.time()
    for i, room in enumerate(rooms, 1):
        out_path = os.path.join(args.output, f"{room['name']}_pred.npy")
        if os.path.exists(out_path):
            print(f"  [{i}/{len(rooms)}] {room['name']} — cached, skipping")
            continue

        t0 = time.time()
        try:
            pred = predict_room(model, room)
            np.save(out_path, pred)
            elapsed = time.time() - t0
            unique = np.unique(pred)
            print(f"  [{i}/{len(rooms)}] {room['name']} — {len(pred):,} pts, "
                  f"{len(unique)} classes, {elapsed:.1f}s")
        except Exception as e:
            print(f"  [{i}/{len(rooms)}] {room['name']} — ERROR: {e}")
            torch.cuda.empty_cache()

    total = time.time() - total_t0
    print(f"\nDone in {total:.0f}s")


if __name__ == "__main__":
    main()
