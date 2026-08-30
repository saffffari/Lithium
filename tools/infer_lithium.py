#!/usr/bin/env python
"""Run single-pass inference on a Lithium-format dataset using a trained
PT-v3 checkpoint.

Sister of ``tools/infer_s3dis.py`` — same single-pass voxel inference, but
parameterised for our model architecture (in_channels=9 with normals, our
own ``Vertebrae`` pdnorm condition) and num_classes auto-detected from
the dataset's ``classes.json`` sidecar.

Usage (run inside the lithium-ptv3 conda env):

    python tools/infer_lithium.py \\
        --checkpoint dataset/training_runs/<run>/model/model_best.pth \\
        --data-root dataset \\
        --split test \\
        --output dataset/training_runs/<run>/predictions \\
        --grid-size 0.5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add Pointcept + the Lithium pointcept_ext dir to path so the dataset
# class is registered (mirrors what the training runner does for the
# main process — inference doesn't need DataLoader workers, so the
# worker_init shim isn't required here).
POINTCEPT_DIR = str(Path(__file__).resolve().parent.parent / "training" / "pointcept")
EXT_DIR = str(Path(__file__).resolve().parent.parent / "src" / "training" / "pointcept_ext")
sys.path.insert(0, POINTCEPT_DIR)
sys.path.insert(0, EXT_DIR)

GRID_SIZE_DEFAULT = 0.5  # millimetres for VerSe-style CT-derived clouds


def load_classes_json(data_root: str) -> dict:
    """Load classes.json from the dataset root. Returns the parsed dict."""
    p = os.path.join(data_root, "classes.json")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"classes.json not found at {p}. Inference needs the dataset's "
            f"classes.json (written by the export step) to know how many "
            f"classes to build the model with.")
    with open(p) as f:
        return json.load(f)


def build_model(checkpoint_path: str, num_classes: int):
    """Build PT-v3m1 with the same architecture used by Lithium training."""
    from pointcept.models import build_model as _build_model

    model_cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=num_classes,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1",
            in_channels=9,            # xyz + color (3) + normal (3)
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
            pdnorm_conditions=("Vertebrae",),
        ),
    )

    model = _build_model(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(clean, strict=True)
    model = model.cuda().eval()
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', '?')}")
    return model


def voxelize(coord, color, normal, grid_size: float):
    """Grid-sample. Returns list of fragment dicts (test mode)."""
    from pointcept.datasets.transform import GridSample
    data = dict(coord=coord, color=color, normal=normal)
    transform = GridSample(
        grid_size=grid_size, hash_type="fnv", mode="test",
        return_grid_coord=True,
    )
    result = transform(data)
    if isinstance(result, dict):
        return [result]
    return result


def find_scenes(data_root: str, split: str) -> list[dict]:
    split_dir = os.path.join(data_root, split)
    scenes = []
    for name in sorted(os.listdir(split_dir)):
        scene_dir = os.path.join(split_dir, name)
        if (os.path.isdir(scene_dir)
                and os.path.exists(os.path.join(scene_dir, "coord.npy"))):
            scenes.append({"name": name, "path": scene_dir})
    return scenes


@torch.no_grad()
def predict_scene(model, scene: dict, grid_size: float) -> np.ndarray:
    """Single-pass inference. Returns (N,) int32 per-point class predictions."""
    coord = np.load(os.path.join(scene["path"], "coord.npy")).astype(np.float32)
    color = np.load(os.path.join(scene["path"], "color.npy")).astype(np.float32)
    if color.max() > 1.0:
        color = color / 255.0
    normal_path = os.path.join(scene["path"], "normal.npy")
    if os.path.exists(normal_path):
        normal = np.load(normal_path).astype(np.float32)
    else:
        normal = np.zeros_like(coord)

    n_points = len(coord)
    coord = coord - coord.mean(0)         # CenterShift
    color_norm = (color - 0.5) / 0.5       # NormalizeColor → [-1, 1]

    fragments = voxelize(coord, color_norm, normal, grid_size)
    frag = fragments[0]                    # single-pass: one fragment

    feat = np.concatenate([frag["coord"], frag["color"], frag["normal"]], axis=1)
    offset = np.array([len(frag["coord"])], dtype=np.int64)

    input_dict = dict(
        coord=torch.from_numpy(frag["coord"]).float().cuda(),
        grid_coord=torch.from_numpy(frag["grid_coord"]).int().cuda(),
        feat=torch.from_numpy(feat).float().cuda(),
        offset=torch.from_numpy(offset).long().cuda(),
    )
    output = model(input_dict)
    logits = output["seg_logits"]
    pred_vox = logits.argmax(dim=1).cpu().numpy().astype(np.int32)
    del input_dict, output, logits
    torch.cuda.empty_cache()

    pred = np.zeros(n_points, dtype=np.int32)
    pred[frag["index"]] = pred_vox
    return pred


def main():
    parser = argparse.ArgumentParser(description="Lithium PT-v3 inference")
    parser.add_argument("--checkpoint", required=True, help="model_best.pth path")
    parser.add_argument("--data-root", required=True, help="Lithium dataset root")
    parser.add_argument("--split", default="test", help="train / val / test")
    parser.add_argument("--output", required=True, help="prediction output dir")
    parser.add_argument("--grid-size", type=float, default=GRID_SIZE_DEFAULT,
                        help="Voxel grid size; must match training grid_size")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Pull num_classes + names + colors from the dataset's classes.json so
    # the import-back step has matching metadata.
    meta = load_classes_json(args.data_root)
    num_classes = int(meta["num_classes"])
    print(f"Dataset classes: {num_classes} ({', '.join(meta.get('class_names', []))})")

    # Carry the dataset's classes.json forward into the predictions dir so
    # the import side has everything it needs in one place.
    out_classes = os.path.join(args.output, "classes.json")
    if not os.path.exists(out_classes):
        with open(out_classes, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Wrote {out_classes}")

    # lithium_dataset import isn't strictly needed here (we're not
    # going through Pointcept's DataLoader) but importing it is cheap and
    # mirrors training so future expansion of the inference path stays
    # consistent.
    try:
        import lithium_dataset  # noqa: F401
    except Exception as e:
        print(f"[warn] lithium_dataset import failed: {e}")

    print(f"Loading model from {args.checkpoint} ...")
    model = build_model(args.checkpoint, num_classes=num_classes)

    scenes = find_scenes(args.data_root, args.split)
    print(f"Found {len(scenes)} scenes in {args.split}")

    total_t0 = time.time()
    for i, scene in enumerate(scenes, 1):
        out_path = os.path.join(args.output, f"{scene['name']}_pred.npy")
        if os.path.exists(out_path):
            print(f"  [{i}/{len(scenes)}] {scene['name']} — cached, skipping")
            continue

        t0 = time.time()
        try:
            pred = predict_scene(model, scene, args.grid_size)
            np.save(out_path, pred)
            elapsed = time.time() - t0
            unique, counts = np.unique(pred, return_counts=True)
            class_summary = ", ".join(
                f"{int(u)}={int(c)}" for u, c in zip(unique, counts))
            print(f"  [{i}/{len(scenes)}] {scene['name']} — {len(pred):,} pts, "
                  f"{elapsed:.1f}s, classes: {class_summary}")
        except Exception as e:
            print(f"  [{i}/{len(scenes)}] {scene['name']} — ERROR: {e}")
            torch.cuda.empty_cache()

    total = time.time() - total_t0
    print(f"\nDone in {total:.0f}s")
    print(f"Predictions in: {args.output}")


if __name__ == "__main__":
    main()
