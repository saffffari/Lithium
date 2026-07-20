#!/usr/bin/env python
"""Single-cloud PT-v3 inference subprocess.

Reads a single point cloud from a temp ``.npz`` (coord, color, normal),
runs one forward pass through the trained checkpoint, writes per-point
class predictions to a ``.npy`` file. Intended to be spawned from the
3Photon GUI's "Predict Current Cloud" command — fast, no dataset
directory dance.

Usage (run from inside the 3photon-ptv3 conda env):

    python tools/infer_single.py \\
        --checkpoint <model_best.pth> \\
        --input <input.npz>      # keys: coord, color, normal (optional)
        --output <pred.npy> \\
        --num-classes 2 \\
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

POINTCEPT_DIR = str(Path(__file__).resolve().parent.parent / "training" / "pointcept")
EXT_DIR = str(Path(__file__).resolve().parent.parent / "src" / "training" / "pointcept_ext")
sys.path.insert(0, POINTCEPT_DIR)
sys.path.insert(0, EXT_DIR)

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def build_model(checkpoint_path: str, num_classes: int | None = None):
    """Build the inference model with the checkpoint's true class count.

    ``num_classes`` from the CLI is treated as a hint only. The
    authoritative value is the shape of ``seg_head.weight`` in the
    checkpoint — that's what the model was actually trained against and
    is the only value ``load_state_dict(strict=True)`` will accept. If
    the caller's hint disagrees we emit a one-line warning and use the
    checkpoint's value anyway. This makes the script robust against the
    caller (the 3Photon GUI) computing num_classes from a stale project
    registry or a different project's ontology — a foot-gun that caused
    repeated load_state_dict mismatches in the Contact Sheets RUN
    INFERENCE flow.
    """
    from pointcept.models import build_model as _build_model

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in ckpt["state_dict"].items()}
    ckpt_head = state.get("seg_head.weight")
    if ckpt_head is None:
        if num_classes is None:
            raise RuntimeError(
                "checkpoint has no seg_head.weight and no --num-classes "
                "given — can't infer model width.")
        true_nc = int(num_classes)
    else:
        true_nc = int(ckpt_head.shape[0])
        if num_classes is not None and int(num_classes) != true_nc:
            print(f"[infer_single] caller said --num-classes={num_classes} "
                  f"but checkpoint seg_head is ({true_nc}, ...); using "
                  f"checkpoint value.", flush=True)

    model_cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=true_nc,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1", in_channels=9,
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
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True, enable_rpe=False,
            enable_flash=False, upcast_attention=False, upcast_softmax=False,
            enc_mode=False, pdnorm_bn=False, pdnorm_ln=False,
            pdnorm_decouple=True, pdnorm_adaptive=False, pdnorm_affine=True,
            pdnorm_conditions=("Vertebrae",),
        ),
    )
    model = _build_model(model_cfg)
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


@torch.no_grad()
def predict(model, coord, color, normal, grid_size: float) -> np.ndarray:
    from pointcept.datasets.transform import GridSample

    n_points = len(coord)
    coord = coord.astype(np.float32) - coord.mean(0)        # CenterShift
    color = color.astype(np.float32)
    if color.max() > 1.0:
        color = color / 255.0
    color_norm = (color - 0.5) / 0.5                         # NormalizeColor
    if normal is None:
        normal = np.zeros_like(coord, dtype=np.float32)
    else:
        normal = normal.astype(np.float32)

    transform = GridSample(
        grid_size=grid_size, hash_type="fnv", mode="test",
        return_grid_coord=True,
    )
    result = transform(dict(coord=coord, color=color_norm, normal=normal))
    fragments = [result] if isinstance(result, dict) else list(result)
    if len(fragments) > 1:
        print(f"[infer_single] note: GridSample produced {len(fragments)} "
              f"fragments for this cloud — running all of them and "
              f"merging predictions so no points are silently left at "
              f"class 0.", flush=True)

    pred = np.zeros(n_points, dtype=np.int32)
    covered = np.zeros(n_points, dtype=bool)
    for frag in fragments:
        feat = np.concatenate(
            [frag["coord"], frag["color"], frag["normal"]], axis=1)
        offset = np.array([len(frag["coord"])], dtype=np.int64)
        inp = dict(
            coord=torch.from_numpy(frag["coord"]).float().cuda(),
            grid_coord=torch.from_numpy(frag["grid_coord"]).int().cuda(),
            feat=torch.from_numpy(feat).float().cuda(),
            offset=torch.from_numpy(offset).long().cuda(),
        )
        output = model(inp)
        pred_vox = output["seg_logits"].argmax(dim=1).cpu().numpy().astype(np.int32)
        del inp, output
        torch.cuda.empty_cache()
        idx = frag["index"]
        pred[idx] = pred_vox
        covered[idx] = True

    uncovered = int((~covered).sum())
    if uncovered:
        print(f"[infer_single] warning: {uncovered:,} of {n_points:,} "
              f"points were not covered by any fragment; left as class 0",
              flush=True)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", required=True, help=".npz with coord/color/normal")
    ap.add_argument("--output", required=True, help="output .npy of class ids")
    # Hint only — the checkpoint's seg_head shape is authoritative. See
    # build_model for the rationale.
    ap.add_argument("--num-classes", type=int, required=False, default=None)
    ap.add_argument("--grid-size", type=float, default=0.5)
    args = ap.parse_args()

    print(f"[infer_single] loading {args.checkpoint}", flush=True)
    t0 = time.time()
    model = build_model(args.checkpoint, args.num_classes)
    print(f"[infer_single] model loaded in {time.time()-t0:.1f}s", flush=True)

    z = np.load(args.input)
    coord = z["coord"]
    color = z["color"] if "color" in z.files else np.full(coord.shape, 0.5, dtype=np.float32)
    normal = z["normal"] if "normal" in z.files else None
    print(f"[infer_single] input: {len(coord):,} points", flush=True)

    t1 = time.time()
    pred = predict(model, coord, color, normal, args.grid_size)
    print(f"[infer_single] inference {time.time()-t1:.2f}s", flush=True)

    np.save(args.output, pred)
    unique, counts = np.unique(pred, return_counts=True)
    summary = ", ".join(f"{int(u)}={int(c)}" for u, c in zip(unique, counts))
    print(f"[infer_single] wrote {args.output}  classes: {summary}", flush=True)


if __name__ == "__main__":
    main()
