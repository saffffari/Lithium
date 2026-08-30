"""In-app inference latency -> paper/tables/bench_infer.json.

Runs tools/infer_single.py (the exact subprocess the INFER button spawns) on
N Gold247 val bones with the Sonata checkpoint, parsing its own timing lines
(model load, inference). GPU: whatever `nvidia-smi` reports.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DS = pathlib.Path(os.environ.get("GOLD247_DATASET",
                  "/home/alex/Projects/spinelab/cloud_models/yamato_gold247_v1/dataset"))
PY = os.environ.get("LITHIUM_TRAIN_PYTHON", os.path.expanduser("~/miniforge3/envs/lithium-ptv3/bin/python"))
CKPT = ROOT / "training/runs/sonata_full6_gold247_v1/model/model_best.pth"
OUT = ROOT / "paper" / "tables" / "bench_infer.json"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

man = json.load(open(DS / "manifest.json"))
val_keys = sorted(k for k in man if k.startswith("val/"))[:N]
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
except Exception:  # noqa: BLE001
    gpu = "unknown"
runs = []
with tempfile.TemporaryDirectory() as td:
    for k in val_keys:
        inp = pathlib.Path(td) / "in.npz"; outp = pathlib.Path(td) / "out.npy"
        np.savez(inp, coord=np.load(DS / k / "coord.npy").astype(np.float32),
                 color=np.load(DS / k / "color.npy").astype(np.float32),
                 normal=np.load(DS / k / "normal.npy").astype(np.float32))
        t0 = time.perf_counter()
        p = subprocess.run([PY, "-u", str(ROOT / "tools/infer_single.py"), "--checkpoint", str(CKPT),
                            "--input", str(inp), "--output", str(outp), "--num-classes", "6", "--grid-size", "0.5"],
                           capture_output=True, text=True)
        wall = time.perf_counter() - t0
        load = re.search(r"model loaded in ([0-9.]+)s", p.stdout)
        inf = re.search(r"inference ([0-9.]+)s", p.stdout)
        npts = re.search(r"input: ([0-9,]+) points", p.stdout)
        if p.returncode != 0 or not (load and inf):
            print(p.stdout[-800:], p.stderr[-800:]); raise SystemExit(f"infer failed on {k}")
        runs.append({"scene": k, "cloud": man[k]["cloud"], "points": int(npts.group(1).replace(",", "")),
                     "model_load_s": float(load.group(1)), "inference_s": float(inf.group(1)), "wall_s": wall})
        print(f"{man[k]['cloud']:24} load {runs[-1]['model_load_s']:.1f}s  infer {runs[-1]['inference_s']:.2f}s  wall {wall:.1f}s")
out = {"gpu": gpu, "checkpoint": str(CKPT.relative_to(ROOT)), "n_bones": len(runs), "runs": runs,
       "inference_s_median": statistics.median(r["inference_s"] for r in runs),
       "model_load_s_median": statistics.median(r["model_load_s"] for r in runs),
       "wall_s_median": statistics.median(r["wall_s"] for r in runs),
       "points_mean": statistics.mean(r["points"] for r in runs)}
OUT.write_text(json.dumps(out, indent=1))
print("median inference", round(out["inference_s_median"], 2), "s; model load", round(out["model_load_s_median"], 1), "s; gpu", gpu)
