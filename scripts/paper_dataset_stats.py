"""Gold247 composition + catalog stats -> paper/tables/dataset_stats.json.

Reads the frozen Gold247 export (public VerSe bones; manifest gives subject +
level per scene) and the live Lithium catalog. Pure numpy, seconds.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import re
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DS = pathlib.Path(os.environ.get("GOLD247_DATASET",
                  "/home/alex/Projects/spinelab/cloud_models/yamato_gold247_v1/dataset"))
LIB = pathlib.Path(os.environ.get("LITHIUM_LIBRARY_DIR", os.path.expanduser("~/.lithium/library")))
OUT = ROOT / "paper" / "tables" / "dataset_stats.json"

man = json.load(open(DS / "manifest.json"))
classes = json.load(open(DS / "classes.json"))
names = classes["class_names"]

def level_of(cloud: str) -> str:
    m = re.search(r"_([CTL]\d+)\.ply$", cloud)
    return m.group(1) if m else "?"

split_counts = Counter(k.split("/")[0] for k in man)
subjects = defaultdict(set)
levels = Counter()
region = Counter()
pts = []
class_pts = np.zeros(len(names), dtype=np.int64)
per_split_class = {s: np.zeros(len(names), dtype=np.int64) for s in split_counts}
for key, rec in man.items():
    split = key.split("/")[0]
    subjects[split].add(rec["subject"])
    lv = level_of(rec["cloud"])
    levels[lv] += 1
    region[lv[0]] += 1
    seg = np.load(DS / key / "segment.npy").astype(np.int64).reshape(-1)
    pts.append(seg.shape[0])
    bc = np.bincount(seg[(seg >= 0) & (seg < len(names))], minlength=len(names))
    class_pts += bc
    per_split_class[split] += bc

order = [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]
pts = np.array(pts)
stats = {
    "n_bones": len(man),
    "n_subjects": len(set().union(*subjects.values())),
    "splits": {s: {"bones": split_counts[s], "subjects": len(subjects[s])} for s in ("train", "val", "test")},
    "points_per_bone": {"mean": float(pts.mean()), "min": int(pts.min()), "max": int(pts.max()),
                        "total": int(pts.sum())},
    "levels": {lv: int(levels[lv]) for lv in order if levels[lv]},
    "regions": {"C": region["C"], "T": region["T"], "L": region["L"]},
    "class_names": names,
    "class_points": class_pts.tolist(),
    "class_fraction": (class_pts / class_pts.sum()).tolist(),
    "per_split_class_points": {s: v.tolist() for s, v in per_split_class.items()},
}
# live catalog (no patient data leaves this script — counts only)
try:
    idx = json.load(open(LIB / "index.json"))
    projs = json.load(open(LIB / "projects.json"))
    n_models = sum(len(json.load(open(f))) for f in glob.glob(str(LIB / "models" / "proj_*.json")))
    stats["catalog"] = {"clouds": len(idx), "projects": len([p for p in projs if p != "proj:sandbox"]),
                        "models": n_models,
                        "label_files": len(glob.glob(str(LIB / "labels" / "*" / "*.npy")))}
except Exception as e:  # noqa: BLE001
    stats["catalog"] = {"error": str(e)}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(stats, indent=1))
print(json.dumps({k: v for k, v in stats.items() if k not in ("class_points", "per_split_class_points")}, indent=1))
