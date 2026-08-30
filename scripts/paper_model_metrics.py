"""EP4 val results for both in-app-trained models -> paper/tables/model_metrics.json.

Sources (all produced by training/runs/eval_precision.py + local_axis.py on the
38 Gold247 val bones, public VerSe data):
  training/runs/sonata_full6_gold247_v1/{precision_val.json,axis_val.json,val_dump}
  training/runs/baselines/yamato_gold247_v1_ep4_val_{precision.json}, yamato_axis_val.json, *_dump
Per-class precision/recall/IoU and confidence-gated precision are re-derived from the
dumps (so the two models are scored by the same code path) and cross-checked against
the JSONs' pooled values.
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "training" / "runs"
OUT = ROOT / "paper" / "tables" / "model_metrics.json"
MODELS = {
    "sonata": {"label": "Sonata (PT-v3m2, fine-tuned)", "prec": RUNS / "sonata_full6_gold247_v1/precision_val.json",
               "axis": RUNS / "sonata_full6_gold247_v1/axis_val.json", "dump": RUNS / "sonata_full6_gold247_v1/val_dump",
               "log": RUNS / "sonata_full6_gold247_v1/train.log"},
    "yamato": {"label": "Yamato (PT-v3m1, from scratch)", "prec": RUNS / "baselines/yamato_gold247_v1_ep4_val_precision.json",
               "axis": RUNS / "baselines/yamato_axis_val.json", "dump": RUNS / "baselines/yamato_gold247_v1_ep4_val_dump",
               "log": pathlib.Path("/home/alex/Projects/spinelab/cloud_models/yamato_gold247_v1/run/train.log")},
}
THRESH = 0.99


def score_dump(dump: pathlib.Path):
    meta = json.load(open(dump / "meta.json"))
    names = meta["names"]; K = len(names)
    conf = np.zeros((K, K), dtype=np.int64)
    hi_tp = np.zeros(K, dtype=np.int64); hi_n = np.zeros(K, dtype=np.int64)
    per_bone = []
    for f in sorted(glob.glob(str(dump / "val_*.npz")), key=lambda p: int(re.search(r"val_(\d+)", p).group(1))):
        d = np.load(f)
        g = d["gt"].astype(np.int64); p = d["pred"].astype(np.int64); pmax = d["pmax"].astype(np.float32)
        conf += np.bincount(g * K + p, minlength=K * K).reshape(K, K)
        rec = {"name": os.path.basename(f)[:-4], "n": int(g.size)}
        for c in range(1, K):
            tp = int(((p == c) & (g == c)).sum()); pp = int((p == c).sum()); gg = int((g == c).sum())
            hi = (p == c) & (pmax >= THRESH)
            hi_n[c] += int(hi.sum()); hi_tp[c] += int((hi & (g == c)).sum())
            rec[names[c]] = {"precision": tp / pp if pp else None, "recall": tp / gg if gg else None,
                             "gt": gg, "pred": pp, "hi_kept": int(hi.sum()),
                             "hi_precision": float((g[hi] == c).mean()) if hi.any() else None}
        per_bone.append(rec)
    tp = np.diag(conf); pp = conf.sum(0); gg = conf.sum(1)
    classes = {}
    for c in range(1, K):
        iou = tp[c] / (pp[c] + gg[c] - tp[c])
        classes[names[c]] = {"precision": float(tp[c] / pp[c]), "recall": float(tp[c] / gg[c]), "iou": float(iou),
                             "hi_precision": float(hi_tp[c] / hi_n[c]) if hi_n[c] else None,
                             "hi_kept_per_bone": float(hi_n[c] / len(per_bone))}
    fg = [classes[n] for n in names[1:]]
    summary = {"n_bones": len(per_bone), "names": names,
               "mean_fg_precision": float(np.mean([x["precision"] for x in fg])),
               "mean_fg_iou": float(np.mean([x["iou"] for x in fg])),
               "mean_fg_hi_precision": float(np.mean([x["hi_precision"] for x in fg]))}
    return names, classes, summary, per_bone


def axis_records(path: pathlib.Path):
    d = json.load(open(path))
    recs = None
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict) and "S_deg" in v[0]:
            recs = v
    if recs is None:
        for k, v in d.items():
            if isinstance(v, dict) and k != "summary":
                vals = list(v.values())
                if vals and isinstance(vals[0], dict) and "S_deg" in vals[0]:
                    recs = [dict(name=kk, **vv) for kk, vv in v.items()]
    return d.get("summary", {}), recs or []


def best_miou_from_log(path: pathlib.Path):
    if not path.exists():
        return None
    best = None
    for line in open(path, errors="replace"):
        m = re.search(r"mIoU/mAcc/allAcc ([0-9.]+)/", line)
        if m:
            best = max(best or 0.0, float(m.group(1)))
        m2 = re.search(r"Best mIoU[:\s]+([0-9.]+)", line)
        if m2:
            best = max(best or 0.0, float(m2.group(1)))
    return best


out = {"threshold": THRESH, "models": {}}
for mid, spec in MODELS.items():
    names, classes, summary, per_bone = score_dump(spec["dump"])
    pooled = json.load(open(spec["prec"])); pooled = pooled.get("summary", pooled)
    # cross-check against the JSON produced at eval time
    for n in names[1:]:
        assert abs(pooled["classes"][n]["precision"] - classes[n]["precision"]) < 2e-3, (mid, n)
    ax_summary, ax_recs = axis_records(spec["axis"])
    out["models"][mid] = {"label": spec["label"], "classes": classes, "summary": summary,
                          "per_bone": per_bone, "axis_summary": ax_summary, "axis_per_bone": ax_recs,
                          "val_miou_6class": best_miou_from_log(spec["log"])}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
for mid, m in out["models"].items():
    print(mid, {n: {k: round(v, 3) for k, v in c.items() if v is not None} for n, c in m["classes"].items()},
          {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m["summary"].items() if k != "names"},
          "6class mIoU", m["val_miou_6class"], "axis", {k: round(v["median"], 2) for k, v in m["axis_summary"].items() if isinstance(v, dict) and "median" in v})
