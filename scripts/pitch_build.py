"""Build the interactive Lithium pitch page (docs/pitch/lithium_pitch.html).

Embeds: two adjacent held-out Gold247 vertebrae (coords + hand labels + Sonata
predictions) for the live labels→frame→measurement demo, the app screenshots,
the paper's graphical abstract / results / qualitative figures, and the headline
numbers from paper/tables/*.json. Output is a body fragment (the Artifact host
wraps it in the document skeleton). Run: .venv/bin/python scripts/pitch_build.py
"""
from __future__ import annotations

import base64
import io
import json
import pathlib
import re

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
DS = pathlib.Path("/home/alex/Projects/spinelab/cloud_models/yamato_gold247_v1/dataset")
DUMP = ROOT / "training/runs/sonata_full6_gold247_v1/val_dump"
TAB, FIG = ROOT / "paper/tables", ROOT / "paper/figures"
OUT = ROOT / "docs/pitch/lithium_pitch.html"
SCALE = 25.0
LEV = [f"C{i}" for i in range(1, 8)] + [f"T{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]

man = json.load(open(DS / "manifest.json"))
mm = json.load(open(TAB / "model_metrics.json")); SON = mm["models"]["sonata"]; YAM = mm["models"]["yamato"]
ds = json.load(open(TAB / "dataset_stats.json")); ops = json.load(open(TAB / "bench_ops.json")); inf = json.load(open(TAB / "bench_infer.json"))
ped = {b["name"]: b["Pedicle"]["precision"] for b in SON["per_bone"]}


def level(cloud):
    m = re.search(r"([CTL])(\d+)", cloud); return f"{m.group(1)}{m.group(2)}" if m else None


rows = []
for k, v in man.items():
    if not k.startswith("val/"):
        continue
    lv = level(v["cloud"]); idx = int(k.split("_")[-1])
    if lv:
        rows.append((v["subject"], LEV.index(lv), lv, idx, ped.get(f"val_{idx}", 0.0), k))
pairs = [(a, b) for a in rows for b in rows if a[0] == b[0] and b[1] == a[1] + 1]
pairs.sort(key=lambda ab: -(ab[0][4] + ab[1][4] + (0.3 if ab[0][2][0] in "TL" else 0)))
up, lo = pairs[0]
print("pair:", up[0], up[2], f"(val_{up[3]}, ped {up[4]:.2f})", "->", lo[2], f"(val_{lo[3]}, ped {lo[4]:.2f})")


def load_bone(row):
    subject, _, lv, idx, _, key = row
    orig = np.load(DS / key / "coord.npy"); mn, mx = orig.min(0), orig.max(0)
    d = np.load(DUMP / f"val_{idx}.npz")
    xyz = d["coord"].astype(np.float64) * SCALE + np.array([(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, mn[2]])
    return dict(name=f"{subject.replace('sub-', '')} {lv}", level=lv, xyz=xyz, gt=d["gt"].astype(np.uint8), pred=d["pred"].astype(np.uint8))


A, B = load_bone(up), load_bone(lo)
center = (A["xyz"].mean(0) + B["xyz"].mean(0)) / 2


def pack(bone):
    q = np.round((bone["xyz"] - center) * 100).astype(np.int16)
    assert np.abs(q).max() < 32000
    return {"name": bone["name"], "level": bone["level"], "n": int(len(q)),
            "xyz": base64.b64encode(q.tobytes()).decode(), "gt": base64.b64encode(bone["gt"].tobytes()).decode(),
            "pred": base64.b64encode(bone["pred"].tobytes()).decode()}


def img(path, width=1400, fmt="JPEG", q=84):
    im = Image.open(path).convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, fmt, quality=q, optimize=True)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


imgs = {
    "light": img(FIG / "screenshot_light.png"), "sheets": img(FIG / "screenshot_sheets.png"), "train": img(FIG / "screenshot_train.png"),
    "ga": img(FIG / "fig_graphical_abstract.png", 1600, "PNG"), "results": img(FIG / "fig_results.png", 1500, "PNG"),
    "qual": img(FIG / "fig_qualitative.png", 1300),
}
c, y = SON["classes"], YAM["classes"]
N = {
    "bones": ds["n_bones"], "subjects": ds["n_subjects"], "pts": f"{ds['points_per_bone']['mean']:,.0f}",
    "train": ds["splits"]["train"]["bones"], "val": ds["splits"]["val"]["bones"], "test": ds["splits"]["test"]["bones"],
    "clouds": ds["catalog"]["clouds"], "projects": ds["catalog"]["projects"], "models": ds["catalog"]["models"],
    "pick": f"{ops['ops']['pick']['ms']:.0f}", "lasso": f"{ops['ops']['lasso']['ms']:.0f}", "apply": f"{ops['ops']['apply']['ms']:.1f}",
    "infer": f"{inf['inference_s_median']:.1f}", "gpu": inf["gpu"].replace("NVIDIA GeForce ", ""),
    "sup": f"{c['Superior_Endplate']['precision']:.2f}", "inf_": f"{c['Inferior_Endplate']['precision']:.2f}", "ped": f"{c['Pedicle']['precision']:.2f}",
    "ysup": f"{y['Superior_Endplate']['precision']:.2f}", "yinf": f"{y['Inferior_Endplate']['precision']:.2f}", "yped": f"{y['Pedicle']['precision']:.2f}",
    "hisup": f"{c['Superior_Endplate']['hi_precision']:.3f}", "hiinf": f"{c['Inferior_Endplate']['hi_precision']:.3f}", "hiped": f"{c['Pedicle']['hi_precision']:.2f}",
    "axS": f"{SON['axis_summary']['S_deg']['median']:.2f}", "axP": f"{SON['axis_summary']['P_deg']['median']:.1f}", "axC": f"{SON['axis_summary']['C_mm']['median']:.1f}",
    "miou": f"{SON['val_miou_6class']:.3f}", "ymiou": f"{YAM['val_miou_6class']:.3f}",
}
html = (ROOT / "scripts/pitch_template.html").read_text()
for k, v in N.items():
    html = html.replace("{{N." + k + "}}", str(v))
for k, v in imgs.items():
    html = html.replace("{{IMG." + k + "}}", v)
html = html.replace("{{DATA}}", json.dumps({"A": pack(A), "B": pack(B)}))
OUT.write_text(html)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
