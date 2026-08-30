"""Result + concept figures for the Lithium manuscript -> paper/figures/*.{pdf,png}.

Every panel reads cached results (paper/tables/*.json, training/runs/*/val_dump,
the public Gold247 export) — no GPU, seconds. Each figure carries a figure
title, per-panel titles and labelled axes. Run: .venv/bin/python scripts/paper_figs.py
"""
from __future__ import annotations

import glob
import json
import os
import pathlib
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec, patches  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training" / "runs"))
from local_axis import axis_from_labels, unit  # noqa: E402

TAB, FIG = ROOT / "paper" / "tables", ROOT / "paper" / "figures"
RUNS = ROOT / "training" / "runs"
DS = pathlib.Path(os.environ.get("GOLD247_DATASET", "/home/alex/Projects/spinelab/cloud_models/yamato_gold247_v1/dataset"))
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, LIGHT, ACC, BASE = "#1a1a1a", "#6b6b6b", "#dcdcdc", "#f28c28", "#a3a3a3"
CLS = {"Unlabeled": "#c9c9c9", "Superior_Endplate": "#d65e00", "Inferior_Endplate": "#0073b2",
       "Pedicle": "#2fb35d", "Body_Wall": "#26bfa6", "Spinous_Process": "#9457eb"}
SHORT = {"Superior_Endplate": "superior endplate", "Inferior_Endplate": "inferior endplate", "Pedicle": "pedicle",
         "Body_Wall": "body wall", "Spinous_Process": "spinous process", "Unlabeled": "unlabelled"}
REGION = {"C": "#0073b2", "T": "#2fb35d", "L": "#d65e00"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.labelsize": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 100, "savefig.dpi": 220, "legend.frameon": False,
                     "figure.titlesize": 12, "figure.titleweight": "bold"})
ds = json.loads((TAB / "dataset_stats.json").read_text())
mm = json.loads((TAB / "model_metrics.json").read_text())
ops = json.loads((TAB / "bench_ops.json").read_text())
inf = json.loads((TAB / "bench_infer.json").read_text())
MAN = json.load(open(DS / "manifest.json"))
SON, YAM = mm["models"]["sonata"], mm["models"]["yamato"]
EP4 = ["Superior_Endplate", "Inferior_Endplate", "Pedicle"]


def save(fig, name, png_only=False):
    if not png_only:
        fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def panel_letter(ax, letter, x=-0.02, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right", color=INK)


def level_of(cloud):
    m = re.search(r"([CTL])(\d+)", cloud)
    return f"{m.group(1)}{m.group(2)}" if m else "?"


def val_cloud_name(dump_name):  # "val_7" -> "sub-verse631_T3"
    key = f"val/scene_{int(dump_name.split('_')[-1]):03d}"
    return MAN[key]["cloud"].replace(".ply", "")


class View:
    """Orthographic view: look along -f with up u; returns 2-D coords (right, up)."""
    def __init__(self, f, up):
        self.f = unit(np.asarray(f, float)); r = np.cross(self.f, up); self.r = unit(r); self.u = np.cross(self.r, self.f)
    def __call__(self, p):
        return np.stack([p @ self.r, p @ self.u], 1)
    def depth(self, p):
        return p @ self.f


def scatter_bone(ax, xy, colors, depth, s=1.2, alpha=0.9):
    order = np.argsort(depth)
    ax.scatter(xy[order, 0], xy[order, 1], s=s, c=np.asarray(colors)[order], linewidths=0, alpha=alpha, rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")


# ---------------------------------------------------------------------------- environment
def fig_environment():
    shots = {k: FIG / f"screenshot_{k}.png" for k in ("light", "sheets", "train")}
    if not all(p.exists() for p in shots.values()):
        print("environment: screenshots missing, skipped"); return
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.0, 1.0], hspace=0.14, wspace=0.05, top=0.92, bottom=0.01, left=0.01, right=0.99)
    spec = [("light", gs[0, :], "A  LIGHT TABLE — paint per-point labels; INFER runs the project's model on this cloud"),
            ("sheets", gs[1, 0], "B  SHEETS — catalog · projects · ontology · gallery"),
            ("train", gs[1, 1], "C  TRAIN — stage a dataset · launch PT-v3 · models")]
    for key, cell, title in spec:
        ax = fig.add_subplot(cell); im = Image.open(shots[key])
        ax.imshow(im, interpolation="lanczos"); ax.axis("off")
        ax.set_title(title, loc="left", fontsize=8.5, pad=4)
        ax.add_patch(patches.Rectangle((0, 0), im.size[0] - 1, im.size[1] - 1, fill=False, ec=LIGHT, lw=0.6))
    fig.suptitle("One desktop app, three tabs: browse → label → train, all local", y=0.975)
    fig.savefig(FIG / "fig_environment.png", dpi=220); plt.close(fig); print("wrote fig_environment")


# ---------------------------------------------------------------------------- concept
def pick_val_bone(level="L2"):
    for k, v in MAN.items():
        if k.startswith("val/") and v["cloud"].endswith(f"_{level}.ply"):
            return k, v["cloud"].replace(".ply", "")
    k = next(k for k in MAN if k.startswith("val/")); return k, MAN[k]["cloud"].replace(".ply", "")


def fig_concept():
    key, name = pick_val_bone("L2")
    xyz = np.load(DS / key / "coord.npy").astype(float); seg = np.load(DS / key / "segment.npy").astype(int).reshape(-1)
    fr = axis_from_labels(xyz, seg)
    C, S, P, L = fr["C"], fr["S"], fr["P"], fr["L"]
    xyz0 = xyz - C
    views = {"lateral": View(-L, S), "anterior": View(P, S), "axial": View(-S, P)}
    ext = xyz.max(0) - xyz.min(0); vox = int(np.prod(np.ceil(ext)))
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.14, top=0.86, bottom=0.07, left=0.06, right=0.98)
    # A: the surface
    ax = fig.add_subplot(gs[0, 0]); v = views["lateral"]
    scatter_bone(ax, v(xyz0), [MUTED] * len(xyz0), v.depth(xyz0), s=0.9)
    ax.set_title("A  A vertebra is a surface", loc="left")
    ax.text(0.5, -0.04, f"{name.replace('sub-', '')} · {len(xyz):,} points", transform=ax.transAxes, ha="center", fontsize=8, color=MUTED)
    # B: voxels vs points
    ax = fig.add_subplot(gs[0, 1]); ax.set_title("B  Elements to label", loc="left")
    vals = [len(xyz), vox]; labels = ["surface\npoints", "1 mm³ voxels\nin the bounding box"]
    ax.bar([0, 1], vals, color=[ACC, LIGHT], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, vox * 1.3); ax.yaxis.set_major_formatter(lambda x, p: f"{x / 1e3:.0f}k")
    ax.text(0, len(xyz) + vox * 0.03, f"{len(xyz):,}", ha="center", fontsize=8, color=INK)
    ax.text(1, vox + vox * 0.03, f"{vox:,}", ha="center", fontsize=8, color=INK)
    ax.text(0.04, 0.9, f"{vox / len(xyz):.0f}× fewer elements", transform=ax.transAxes, ha="left", fontsize=8, color=INK, fontweight="bold")
    ax.text(0.04, 0.82, "carry the same anatomy", transform=ax.transAxes, ha="left", fontsize=7.5, color=MUTED)
    # C: EP4 labels, three views
    axC = fig.add_subplot(gs[0, 2]); v = views["axial"]
    col = np.array([CLS[ds["class_names"][s]] if ds["class_names"][s] in EP4 else "#d0d0d0" for s in seg])
    scatter_bone(axC, v(xyz0), col, v.depth(xyz0), s=0.9); axC.set_title("C  EP4 labels, axial view", loc="left")
    for i, (vn, ttl) in enumerate((("lateral", "D  Frame from labels — lateral"),
                                   ("anterior", "E  anterior"), ("axial", "F  axial"))):
        ax = fig.add_subplot(gs[1, i]); v = views[vn]
        scatter_bone(ax, v(xyz0), col, v.depth(xyz0), s=0.8, alpha=0.55); ax.set_title(ttl, loc="left")
        o = v(np.zeros((1, 3)))[0]; scale = 0.45 * np.ptp(v(xyz0)[:, 1])
        for vec, c, lab in ((S, "#1a1a1a", "S"), (P, "#7a3fd1", "P"), (L, "#c0392b", "L")):
            d = v(vec[None, :])[0] * scale
            if np.hypot(*d) < 0.15 * scale:
                ax.plot(o[0], o[1], "o", ms=4, color=c); ax.text(o[0] + 0.05 * scale, o[1] + 0.05 * scale, lab, color=c, fontsize=8, fontweight="bold")
                continue
            ax.annotate("", xy=o + d, xytext=o, arrowprops=dict(arrowstyle="-|>", lw=1.6, color=c, shrinkA=0, shrinkB=0))
            ax.text(*(o + d * 1.12), lab, color=c, fontsize=8, fontweight="bold", ha="center", va="center")
        for cpt, n, cc in ((fr["cs"], fr["ns"], CLS["Superior_Endplate"]), (fr["ci"], fr["ni"], CLS["Inferior_Endplate"])):
            a = v((cpt - C)[None, :])[0]; b = v((cpt - C + n * 0.35 * scale / 0.45)[None, :])[0]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=cc, lw=1.2)
    handles = [patches.Patch(color=CLS[c], label=SHORT[c]) for c in EP4] + [patches.Patch(color="#d0d0d0", label="other")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.suptitle("Label the surface, not the volume — and read the anatomy off the labels", y=0.97)
    fig.text(0.5, 0.915, "S = cranio-caudal (endplate normals), P = posterior (pedicle pair), L = S × P", ha="center", fontsize=8, color=MUTED)
    fig.savefig(FIG / "fig_concept.pdf"); fig.savefig(FIG / "fig_concept.png", dpi=220); plt.close(fig); print("wrote fig_concept")


# ---------------------------------------------------------------------------- dataset
def fig_dataset():
    fig = plt.figure(figsize=(7.2, 2.7)); gs = gridspec.GridSpec(1, 3, width_ratios=[2.3, 1.5, 0.8], wspace=0.75)
    ax = fig.add_subplot(gs[0]); lv = list(ds["levels"].items())
    ax.bar(range(len(lv)), [n for _, n in lv], color=[REGION[k[0]] for k, _ in lv], width=0.75)
    ax.set_xticks(range(len(lv))); ax.set_xticklabels([k for k, _ in lv], rotation=90, fontsize=7)
    ax.set_ylabel("bones"); ax.set_title(f"A  {ds['n_bones']} vertebrae over {len(lv)} levels", loc="left")
    for r, n in ds["regions"].items():
        ax.plot([], [], "s", color=REGION[r], label=f"{ {'C': 'cervical', 'T': 'thoracic', 'L': 'lumbar'}[r]} ({n})")
    ax.set_ylim(0, 19); ax.legend(fontsize=6.5, loc="upper right", ncol=3, handlelength=0.9, columnspacing=0.8)
    ax = fig.add_subplot(gs[1]); names = ds["class_names"]; fr = np.array(ds["class_fraction"]) * 100
    order = np.argsort(fr)
    ax.barh(range(len(names)), fr[order], color=[CLS[names[i]] for i in order])
    ax.set_yticks(range(len(names))); ax.set_yticklabels([SHORT[names[i]] for i in order], fontsize=7.5); ax.tick_params(axis="y", length=0, pad=3)
    for j, i in enumerate(order):
        ax.text(fr[i] + 0.8, j, f"{fr[i]:.1f}%", va="center", fontsize=7, color=INK)
    ax.set_xlim(0, max(fr) * 1.28); ax.set_xlabel("% of points"); ax.set_title("B  Points per class", loc="left")
    ax = fig.add_subplot(gs[2]); sp = ds["splits"]; keys = ["train", "val", "test"]; cols = [INK, ACC, BASE]
    bottom = 0
    for k, c in zip(keys, cols):
        ax.bar(0, sp[k]["bones"], bottom=bottom, color=c, width=0.55)
        ax.text(0.42, bottom + sp[k]["bones"] / 2, f"{k}: {sp[k]['bones']} bones\n{sp[k]['subjects']} subjects", va="center", fontsize=7.2, color=INK)
        bottom += sp[k]["bones"]
    ax.set_xlim(-0.5, 1.6); ax.set_xticks([]); ax.set_ylabel("bones"); ax.set_title("C  Subject-level split", loc="left")
    fig.suptitle(f"Gold247: {ds['n_bones']} hand-labelled VerSe vertebrae, {ds['n_subjects']} subjects, {ds['points_per_bone']['mean']:,.0f} surface points each", y=1.04)
    save(fig, "fig_dataset")


# ---------------------------------------------------------------------------- results
def per_bone(model, cls, key="precision"):
    return np.array([b[cls][key] if b[cls][key] is not None else np.nan for b in model["per_bone"]])


def fig_results():
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = gridspec.GridSpec(2, 12, height_ratios=[1, 1.15], hspace=0.55, wspace=2.2, top=0.88, bottom=0.13, left=0.07, right=0.99)
    top = [gs[0, 0:4], gs[0, 4:8], gs[0, 8:12]]; bot = [gs[1, 0:3], gs[1, 3:6], gs[1, 6:9], gs[1, 9:12]]
    x = np.arange(3); w = 0.36
    for i, (metric, ttl) in enumerate((("precision", "A  Precision (val, pooled)"), ("iou", "B  IoU"),
                                       ("hi_precision", f"C  Precision at ≥{mm['threshold']:.2f} conf."))):
        ax = fig.add_subplot(top[i])
        ys = [YAM["classes"][c][metric] for c in EP4]; ys2 = [SON["classes"][c][metric] for c in EP4]
        ax.bar(x - w / 2, ys, w, color=BASE, label="Yamato (PT-v3m1, scratch)")
        ax.bar(x + w / 2, ys2, w, color=ACC, label="Sonata (PT-v3m2, fine-tuned)")
        for xx, y in zip(x - w / 2, ys): ax.text(xx, y + 0.012, f"{y:.2f}", ha="center", fontsize=6.5, color=MUTED)
        for xx, y in zip(x + w / 2, ys2): ax.text(xx, y + 0.012, f"{y:.2f}", ha="center", fontsize=6.5, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(["sup.\nendplate", "inf.\nendplate", "pedicle"], fontsize=7.5)
        ax.set_ylim(0.5, 1.03); ax.set_ylabel(metric.replace("hi_", "")); ax.set_title(ttl, loc="left")
        if i == 0: handles_models = ax.get_legend_handles_labels()
    # D: per-bone pedicle precision scatter
    ax = fig.add_subplot(bot[0]); py, ps = per_bone(YAM, "Pedicle"), per_bone(SON, "Pedicle")
    names = [val_cloud_name(b["name"]) for b in SON["per_bone"]]
    cols = [REGION.get(level_of(n)[0], MUTED) for n in names]
    ax.plot([0.2, 1], [0.2, 1], color=LIGHT, lw=1, zorder=0)
    ax.scatter(py, ps, s=22, c=cols, edgecolors="white", linewidths=0.5)
    ax.set_xlabel("Yamato pedicle precision"); ax.set_ylabel("Sonata pedicle precision")
    ax.set_xlim(0.2, 1.0); ax.set_ylim(0.2, 1.0); ax.set_aspect("equal")
    ax.set_title("D  Pedicle prec. / bone", loc="left")
    for r, c in REGION.items(): ax.plot([], [], "o", color=c, label={"C": "cervical", "T": "thoracic", "L": "lumbar"}[r])
    ax.legend(fontsize=6.5, loc="lower right")
    ax.text(0.03, 0.95, f"median  {np.nanmedian(py):.2f} → {np.nanmedian(ps):.2f}", transform=ax.transAxes, fontsize=7, va="top", color=MUTED)
    # E–G: axis errors
    for j, (k, ttl, unit_, cap) in enumerate((("S_deg", "E  S-axis error", "degrees", None), ("P_deg", "F  P-axis error", "degrees", 20.0),
                                               ("C_mm", "G  Centre error", "mm", None))):
        ax = fig.add_subplot(bot[1 + j])
        data = [np.array([b[k] for b in m["axis_per_bone"] if b.get("ok")]) for m in (YAM, SON)]
        for i, (d, c) in enumerate(zip(data, (BASE, ACC))):
            jitter = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.28
            shown = d if cap is None else np.minimum(d, cap)
            ax.scatter(i + jitter, shown, s=10, color=c, alpha=0.75, linewidths=0)
            ax.hlines(np.median(d), i - 0.25, i + 0.25, color=INK, lw=1.4)
            ax.text(i + 0.3, np.median(d), f"{np.median(d):.2f}", ha="left", va="center", fontsize=7, color=INK)
            if cap is not None and (d > cap).any():
                ax.text(i, cap * 1.02, f"▲ {(d > cap).sum()} above {cap:.0f}°", ha="center", va="bottom", fontsize=6, color=MUTED)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Yamato", "Sonata"]); ax.set_ylabel(unit_); ax.set_title(ttl, loc="left")
        ax.set_ylim(0, (cap * 1.15) if cap else None); ax.set_xlim(-0.5, 1.75)
    fig.suptitle(f"Two in-app models on the {SON['summary']['n_bones']} held-out Gold247 bones (EP4)", y=0.985)
    fig.text(0.5, 0.94, "each dot is one vertebra; bars mark the median; frame errors are against the frame from the hand labels",
             ha="center", fontsize=7.5, color=MUTED)
    fig.legend(*handles_models, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2, fontsize=7.5)
    fig.savefig(FIG / "fig_results.pdf"); fig.savefig(FIG / "fig_results.png", dpi=220); plt.close(fig); print("wrote fig_results")


# ---------------------------------------------------------------------------- qualitative
def fig_qualitative():
    dump = RUNS / "sonata_full6_gold247_v1" / "val_dump"; meta = json.load(open(dump / "meta.json")); names = meta["names"]
    ped = per_bone(SON, "Pedicle"); order = np.argsort(ped)
    picks = [int(order[-1]), int(order[len(order) // 2]), int(order[0])]
    labels = ["best pedicle precision", "median", "worst"]
    fig, axes = plt.subplots(len(picks), 3, figsize=(7.2, 1.75 * len(picks)))
    for r, (bi, tag) in enumerate(zip(picks, labels)):
        b = SON["per_bone"][bi]; d = np.load(dump / f"{b['name']}.npz")
        xyz = d["coord"].astype(float); gt = d["gt"].astype(int); pr = d["pred"].astype(int)
        fr = axis_from_labels(xyz, gt) or axis_from_labels(xyz, pr)
        C = fr["C"]; v = View(-fr["L"], fr["S"]); xy = v(xyz - C); dep = v.depth(xyz - C)
        cmap = lambda lab: np.array([CLS[names[s]] if s else "#d5d5d5" for s in lab])
        err = np.where(pr == gt, 0, np.where(pr > 0, 1, 2))
        errc = np.array(["#dedede", "#e0231f", "#3b5bdb"])[err]
        errc[(err == 0) & (gt > 0)] = [CLS[names[s]] for s in gt[(err == 0) & (gt > 0)]]
        for c_, (colors, ttl) in enumerate(((cmap(gt), "hand labels"), (cmap(pr), "Sonata"), (errc, "errors (red FP · blue FN)"))):
            ax = axes[r, c_]; scatter_bone(ax, xy, colors, dep, s=1.6)
            if r == 0: ax.set_title(ttl, loc="center", fontsize=8.5)
            if c_ == 0:
                ax.text(-0.04, 0.5, f"{val_cloud_name(b['name']).replace('sub-', '')}\n{tag}\npedicle prec. {b['Pedicle']['precision']:.2f}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=7.5, color=INK, rotation=0)
    handles = [patches.Patch(color=CLS[c], label=SHORT[c]) for c in EP4]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=8)
    fig.suptitle("What the model gets right and wrong: held-out bones, lateral view", y=0.99)
    fig.text(0.5, 0.925, "errors concentrate at the pedicle → lamina boundary, not at random", ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.2, right=0.99, bottom=0.08, top=0.86, wspace=0.02, hspace=0.02)
    fig.savefig(FIG / "fig_qualitative.png", dpi=220); plt.close(fig); print("wrote fig_qualitative")


# ---------------------------------------------------------------------------- latency
def fig_latency():
    fig = plt.figure(figsize=(7.2, 2.5)); gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1], wspace=0.5)
    ax = fig.add_subplot(gs[0]); keys = ["pick", "box", "lasso", "brush", "apply", "undo", "redo"]
    vals = [ops["ops"][k]["ms"] for k in keys]; labs = [ops["ops"][k]["label"] for k in keys]
    ax.barh(range(len(keys))[::-1], vals, color=[ACC] * 4 + [INK] * 3, height=0.62)
    for i, v in enumerate(vals): ax.text(v * 1.12, len(keys) - 1 - i, f"{v:.1f} ms", va="center", fontsize=7.5)
    ax.set_yticks(range(len(keys))[::-1]); ax.set_yticklabels(labs, fontsize=7.5); ax.set_xscale("log")
    ax.set_xlim(0.3, max(vals) * 3.5); ax.set_xlabel("time (ms, log)")
    ax.set_title(f"A  Interaction at {ops['n_points'] / 1e6:.0f} M points (median of {ops['repeats']})", loc="left")
    ax.axvline(16.7, color=LIGHT, lw=1, ls="--"); ax.text(16.7, len(keys) - 0.45, "one 60 Hz frame", fontsize=6.5, color=MUTED, ha="center")
    ax = fig.add_subplot(gs[1]); runs = inf["runs"]
    ax.bar(range(len(runs)), [r["model_load_s"] for r in runs], color=LIGHT, label="model load")
    ax.bar(range(len(runs)), [r["inference_s"] for r in runs], bottom=[r["model_load_s"] for r in runs], color=ACC, label="inference")
    ax.set_xticks(range(len(runs))); ax.set_xticklabels([level_of(r["cloud"]) for r in runs], fontsize=7.5)
    ax.set_ylabel("seconds"); ax.set_xlabel(f"held-out bone ({int(inf['points_mean']):,} points each)")
    ax.set_title(f"B  INFER on one bone, {inf['gpu'].replace('NVIDIA GeForce ', '')}", loc="left")
    ax.set_ylim(0, max(r["model_load_s"] + r["inference_s"] for r in runs) * 1.45)
    ax.legend(fontsize=6.5, loc="upper left", handlelength=1.0)
    ax.text(0.98, 0.96, f"median inference\n{inf['inference_s_median']:.1f} s", transform=ax.transAxes, fontsize=7.5, color=INK, ha="right", va="top")
    fig.suptitle("Every stroke lands within a frame; a model pass is seconds", y=1.05)
    save(fig, "fig_latency")


# ---------------------------------------------------------------------------- timing study design
def fig_timing_design():
    fig = plt.figure(figsize=(7.2, 3.0)); gs = gridspec.GridSpec(1, 3, width_ratios=[2.8, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0]); ax.axis("off"); ax.set_xlim(0, 12.4); ax.set_ylim(0, 10)
    ax.set_title("A  Protocol (planned)", loc="left")
    def box(x, y, w, h, text, fc="#ffffff", ec=MUTED, fs=7.5, bold=False):
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal")
    box(0.1, 7.2, 4.2, 2.4, "30 bones\n10 C · 10 T · 10 L\nunseen subjects", fc="#f6f6f6", fs=6.5)
    box(0.1, 3.9, 4.2, 2.4, "2 annotators\nblind to each other", fc="#f6f6f6", fs=6.5)
    box(0.1, 0.6, 4.2, 2.4, "2 conditions\nsurface (Lithium, EP4)\nvolume (slice painting)", fc="#f6f6f6", fs=6.5)
    box(4.9, 3.7, 3.1, 2.8, "per bone:\nminutes to done\n+ label file", fc="#fff3e8", ec=ACC, fs=6.5)
    box(8.6, 7.2, 3.7, 2.4, "minutes per bone\nper condition", fc="#ffffff", fs=6.5)
    box(8.6, 3.9, 3.7, 2.4, "inter-annotator\npoint precision", fc="#ffffff", fs=6.5)
    box(8.6, 0.6, 3.7, 2.4, "frame error\nlabels → S, P, L", fc="#ffffff", fs=6.5)
    for y in (8.4, 5.1, 1.8): ax.annotate("", xy=(4.85, 5.1), xytext=(4.35, y), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    for y in (8.4, 5.1, 1.8): ax.annotate("", xy=(8.55, y), xytext=(8.05, 5.1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    for i, (ttl, yl, cats) in enumerate((("B  Minutes per bone", "minutes", ["surface", "volume"]), ("C  Agreement", "point precision", ["surface", "volume"]))):
        ax = fig.add_subplot(gs[1 + i]); ax.set_title(ttl, loc="left")
        for j, c in enumerate(cats):
            ax.bar(j, 1, color="white", edgecolor=LIGHT, hatch="////", lw=0.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(cats); ax.set_yticks([]); ax.set_ylabel(yl); ax.set_ylim(0, 1.25)
        ax.text(0.5, 0.6, "DATA PENDING", transform=ax.transAxes, ha="center", va="center", fontsize=9, color=ACC, fontweight="bold")
        ax.text(0.5, 0.42, "study not yet run", transform=ax.transAxes, ha="center", va="center", fontsize=7, color=MUTED)
    fig.suptitle("The timing study this paper still owes: two annotators, 30 bones, minutes and agreement", y=1.06)
    save(fig, "fig_timing_design")


if __name__ == "__main__":
    only = sys.argv[1:]
    for fn in (fig_environment, fig_concept, fig_dataset, fig_results, fig_qualitative, fig_latency, fig_timing_design):
        if only and fn.__name__ not in only:
            continue
        try:
            fn()
        except Exception:  # noqa: BLE001
            print("FAILED", fn.__name__); traceback.print_exc()
