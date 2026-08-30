"""Vector concept diagrams (SVG -> PDF via rsvg-convert, + PNG preview):
  fig_graphical_abstract  — the whole paper in one two-row strip, numbers from tables/*.json
  fig_data_model          — catalog-is-the-dataset: clouds x projects x namespaces, layers, models
Type sizes are chosen for a 1000 px canvas printed at 6.5 in (14 px ≈ 6.6 pt).
Run: .venv/bin/python scripts/paper_concept_figs.py
"""
from __future__ import annotations

import json
import math
import pathlib
import random
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
TAB, FIG = ROOT / "paper" / "tables", ROOT / "paper" / "figures"
ds = json.loads((TAB / "dataset_stats.json").read_text())
mm = json.loads((TAB / "model_metrics.json").read_text())
ops = json.loads((TAB / "bench_ops.json").read_text())
inf = json.loads((TAB / "bench_infer.json").read_text())
SON = mm["models"]["sonata"]

INK, MUTED, LIGHT, ACC, PALE = "#1a1a1a", "#6b6b6b", "#d9d9d9", "#f28c28", "#fff1e3"
SUP, INF, PED = "#d65e00", "#0073b2", "#2fb35d"
FONT = "font-family='DejaVu Sans, Helvetica, Arial, sans-serif'"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h; self.parts = []
    def rect(self, x, y, w, h, fill="#fff", stroke=MUTED, sw=1, r=8, dash=None):
        d = f" stroke-dasharray='{dash}'" if dash else ""
        self.parts.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{r}' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'{d}/>")
    def text(self, x, y, s, size=14, fill=INK, anchor="middle", weight="normal"):
        self.parts.append(f"<text x='{x}' y='{y}' {FONT} font-size='{size}' fill='{fill}' text-anchor='{anchor}' font-weight='{weight}'>{esc(s)}</text>")
    def lines(self, x, y, rows, size=14, fill=INK, anchor="middle", lh=None, weight="normal"):
        lh = lh or size * 1.35
        for i, r in enumerate(rows):
            self.text(x, y + i * lh, r, size, fill, anchor, weight)
    def arrow(self, x1, y1, x2, y2, color=MUTED, sw=1.6):
        self.parts.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{color}' stroke-width='{sw}' marker-end='url(#ah)'/>")
    def circle(self, x, y, r, fill, stroke="none"):
        self.parts.append(f"<circle cx='{x}' cy='{y}' r='{r}' fill='{fill}' stroke='{stroke}'/>")
    def raw(self, s):
        self.parts.append(s)
    def write(self, name):
        head = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{self.w}' height='{self.h}' viewBox='0 0 {self.w} {self.h}'>"
                f"<defs><marker id='ah' markerWidth='8' markerHeight='8' refX='7' refY='4' orient='auto'>"
                f"<path d='M0,0 L8,4 L0,8 z' fill='{MUTED}'/></marker></defs>"
                f"<rect width='{self.w}' height='{self.h}' fill='white'/>")
        svg = FIG / f"{name}.svg"; svg.write_text(head + "\n".join(self.parts) + "</svg>")
        subprocess.run(["rsvg-convert", "-f", "pdf", "-o", str(FIG / f"{name}.pdf"), str(svg)], check=True)
        subprocess.run(["rsvg-convert", "-f", "png", "-z", "2.0", "-o", str(FIG / f"{name}.png"), str(svg)], check=True)
        print("wrote", name)


def bone_glyph(s, cx, cy, sc=1.0, labelled=True):
    """A stylised vertebra (body + arch + spinous) as dots, optionally EP4-coloured."""
    rnd = random.Random(7)
    for _ in range(160):
        a = rnd.random() * 2 * math.pi; r = 0.92 + 0.08 * rnd.random()
        x = cx + math.cos(a) * 30 * sc * r; y = cy + math.sin(a) * 17 * sc * r
        top, bot = y < cy - 11 * sc, y > cy + 11 * sc
        col = (SUP if top else INF if bot else "#b7b7b7") if labelled else "#8d8d8d"
        s.circle(round(x, 1), round(y, 1), 1.6 * sc, col)
    for _ in range(80):
        a = rnd.random() * 2 * math.pi; x = cx + 40 * sc + math.cos(a) * 16 * sc; y = cy + math.sin(a) * 14 * sc
        near = abs(math.cos(a) + 1) < 0.5
        s.circle(round(x, 1), round(y, 1), 1.6 * sc, (PED if near else "#b7b7b7") if labelled else "#8d8d8d")
    for _ in range(20):
        t = rnd.random(); s.circle(round(cx + 56 * sc + t * 18 * sc, 1), round(cy + t * 7 * sc, 1), 1.6 * sc, "#b7b7b7" if labelled else "#8d8d8d")


def graphical_abstract():
    W, H = 1000, 640; s = SVG(W, H)
    s.text(16, 34, "Lithium — label the surface of anatomy, train in the loop, read the frame off the labels", 20, INK, "start", "bold")
    pw, ph, gap, y1, y2 = 306, 250, 30, 60, 340
    xs = [16, 16 + pw + gap, 16 + 2 * (pw + gap)]
    titles = ["1  Surface point clouds", "2  Paint labels on the surface", "3  Train PT-v3 in the app",
              "4  Predict, correct, repeat", "5  Anatomy from labels", "Held-out results"]
    cells = [(xs[0], y1), (xs[1], y1), (xs[2], y1), (xs[0], y2), (xs[1], y2), (xs[2], y2)]
    for (x, y), t in zip(cells, titles):
        s.rect(x, y, pw, ph, fill="#fbfbfb" if t[0].isdigit() else PALE, stroke=LIGHT if t[0].isdigit() else ACC, sw=1.2, r=12)
        s.text(x + 14, y + 28, t, 17, INK, "start", "bold")
    for i in (0, 1):
        s.arrow(xs[i] + pw + 4, y1 + ph / 2, xs[i + 1] - 4, y1 + ph / 2, ACC, 2.4)
    s.arrow(xs[0] + pw + 4, y2 + ph / 2, xs[1] - 4, y2 + ph / 2, ACC, 2.4)
    s.arrow(xs[1] + pw + 4, y2 + ph / 2, xs[2] - 4, y2 + ph / 2, ACC, 2.4)
    # wrap arrow 3 -> 4
    s.raw(f"<path d='M {xs[2] + pw / 2} {y1 + ph + 4} L {xs[2] + pw / 2} {y1 + ph + 22} L {xs[0] + pw / 2} {y1 + ph + 22} L {xs[0] + pw / 2} {y2 - 6}' fill='none' stroke='{ACC}' stroke-width='2.4' marker-end='url(#ah)'/>")
    # 1
    x, y = cells[0]; bone_glyph(s, x + 120, y + 105, 1.25, labelled=False)
    s.lines(x + pw / 2, y + 180, [f"{ds['points_per_bone']['mean']:,.0f} points per bone", "instead of ~500k voxels",
                                   f"{ds['n_bones']} VerSe vertebrae · {ds['n_subjects']} subjects"], 14, MUTED)
    # 2
    x, y = cells[1]; bone_glyph(s, x + 120, y + 95, 1.25, labelled=True)
    for i, (c, n) in enumerate(((SUP, "superior endplate"), (INF, "inferior endplate"), (PED, "pedicle"))):
        s.circle(x + 26, y + 160 + i * 22, 6, c); s.text(x + 40, y + 165 + i * 22, n, 14, MUTED, "start")
    s.text(x + pw - 14, y + ph - 14, f"pick {ops['ops']['pick']['ms']:.0f} ms at 1 M points", 13, MUTED, "end")
    # 3
    x, y = cells[2]; bx = x + 24
    for i, (lab, col) in enumerate((("catalog = dataset", "#eeeeee"), ("export PT-v3 scenes", "#eeeeee"), ("train (sidecar process)", PALE), ("model registry", "#eeeeee"))):
        s.rect(bx, y + 44 + i * 50, 258, 36, fill=col, stroke=LIGHT, r=6); s.text(bx + 129, y + 68 + i * 50, lab, 14, INK)
        if i < 3: s.arrow(bx + 129, y + 82 + i * 50, bx + 129, y + 92 + i * 50, MUTED, 1.2)
    # 4
    x, y = cells[3]; bone_glyph(s, x + 120, y + 100, 1.15, labelled=True)
    s.lines(x + pw / 2, y + 178, [f"INFER: {inf['inference_s_median']:.1f} s per bone", f"on one {inf['gpu'].replace('NVIDIA GeForce ', '')}",
                                   "correct, retrain, run again"], 14, MUTED)
    # 5: frame
    x, y = cells[4]; cx, cy = x + pw / 2 - 10, y + 128
    s.raw(f"<ellipse cx='{cx}' cy='{cy}' rx='50' ry='28' fill='none' stroke='{LIGHT}' stroke-width='2'/>")
    s.raw(f"<line x1='{cx - 50}' y1='{cy - 28}' x2='{cx + 50}' y2='{cy - 28}' stroke='{SUP}' stroke-width='4'/>")
    s.raw(f"<line x1='{cx - 50}' y1='{cy + 28}' x2='{cx + 50}' y2='{cy + 28}' stroke='{INF}' stroke-width='4'/>")
    s.arrow(cx, cy, cx, cy - 82, INK, 2.4); s.text(cx + 12, cy - 70, "S", 15, INK, "start", "bold")
    s.arrow(cx, cy, cx + 80, cy + 8, "#7a3fd1", 2.4); s.text(cx + 72, cy + 30, "P", 15, "#7a3fd1", "start", "bold")
    s.circle(cx + 62, cy - 8, 5.5, PED); s.circle(cx + 62, cy + 24, 5.5, PED)
    s.lines(x + pw / 2, y + 195, ["endplate normals → S", "pedicle pair → P,   L = S × P"], 14, MUTED)
    # 6: results
    x, y = cells[5]; c = SON["classes"]
    s.lines(x + 18, y + 62, [f"{SON['summary']['n_bones']} bones, subjects unseen in training"], 13, MUTED, "start")
    rows = [("endplate precision", f"{c['Superior_Endplate']['precision']:.2f} / {c['Inferior_Endplate']['precision']:.2f}", SUP),
            ("pedicle precision", f"{c['Pedicle']['precision']:.2f}  ← the open gap", PED),
            ("axis error (median)", f"{SON['axis_summary']['S_deg']['median']:.2f}°", INK),
            ("centre error (median)", f"{SON['axis_summary']['C_mm']['median']:.1f} mm", INK)]
    for i, (k, v, col) in enumerate(rows):
        yy = y + 96 + i * 38
        s.text(x + 18, yy, k, 13, MUTED, "start"); s.text(x + 18, yy + 19, v, 17, col, "start", "bold")
    s.text(16, H - 12, "Open source (MIT) · Python · one GPU workstation · no cloud, no accounts · github.com/saffffari/Lithium", 13, MUTED, "start")
    s.write("fig_graphical_abstract")


def data_model():
    W, H = 1000, 560; s = SVG(W, H)
    s.text(16, 34, "The catalog is the dataset: one copy of each cloud, one label array per (cloud, project)", 19, INK, "start", "bold")
    # catalog column
    s.rect(16, 60, 270, 470, fill="#fbfbfb", stroke=LIGHT, r=12); s.text(30, 90, "CATALOG", 16, INK, "start", "bold")
    s.lines(30, 116, [f"{ds['catalog']['clouds']} clouds, content-hashed", "data/<key>.npz", "previews/, index.json"], 13, MUTED, "start")
    ys = [200, 300, 400]
    for y, name in zip(ys, ("cloud A", "cloud B", "cloud C")):
        s.rect(32, y, 238, 66, fill="white", stroke=MUTED, r=8); s.text(46, y + 28, name, 15, INK, "start", "bold")
        s.text(46, y + 52, "key = sha256(bytes)[:16]", 12, MUTED, "start")
    # projects
    px = 340
    s.rect(px, 60, 300, 210, fill=PALE, stroke=ACC, r=12); s.text(px + 14, 90, "PROJECT  Gold247 · 6-class", 16, INK, "start", "bold")
    s.lines(px + 14, 118, ["ontology: 6 classes, locked", "labels/proj_1/<key>.npy", "models/proj_1.json", "settings: active model"], 13, MUTED, "start")
    s.rect(px, 320, 300, 210, fill="#f3f6fb", stroke=INF, r=12); s.text(px + 14, 350, "PROJECT  Endplate · 2-class", 16, INK, "start", "bold")
    s.lines(px + 14, 378, ["ontology: 2 classes", "labels/proj_2/<key>.npy", "same clouds,", "independent labels"], 13, MUTED, "start")
    for y in ys[:2]:
        s.arrow(270, y + 33, px - 4, 190, ACC, 1.6)
    s.arrow(270, ys[0] + 33, px - 4, 430, INF, 1.6); s.arrow(270, ys[2] + 33, px - 4, 445, INF, 1.6)
    # sandbox + models
    sx = 692
    s.rect(sx, 60, 292, 210, fill="#fff8ef", stroke=ACC, r=12, dash="6,5"); s.text(sx + 14, 90, "SANDBOX  (always present)", 16, INK, "start", "bold")
    s.lines(sx + 14, 118, ["no ontology, no own labels", "any model on any cloud", "output = cloud-level layer", "labels/layer:<model>/<key>.npy", "+ _layer.json"], 13, MUTED, "start")
    s.arrow(270, ys[2] + 33, sx - 4, 200, ACC, 1.6)
    s.rect(sx, 320, 292, 210, fill="#f6f6f6", stroke=MUTED, r=12); s.text(sx + 14, 350, f"MODELS  ({ds['catalog']['models']} registered)", 16, INK, "start", "bold")
    s.lines(sx + 14, 378, ["checkpoint + frozen class map", "runnable in any project whose", "ontology has its classes (by name)", "best mIoU is the default"], 13, MUTED, "start")
    s.arrow(640, 200, sx - 4, 400, MUTED, 1.6); s.arrow(sx + 146, 316, sx + 146, 276, MUTED, 1.6)
    s.text(16, H - 12, "Every stroke is written atomically under the active namespace — no “save project” step, nothing lost by closing the app.", 13, MUTED, "start")
    s.write("fig_data_model")


if __name__ == "__main__":
    graphical_abstract(); data_model()
