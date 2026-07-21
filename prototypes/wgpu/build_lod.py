#!/usr/bin/env python
"""Build a chunked multi-LOD point store (".3pc c-store") from LAZ tiles.

Layout (see docs/perf-1.1.md):

    <out>.3pc/
      index.json          tiles, UTM bounds, per-level counts, palette
      L0/<tile>.bin       full-res points, 12 B/pt
      L1/<tile>.bin       voxel-decimated ~1/4 density
      L2../               voxel size doubles per level

Record (12 B, little-endian): uint16 qx, qy, qz (position quantized to
the tile's own bounds), uint16 pad, uint8 r, g, b, a. The vertex shader
dequantizes with the tile's bounds — ≤ 1.2 cm error on 750 m tiles,
below sensor noise, at 25 % of the old resident footprint.

Coloring matches convert_laz.py (elevation ramp × intensity, ASPRS
class tints, noise returns dropped). Tiles convert in parallel.

Usage:
    build_lod.py <laz_dir> <out.3pc> [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import laspy
import numpy as np

from convert_laz import _ramp, _CLASS_TINT  # same visual grammar

CHUNK = 4_000_000
BASE_VOXEL = 0.5          # m — L1 voxel; doubles per level
MIN_LEVEL_PTS = 30_000    # stop pyramiding below this

REC = np.dtype([("q", "<u2", 3), ("pad", "<u2"), ("rgba", "u1", 4)])


def _color(z, inten, cls, z_lo, z_hi):
    tz = ((z - z_lo) / max(z_hi - z_lo, 1e-6)).astype(np.float32)
    col = _ramp(tz)
    med = max(float(np.median(inten)), 1.0)
    iv = np.clip(np.log1p(inten) / np.log1p(med * 4), 0.0, 1.2)
    col *= (0.78 + 0.22 * iv)[:, None]
    for cid, (tint, k) in _CLASS_TINT.items():
        m = cls == cid
        if m.any():
            col[m] = col[m] * (1 - k) + np.asarray(tint) * k
    return np.clip(col * 255, 0, 255).astype(np.uint8)


def _load_tile(path, z_lo, z_hi):
    """Read a LAZ tile → (float64 xyz Nx3, rgba u8 Nx4, bounds)."""
    xs, cs = [], []
    with laspy.open(path) as f:
        for pts in f.chunk_iterator(CHUNK):
            cls = np.asarray(pts.classification, dtype=np.uint8)
            keep = (cls != 7) & (cls != 18)
            try:
                keep &= ~np.asarray(pts.withheld, dtype=bool)
            except AttributeError:
                pass
            if not keep.any():
                continue
            xyz = np.column_stack([
                np.asarray(pts.x, dtype=np.float64)[keep],
                np.asarray(pts.y, dtype=np.float64)[keep],
                np.asarray(pts.z, dtype=np.float64)[keep]])
            rgba = np.empty((len(xyz), 4), dtype=np.uint8)
            rgba[:, :3] = _color(
                xyz[:, 2], np.asarray(pts.intensity, np.float32)[keep],
                cls[keep], z_lo, z_hi)
            rgba[:, 3] = 255
            xs.append(xyz)
            cs.append(rgba)
    if not xs:
        return None
    xyz = np.concatenate(xs)
    rgba = np.concatenate(cs)
    return xyz, rgba


def _voxel_decimate(xyz, keep_from, voxel):
    """Indices of one representative point per voxel of size ``voxel``."""
    q = np.floor(xyz[keep_from] / voxel).astype(np.int64)
    # pack 3×21-bit — tile extents / 0.5 m are far below 2^21
    key = (q[:, 0] << 42) ^ ((q[:, 1] & 0x1FFFFF) << 21) ^ (q[:, 2] & 0x1FFFFF)
    _, idx = np.unique(key, return_index=True)
    return keep_from[idx]


def _write_level(out_dir, name, xyz, rgba, bmin, span):
    qn = np.empty(len(xyz), dtype=REC)
    qs = np.clip((xyz - bmin) / span * 65535.0, 0, 65535)
    qn["q"] = qs.astype(np.uint16)
    qn["pad"] = 0
    qn["rgba"] = rgba
    out = out_dir / f"{name}.bin"
    qn.tofile(out)
    return len(qn)


def convert_tile(job):
    path_s, out_root_s, z_lo, z_hi = job
    path = Path(path_s)
    out_root = Path(out_root_s)
    name = path.stem.split("_")[-1]     # 11SLTxxxxxxx id
    t0 = time.perf_counter()
    try:
        loaded = _load_tile(path, z_lo, z_hi)
    except Exception as e:
        return {"error": f"{path.name}: {type(e).__name__}: {e}"}
    if loaded is None:
        return None
    xyz, rgba = loaded
    bmin = xyz.min(axis=0)
    bmax = xyz.max(axis=0)
    span = np.maximum(bmax - bmin, 1e-6)

    levels = {}
    keep = np.arange(len(xyz))
    lvl = 0
    while True:
        d = out_root / f"L{lvl}"
        d.mkdir(parents=True, exist_ok=True)
        levels[str(lvl)] = _write_level(
            d, name, xyz[keep], rgba[keep], bmin, span)
        if len(keep) <= MIN_LEVEL_PTS:
            break
        lvl += 1
        keep = _voxel_decimate(xyz, keep, BASE_VOXEL * (2 ** (lvl - 1)))

    return {"name": name, "file": path.name,
            "bounds_min": bmin.tolist(), "bounds_max": bmax.tolist(),
            "levels": levels, "secs": round(time.perf_counter() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("laz_dir")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tiles = sorted(Path(args.laz_dir).glob("*.laz"))
    if args.limit:
        tiles = tiles[:args.limit]
    if not tiles:
        print("no tiles"); return 1
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Global elevation ramp: probe every 8th tile's first chunk.
    # Truncated/corrupt tiles are skipped here and reported by the
    # convert pass instead of killing the build.
    zs = []
    for t in tiles[::8] or tiles[:1]:
        try:
            with laspy.open(t) as f:
                zs.append(np.asarray(next(f.chunk_iterator(500_000)).z,
                                     dtype=np.float32))
        except Exception as e:
            print(f"probe skip {t.name}: {e}", flush=True)
    zcat = np.concatenate(zs)
    z_lo = float(np.percentile(zcat, 2))
    z_hi = float(np.percentile(zcat, 99.5))
    print(f"{len(tiles)} tiles | ramp {z_lo:.0f}..{z_hi:.0f} m | "
          f"{args.workers} workers")

    jobs = [(str(t), str(out_root), z_lo, z_hi) for t in tiles]
    entries = []
    t0 = time.perf_counter()
    # spawn, not fork: the parent's ramp probe already started lazrs's
    # Rust thread pool, and forking after that inherits locked mutexes
    # (workers deadlock in futex_wait on their first read).
    with get_context("spawn").Pool(args.workers) as pool:
        bad = []
        for i, e in enumerate(pool.imap_unordered(convert_tile, jobs)):
            if e is None:
                continue
            if "error" in e:
                bad.append(e["error"])
                print(f"[{i + 1}/{len(tiles)}] BAD: {e['error']}",
                      flush=True)
                continue
            entries.append(e)
            total_l0 = sum(x["levels"]["0"] for x in entries)
            print(f"[{i + 1}/{len(tiles)}] {e['name']} "
                  f"L0={e['levels']['0']:,} lv={len(e['levels'])} "
                  f"({e['secs']}s)  cum L0 {total_l0 / 1e9:.2f}B",
                  flush=True)

    entries.sort(key=lambda e: e["name"])
    gmin = np.min([e["bounds_min"] for e in entries], axis=0)
    gmax = np.max([e["bounds_max"] for e in entries], axis=0)
    n_levels = max(len(e["levels"]) for e in entries)
    index = {
        "format": "3pc-v1 u16x3+pad+rgba8 (12B)",
        "crs": "EPSG:26911",
        "bounds_min": gmin.tolist(), "bounds_max": gmax.tolist(),
        "base_voxel": BASE_VOXEL, "n_levels": n_levels,
        "z_ramp": [z_lo, z_hi],
        "tiles": entries,
    }
    (out_root / "index.json").write_text(json.dumps(index))
    total = sum(int(c) for e in entries for c in e["levels"].values())
    l0 = sum(e["levels"]["0"] for e in entries)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min — "
          f"{len(entries)} tiles, L0 {l0:,} pts, "
          f"all levels {total:,} pts, "
          f"{total * 12 / 1e9:.1f} GB")
    if bad:
        (out_root / "bad_tiles.txt").write_text("\n".join(bad) + "\n")
        print(f"{len(bad)} BAD tiles listed in bad_tiles.txt — "
              f"re-download and re-run to fill in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
