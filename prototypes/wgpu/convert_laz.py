#!/usr/bin/env python
"""Convert a directory of LAZ tiles into the wgpu demo's packed binary.

Output: one interleaved little-endian record per point —
    float32 x, y, z   (meters, centered on the block centroid)
    uint8   r, g, b, a
16 bytes/point, written per-tile into a single memmapped file so the
demo can load hundreds of millions of points without a parse step.

Coloring (USGS LPC format 6 has no RGB): elevation ramp blended with
log-scaled intensity, tinted by ASPRS classification (vegetation greens,
building grey, water blue, ground warm tan). On post-fire collections
the burn scar reads as bare warm ground against surviving canopy.

Usage:
    convert_laz.py <tiles_dir> <out_prefix> [--max-points N]

Writes <out_prefix>.bin + <out_prefix>.json (counts, bounds, tile spans).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np

CHUNK = 4_000_000

# Elevation ramp control points (t in 0..1) — muted terrain palette.
_RAMP_T = np.array([0.00, 0.20, 0.45, 0.70, 0.88, 1.00], dtype=np.float32)
_RAMP_C = np.array([
    [0.16, 0.22, 0.30],   # low: cool slate
    [0.38, 0.34, 0.26],   # bluff brown
    [0.62, 0.51, 0.34],   # warm tan
    [0.78, 0.66, 0.44],   # sunlit slope
    [0.88, 0.82, 0.66],   # high wash
    [0.97, 0.96, 0.92],   # ridgeline
], dtype=np.float32)

# Classification tints (ASPRS): blend factor toward tint color.
_CLASS_TINT = {
    3: ((0.30, 0.52, 0.24), 0.55),   # low veg
    4: ((0.26, 0.50, 0.22), 0.65),   # med veg
    5: ((0.20, 0.46, 0.20), 0.75),   # high veg / canopy
    6: ((0.72, 0.70, 0.68), 0.60),   # building
    9: ((0.15, 0.32, 0.55), 0.80),   # water
    17: ((0.55, 0.52, 0.50), 0.50),  # bridge deck
}


def _ramp(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    idx = np.searchsorted(_RAMP_T, t, side="right").clip(1, len(_RAMP_T) - 1)
    t0, t1 = _RAMP_T[idx - 1], _RAMP_T[idx]
    c0, c1 = _RAMP_C[idx - 1], _RAMP_C[idx]
    w = ((t - t0) / np.maximum(t1 - t0, 1e-6))[:, None]
    return c0 + (c1 - c0) * w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles_dir")
    ap.add_argument("out_prefix")
    ap.add_argument("--max-points", type=int, default=0,
                    help="uniform-stride cap on total points (0 = all)")
    args = ap.parse_args()

    tiles = sorted(Path(args.tiles_dir).glob("*.laz"))
    if not tiles:
        print("no .laz tiles found"); return 1

    # Pass 1: headers — total count, bounds, elevation percentile probe.
    total = 0
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for t in tiles:
        with laspy.open(t) as f:
            total += f.header.point_count
            mins = np.minimum(mins, f.header.mins)
            maxs = np.maximum(maxs, f.header.maxs)
    stride = 1
    if args.max_points and total > args.max_points:
        stride = int(np.ceil(total / args.max_points))
    est = total // stride + len(tiles)  # stride rounding slack per tile
    center = (mins + maxs) * 0.5
    print(f"{len(tiles)} tiles, {total:,} points, stride {stride} "
          f"(~{total // stride:,} kept)")
    print(f"bounds {mins} .. {maxs}")

    # Elevation normalization from a probe across EVERY tile — a
    # single-tile probe skews the ramp badly when the block spans
    # coastal flats up into the hills.
    probes = []
    for t in tiles:
        with laspy.open(t) as f:
            chunk = next(f.chunk_iterator(1_000_000))
            probes.append(np.asarray(chunk.z, dtype=np.float32))
    zs = np.concatenate(probes)
    z_lo = float(np.percentile(zs, 2))
    z_hi = float(np.percentile(zs, 99))
    del probes, zs
    print(f"elevation ramp {z_lo:.1f} .. {z_hi:.1f} m")

    out_bin = Path(args.out_prefix + ".bin")
    rec = np.dtype([("pos", "<f4", 3), ("rgba", "u1", 4)])
    mm = np.lib.format.open_memmap(  # .npy container: mmap-able + typed
        str(out_bin), mode="w+", dtype=rec, shape=(est,))

    written = 0
    tile_spans = []
    for ti, t in enumerate(tiles):
        t_start = written
        with laspy.open(t) as f:
            for pts in f.chunk_iterator(CHUNK):
                n0 = len(pts.x)
                sl = slice(0, n0, stride)
                x = np.asarray(pts.x[sl], dtype=np.float64)
                y = np.asarray(pts.y[sl], dtype=np.float64)
                z = np.asarray(pts.z[sl], dtype=np.float64)
                inten = np.asarray(pts.intensity[sl], dtype=np.float32)
                cls = np.asarray(pts.classification[sl], dtype=np.uint8)
                # Drop noise returns: ASPRS 7 (low noise), 18 (high
                # noise), and the withheld flag — they read as dark
                # speckle punched into every surface under EDL.
                keep = (cls != 7) & (cls != 18)
                try:
                    keep &= ~np.asarray(pts.withheld[sl], dtype=bool)
                except AttributeError:
                    pass
                x, y, z = x[keep], y[keep], z[keep]
                inten, cls = inten[keep], cls[keep]
                n = len(x)
                if n == 0:
                    continue

                tz = ((z - z_lo) / max(z_hi - z_lo, 1e-6)).astype(np.float32)
                col = _ramp(tz)
                # Intensity: log-scaled around its own median for local
                # contrast; modulates brightness +-35%.
                med = max(float(np.median(inten)), 1.0)
                iv = np.clip(np.log1p(inten) / np.log1p(med * 4), 0.0, 1.2)
                col *= (0.78 + 0.22 * iv)[:, None]
                for cid, (tint, k) in _CLASS_TINT.items():
                    m = cls == cid
                    if m.any():
                        col[m] = col[m] * (1 - k) + np.asarray(tint) * k

                out = mm[written:written + n]
                out["pos"][:, 0] = x - center[0]
                out["pos"][:, 1] = y - center[1]
                out["pos"][:, 2] = z - center[2]
                out["rgba"][:, :3] = np.clip(col * 255, 0, 255)
                out["rgba"][:, 3] = 255
                written += n
        tile_spans.append({"file": t.name, "start": t_start,
                           "count": written - t_start})
        print(f"[{ti + 1}/{len(tiles)}] {t.name}: "
              f"+{written - t_start:,} (total {written:,})")

    mm.flush()
    del mm
    # Trim the .npy to the real count (header rewrite via resize-copy of
    # the header only: simplest correct approach is a truncate + header
    # patch; np.load with mmap handles shape from header, so rewrite it).
    final = np.lib.format.open_memmap(str(out_bin), mode="r+")
    if len(final) != written:
        data = final[:written]
        tmp = Path(str(out_bin) + ".tmp")
        out2 = np.lib.format.open_memmap(str(tmp), mode="w+",
                                         dtype=rec, shape=(written,))
        step = 50_000_000
        for i in range(0, written, step):
            out2[i:i + step] = data[i:i + step]
        out2.flush(); del out2, data, final
        tmp.replace(out_bin)

    meta = {
        "points": int(written),
        "center_utm": center.tolist(),
        "bounds_min": (mins - center).tolist(),
        "bounds_max": (maxs - center).tolist(),
        "z_ramp": [z_lo, z_hi],
        "tiles": tile_spans,
        "crs": "EPSG:26911 (UTM 11N, NAD83) — demo-local, centered",
    }
    Path(args.out_prefix + ".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_bin} ({out_bin.stat().st_size / 1e9:.2f} GB), "
          f"{written:,} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
