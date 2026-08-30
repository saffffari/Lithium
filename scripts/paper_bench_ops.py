"""Interaction latency at 1M points -> paper/tables/bench_ops.json.

Times the exact selection / label / undo functions the app calls
(src/core/tools/*, src/core/undo.py) on a synthetic 1M-point cloud at 1920x1080,
mirroring tests/test_phase13_performance.py. CPU path (the GPU upload that
follows a stroke is a 4 MB label-buffer update, not timed here). Median of
REPEATS runs after one warm-up.
"""
from __future__ import annotations

import json
import pathlib
import platform
import statistics
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.core.tools.box_tool import box_select  # noqa: E402
from src.core.tools.brush_tool import brush_select  # noqa: E402
from src.core.tools.lasso_tool import lasso_select  # noqa: E402
from src.core.tools.pick_tool import pick_point  # noqa: E402
from src.core.undo import UndoStack, apply_label  # noqa: E402
from src.data.point_cloud import PointCloudData  # noqa: E402
from src.utils.math_utils import look_at, perspective  # noqa: E402

N = 1_000_000
W, H = 1920, 1080
REPEATS = 7
OUT = ROOT / "paper" / "tables" / "bench_ops.json"

rng = np.random.default_rng(42)
cloud = PointCloudData(positions=rng.random((N, 3)).astype(np.float32) * 10,
                       colors=rng.random((N, 3)).astype(np.float32))
view = look_at(np.array([5.0, 5.0, 15.0], np.float32), np.array([5.0, 5.0, 5.0], np.float32),
               np.array([0.0, 1.0, 0.0], np.float32))
mvp = perspective(np.radians(60.0), W / H, 0.1, 100.0) @ view
polygon = [(200, 200), (1700, 200), (1700, 800), (200, 800)]
center = np.array([5.0, 5.0, 5.0], np.float32)
half = np.arange(0, 500_000, dtype=np.int32)


def timed(fn):
    fn()  # warm-up
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter(); r = fn(); ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), min(ts), r


def undo_redo():
    st = UndoStack(); apply_label(cloud, half, 1, st)
    t0 = time.perf_counter(); st.undo(cloud); u = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); st.redo(cloud); rd = (time.perf_counter() - t0) * 1000
    return u, rd


ops = {}
ms, mn, r = timed(lambda: pick_point(cloud.positions, mvp, 960.0, 540.0, W, H, 15.0))
ops["pick"] = {"label": "pick (nearest point)", "ms": ms, "min_ms": mn, "points": N}
ms, mn, r = timed(lambda: box_select(cloud.positions, mvp, 100, 100, 1800, 900, W, H))
ops["box"] = {"label": "box select", "ms": ms, "min_ms": mn, "points": N, "selected": int(len(r))}
ms, mn, r = timed(lambda: lasso_select(cloud.positions, mvp, polygon, W, H))
ops["lasso"] = {"label": "lasso select", "ms": ms, "min_ms": mn, "points": N, "selected": int(len(r))}
ms, mn, r = timed(lambda: brush_select(cloud.positions, center, 1.0))
ops["brush"] = {"label": "brush select (r=1)", "ms": ms, "min_ms": mn, "points": N, "selected": int(len(r))}
ms, mn, _ = timed(lambda: apply_label(cloud, half, 1, UndoStack()))
ops["apply"] = {"label": "apply label (500k pts)", "ms": ms, "min_ms": mn, "points": 500_000}
undo_redo()
us, rs = [], []
for _ in range(REPEATS):
    u, rd = undo_redo(); us.append(u); rs.append(rd)
ops["undo"] = {"label": "undo (500k pts)", "ms": statistics.median(us), "min_ms": min(us), "points": 500_000}
ops["redo"] = {"label": "redo (500k pts)", "ms": statistics.median(rs), "min_ms": min(rs), "points": 500_000}

out = {"n_points": N, "resolution": [W, H], "repeats": REPEATS, "stat": "median",
       "cpu": platform.processor() or platform.machine(), "python": platform.python_version(),
       "numpy": np.__version__, "ops": ops}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=1))
for k, v in ops.items():
    print(f"{k:6} {v['ms']:8.1f} ms  (min {v['min_ms']:.1f})")
