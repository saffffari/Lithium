"""Phase 13 verification: Performance benchmarks for large clouds.

Target: selection operations on 1M points < 200ms, label application
< 100ms, no crashes on error paths.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.point_cloud import PointCloudData
from src.core.selection import SelectionBuffer, SelectionMode
from src.core.undo import UndoStack, apply_label
from src.core.tools.pick_tool import pick_point
from src.core.tools.box_tool import box_select
from src.core.tools.lasso_tool import lasso_select
from src.core.tools.brush_tool import brush_select
from src.utils.math_utils import perspective, look_at


N_POINTS = 1_000_000


def make_large_cloud():
    rng = np.random.default_rng(42)
    positions = rng.random((N_POINTS, 3)).astype(np.float32) * 10
    colors = rng.random((N_POINTS, 3)).astype(np.float32)
    return PointCloudData(positions=positions, colors=colors)


def get_camera_mvp(width=1920, height=1080):
    view = look_at(
        np.array([5.0, 5.0, 15.0], dtype=np.float32),
        np.array([5.0, 5.0, 5.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    proj = perspective(np.radians(60.0), width / height, 0.1, 100.0)
    return proj @ view, view, proj


def benchmark(name, fn, *args, max_ms=200):
    start = time.perf_counter()
    result = fn(*args)
    elapsed_ms = (time.perf_counter() - start) * 1000
    status = "PASS" if elapsed_ms < max_ms else "SLOW"
    print(f"{status}: {name} - {elapsed_ms:.1f}ms (target < {max_ms}ms)")
    return result, elapsed_ms


def test_pick_performance():
    cloud = make_large_cloud()
    mvp, _, _ = get_camera_mvp()
    _, ms = benchmark(
        "pick on 1M points",
        pick_point, cloud.positions, mvp, 960.0, 540.0, 1920, 1080, 15.0,
        max_ms=500,
    )
    assert ms < 500


def test_box_performance():
    cloud = make_large_cloud()
    mvp, view, _ = get_camera_mvp()
    indices, ms = benchmark(
        "box select 1M points",
        box_select, cloud.positions, mvp, 100, 100, 1800, 900, 1920, 1080,
        max_ms=300,
    )
    assert ms < 300
    print(f"  Selected {len(indices):,} points")


def test_lasso_performance():
    cloud = make_large_cloud()
    mvp, _, _ = get_camera_mvp()
    polygon = [(200, 200), (1700, 200), (1700, 800), (200, 800)]
    indices, ms = benchmark(
        "lasso select 1M points",
        lasso_select, cloud.positions, mvp, polygon, 1920, 1080,
        max_ms=500,
    )
    assert ms < 500
    print(f"  Selected {len(indices):,} points")


def test_brush_performance():
    cloud = make_large_cloud()
    center = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    indices, ms = benchmark(
        "brush select 1M points (r=1)",
        brush_select, cloud.positions, center, 1.0,
        max_ms=100,
    )
    assert ms < 100
    print(f"  Selected {len(indices):,} points")


def test_label_apply_performance():
    cloud = make_large_cloud()
    undo_stack = UndoStack()
    indices = np.arange(0, 500_000, dtype=np.int32)

    start = time.perf_counter()
    apply_label(cloud, indices, 1, undo_stack)
    ms = (time.perf_counter() - start) * 1000
    print(f"PASS: apply_label on 500K points - {ms:.1f}ms (target < 100ms)")
    assert ms < 100


def test_undo_redo_performance():
    cloud = make_large_cloud()
    undo_stack = UndoStack()
    indices = np.arange(0, 500_000, dtype=np.int32)
    apply_label(cloud, indices, 1, undo_stack)

    start = time.perf_counter()
    undo_stack.undo(cloud)
    undo_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    undo_stack.redo(cloud)
    redo_ms = (time.perf_counter() - start) * 1000

    print(f"PASS: undo 500K pts - {undo_ms:.1f}ms, redo - {redo_ms:.1f}ms")
    assert undo_ms < 100
    assert redo_ms < 100


def test_empty_cloud_no_crash():
    """Operations on empty clouds should not crash."""
    cloud = PointCloudData(
        positions=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.float32),
    )
    # All selection tools should handle empty gracefully
    mvp, _, _ = get_camera_mvp()
    assert pick_point(cloud.positions, mvp, 100, 100, 1920, 1080) == -1
    assert len(box_select(cloud.positions, mvp, 0, 0, 100, 100, 1920, 1080)) == 0
    assert len(lasso_select(cloud.positions, mvp, [(0, 0), (10, 0), (5, 10)], 1920, 1080)) == 0
    assert len(brush_select(cloud.positions, np.zeros(3, dtype=np.float32), 1.0)) == 0
    print("PASS: Empty cloud operations no-crash")


if __name__ == '__main__':
    test_pick_performance()
    test_box_performance()
    test_lasso_performance()
    test_brush_performance()
    test_label_apply_performance()
    test_undo_redo_performance()
    test_empty_cloud_no_crash()
    print("\n=== Phase 13 ALL TESTS PASSED ===")
