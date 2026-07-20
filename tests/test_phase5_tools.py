"""Phase 5 verification: Box, lasso, and brush selection."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.tools.box_tool import box_select
from src.core.tools.lasso_tool import lasso_select, _point_in_polygon
from src.core.tools.brush_tool import brush_select
from src.utils.math_utils import perspective, look_at


def make_camera_mvp(width=256, height=256):
    view = look_at(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    proj = perspective(np.radians(60.0), width / height, 0.1, 100.0)
    mvp = proj @ view
    return mvp, view


def test_box_select_center():
    """Box covering screen center should select a point at origin."""
    positions = np.array([
        [0.0, 0.0, 0.0],   # origin - should be selected
        [2.0, 2.0, 0.0],   # far off-axis - not selected
        [-2.0, -2.0, 0.0], # far off-axis - not selected
    ], dtype=np.float32)
    mvp, _ = make_camera_mvp(256, 256)
    indices = box_select(positions, mvp, 100, 100, 156, 156, 256, 256)
    assert 0 in indices, f"origin should be selected, got {indices}"
    assert len(indices) == 1, f"only origin, got {len(indices)}"
    print(f"PASS: Box select center")


def test_box_select_all():
    """Box covering entire screen selects all visible points."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [-0.5, -0.5, 0.0],
    ], dtype=np.float32)
    mvp, _ = make_camera_mvp()
    indices = box_select(positions, mvp, 0, 0, 256, 256, 256, 256)
    assert len(indices) == 3, f"expected 3, got {len(indices)}"
    print("PASS: Box select all visible")


def test_box_depth_limit():
    """Depth limit should exclude far points."""
    positions = np.array([
        [0.0, 0.0, 3.0],   # near camera (z=5)
        [0.0, 0.0, -3.0],  # far from camera
    ], dtype=np.float32)
    mvp, view = make_camera_mvp()
    # max_depth=3 means cut at view-space distance 3
    # Camera at z=5, point at z=3 is view_z=2, point at z=-3 is view_z=8
    indices = box_select(positions, mvp, 0, 0, 256, 256, 256, 256,
                         max_depth=5.0, view_matrix=view)
    assert 0 in indices, "near point should be selected"
    assert 1 not in indices, "far point should be excluded"
    print("PASS: Box depth limit")


def test_point_in_polygon():
    """Winding number test for basic polygon shapes."""
    # Triangle
    poly = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]], dtype=np.float32)
    # Inside
    sx = np.array([5.0, 2.0, 8.0], dtype=np.float32)
    sy = np.array([2.0, 1.0, 1.0], dtype=np.float32)
    inside = _point_in_polygon(sx, sy, poly)
    assert inside.all(), f"expected all inside, got {inside}"

    # Outside
    sx = np.array([-1.0, 15.0, 5.0], dtype=np.float32)
    sy = np.array([5.0, 5.0, 15.0], dtype=np.float32)
    inside = _point_in_polygon(sx, sy, poly)
    assert not inside.any(), f"expected none inside, got {inside}"
    print("PASS: Point in polygon")


def test_lasso_select():
    """Lasso around the origin should select the center point."""
    positions = np.array([
        [0.0, 0.0, 0.0],   # center - should be selected
        [2.0, 0.0, 0.0],   # outside lasso
    ], dtype=np.float32)
    mvp, _ = make_camera_mvp(256, 256)
    # Draw a box polygon around screen center
    polygon = [(100, 100), (156, 100), (156, 156), (100, 156)]
    indices = lasso_select(positions, mvp, polygon, 256, 256)
    assert 0 in indices
    assert 1 not in indices
    print("PASS: Lasso select")


def test_brush_select():
    """Brush sphere selects points within radius."""
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=np.float32)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    indices = brush_select(positions, center, 0.3)
    assert list(sorted(indices)) == [0, 1], f"expected [0,1], got {indices}"

    indices = brush_select(positions, center, 1.0)
    assert list(sorted(indices)) == [0, 1, 2], f"expected [0,1,2], got {indices}"
    print("PASS: Brush select")


if __name__ == '__main__':
    test_box_select_center()
    test_box_select_all()
    test_box_depth_limit()
    test_point_in_polygon()
    test_lasso_select()
    test_brush_select()
    print("\n=== Phase 5 ALL TESTS PASSED ===")
