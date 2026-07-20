"""Phase 4 verification: Selection buffer + pick tool."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.selection import SelectionBuffer, SelectionMode, mode_from_mods
from src.core.tools.pick_tool import pick_point, _project_to_screen
from src.utils.math_utils import perspective, look_at


def test_selection_buffer_basic():
    buf = SelectionBuffer(1000)
    assert buf.selected_count == 0

    buf.select(np.arange(10, 20))
    assert buf.selected_count == 10
    assert list(buf.selected_indices()) == list(range(10, 20))

    # REPLACE overwrites
    buf.select(np.arange(30, 40))
    assert buf.selected_count == 10
    assert list(buf.selected_indices()) == list(range(30, 40))

    # ADD unions
    buf.select(np.arange(50, 60), SelectionMode.ADD)
    assert buf.selected_count == 20

    # SUBTRACT removes
    buf.select(np.arange(30, 35), SelectionMode.SUBTRACT)
    assert buf.selected_count == 15

    buf.clear()
    assert buf.selected_count == 0
    print("PASS: SelectionBuffer modes")


def test_mode_from_mods():
    assert mode_from_mods(False, False) == SelectionMode.REPLACE
    assert mode_from_mods(True, False) == SelectionMode.ADD
    assert mode_from_mods(False, True) == SelectionMode.SUBTRACT
    assert mode_from_mods(True, True) == SelectionMode.SUBTRACT  # Ctrl wins
    print("PASS: Modifier mapping")


def test_selection_invert():
    buf = SelectionBuffer(10)
    buf.select(np.arange(0, 5))
    assert buf.selected_count == 5
    buf.invert()
    assert buf.selected_count == 5
    assert list(buf.selected_indices()) == [5, 6, 7, 8, 9]
    print("PASS: Invert")


def test_selection_resize():
    buf = SelectionBuffer(100)
    buf.select(np.arange(10, 20))
    buf.resize(1000)
    assert buf.point_count == 1000
    assert buf.selected_count == 0  # reset on resize
    print("PASS: Resize clears")


def test_pick_simple():
    """Place a known point, project with identity camera, pick at its screen coord."""
    # Single point at origin
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    # Identity camera looking down -z from distance 5
    view = look_at(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    proj = perspective(np.radians(60.0), 1.0, 0.1, 100.0)
    mvp = proj @ view

    width, height = 256, 256

    # Origin projects to screen center
    screen, depth = _project_to_screen(positions, mvp, width, height)
    cx, cy = screen[0]
    assert abs(cx - 128) < 2, f"expected x~128, got {cx}"
    assert abs(cy - 128) < 2, f"expected y~128, got {cy}"

    # Pick at screen center should find the point
    idx = pick_point(positions, mvp, 128.0, 128.0, width, height, radius_px=20.0)
    assert idx == 0, f"expected 0, got {idx}"

    # Pick far away returns -1
    idx = pick_point(positions, mvp, 10.0, 10.0, width, height, radius_px=5.0)
    assert idx == -1, f"expected -1, got {idx}"
    print("PASS: Pick tool")


def test_pick_nearest_camera():
    """Two overlapping points at different depths - pick the nearer one."""
    positions = np.array([
        [0.0, 0.0, 0.0],   # further from camera
        [0.0, 0.0, 2.0],   # closer to camera at (0,0,5)
    ], dtype=np.float32)

    view = look_at(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    proj = perspective(np.radians(60.0), 1.0, 0.1, 100.0)
    mvp = proj @ view

    width, height = 256, 256
    idx = pick_point(positions, mvp, 128.0, 128.0, width, height, radius_px=20.0)
    assert idx == 1, f"expected closer point (index 1), got {idx}"
    print("PASS: Pick selects nearest to camera")


if __name__ == '__main__':
    test_selection_buffer_basic()
    test_mode_from_mods()
    test_selection_invert()
    test_selection_resize()
    test_pick_simple()
    test_pick_nearest_camera()
    print("\n=== Phase 4 ALL TESTS PASSED ===")
