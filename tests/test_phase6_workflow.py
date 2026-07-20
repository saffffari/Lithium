"""Phase 6 verification: Label panel + polygon + curve tools + full workflow."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.point_cloud import PointCloudData
from src.data.labels import LabelRegistry
from src.core.tools.curve_tool import curve_select, _point_to_segment_distance_2d
from src.core.tools.polygon_tool import polygon_select
from src.core.undo import UndoStack, apply_label
from src.core.selection import SelectionBuffer, SelectionMode
from src.utils.math_utils import perspective, look_at


def test_point_to_segment():
    # Point directly above segment midpoint
    px = np.array([5.0], dtype=np.float32)
    py = np.array([3.0], dtype=np.float32)
    d = _point_to_segment_distance_2d(px, py, 0.0, 0.0, 10.0, 0.0)
    assert abs(d[0] - 3.0) < 0.01, f"expected 3.0, got {d[0]}"

    # Point at segment endpoint
    px = np.array([0.0], dtype=np.float32)
    py = np.array([0.0], dtype=np.float32)
    d = _point_to_segment_distance_2d(px, py, 0.0, 0.0, 10.0, 0.0)
    assert d[0] < 0.01

    # Point past endpoint (should use endpoint distance)
    px = np.array([15.0], dtype=np.float32)
    py = np.array([0.0], dtype=np.float32)
    d = _point_to_segment_distance_2d(px, py, 0.0, 0.0, 10.0, 0.0)
    assert abs(d[0] - 5.0) < 0.01
    print("PASS: Point to segment distance")


def test_curve_select():
    """Curve along x-axis at y=0 should select points near y=0."""
    positions = np.array([
        [0.0, 0.0, 0.0],   # on curve
        [0.5, 0.0, 0.0],   # on curve
        [0.0, 0.5, 0.0],   # off curve
    ], dtype=np.float32)
    view = look_at(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    proj = perspective(np.radians(60.0), 1.0, 0.1, 100.0)
    mvp = proj @ view
    # Polyline across screen center horizontally
    polyline = [(50, 128), (200, 128)]
    indices = curve_select(positions, mvp, polyline, 256, 256, threshold_px=20.0)
    # points on x-axis should be selected, point at y=0.5 world should not
    assert 0 in indices
    assert 1 in indices
    print(f"PASS: Curve select ({len(indices)} points)")


def test_polygon_select():
    """Polygon lasso is the same as lasso."""
    positions = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    view = look_at(
        np.array([0.0, 0.0, 5.0], dtype=np.float32),
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    proj = perspective(np.radians(60.0), 1.0, 0.1, 100.0)
    mvp = proj @ view
    polygon = [(100, 100), (156, 100), (156, 156), (100, 156)]
    indices = polygon_select(positions, mvp, polygon, 256, 256)
    assert 0 in indices
    print("PASS: Polygon select")


def test_full_annotation_workflow():
    """Full workflow: create labels, select, apply, undo."""
    positions = np.random.rand(1000, 3).astype(np.float32)
    colors = np.random.rand(1000, 3).astype(np.float32)
    cloud = PointCloudData(positions=positions, colors=colors)

    registry = LabelRegistry()
    vertebrae = registry.add_label("Vertebrae")
    disc = registry.add_label("Disc")

    undo_stack = UndoStack()
    selection = SelectionBuffer(1000)

    # Select and apply Vertebrae
    selection.select(np.arange(0, 300))
    apply_label(cloud, selection.selected_indices(), vertebrae, undo_stack)
    assert (cloud.labels[0:300] == vertebrae).all()

    # Select and apply Disc
    selection.clear()
    selection.select(np.arange(300, 500))
    apply_label(cloud, selection.selected_indices(), disc, undo_stack)
    assert (cloud.labels[300:500] == disc).all()
    assert (cloud.labels[500:] == 0).all()

    # Undo disc
    undo_stack.undo(cloud)
    assert (cloud.labels[300:500] == 0).all()
    assert (cloud.labels[0:300] == vertebrae).all()

    # Hide vertebrae
    registry.set_visible(vertebrae, False)
    lut = registry.get_color_lut()
    assert lut[vertebrae, 3] == 0.0, "hidden label alpha should be 0"

    # Lock disc
    registry.set_locked(disc, True)
    assert registry.get(disc).locked is True

    # Hierarchy
    thoracic = registry.add_label("Thoracic", parent_id=vertebrae)
    t7 = registry.add_label("T7", parent_id=thoracic)
    hierarchy = registry.get_hierarchy()
    names = [l.name for l, d in hierarchy]
    assert "T7" in names
    assert "Thoracic" in names
    print("PASS: Full annotation workflow")


if __name__ == '__main__':
    test_point_to_segment()
    test_curve_select()
    test_polygon_select()
    test_full_annotation_workflow()
    print("\n=== Phase 6 ALL TESTS PASSED ===")
