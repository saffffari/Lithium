"""LabelStrokeRecorder verification — direct-paint undo grouping."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.point_cloud import PointCloudData
from src.core.undo import UndoStack
from src.core.stroke_recorder import LabelStrokeRecorder


def make_cloud(n=1000):
    positions = np.random.rand(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    return PointCloudData(positions=positions, colors=colors)


def test_single_stroke_one_command():
    """A brush stroke that touches many indices produces one undo command."""
    cloud = make_cloud(500)
    stack = UndoStack()
    rec = LabelStrokeRecorder()

    rec.begin(cloud, new_label=5, description="test stroke")
    # Simulate four ticks of a brush drag with overlapping hits.
    rec.add_points(np.arange(0, 50))
    rec.add_points(np.arange(30, 80))   # 20 overlap
    rec.add_points(np.arange(60, 110))  # 20 overlap with previous
    rec.add_points(np.array([5, 5, 5]))  # repeats, all skipped
    cmd = rec.end(stack)

    assert cmd is not None
    # Union of all unique indices touched: [0..109]
    assert len(cmd.point_indices) == 110
    assert (cmd.new_label == 5)
    # Undo-stack only got ONE command, not four.
    assert stack.undo_count == 1
    print("PASS: Single stroke → single LabelCommand")


def test_overlapping_touches_preserve_first_label():
    """A point touched twice records its PRE-stroke label, not the intermediate."""
    cloud = make_cloud(50)
    # Seed point 7 with label 3 before the stroke starts.
    cloud.labels[7] = 3
    stack = UndoStack()
    rec = LabelStrokeRecorder()

    rec.begin(cloud, new_label=9, description="double-touch")
    rec.add_points(np.array([7]))
    # Re-touch point 7 in a later tick — should be ignored (first-touch wins).
    rec.add_points(np.array([7, 8]))
    cmd = rec.end(stack)

    # After the stroke, point 7 should carry label 9.
    assert cloud.labels[7] == 9
    assert cloud.labels[8] == 9

    # Undo should restore point 7 to label 3 (not the intermediate new label).
    stack.undo(cloud)
    assert cloud.labels[7] == 3, f"expected 3, got {cloud.labels[7]}"
    assert cloud.labels[8] == 0
    print("PASS: First-touch original label preserved across re-touches")


def test_cancel_reverts_writes():
    """Cancelling a stroke mid-drag reverts cloud.labels to pre-stroke state."""
    cloud = make_cloud(20)
    cloud.labels[0:5] = 4
    snapshot = cloud.labels.copy()
    rec = LabelStrokeRecorder()

    rec.begin(cloud, new_label=9)
    rec.add_points(np.arange(0, 10))
    # Mid-stroke cancellation (e.g. user hit ESC).
    rec.cancel()

    assert (cloud.labels == snapshot).all(), \
        f"labels should match pre-stroke snapshot, got {cloud.labels!r}"
    assert not rec.is_active
    print("PASS: Cancel reverts in-progress stroke")


def test_empty_stroke_returns_none():
    """Ending a stroke that touched no points returns None (no undo entry)."""
    cloud = make_cloud(10)
    stack = UndoStack()
    rec = LabelStrokeRecorder()

    rec.begin(cloud, new_label=1)
    cmd = rec.end(stack)

    assert cmd is None
    assert stack.undo_count == 0
    print("PASS: Empty stroke → no undo command")


def test_add_points_before_begin_is_noop():
    """Calling add_points without begin is a harmless no-op."""
    cloud = make_cloud(10)
    rec = LabelStrokeRecorder()
    # No begin — should not crash or write anything.
    n = rec.add_points(np.array([0, 1, 2]))
    assert n == 0
    assert (cloud.labels == 0).all()
    print("PASS: add_points without begin is a no-op")


def test_stroke_is_undoable_and_redoable():
    """After end, undo restores originals and redo re-applies the new label."""
    cloud = make_cloud(30)
    cloud.labels[0:15] = 7
    stack = UndoStack()
    rec = LabelStrokeRecorder()

    rec.begin(cloud, new_label=3)
    rec.add_points(np.arange(0, 25))
    rec.end(stack)

    # Labels [0..24] now = 3, rest still 0
    assert (cloud.labels[0:25] == 3).all()
    assert (cloud.labels[25:] == 0).all()

    # Undo → points 0..14 back to 7, points 15..24 back to 0
    stack.undo(cloud)
    assert (cloud.labels[0:15] == 7).all()
    assert (cloud.labels[15:] == 0).all()

    # Redo → all 25 back to 3
    stack.redo(cloud)
    assert (cloud.labels[0:25] == 3).all()
    assert (cloud.labels[25:] == 0).all()
    print("PASS: Stroke round-trip undo/redo")


if __name__ == '__main__':
    test_single_stroke_one_command()
    test_overlapping_touches_preserve_first_label()
    test_cancel_reverts_writes()
    test_empty_stroke_returns_none()
    test_add_points_before_begin_is_noop()
    test_stroke_is_undoable_and_redoable()
    print("\nAll LabelStrokeRecorder tests passed.")
