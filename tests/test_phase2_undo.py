"""Phase 2 verification: Undo/redo command stack."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data.point_cloud import PointCloudData
from src.core.undo import UndoStack, LabelCommand, apply_label


def make_cloud(n=1000):
    positions = np.random.rand(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    return PointCloudData(positions=positions, colors=colors)


def test_basic_undo_redo():
    cloud = make_cloud()
    stack = UndoStack()

    # Apply label 1 to [0..99]
    apply_label(cloud, np.arange(0, 100), 1, stack)
    # Apply label 2 to [50..149]
    apply_label(cloud, np.arange(50, 150), 2, stack)

    # Current state: 0..49 = 1, 50..149 = 2, rest = 0
    assert (cloud.labels[0:50] == 1).all()
    assert (cloud.labels[50:150] == 2).all()
    assert (cloud.labels[150:] == 0).all()

    # Undo last command (label 2)
    stack.undo(cloud)
    assert (cloud.labels[0:100] == 1).all(), "0..99 should still be 1"
    assert (cloud.labels[100:] == 0).all(), "100..end should revert to 0"

    # Undo first command (label 1)
    stack.undo(cloud)
    assert (cloud.labels == 0).all(), "all labels should be 0"

    # Redo
    stack.redo(cloud)
    assert (cloud.labels[0:100] == 1).all()
    assert (cloud.labels[100:] == 0).all()

    stack.redo(cloud)
    assert (cloud.labels[0:50] == 1).all()
    assert (cloud.labels[50:150] == 2).all()

    print("PASS: Basic undo/redo")


def test_redo_truncation():
    cloud = make_cloud()
    stack = UndoStack()

    apply_label(cloud, np.arange(0, 100), 1, stack)
    apply_label(cloud, np.arange(100, 200), 2, stack)
    apply_label(cloud, np.arange(200, 300), 3, stack)

    # Undo twice, then push a new command - redo history should be truncated
    stack.undo(cloud)
    stack.undo(cloud)
    assert stack.redo_count == 2

    apply_label(cloud, np.arange(300, 400), 5, stack)
    assert stack.redo_count == 0, "pushing after undo should truncate redo"
    assert stack.undo_count == 2  # first command + new command

    print("PASS: Redo truncation on new push")


def test_can_undo_redo_flags():
    cloud = make_cloud()
    stack = UndoStack()

    assert not stack.can_undo
    assert not stack.can_redo

    apply_label(cloud, np.arange(0, 100), 1, stack)
    assert stack.can_undo
    assert not stack.can_redo

    stack.undo(cloud)
    assert not stack.can_undo
    assert stack.can_redo

    print("PASS: can_undo/can_redo flags")


def test_memory_tracking():
    cloud = make_cloud(10000)
    stack = UndoStack()

    apply_label(cloud, np.arange(0, 5000), 1, stack)
    # 5000 int32 indices + 5000 int32 old_labels = 40000 bytes
    assert stack.memory_usage == 40000, f"expected 40000, got {stack.memory_usage}"

    apply_label(cloud, np.arange(5000, 10000), 2, stack)
    assert stack.memory_usage == 80000

    print(f"PASS: Memory tracking ({stack.memory_usage} bytes for 2 commands)")


def test_empty_indices_noop():
    cloud = make_cloud()
    stack = UndoStack()

    cmd = apply_label(cloud, np.array([], dtype=np.int32), 1, stack)
    assert cmd is None
    assert not stack.can_undo
    print("PASS: Empty indices is no-op")


def test_undo_with_no_history():
    cloud = make_cloud()
    stack = UndoStack()
    assert stack.undo(cloud) is None
    assert stack.redo(cloud) is None
    print("PASS: Undo/redo with no history returns None")


if __name__ == '__main__':
    test_basic_undo_redo()
    test_redo_truncation()
    test_can_undo_redo_flags()
    test_memory_tracking()
    test_empty_indices_noop()
    test_undo_with_no_history()
    print("\n=== Phase 2 ALL TESTS PASSED ===")
