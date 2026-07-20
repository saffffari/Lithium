"""Brush-stroke undo grouping for direct-paint label workflow.

A ``LabelStrokeRecorder`` is begun at mouse-press, fed incremental
indices as the user drags, and ended at mouse-release. It writes the
new label to ``cloud.labels`` immediately so the user sees the paint
happen live, captures the pre-stroke label of each point the first
time it is touched, and pushes a single ``LabelCommand`` onto the undo
stack at stroke end. Subsequent touches on the same point are ignored
(first-touch wins) so a stroke that passes over the same point twice
still produces a correct undo.

Used by brush/box/lasso/curve/pick handlers in ``src.main.App``.
"""

from __future__ import annotations

import numpy as np

from src.core.undo import LabelCommand, UndoStack


class LabelStrokeRecorder:
    """Collects the points touched by a single paint stroke.

    Lifecycle:
        recorder.begin(cloud, new_label, description)
        recorder.add_points(indices)   # repeat during drag
        ...
        cmd = recorder.end(undo_stack)  # pushes one LabelCommand
    """

    def __init__(self) -> None:
        self._cloud = None
        self._new_label: int = 0
        self._description: str = ""
        # Boolean mask: True for points already touched this stroke
        self._touched_mask: np.ndarray | None = None
        # Collected indices and their original labels (built incrementally)
        self._touched_indices: list[np.ndarray] = []
        self._touched_originals: list[np.ndarray] = []

    @property
    def is_active(self) -> bool:
        return self._cloud is not None

    def begin(self, cloud, new_label: int, description: str = "") -> None:
        """Start a new stroke on ``cloud``. Any previous stroke is discarded."""
        self._cloud = cloud
        self._new_label = int(new_label)
        self._description = description or f"Paint as label {new_label}"
        self._touched_mask = np.zeros(cloud.point_count, dtype=np.bool_)
        self._touched_indices = []
        self._touched_originals = []

    def add_points(self, indices: np.ndarray) -> int:
        """Apply the new label to ``indices``, recording first-touch originals.

        Returns the number of *newly touched* points. Uses a numpy
        boolean mask for O(1) per-point membership testing instead of
        a Python dict, avoiding thousands of int() conversions per tick.
        """
        if self._cloud is None or len(indices) == 0:
            return 0
        idx = np.unique(np.asarray(indices, dtype=np.int64).ravel())
        # Filter to points not yet touched this stroke
        new_mask = ~self._touched_mask[idx]
        if not new_mask.any():
            return 0
        new_arr = idx[new_mask]
        labels = self._cloud.labels
        originals = labels[new_arr].astype(np.int32, copy=True)
        self._touched_mask[new_arr] = True
        self._touched_indices.append(new_arr)
        self._touched_originals.append(originals)
        labels[new_arr] = self._new_label
        return len(new_arr)

    def end(self, undo_stack: UndoStack) -> LabelCommand | None:
        """Finish the stroke and push a single LabelCommand onto the stack."""
        if self._cloud is None:
            return None
        t_indices = self._touched_indices
        t_originals = self._touched_originals
        self._cloud = None
        self._touched_mask = None
        self._touched_indices = []
        self._touched_originals = []
        if not t_indices:
            return None
        point_indices = np.concatenate(t_indices).astype(np.int32)
        old_labels = np.concatenate(t_originals).astype(np.int32)
        # Sort for deterministic order
        order = np.argsort(point_indices)
        point_indices = point_indices[order]
        old_labels = old_labels[order]
        cmd = LabelCommand(
            point_indices=point_indices,
            old_labels=old_labels,
            new_label=self._new_label,
            description=self._description,
        )
        undo_stack.push(cmd)
        return cmd

    def cancel(self) -> None:
        """Revert in-progress writes and discard the stroke."""
        if self._cloud is None:
            return
        labels = self._cloud.labels
        for idx_arr, orig_arr in zip(self._touched_indices, self._touched_originals):
            labels[idx_arr] = orig_arr
        self._cloud = None
        self._touched_mask = None
        self._touched_indices = []
        self._touched_originals = []
