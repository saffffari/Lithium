"""Selection buffer and modes for point cloud annotation.

A SelectionBuffer wraps a boolean mask over all points in a cloud.
Selection tools modify this buffer via SelectionMode (replace, add, subtract).
"""

from enum import IntEnum
import numpy as np


class SelectionMode(IntEnum):
    REPLACE = 0   # clear existing selection, use new
    ADD = 1       # union with existing
    SUBTRACT = 2  # remove from existing


class SelectionBuffer:
    """Boolean mask over all points in a cloud."""

    def __init__(self, point_count: int):
        self.point_count = point_count
        self._mask = np.zeros(point_count, dtype=bool)

    def resize(self, point_count: int):
        """Resize the buffer (e.g., when switching clouds)."""
        if point_count != self.point_count:
            self.point_count = point_count
            self._mask = np.zeros(point_count, dtype=bool)

    def select(self, indices: np.ndarray, mode: SelectionMode = SelectionMode.REPLACE):
        """Apply a selection operation.

        indices: int array of indices to select
        mode: how to combine with the current selection
        """
        indices = np.asarray(indices, dtype=np.int64)
        if mode == SelectionMode.REPLACE:
            self._mask[:] = False
            if len(indices) > 0:
                self._mask[indices] = True
        elif mode == SelectionMode.ADD:
            if len(indices) > 0:
                self._mask[indices] = True
        elif mode == SelectionMode.SUBTRACT:
            if len(indices) > 0:
                self._mask[indices] = False

    def select_mask(self, mask: np.ndarray, mode: SelectionMode = SelectionMode.REPLACE):
        """Apply a boolean mask directly."""
        if mode == SelectionMode.REPLACE:
            self._mask[:] = mask
        elif mode == SelectionMode.ADD:
            self._mask |= mask
        elif mode == SelectionMode.SUBTRACT:
            self._mask &= ~mask

    def clear(self):
        self._mask[:] = False

    def invert(self):
        self._mask[:] = ~self._mask

    def selected_indices(self) -> np.ndarray:
        """Return indices of currently selected points."""
        return np.nonzero(self._mask)[0].astype(np.int32)

    @property
    def selected_count(self) -> int:
        return int(self._mask.sum())

    @property
    def mask(self) -> np.ndarray:
        return self._mask


def mode_from_mods(shift: bool, ctrl: bool) -> SelectionMode:
    """Map keyboard modifiers to selection mode.

    Shift = ADD, Ctrl = SUBTRACT, neither = REPLACE.
    """
    if ctrl:
        return SelectionMode.SUBTRACT
    if shift:
        return SelectionMode.ADD
    return SelectionMode.REPLACE
