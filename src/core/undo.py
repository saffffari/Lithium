"""Undo/redo command stack for label operations.

Each label change is captured as a LabelCommand storing the old labels
at the modified indices. Undo restores old labels; redo re-applies the
new label.
"""

from dataclasses import dataclass
import numpy as np


MAX_MEMORY_BYTES = 500 * 1024 * 1024  # 500 MB cap


@dataclass
class LabelCommand:
    """A single undoable label operation."""
    point_indices: np.ndarray   # int32, indices that were modified
    old_labels: np.ndarray      # int32, previous label values at those indices
    new_label: int              # the label that was applied
    description: str = ""       # human-readable summary

    @property
    def memory_bytes(self) -> int:
        return int(self.point_indices.nbytes + self.old_labels.nbytes)


class UndoStack:
    """Command stack with undo/redo cursor and memory limit."""

    def __init__(self, max_memory: int = MAX_MEMORY_BYTES):
        self._commands: list[LabelCommand] = []
        self._cursor: int = 0  # points to next redo position
        self._max_memory = max_memory
        self._total_bytes: int = 0

    def push(self, cmd: LabelCommand):
        """Push a command. Truncates any redo history after cursor."""
        # Drop any redo history beyond current cursor
        if self._cursor < len(self._commands):
            for dropped in self._commands[self._cursor:]:
                self._total_bytes -= dropped.memory_bytes
            self._commands = self._commands[:self._cursor]
        self._commands.append(cmd)
        self._total_bytes += cmd.memory_bytes
        self._cursor = len(self._commands)

        # Enforce memory cap from the oldest end
        while self._commands and self._total_bytes > self._max_memory:
            dropped = self._commands.pop(0)
            self._total_bytes -= dropped.memory_bytes
            self._cursor -= 1
            if self._cursor < 0:
                self._cursor = 0

    def undo(self, cloud) -> LabelCommand | None:
        """Restore old labels at cmd.point_indices. Returns the command or None.

        Drops the command (and the rest of the stack) if its indices
        don't fit the current cloud — happens when the user switched
        clouds without clearing the stack and a stale stroke from a
        bigger cloud would IndexError. The drop is deliberate: a stroke
        recorded against a different cloud has no meaning here.
        """
        if self._cursor == 0:
            return None
        self._cursor -= 1
        cmd = self._commands[self._cursor]
        n = len(cloud.labels)
        if cmd.point_indices.size and int(cmd.point_indices.max()) >= n:
            # Stale command — purge it and everything cursor-and-after.
            del self._commands[self._cursor:]
            return None
        cloud.labels[cmd.point_indices] = cmd.old_labels
        return cmd

    def redo(self, cloud) -> LabelCommand | None:
        """Re-apply new label at cmd.point_indices. Returns the command or None."""
        if self._cursor >= len(self._commands):
            return None
        cmd = self._commands[self._cursor]
        n = len(cloud.labels)
        if cmd.point_indices.size and int(cmd.point_indices.max()) >= n:
            # Stale forward-history — discard from here.
            del self._commands[self._cursor:]
            return None
        cloud.labels[cmd.point_indices] = cmd.new_label
        self._cursor += 1
        return cmd

    def clear(self):
        self._commands.clear()
        self._cursor = 0
        self._total_bytes = 0

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._commands)

    @property
    def undo_count(self) -> int:
        return self._cursor

    @property
    def redo_count(self) -> int:
        return len(self._commands) - self._cursor

    @property
    def memory_usage(self) -> int:
        return self._total_bytes


def apply_label(cloud, indices: np.ndarray, new_label: int,
                undo_stack: UndoStack, description: str = "") -> LabelCommand | None:
    """Apply a label to points at the given indices and push to undo stack.

    Returns the LabelCommand that was pushed, or ``None`` if ``indices`` was
    empty (no-op, nothing pushed to the undo stack).
    """
    indices = np.asarray(indices, dtype=np.int32)
    if len(indices) == 0:
        return None
    old_labels = cloud.labels[indices].copy()
    cloud.labels[indices] = new_label
    if not description:
        description = f"Label {len(indices)} pts as {new_label}"
    cmd = LabelCommand(
        point_indices=indices,
        old_labels=old_labels,
        new_label=new_label,
        description=description,
    )
    undo_stack.push(cmd)
    return cmd
