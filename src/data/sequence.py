"""Time-series point cloud sequence support.

A PointCloudSequence is an ordered list of frame files with LRU caching
to keep memory bounded. Labels persist per-frame.
"""

import os
import re
import numpy as np

from src.data.point_cloud import PointCloudData
from src.data.loader import load_point_cloud


_NUMBERED_PATTERN = re.compile(r'(\d+)', re.IGNORECASE)


def _numeric_sort_key(name: str) -> tuple:
    """Sort key that handles numeric suffixes (frame_1 < frame_2 < frame_10)."""
    match = _NUMBERED_PATTERN.search(name)
    if match:
        return (0, int(match.group(1)), name)
    return (1, 0, name)


def detect_sequence(directory: str) -> list[str] | None:
    """Return a list of files that appear to form a numbered sequence.

    Returns None if the directory doesn't look like a sequence.
    """
    from src.data.loader import SUPPORTED_EXTENSIONS
    files = []
    for name in os.listdir(directory):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            files.append(name)
    if len(files) < 2:
        return None
    # All files must share a common stem and have a numeric suffix
    numbered = [(name, _NUMBERED_PATTERN.search(name)) for name in files]
    if not all(m for _, m in numbered):
        return None
    files.sort(key=_numeric_sort_key)
    return [os.path.join(directory, f) for f in files]


class PointCloudSequence:
    """Time-series of point cloud frames with LRU caching."""

    def __init__(self, frame_paths: list[str], cache_size: int = 3):
        self.frame_paths = list(frame_paths)
        self.cache_size = cache_size
        self.current_index = 0
        self._cache: dict[int, PointCloudData] = {}
        self._access_order: list[int] = []  # LRU order
        # Persistent "does frame N have any non-zero labels?" map. Survives
        # LRU eviction so the timeline markers stay accurate across scrubs.
        self._has_labels: dict[int, bool] = {}

    @property
    def frame_count(self) -> int:
        return len(self.frame_paths)

    def get_frame(self, index: int) -> PointCloudData:
        """Return the frame at index, loading from disk if not cached."""
        if not (0 <= index < self.frame_count):
            raise IndexError(f"Frame {index} out of range [0, {self.frame_count})")
        if index in self._cache:
            # Bump LRU
            self._access_order.remove(index)
            self._access_order.append(index)
            return self._cache[index]

        # Load from disk
        cloud = load_point_cloud(self.frame_paths[index])
        # Ensure labels exist
        if cloud.labels is None:
            cloud.labels = np.zeros(cloud.point_count, dtype=np.int32)

        self._cache[index] = cloud
        self._access_order.append(index)
        # Record whether this frame already has any non-zero labels
        # (projects loaded from disk may carry labels even on first load)
        self._has_labels[index] = bool((cloud.labels != 0).any())

        # Evict oldest entries until within budget
        while len(self._cache) > self.cache_size:
            oldest = self._access_order.pop(0)
            if oldest != index:
                del self._cache[oldest]
            else:
                # Don't evict the just-loaded frame; put at end and continue
                self._access_order.append(oldest)

        return cloud

    def current_frame(self) -> PointCloudData:
        return self.get_frame(self.current_index)

    def step(self, delta: int) -> PointCloudData:
        """Advance by delta frames (with clamping), return new frame."""
        self.current_index = max(0, min(self.frame_count - 1, self.current_index + delta))
        return self.current_frame()

    def seek(self, index: int) -> PointCloudData:
        self.current_index = max(0, min(self.frame_count - 1, index))
        return self.current_frame()

    def frame_name(self, index: int = None) -> str:
        if index is None:
            index = self.current_index
        return os.path.basename(self.frame_paths[index])

    def has_labels(self, index: int) -> bool:
        """Return True if the frame at index has any non-zero labels.

        For cached frames we re-scan live (so in-place label mutations
        show up immediately). For evicted frames we fall back to the
        persistent bool cache — keeps timeline green dots accurate
        even for pages swapped out of the LRU.
        """
        cloud = self._cache.get(index)
        if cloud is not None and cloud.labels is not None:
            has = bool((cloud.labels != 0).any())
            self._has_labels[index] = has
            return has
        return self._has_labels.get(index, False)
