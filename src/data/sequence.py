"""Time-series point cloud sequence support.

A PointCloudSequence is an ordered list of frame files with LRU caching
to keep memory bounded. Labels persist per-frame.

LS-1: each frame can carry a catalog ``file_key`` (``frame_keys``,
populated by ``App._load_as_sequence`` via ``catalog.register_files``).
When a key is present:

- ``get_frame`` re-applies the persisted catalog labels after loading a
  frame from disk, so a frame paged back in after LRU eviction shows
  the labels the user painted on it earlier in the session (or in a
  previous session).
- LRU eviction persists the evicted frame's labels first, so paint on
  a frame that scrolls out of the 3-frame cache is never discarded.
- ``flush_labels`` persists every cached frame — App._cleanup calls it
  at shutdown so frames other than the one currently on the GPU are
  covered too.

Persistence goes through ``cloud_store.save_cloud_labels`` (atomic
tmp + os.replace, namespace-aware). All sequence methods run on the
main thread, so the process-active label namespace applies.
"""

import os
import re
import numpy as np

from src.data import cloud_store
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
        # LS-1: per-frame catalog file_keys. None = no persistence for
        # that frame (catalog unavailable / registration failed).
        self.frame_keys: list[str | None] = [None] * len(self.frame_paths)
        # Optional callable(str) invoked with a human-readable message
        # when a persist fails — App wires this to set_status_banner so
        # eviction-time save errors aren't print-only.
        self.persist_error_cb = None

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

        # LS-1: re-apply persisted catalog labels so a frame paged back
        # in after LRU eviction (or opened in a later session) keeps
        # the paint it received earlier.
        key = self.frame_key(index)
        if key:
            stored = cloud_store.load_cloud_labels(key)
            if stored is not None:
                stored = np.asarray(stored, dtype=np.int32).reshape(-1)
                if stored.shape[0] == cloud.point_count:
                    cloud.labels[:] = stored
                else:
                    print(f"[sequence] persisted labels for "
                          f"{self.frame_name(index)}: length "
                          f"{stored.shape[0]} != {cloud.point_count}; "
                          f"ignoring stale array.")

        self._cache[index] = cloud
        self._access_order.append(index)
        # Record whether this frame already has any non-zero labels
        # (projects loaded from disk may carry labels even on first load)
        self._has_labels[index] = bool((cloud.labels != 0).any())

        # Evict oldest entries until within budget. Persist the evicted
        # frame's labels first (LS-1) — automation's propagate-all paints
        # frames that were never the *current* frame, so eviction is the
        # only persistence point they're guaranteed to hit.
        while len(self._cache) > self.cache_size:
            oldest = self._access_order.pop(0)
            if oldest != index:
                self._persist_frame_labels(oldest)
                del self._cache[oldest]
            else:
                # Don't evict the just-loaded frame; put at end and continue
                self._access_order.append(oldest)

        return cloud

    def frame_key(self, index: int | None = None) -> str | None:
        """Catalog file_key for the frame at ``index`` (default current)."""
        if index is None:
            index = self.current_index
        if 0 <= index < len(self.frame_keys):
            return self.frame_keys[index]
        return None

    def _persist_frame_labels(self, index: int) -> str | None:
        """Persist a cached frame's labels to the catalog.

        Returns None on success/no-op, or an error string. Skips frames
        with no key and frames that are all-unlabeled UNLESS a labels
        file already exists on disk (erase-everything must not
        resurrect the old paint on reload).
        """
        cloud = self._cache.get(index)
        key = self.frame_key(index)
        if cloud is None or not key or cloud.labels is None:
            return None
        if (not (cloud.labels != 0).any()
                and not cloud_store.has_cloud_labels(key)):
            return None  # never painted, nothing on disk — skip the write
        err = cloud_store.save_cloud_labels(key, cloud.labels)
        if err and self.persist_error_cb is not None:
            try:
                self.persist_error_cb(
                    f"Sequence frame {self.frame_name(index)}: {err}")
            except Exception:
                pass
        return err

    def flush_labels(self) -> int:
        """Persist labels for every cached frame. Returns frames written.

        Called by App._cleanup at shutdown so cached frames other than
        the one currently on the GPU don't lose their paint.
        """
        flushed = 0
        for idx in list(self._cache.keys()):
            if self._persist_frame_labels(idx) is None:
                flushed += 1
        return flushed

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
