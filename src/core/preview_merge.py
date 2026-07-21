"""LS-6: carry paint applied to a preview cloud over to the full cloud.

While a cloud only has ``preview_gpu`` (async full-res load still in
flight), brush/box/lasso strokes write PREVIEW-length labels. The
full-res labels file is the catalog's source of truth and its persist
path rejects wrong-length arrays, so pre-fix that paint was orphaned:
never persisted, and visually discarded the moment full_gpu arrived.

The fix projects the painted (non-zero) preview labels onto the full
cloud through a nearest-neighbour index map:

- ``build_preview_to_full_idx`` builds the map with a scipy cKDTree.
  It is O(N log N) over the full cloud — App runs it on the full-res
  loader background thread (``_full_res_load_job``), never the frame
  loop. scipy is imported lazily so it stays off the startup path.
- ``merge_preview_labels_into_full`` scatters the non-zero preview
  labels through the map into the full array, in place. Zero (erased /
  never-painted) preview entries are ignored so labels already present
  on the full cloud are never clobbered by the sparser preview.

Previews are voxel-downsampled subsets of the full positions, so the
nearest-neighbour distance is zero and the mapping is exact — the same
property ``App._propagate_labels_to_preview`` relies on for the
opposite direction.
"""

from __future__ import annotations

import numpy as np


def build_preview_to_full_idx(preview_positions, full_positions,
                              workers: int = 4) -> np.ndarray | None:
    """Map each preview point to the index of its nearest full-res point.

    Returns an int64 array of length ``len(preview_positions)``, or
    None when the map can't be built (scipy unavailable, empty input,
    tree failure). Callers must treat None as "fall back + warn", not
    an error to raise.

    ``workers=4`` mirrors _propagate_labels_to_preview: enough
    parallelism to keep a 2M-point query around a second without
    pinning every core of the user's machine.
    """
    try:
        if preview_positions is None or full_positions is None:
            return None
        prev = np.ascontiguousarray(preview_positions, dtype=np.float32)
        full = np.ascontiguousarray(full_positions, dtype=np.float32)
        if prev.size == 0 or full.size == 0:
            return None
        from scipy.spatial import cKDTree  # lazy — keep scipy off startup
        tree = cKDTree(full)
        _, idx = tree.query(prev, k=1, workers=workers)
        return np.asarray(idx, dtype=np.int64).reshape(-1)
    except Exception as e:
        print(f"[preview merge] index build failed: {e}")
        return None


def merge_preview_labels_into_full(full_labels: np.ndarray,
                                   preview_labels: np.ndarray,
                                   preview_to_full_idx: np.ndarray) -> int:
    """Scatter non-zero preview labels onto the full array, in place.

    Returns the number of preview points merged (0 = nothing to do or
    inputs unusable). Only non-zero preview labels transfer, so the
    merge can add/overwrite paint where the user actually stroked but
    can never erase existing full-res labels.
    """
    if (full_labels is None or preview_labels is None
            or preview_to_full_idx is None):
        return 0
    if len(preview_labels) != len(preview_to_full_idx):
        print(f"[preview merge] map length {len(preview_to_full_idx)} != "
              f"preview labels {len(preview_labels)}; skipping merge.")
        return 0
    nz = preview_labels != 0
    if not nz.any():
        return 0
    idx = preview_to_full_idx[nz]
    if idx.size and (idx.min() < 0 or idx.max() >= len(full_labels)):
        print("[preview merge] map indices out of range for full cloud; "
              "skipping merge.")
        return 0
    full_labels[idx] = preview_labels[nz]
    return int(nz.sum())
