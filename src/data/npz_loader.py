"""NPZ file loader — flexible numpy archive format.

Recognized keys (first match wins):
    positions: 'positions', 'points', 'xyz', 'coord', 'coords', 'pts'
    colors:    'colors', 'color', 'rgb', 'colours'
    intensity: 'intensity', 'intensities'

Any label-like array ('label', 'seg', 'class', …) is exposed as a
scalar field rather than being written into cloud.labels — source-file
label ids are not guaranteed to match our registry and would make
points render as invisible. See ply_loader for the same rationale.

Any other 1D array of length N is exposed as a scalar field.
"""

import os
import numpy as np

from src.data.point_cloud import PointCloudData


_POS_KEYS = ('positions', 'points', 'xyz', 'coord', 'coords', 'pts')
_COLOR_KEYS = ('colors', 'color', 'rgb', 'colours')
_LABEL_KEYS = ('labels', 'label', 'seg', 'segmentation', 'class', 'classes')


def _find(d, keys):
    for k in keys:
        if k in d:
            return k
    # case-insensitive fallback
    lower = {k.lower(): k for k in d}
    for k in keys:
        if k in lower:
            return lower[k]
    return None


def load_npz(path: str) -> PointCloudData:
    """Load a point cloud from an .npz archive."""
    try:
        # Process arrays lazily inside the context manager to avoid
        # materializing the entire archive into a separate dict (which
        # doubles peak memory for large files).
        with np.load(path, allow_pickle=False) as npz:
            file_keys = set(npz.files)

            pos_key = _find(npz, _POS_KEYS)
            if pos_key is None:
                # Fallback: first (N, 3) float-ish array
                for k in npz.files:
                    v = npz[k]
                    if v.ndim == 2 and v.shape[1] == 3 and np.issubdtype(v.dtype, np.number):
                        pos_key = k
                        break
            if pos_key is None:
                raise ValueError(
                    f"NPZ has no recognizable positions array (keys: {list(file_keys)})")

            positions = np.ascontiguousarray(npz[pos_key], dtype=np.float32)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError(
                    f"Positions key '{pos_key}' has shape {positions.shape}, expected (N, 3)")
            n = len(positions)

            # Colors — accept (N, 3) or (N, 4+); we only use RGB.
            color_key = _find(npz, _COLOR_KEYS)
            color_arr = npz[color_key] if color_key is not None else None
            if (color_arr is not None
                    and color_arr.ndim == 2
                    and color_arr.shape[0] == n
                    and color_arr.shape[1] >= 3):
                c = np.asarray(color_arr[:, :3], dtype=np.float32)
                if c.size > 0 and float(c.max()) > 1.0:
                    c = c / 255.0
                colors = np.ascontiguousarray(c)
            else:
                colors = np.ones((n, 3), dtype=np.float32)

            # Scalars — any remaining 1D arrays of length n
            consumed = {pos_key, color_key}
            scalars: dict[str, np.ndarray] = {}
            for k in npz.files:
                if k in consumed:
                    continue
                v = npz[k]
                if v.ndim == 1 and v.shape[0] == n and np.issubdtype(v.dtype, np.number):
                    scalars[k] = np.asarray(v, dtype=np.float32)

    except (OSError, ValueError, EOFError) as e:
        if isinstance(e, ValueError) and "NPZ has no" in str(e):
            raise
        raise IOError(f"Failed to read NPZ file {path}: {e}") from e

    return PointCloudData(
        positions=positions,
        colors=colors,
        scalars=scalars,
        file_path=path,
        metadata={'source_unit': 'raw'},
    )
