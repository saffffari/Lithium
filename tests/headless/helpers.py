"""Helper utilities for the headless test harness."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def make_tiny_ply(path: str, n: int = 100, *,
                  with_colors: bool = True, seed: int = 42) -> str:
    """Write a small binary PLY at ``path`` with ``n`` points.

    Deterministic for a given seed so cross-session tests can rely on
    identical file_keys after a copy. Returns the absolute path.
    """
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3)).astype(np.float32) * 10.0
    if with_colors:
        cols = (rng.random((n, 3)) * 255.0).astype(np.uint8)
        dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        arr = np.empty(n, dtype=dtype)
        arr['x'] = pts[:, 0]
        arr['y'] = pts[:, 1]
        arr['z'] = pts[:, 2]
        arr['red'] = cols[:, 0]
        arr['green'] = cols[:, 1]
        arr['blue'] = cols[:, 2]
    else:
        dtype = [('x', '<f4'), ('y', '<f4'), ('z', '<f4')]
        arr = np.empty(n, dtype=dtype)
        arr['x'] = pts[:, 0]
        arr['y'] = pts[:, 1]
        arr['z'] = pts[:, 2]
    el = PlyElement.describe(arr, 'vertex')
    PlyData([el], text=False, byte_order='<').write(path)
    return os.path.abspath(path)


def make_companion_label_npy(ply_path: str, labels: np.ndarray) -> str:
    """Write ``{stem}_labels.npy`` next to ``ply_path``. Returns the path."""
    stem, _ext = os.path.splitext(ply_path)
    out = stem + "_labels.npy"
    np.save(out, labels.astype(np.int32, copy=False))
    return out


def count_label_files_on_disk(library_dir: str) -> int:
    """Number of non-tmp ``.npy`` files in ``<library_dir>/labels/``."""
    labels_dir = Path(library_dir) / "labels"
    if not labels_dir.exists():
        return 0
    return sum(
        1 for p in labels_dir.iterdir()
        if p.is_file() and p.suffix == ".npy"
        and not p.name.startswith("_tmp_")
    )


def labels_npy_path(library_dir: str, file_key: str) -> Path:
    """Compute the catalog labels file path for ``file_key``."""
    return Path(library_dir) / "labels" / f"{file_key}.npy"
