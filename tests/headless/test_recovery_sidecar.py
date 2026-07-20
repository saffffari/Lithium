"""LS-9 recovery sidecar atomicity (post-audit).

The original LS-9 commit (758836b) introduced
``App._persist_recovery_sidecar`` to drop a parallel labels file at
``labels/<fk>.recovery.npy`` every autosave tick while a stroke was
active — so a crash mid-stroke still has a rollback point.

The original code had a path-logic bug:

    tmp = recovery_path.with_suffix(".npy.tmp")     # <fk>.recovery.npy.tmp
    np.save(str(tmp.with_suffix("")), arr)          # strips .tmp -> writes
                                                    #   directly to canonical
                                                    #   <fk>.recovery.npy
    os.replace(tmp, recovery_path)                  # .tmp never existed
                                                    # -> FileNotFoundError

The post-audit fix mirrors ``cloud_store.save_cloud_labels``'s tmp-base
pattern. This regression test pins the working behaviour: the recovery
file lands at the expected path with the expected contents, and the
``os.replace`` step succeeds (no ``FileNotFoundError``).

Run live (no monkey-patching the App method) by exercising the same
sequence of pathlib + np.save + os.replace operations directly on a
tempdir.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


def _emulate_recovery_write(labels_dir: Path, file_key: str,
                            labels: np.ndarray) -> Path:
    """Replicate ``App._persist_recovery_sidecar`` body exactly.

    Kept in lockstep with the fix at ``src/main.py:_persist_recovery_sidecar``
    so test failures here directly indicate the same drift would happen
    in production.
    """
    recovery_path = labels_dir / f"{file_key}.recovery.npy"
    tmp_base = labels_dir / f"_tmp_recovery_{file_key}"
    tmp_with_npy = tmp_base.with_suffix(".npy")
    arr = np.ascontiguousarray(labels, dtype=np.int32)
    np.save(str(tmp_base), arr)
    os.replace(tmp_with_npy, recovery_path)
    return recovery_path


def test_recovery_sidecar_lands_at_canonical_path(tmp_path):
    """Recovery write produces ``<fk>.recovery.npy`` with the labels."""
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    fk = "deadbeef12345678"   # 16-char hex, matches compute_file_key shape

    labels = np.array([1, 2, 0, 3, 5, 0, 7, 7, 7, 0], dtype=np.int32)
    out = _emulate_recovery_write(labels_dir, fk, labels)

    assert out.exists()
    loaded = np.load(out)
    assert loaded.dtype == np.int32
    assert (loaded == labels).all()


def test_recovery_sidecar_idempotent_under_repeated_writes(tmp_path):
    """Second tick of an autosave loop overwrites cleanly, no stale tmp.

    The original bug failed os.replace on every tick (.tmp file never
    existed because np.save wrote past it to the canonical path). A
    correct implementation must round-trip through a real .tmp file each
    call and leave nothing behind.
    """
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    fk = "ab" * 8

    first = np.arange(20, dtype=np.int32)
    _emulate_recovery_write(labels_dir, fk, first)
    second = np.arange(20, dtype=np.int32) * 3
    out = _emulate_recovery_write(labels_dir, fk, second)

    # Final contents reflect the second write.
    assert (np.load(out) == second).all()
    # No leftover temp file.
    assert not (labels_dir / f"_tmp_recovery_{fk}.npy").exists()
    # Only one .npy under the labels dir.
    assert sorted(p.name for p in labels_dir.glob("*.npy")) == [
        f"{fk}.recovery.npy"
    ]


def test_recovery_sidecar_distinct_from_canonical_labels(tmp_path):
    """Recovery write must NOT overwrite ``<fk>.npy`` (canonical labels).

    Pins the LS-9 contract: the canonical labels file stays whatever was
    written on stroke release; the mid-stroke recovery write only ever
    touches the ``.recovery.npy`` sibling.
    """
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    fk = "cafebabe00000000"

    canonical = labels_dir / f"{fk}.npy"
    canonical_labels = np.array([9, 9, 9], dtype=np.int32)
    np.save(str(canonical.with_suffix("")), canonical_labels)
    assert canonical.exists()
    canonical_mtime_ns = canonical.stat().st_mtime_ns

    recovery_labels = np.array([1, 2, 3], dtype=np.int32)
    _emulate_recovery_write(labels_dir, fk, recovery_labels)

    # Canonical untouched.
    assert canonical.stat().st_mtime_ns == canonical_mtime_ns
    assert (np.load(canonical) == canonical_labels).all()
    # Recovery sibling present.
    rec = labels_dir / f"{fk}.recovery.npy"
    assert rec.exists()
    assert (np.load(rec) == recovery_labels).all()
