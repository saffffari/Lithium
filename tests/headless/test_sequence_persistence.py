"""LS-1 regression tests: 4D sequence frames persist labels on disk.

Pre-fix, sequence frames had no catalog file_key: per-stroke persist
no-oped and LRU eviction (cache_size=3) silently discarded painted
labels. These tests exercise the data-layer half of the fix — frame
registration via ``catalog.register_files``, eviction-time persist,
page-back-in reload, and the shutdown ``flush_labels`` path — without
a GL context, mirroring the tier-1 headless harness patterns.
"""

from __future__ import annotations

import numpy as np

from src.core.undo import UndoStack, apply_label
from src.data import cloud_store
from src.data.library_catalog import LibraryCatalog
from src.data.sequence import PointCloudSequence

from tests.headless import helpers


def _make_sequence(tmp_path, n_frames=5, n_points=100, cache_size=3):
    """Numbered PLY frames + a catalog-registered PointCloudSequence."""
    paths = []
    for i in range(n_frames):
        p = str(tmp_path / f"frame_{i:03d}.ply")
        helpers.make_tiny_ply(p, n=n_points, seed=i)
        paths.append(p)
    catalog = LibraryCatalog()
    seq = PointCloudSequence(paths, cache_size=cache_size)
    # Mirrors App._load_as_sequence: batch-register frames, adopt keys.
    seq.frame_keys = [
        (le.file_key if le is not None else None)
        for le in catalog.register_files(paths)
    ]
    return seq, catalog, paths


def _close(catalog: LibraryCatalog) -> None:
    try:
        catalog._executor.shutdown(wait=False, cancel_futures=True)
    except (AttributeError, RuntimeError):
        pass


def test_frame_keys_registered(tmp_library, tmp_path):
    seq, catalog, paths = _make_sequence(tmp_path)
    try:
        assert len(seq.frame_keys) == len(paths)
        assert all(k for k in seq.frame_keys)
        assert len(set(seq.frame_keys)) == len(paths)  # distinct frames
    finally:
        _close(catalog)


def test_labels_survive_lru_eviction(tmp_library, tmp_path):
    """Paint frame 0, seek away far enough to evict it, seek back —
    the labels must come back from the catalog."""
    seq, catalog, _ = _make_sequence(tmp_path, cache_size=3)
    try:
        cloud0 = seq.get_frame(0)
        # Label writes go through core.undo.apply_label (undoable path).
        apply_label(cloud0, np.arange(10, dtype=np.int32), 3, UndoStack(),
                    description="test paint")
        assert (cloud0.labels[:10] == 3).all()

        # Seek forward until frame 0 falls out of the 3-frame LRU.
        seq.seek(1)
        seq.seek(2)
        seq.seek(3)
        assert 0 not in seq._cache, "frame 0 should have been evicted"

        # Eviction must have persisted the paint to the catalog.
        stored = cloud_store.load_cloud_labels(seq.frame_keys[0])
        assert stored is not None
        assert (stored[:10] == 3).all()

        # Page frame 0 back in: labels reload from the catalog.
        back = seq.seek(0)
        assert back is not cloud0, "expected a fresh load after eviction"
        assert (back.labels[:10] == 3).all()
        assert (back.labels[10:] == 0).all()
        assert seq.has_labels(0)
    finally:
        _close(catalog)


def test_erase_all_does_not_resurrect_labels(tmp_library, tmp_path):
    """Erasing every label then evicting must persist the zeros —
    reload must not bring the old paint back."""
    seq, catalog, _ = _make_sequence(tmp_path, cache_size=3)
    try:
        cloud0 = seq.get_frame(0)
        stack = UndoStack()
        apply_label(cloud0, np.arange(10, dtype=np.int32), 7, stack,
                    description="paint")
        seq.seek(1)
        seq.seek(2)
        seq.seek(3)  # evict + persist frame 0

        cloud0b = seq.seek(0)
        assert (cloud0b.labels[:10] == 7).all()
        apply_label(cloud0b, np.arange(10, dtype=np.int32), 0, stack,
                    description="erase")
        assert not (cloud0b.labels != 0).any()
        seq.seek(1)
        seq.seek(2)
        seq.seek(3)  # evict again — must overwrite with zeros

        final = seq.seek(0)
        assert not (final.labels != 0).any(), \
            "erased labels came back from a stale catalog file"
    finally:
        _close(catalog)


def test_unpainted_frames_write_no_label_files(tmp_library, tmp_path):
    """Scrubbing through a never-painted sequence must not litter the
    catalog with all-zero label files."""
    seq, catalog, _ = _make_sequence(tmp_path, cache_size=3)
    try:
        for i in range(5):
            seq.seek(i)
        for key in seq.frame_keys:
            assert not cloud_store.has_cloud_labels(key)
    finally:
        _close(catalog)


def test_flush_labels_covers_cached_frames(tmp_library, tmp_path):
    """Shutdown path: flush_labels persists every cached frame, then a
    fresh sequence (new session) sees the paint."""
    seq, catalog, paths = _make_sequence(tmp_path, cache_size=3)
    try:
        stack = UndoStack()
        # Paint two different cached frames without ever evicting them.
        c1 = seq.seek(1)
        apply_label(c1, np.arange(5, dtype=np.int32), 2, stack,
                    description="paint f1")
        c2 = seq.seek(2)
        apply_label(c2, np.arange(5, 15, dtype=np.int32), 4, stack,
                    description="paint f2")

        seq.flush_labels()  # what App._cleanup calls

        # New session: fresh sequence + fresh catalog over the same dir.
        seq2 = PointCloudSequence(paths, cache_size=3)
        catalog2 = LibraryCatalog()
        seq2.frame_keys = [
            (le.file_key if le is not None else None)
            for le in catalog2.register_files(paths)
        ]
        try:
            f1 = seq2.get_frame(1)
            f2 = seq2.get_frame(2)
            assert (f1.labels[:5] == 2).all()
            assert (f2.labels[5:15] == 4).all()
            assert seq2.has_labels(1) and seq2.has_labels(2)
            assert not seq2.get_frame(0).labels.any()
        finally:
            _close(catalog2)
    finally:
        _close(catalog)


def test_persist_error_surfaces_via_callback(tmp_library, tmp_path, monkeypatch):
    """A failed eviction-time save must reach the app-visible callback
    (App wires it to set_status_banner), not just stdout."""
    seq, catalog, _ = _make_sequence(tmp_path, cache_size=3)
    messages: list[str] = []
    seq.persist_error_cb = messages.append
    try:
        cloud0 = seq.get_frame(0)
        apply_label(cloud0, np.arange(3, dtype=np.int32), 1, UndoStack(),
                    description="paint")
        monkeypatch.setattr(
            cloud_store, "save_cloud_labels",
            lambda *a, **k: "disk full (simulated)")
        seq.seek(1)
        seq.seek(2)
        seq.seek(3)  # evict frame 0 -> persist fails
        assert messages, "persist failure never reached the error callback"
        assert "disk full" in messages[0]
    finally:
        _close(catalog)
