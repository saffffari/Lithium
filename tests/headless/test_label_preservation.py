"""Label-preservation regression tests.

Each test pins down one slice of the label-safety contract. New
regression tests land here as Phase 9 / Phase 10 fixes ship — keep this
file the single source of truth for what Lithium promises about label
state across sessions, project switches, and overwrites.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from src.data import cloud_store
from src.data.labels_io import apply_label_array_to_cloud

from tests.headless.app_fixture import HeadlessApp
from tests.headless.helpers import (
    count_label_files_on_disk,
    labels_npy_path,
    make_companion_label_npy,
    make_tiny_ply,
)


# ---------------------------------------------------------------------------
# Round-trip + idempotence
# ---------------------------------------------------------------------------

def test_paint_save_reload_preserves_labels(tmp_library, tiny_ply):
    """Painted labels survive a HeadlessApp close + new HeadlessApp open."""
    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    assert entry is not None
    indices = np.array([0, 5, 10, 99])
    app.paint_labels(indices, 3)
    app.save_labels()
    app.close()

    app2 = HeadlessApp()
    entry2 = app2.load_file(tiny_ply)
    labels = app2.get_labels()
    assert labels is not None
    assert (labels[indices] == 3).all()
    # Every other index stays unlabeled.
    mask = np.ones(entry2.point_count, dtype=bool)
    mask[indices] = False
    assert (labels[mask] == 0).all()
    app2.close()


def test_double_save_is_idempotent(tmp_library, tiny_ply):
    """Calling save_labels twice with no changes leaves disk state stable."""
    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    app.paint_labels(np.arange(10), 2)
    app.save_labels()
    path = labels_npy_path(tmp_library, entry.file_key)
    first_mtime_ns = path.stat().st_mtime_ns
    first_size = path.stat().st_size
    # Second save with identical labels — content shouldn't change.
    app.save_labels()
    assert path.stat().st_size == first_size
    # mtime may bump (atomic replace) — content is what matters.
    arr = np.load(path)
    assert (arr[:10] == 2).all()
    assert (arr[10:] == 0).all()
    _ = first_mtime_ns  # explicit: we don't pin mtime — replace updates it
    app.close()


def test_load_returns_none_for_unknown_key(tmp_library):
    """Catalog labels load gracefully returns None for missing file_keys."""
    assert cloud_store.load_cloud_labels("nonexistent_key_abc123") is None


# ---------------------------------------------------------------------------
# Length-mismatch guards
# ---------------------------------------------------------------------------

def test_apply_label_array_rejects_wrong_length(tmp_library, tiny_ply):
    """apply_label_array_to_cloud raises ValueError on length mismatch.

    Persistence layer's contract — feeding a wrong-length array must
    fail loudly, never silently truncate or broadcast. LS-2's fix tightens
    behaviour ON match; the rejection on mismatch is already in place
    and is regression-pinned here so future edits don't relax it.
    """
    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    assert entry is not None
    bogus = np.zeros(entry.point_count + 5, dtype=np.int32)
    bogus_path = os.path.splitext(tiny_ply)[0] + "_bogus_labels.npy"
    np.save(bogus_path, bogus)
    with pytest.raises(ValueError, match="length"):
        apply_label_array_to_cloud(entry.cloud, bogus_path, app.label_registry)
    app.close()


def test_persist_skips_wrong_length_array(tmp_library, tiny_ply, capsys):
    """HeadlessApp._persist_entry skips if labels length != point_count.

    Mirror of src/main.py:_persist_cloud_labels' guard so the harness
    catches regressions in the same wrong-length-write scenario without
    requiring a full App boot. The save must NOT touch the labels file.
    """
    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    # Force a wrong-length array, simulating a stale preview-resolution
    # labels object leaking into the persist call.
    entry.cloud.labels = np.ones(entry.point_count - 1, dtype=np.int32)
    app.save_labels()
    path = labels_npy_path(tmp_library, entry.file_key)
    assert not path.exists(), (
        "wrong-length labels must not land in the catalog labels store"
    )
    captured = capsys.readouterr()
    assert "labels length" in captured.out
    app.close()


# ---------------------------------------------------------------------------
# Companion .npy auto-apply
# ---------------------------------------------------------------------------

def test_companion_label_array_applies(tmp_library, tiny_ply):
    """A ``{stem}_labels.npy`` next to the source is picked up on import."""
    arr = np.zeros(100, dtype=np.int32)
    arr[10:40] = 5
    arr[40:70] = 9
    make_companion_label_npy(tiny_ply, arr)

    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    assert entry is not None
    n = app.apply_companion_label_array(entry)
    assert n == 60, "60 points are non-zero in the companion array"

    labels = app.get_labels()
    assert (labels[10:40] == 5).all() or (labels[10:40] != 0).all()
    # Companion application uses the registry's id allocation; the
    # exact id used for a source-value of N may differ from N when the
    # registry already has labels under those slots. What's guaranteed
    # is: every non-zero source position ends up non-zero in the cloud.
    assert (labels[10:70] != 0).all()
    assert (labels[:10] == 0).all()
    assert (labels[70:] == 0).all()

    # Persistence after auto-apply: the catalog labels file must exist.
    assert labels_npy_path(tmp_library, entry.file_key).exists()
    app.close()


# ---------------------------------------------------------------------------
# Cross-session catalog state
# ---------------------------------------------------------------------------

def test_two_sessions_see_same_file_key(tmp_library, tiny_ply):
    """The catalog's file_key is stable for the same source file."""
    app1 = HeadlessApp()
    entry1 = app1.load_file(tiny_ply)
    fk1 = entry1.file_key
    app1.close()

    app2 = HeadlessApp()
    entry2 = app2.load_file(tiny_ply)
    fk2 = entry2.file_key
    app2.close()

    assert fk1 == fk2, "file_key must be deterministic across sessions"


def test_count_label_files_grows_on_save(tmp_library, tmp_path):
    """Each distinct cloud added gets its own labels/<file_key>.npy on save."""
    p1 = str(tmp_path / "a.ply")
    p2 = str(tmp_path / "b.ply")
    make_tiny_ply(p1, n=50, seed=1)
    make_tiny_ply(p2, n=50, seed=2)

    app = HeadlessApp()
    e1 = app.load_file(p1)
    e2 = app.load_file(p2)
    assert e1.file_key != e2.file_key

    app.paint_labels(np.array([0, 1, 2]), 1, entry=e1)
    app.save_labels(entry=e1)
    assert count_label_files_on_disk(tmp_library) == 1

    app.paint_labels(np.array([0, 1, 2]), 2, entry=e2)
    app.save_labels(entry=e2)
    assert count_label_files_on_disk(tmp_library) == 2

    app.close()


def test_empty_cloud_save_is_safe(tmp_library, tmp_path):
    """save_cloud_labels for a zero-length array is a no-op, no exception."""
    cloud_store.save_cloud_labels("anykey", np.zeros(0, dtype=np.int32))
    # The 0-byte guard short-circuits before any file is written.
    assert not (Path(tmp_library) / "labels" / "anykey.npy").exists()


def test_ls2_companion_merge_does_not_wipe_existing(tmp_library, tiny_ply):
    """LS-2: applying a companion .npy must NOT zero the cloud's pre-existing
    labels at indices the source array marks 0.

    Scenario: user has painted 100 points to label 7. A companion
    array arrives with non-zero values only at indices [40, 60).
    Old behaviour zeroed everything first, losing the 0-30 paint.
    New behaviour merges — points 0-40 keep label 7, 40-60 get the
    companion's id, 60+ keep label 7.
    """
    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    assert entry is not None

    # Paint the entire cloud to label 7.
    app.paint_labels(np.arange(entry.point_count), 7)
    app.save_labels()
    assert (app.get_labels() == 7).all()

    # Companion array — zero except for a contiguous block [40, 60).
    arr = np.zeros(entry.point_count, dtype=np.int32)
    arr[40:60] = 9
    make_companion_label_npy(tiny_ply, arr)

    n = app.apply_companion_label_array(entry)
    assert n > 0
    labels = app.get_labels()
    # The merged block carries the companion's label id (resolved
    # against the registry, but always non-zero).
    assert (labels[40:60] != 0).all()
    assert (labels[40:60] != 7).all(), (
        "companion's block should overwrite the painted 7s with its id"
    )
    # Everything OUTSIDE the companion's non-zero block keeps the 7.
    assert (labels[:40] == 7).all(), (
        "merge-not-zero must not wipe existing labels at indices the "
        "source array marks 0"
    )
    assert (labels[60:] == 7).all()
    app.close()
