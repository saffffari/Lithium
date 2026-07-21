"""LS-6 regression tests: paint on a preview during the preview->full-res
window must be projected onto the full cloud and persisted.

Pre-fix, the full-res merge only fired when preview/full lengths were
equal (they never are — previews are subsamples) and the persist path's
length guard rejected the preview-length array, so the stroke was
silently orphaned. These tests cover the projection helpers plus the
persist round-trip through cloud_store, headless (no GL / no App).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.preview_merge import (
    build_preview_to_full_idx,
    merge_preview_labels_into_full,
)
from src.data import cloud_store


def _full_and_preview(n_full=1000, stride=10, seed=7):
    """A synthetic full cloud + an exact-subset preview (mirrors the
    voxel-downsample previews, which are byte-exact position subsets)."""
    rng = np.random.default_rng(seed)
    full = rng.standard_normal((n_full, 3)).astype(np.float32) * 5.0
    subset_idx = np.arange(0, n_full, stride)
    preview = full[subset_idx].copy()
    return full, preview, subset_idx


def test_index_map_is_exact_for_subset_previews():
    full, preview, subset_idx = _full_and_preview()
    mapping = build_preview_to_full_idx(preview, full)
    assert mapping is not None
    assert mapping.shape == (len(preview),)
    np.testing.assert_array_equal(mapping, subset_idx)


def test_merge_projects_painted_labels_onto_full():
    full, preview, subset_idx = _full_and_preview()
    full_labels = np.zeros(len(full), dtype=np.int32)
    preview_labels = np.zeros(len(preview), dtype=np.int32)
    preview_labels[:20] = 5  # the "stroke" during the load window

    mapping = build_preview_to_full_idx(preview, full)
    merged = merge_preview_labels_into_full(
        full_labels, preview_labels, mapping)

    assert merged == 20
    np.testing.assert_array_equal(full_labels[subset_idx[:20]], 5)
    # Nothing outside the painted projection got touched.
    untouched = np.ones(len(full), dtype=bool)
    untouched[subset_idx[:20]] = False
    assert not full_labels[untouched].any()


def test_merge_never_erases_existing_full_labels():
    """Zero (unpainted) preview entries must not clobber full-res paint
    that was already in the catalog."""
    full, preview, subset_idx = _full_and_preview()
    full_labels = np.full(len(full), 9, dtype=np.int32)  # pre-existing paint
    preview_labels = np.zeros(len(preview), dtype=np.int32)
    preview_labels[3] = 2

    mapping = build_preview_to_full_idx(preview, full)
    merged = merge_preview_labels_into_full(
        full_labels, preview_labels, mapping)

    assert merged == 1
    assert full_labels[subset_idx[3]] == 2
    mask = np.ones(len(full), dtype=bool)
    mask[subset_idx[3]] = False
    assert (full_labels[mask] == 9).all()


@pytest.mark.parametrize(
    "mapping", [None, np.arange(3, dtype=np.int64)],
    ids=["map-missing", "map-wrong-length"])
def test_merge_degrades_to_noop_on_bad_map(mapping):
    """Fallback contract: an unusable map returns 0 (caller shows a
    status banner) and leaves the full labels untouched."""
    full_labels = np.zeros(100, dtype=np.int32)
    preview_labels = np.zeros(10, dtype=np.int32)
    preview_labels[0] = 4
    merged = merge_preview_labels_into_full(
        full_labels, preview_labels, mapping)
    assert merged == 0
    assert not full_labels.any()


def test_build_map_degrades_to_none_on_empty_input():
    assert build_preview_to_full_idx(
        np.empty((0, 3), np.float32), np.ones((5, 3), np.float32)) is None
    assert build_preview_to_full_idx(None, np.ones((5, 3), np.float32)) is None


def test_merged_labels_persist_and_reload(tmp_library):
    """End-to-end data path: merge preview paint into the full array,
    persist through cloud_store (the normal path — full-length array
    passes the guard), reload as a fresh session would."""
    full, preview, subset_idx = _full_and_preview()
    file_key = "abcd1234abcd1234"
    full_labels = np.zeros(len(full), dtype=np.int32)
    preview_labels = np.zeros(len(preview), dtype=np.int32)
    preview_labels[10:30] = 6

    mapping = build_preview_to_full_idx(preview, full)
    merged = merge_preview_labels_into_full(
        full_labels, preview_labels, mapping)
    assert merged == 20

    err = cloud_store.save_cloud_labels(file_key, full_labels)
    assert err is None

    stored = cloud_store.load_cloud_labels(file_key)
    assert stored is not None
    assert stored.shape[0] == len(full)  # full-length, not preview-length
    np.testing.assert_array_equal(stored, full_labels)
    np.testing.assert_array_equal(stored[subset_idx[10:30]], 6)
