"""cloud_store verification — catalog-side full-cloud + labels persistence."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pathlib
import tempfile
import time

import numpy as np

from src.data.point_cloud import PointCloudData
from src.data import cloud_store
from src.data import library_paths


def _with_temp_library(func):
    """Run func with library_paths.library_dir() patched to a tempdir."""
    def wrapper():
        orig_factory = library_paths.library_dir
        with tempfile.TemporaryDirectory() as tmp:
            library_paths.library_dir = lambda: tmp
            try:
                func()
            finally:
                library_paths.library_dir = orig_factory
    wrapper.__name__ = func.__name__
    return wrapper


def _make_cloud(n=1000, with_scalars=True):
    rng = np.random.default_rng(42)
    pos = rng.random((n, 3)).astype(np.float32) * 100.0
    col = rng.random((n, 3)).astype(np.float32)
    scalars = {}
    if with_scalars:
        scalars['intensity'] = rng.random(n).astype(np.float32)
        scalars['height'] = pos[:, 2].copy()
    return PointCloudData(positions=pos, colors=col, scalars=scalars)


@_with_temp_library
def test_data_roundtrip_preserves_positions_colors_scalars():
    cloud = _make_cloud(2000)
    key = "deadbeef"
    cloud_store.save_cloud_data(key, cloud, source_path="/fake/source.ply")
    assert cloud_store.has_cloud_data(key)

    result = cloud_store.load_cloud_data(key)
    assert result is not None
    loaded, meta = result

    assert loaded.point_count == cloud.point_count
    assert np.allclose(loaded.positions, cloud.positions)
    assert np.allclose(loaded.colors, cloud.colors)
    assert set(loaded.scalars.keys()) == set(cloud.scalars.keys())
    for k in cloud.scalars:
        assert np.allclose(loaded.scalars[k], cloud.scalars[k]), \
            f"scalar '{k}' mismatch"
    assert meta.get("source_path") == "/fake/source.ply"
    assert meta.get("point_count") == 2000
    print("PASS: data round-trip preserves positions, colors, scalars, metadata")


@_with_temp_library
def test_labels_roundtrip():
    n = 500
    labels = np.zeros(n, dtype=np.int32)
    labels[100:200] = 1
    labels[200:300] = 2
    labels[300:400] = 3
    key = "labelkey"
    cloud_store.save_cloud_labels(key, labels)
    assert cloud_store.has_cloud_labels(key)
    loaded = cloud_store.load_cloud_labels(key)
    assert loaded is not None
    assert loaded.dtype == np.int32
    assert (loaded == labels).all()
    print("PASS: labels round-trip preserves int32 contents")


@_with_temp_library
def test_missing_files_return_none():
    assert cloud_store.load_cloud_data("nope") is None
    assert cloud_store.load_cloud_labels("nope") is None
    print("PASS: missing data and labels files return None")


@_with_temp_library
def test_atomic_save_cleans_up_tmp():
    cloud = _make_cloud(100)
    key = "atomic"
    cloud_store.save_cloud_data(key, cloud)
    cloud_store.save_cloud_labels(key, np.ones(100, dtype=np.int32))
    data_dir = cloud_store._data_dir()
    labels_dir = cloud_store._labels_dir()
    data_files = sorted(p.name for p in data_dir.iterdir())
    labels_files = sorted(p.name for p in labels_dir.iterdir())
    assert "atomic.npz" in data_files
    assert "atomic.npy" in labels_files
    assert not any(name.endswith(".tmp") for name in data_files + labels_files)
    print("PASS: atomic save leaves no .tmp leftovers")


@_with_temp_library
def test_is_source_stale_detects_newer_mtime():
    # Create a fake source file
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        f.write(b"placeholder")
        src_path = f.name
    try:
        # Capture the file's mtime when we "import" it
        cloud = _make_cloud(50)
        key = "stale"
        cloud_store.save_cloud_data(key, cloud, source_path=src_path)
        result = cloud_store.load_cloud_data(key)
        assert result is not None
        _, meta = result
        # Right after save the source is NOT stale
        assert not cloud_store.is_source_stale(meta, src_path)
        # Bump the source file's mtime forward
        future = time.time() + 60.0
        os.utime(src_path, (future, future))
        assert cloud_store.is_source_stale(meta, src_path)
        print("PASS: is_source_stale detects newer source mtime")
    finally:
        try:
            os.unlink(src_path)
        except OSError:
            pass


@_with_temp_library
def test_is_source_stale_handles_missing_metadata():
    assert cloud_store.is_source_stale({}, "/fake/path") is False
    assert cloud_store.is_source_stale({"source_mtime": 1.0}, None) is False
    assert cloud_store.is_source_stale({"source_mtime": 1.0}, "/no/such/file") is False
    print("PASS: is_source_stale safely handles missing metadata / paths")


@_with_temp_library
def test_drop_cloud_removes_both_files():
    cloud = _make_cloud(100)
    key = "dropme"
    cloud_store.save_cloud_data(key, cloud)
    cloud_store.save_cloud_labels(key, np.zeros(100, dtype=np.int32))
    assert cloud_store.has_cloud_data(key)
    assert cloud_store.has_cloud_labels(key)
    cloud_store.drop_cloud(key)
    assert not cloud_store.has_cloud_data(key)
    assert not cloud_store.has_cloud_labels(key)
    print("PASS: drop_cloud removes data and labels")


@_with_temp_library
def test_save_with_empty_cloud_is_noop():
    empty_pos = np.zeros((0, 3), dtype=np.float32)
    empty_col = np.zeros((0, 3), dtype=np.float32)
    cloud = PointCloudData(positions=empty_pos, colors=empty_col)
    cloud_store.save_cloud_data("empty", cloud)
    # No file should have been written
    assert not cloud_store.has_cloud_data("empty")
    print("PASS: empty cloud save is a no-op")


@_with_temp_library
def test_save_with_empty_key_is_noop():
    cloud = _make_cloud(10)
    cloud_store.save_cloud_data("", cloud)
    cloud_store.save_cloud_labels("", np.zeros(10, dtype=np.int32))
    # Nothing should land — empty file_key means session-only entry.
    print("PASS: empty file_key is a silent no-op")


if __name__ == '__main__':
    test_data_roundtrip_preserves_positions_colors_scalars()
    test_labels_roundtrip()
    test_missing_files_return_none()
    test_atomic_save_cleans_up_tmp()
    test_is_source_stale_detects_newer_mtime()
    test_is_source_stale_handles_missing_metadata()
    test_drop_cloud_removes_both_files()
    test_save_with_empty_cloud_is_noop()
    test_save_with_empty_key_is_noop()
    print("\nAll cloud_store tests passed.")
