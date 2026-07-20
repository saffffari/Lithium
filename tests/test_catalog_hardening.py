"""Catalog hardening: wipe, integrity, write verification, lock file."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pathlib
import tarfile
import tempfile

import numpy as np

from src.data import library_catalog, cloud_store, catalog_lock, library_paths
from src.data.point_cloud import PointCloudData


def _with_temp_library(func):
    """Run func with library_paths.library_dir() patched to a tempdir.

    Provides an isolated catalog root + a sibling backups dir so wipe
    tests don't pollute the real ~/.3photon. Patches the canonical
    library_paths.library_dir; cloud_store + catalog_lock + library_catalog
    all dispatch through there.
    """
    def wrapper():
        orig_factory = library_paths.library_dir
        with tempfile.TemporaryDirectory() as parent:
            lib = os.path.join(parent, "library")
            os.makedirs(lib, exist_ok=True)
            library_paths.library_dir = lambda: lib
            try:
                func()
            finally:
                library_paths.library_dir = orig_factory
    wrapper.__name__ = func.__name__
    return wrapper


def _make_cloud(n=200):
    rng = np.random.default_rng(42)
    return PointCloudData(
        positions=rng.random((n, 3)).astype(np.float32),
        colors=rng.random((n, 3)).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# wipe_catalog
# ---------------------------------------------------------------------------

@_with_temp_library
def test_wipe_catalog_backs_up_and_clears():
    # Populate fake catalog
    cloud = _make_cloud(100)
    cloud_store.save_cloud_data("alpha", cloud, source_path="/tmp/a.ply")
    cloud_store.save_cloud_data("beta", cloud, source_path="/tmp/b.ply")
    cloud_store.save_cloud_labels("alpha", np.ones(100, dtype=np.int32))

    # Sanity-check pre-state
    assert cloud_store.has_cloud_data("alpha")
    assert cloud_store.has_cloud_data("beta")
    assert cloud_store.has_cloud_labels("alpha")

    backup_path = cloud_store.wipe_catalog(backup=True)
    assert backup_path is not None
    assert backup_path.exists()
    assert backup_path.suffix == ".gz"

    # Backup is restorable (can list contents without error)
    with tarfile.open(backup_path, "r:gz") as tar:
        names = tar.getnames()
        assert any(n.endswith("alpha.npz") for n in names)
        assert any(n.endswith("beta.npz") for n in names)

    # Catalog is empty after wipe
    assert not cloud_store.has_cloud_data("alpha")
    assert not cloud_store.has_cloud_data("beta")
    assert not cloud_store.has_cloud_labels("alpha")
    # Empty subdirs were recreated
    lib = pathlib.Path(library_catalog.library_dir())
    assert (lib / "data").is_dir()
    assert (lib / "labels").is_dir()
    assert (lib / "previews").is_dir()
    print("PASS: wipe_catalog backs up + clears + recreates skeleton")


@_with_temp_library
def test_wipe_catalog_no_backup_skips_tarball():
    cloud_store.save_cloud_data("solo", _make_cloud(50))
    backup_path = cloud_store.wipe_catalog(backup=False)
    assert backup_path is None
    # Wipe still clears
    assert not cloud_store.has_cloud_data("solo")
    print("PASS: wipe_catalog(backup=False) skips tarball")


# ---------------------------------------------------------------------------
# check_catalog_integrity
# ---------------------------------------------------------------------------

@_with_temp_library
def test_check_integrity_reports_orphans():
    # Save a data file but don't add it to any index — that's an orphan.
    cloud_store.save_cloud_data("ghost", _make_cloud(20))
    status = cloud_store.check_catalog_integrity(catalog=None)
    assert "ghost" in status["orphans"]
    assert status["entries_count"] == 0  # no index entries
    print("PASS: check_catalog_integrity reports orphans")


@_with_temp_library
def test_check_integrity_reports_missing_data():
    # Fake an index.json entry pointing at a file_key with no data file.
    lib = pathlib.Path(library_catalog.library_dir())
    index = lib / "index.json"
    index.write_text(json.dumps({
        "phantom": {
            "file_key": "phantom",
            "file_path": "/tmp/phantom.ply",
            "point_count": 100,
            "bounds_min": [0, 0, 0],
            "bounds_max": [1, 1, 1],
            "import_date": 0,
            "last_opened": 0,
        }
    }))
    status = cloud_store.check_catalog_integrity(catalog=None)
    assert status["entries_count"] == 1
    assert "phantom" in status["missing_data"]
    print("PASS: check_catalog_integrity reports missing data files")


@_with_temp_library
def test_check_integrity_clean_state():
    cloud = _make_cloud(50)
    cloud_store.save_cloud_data("alpha", cloud)
    cloud_store.save_cloud_labels("alpha", np.zeros(50, dtype=np.int32))
    lib = pathlib.Path(library_catalog.library_dir())
    (lib / "index.json").write_text(json.dumps({
        "alpha": {
            "file_key": "alpha", "file_path": "/tmp/a.ply",
            "point_count": 50, "bounds_min": [0, 0, 0],
            "bounds_max": [1, 1, 1],
            "import_date": 0, "last_opened": 0,
        }
    }))
    status = cloud_store.check_catalog_integrity(catalog=None)
    assert status["entries_count"] == 1
    assert status["with_data"] == 1
    assert status["with_labels"] == 1
    assert status["orphans"] == []
    assert status["missing_data"] == []
    print("PASS: check_catalog_integrity clean state")


# ---------------------------------------------------------------------------
# catalog_lock
# ---------------------------------------------------------------------------

@_with_temp_library
def test_acquire_lock_succeeds_on_clean_state():
    ok, blocking = catalog_lock.acquire_lock()
    assert ok is True
    assert blocking is None
    assert catalog_lock.lock_path().exists()
    # Released cleanly
    catalog_lock.release_lock()
    assert not catalog_lock.lock_path().exists()
    print("PASS: lock acquire/release on clean state")


@_with_temp_library
def test_acquire_lock_refuses_when_pid_alive():
    # Write a lock file with our own PID — that's a "live" PID by definition.
    path = catalog_lock.lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": os.getpid(), "started": 0.0}
    with open(path, "w") as f:
        json.dump(payload, f)
    ok, blocking = catalog_lock.acquire_lock()
    assert ok is False
    assert blocking == os.getpid()
    print("PASS: lock refuses when PID is alive")


@_with_temp_library
def test_acquire_lock_clears_stale_lock():
    # Write a lock file with a PID that almost certainly doesn't exist.
    path = catalog_lock.lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"pid": 9999999, "started": 0.0}
    with open(path, "w") as f:
        json.dump(payload, f)
    ok, blocking = catalog_lock.acquire_lock()
    assert ok is True
    assert blocking is None
    # File should now contain our PID
    new_payload = json.loads(path.read_text())
    assert new_payload["pid"] == os.getpid()
    catalog_lock.release_lock()
    print("PASS: stale lock is cleared and re-acquired")


@_with_temp_library
def test_acquire_lock_handles_corrupt_file():
    path = catalog_lock.lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json")
    ok, blocking = catalog_lock.acquire_lock()
    # Corrupt file → blocking_pid parsed as 0 → not alive → acquire
    assert ok is True
    assert blocking is None
    catalog_lock.release_lock()
    print("PASS: corrupt lock file is treated as stale")


# ---------------------------------------------------------------------------
# write verification
# ---------------------------------------------------------------------------

@_with_temp_library
def test_save_cloud_data_verify_catches_zero_byte_file():
    """If os.replace lands a 0-byte file, _verify_write should warn."""
    cloud = _make_cloud(50)
    # Monkey-patch np.savez_compressed to write a 0-byte file
    real_savez = np.savez_compressed

    def fake_savez(path, **kwargs):
        # Honor the path arg numpy uses, append .npz
        out = path if str(path).endswith(".npz") else (str(path) + ".npz")
        with open(out, "wb") as f:
            pass  # zero bytes

    np.savez_compressed = fake_savez
    try:
        # Capture stdout to verify the warning is printed
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cloud_store.save_cloud_data("zerobyte", cloud)
        # The save itself doesn't raise — it just prints the verify warning.
        out = buf.getvalue()
        assert "verify failed" in out.lower(), f"expected verify warning, got: {out!r}"
    finally:
        np.savez_compressed = real_savez
    print("PASS: zero-byte write triggers verify warning")


# ---------------------------------------------------------------------------
# _migrate_meta scaffold
# ---------------------------------------------------------------------------

def test_migrate_meta_v1_passthrough():
    meta = {"format_version": 1, "source_path": "x", "scalar_keys": []}
    out = cloud_store._migrate_meta(meta)
    assert out is meta or out == meta
    assert out["format_version"] == 1
    print("PASS: _migrate_meta is a no-op for v1")


def test_migrate_meta_handles_missing_version():
    meta = {"source_path": "y"}
    out = cloud_store._migrate_meta(meta)
    assert out["format_version"] == cloud_store.FORMAT_VERSION_CURRENT
    print("PASS: _migrate_meta stamps missing version")


def test_migrate_meta_handles_garbage_input():
    out = cloud_store._migrate_meta("not a dict")
    assert isinstance(out, dict)
    assert out["format_version"] == cloud_store.FORMAT_VERSION_CURRENT
    print("PASS: _migrate_meta handles non-dict input")


if __name__ == '__main__':
    test_wipe_catalog_backs_up_and_clears()
    test_wipe_catalog_no_backup_skips_tarball()
    test_check_integrity_reports_orphans()
    test_check_integrity_reports_missing_data()
    test_check_integrity_clean_state()
    test_acquire_lock_succeeds_on_clean_state()
    test_acquire_lock_refuses_when_pid_alive()
    test_acquire_lock_clears_stale_lock()
    test_acquire_lock_handles_corrupt_file()
    test_save_cloud_data_verify_catches_zero_byte_file()
    test_migrate_meta_v1_passthrough()
    test_migrate_meta_handles_missing_version()
    test_migrate_meta_handles_garbage_input()
    print("\nAll catalog hardening tests passed.")
