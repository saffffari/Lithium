"""Phase 16: recursive directory import for hierarchical datasets (VerSe).

Covers ``scan_directory_recursive`` — the os.walk-based pipeline that
finds every supported point cloud file under a root. The app-side
``load_directory_recursive`` integration is not unit-testable without
a GL context; tested manually via the ``load-recursive`` CLI command.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import scan_directory, scan_directory_recursive


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"")


def test_scan_recursive_flat_dir():
    """Same behavior as scan_directory for a flat directory."""
    with tempfile.TemporaryDirectory() as tmp:
        _touch(os.path.join(tmp, "a.ply"))
        _touch(os.path.join(tmp, "b.ply"))
        _touch(os.path.join(tmp, "readme.txt"))
        found = scan_directory_recursive(tmp)
        assert len(found) == 2
        assert all(f.endswith(".ply") for f in found)
    print("PASS: scan_recursive flat dir")


def test_scan_recursive_nested_verse_layout():
    """Simulate verse_points/<split>/<subject>/<NNN.ply>."""
    with tempfile.TemporaryDirectory() as tmp:
        for split in ("01_training", "02_validation"):
            for subj in (f"sub-{i:03d}" for i in range(3)):
                for idx in range(1, 11):
                    _touch(os.path.join(tmp, split, subj, f"{idx:03d}.ply"))
        # A stray junk file that should be ignored
        _touch(os.path.join(tmp, "01_training", "README.md"))
        found = scan_directory_recursive(tmp)
        # 2 splits * 3 subjects * 10 files = 60
        assert len(found) == 60, f"expected 60, got {len(found)}"
        assert all(f.endswith(".ply") for f in found)
        # Stable sort
        assert found == sorted(found)
    print("PASS: scan_recursive nested verse layout")


def test_scan_recursive_skips_hidden_and_pycache():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(os.path.join(tmp, "data", "good.ply"))
        _touch(os.path.join(tmp, ".hidden", "bad.ply"))
        _touch(os.path.join(tmp, "__pycache__", "bad.ply"))
        _touch(os.path.join(tmp, ".git", "objects", "bad.ply"))
        found = scan_directory_recursive(tmp)
        assert len(found) == 1
        assert found[0].endswith("good.ply")
    print("PASS: scan_recursive skips hidden + pycache")


def test_scan_recursive_mixed_extensions():
    with tempfile.TemporaryDirectory() as tmp:
        _touch(os.path.join(tmp, "a", "1.ply"))
        _touch(os.path.join(tmp, "a", "2.las"))
        _touch(os.path.join(tmp, "a", "3.laz"))
        _touch(os.path.join(tmp, "a", "4.npz"))
        _touch(os.path.join(tmp, "a", "5.txt"))
        # .tif / .nii.gz medical-volume loaders left with the SpineLab cut;
        # the research build supports only point-cloud extensions now.
        _touch(os.path.join(tmp, "b", "6.tif"))
        _touch(os.path.join(tmp, "b", "7.nii.gz"))
        found = scan_directory_recursive(tmp)
        # Only .ply/.las/.laz/.npz are supported; .txt/.tif/.nii.gz ignored.
        assert len(found) == 4, f"expected 4, got {len(found)}: {found}"
        assert all(f.endswith((".ply", ".las", ".laz", ".npz")) for f in found)
    print("PASS: scan_recursive mixed extensions")


def test_scan_recursive_empty_dir():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "sub"))
        found = scan_directory_recursive(tmp)
        assert found == []
    print("PASS: scan_recursive empty dir")


def test_scan_recursive_max_files_cap():
    """Cap should prevent runaway walks into giant trees."""
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(20):
            _touch(os.path.join(tmp, f"{i:03d}.ply"))
        found = scan_directory_recursive(tmp, max_files=5)
        assert len(found) == 5
    print("PASS: scan_recursive max_files cap")


def test_scan_recursive_preserves_scan_directory_semantics_at_leaf():
    """At a leaf, scan_directory_recursive should find the same files
    scan_directory would find (modulo nested subdirs with their own
    files). This is a sanity check that the new function doesn't
    silently disagree with the flat one on basic layouts.
    """
    with tempfile.TemporaryDirectory() as tmp:
        flat = os.path.join(tmp, "leaf")
        os.makedirs(flat)
        for n in ("a.ply", "b.ply", "c.las"):
            _touch(os.path.join(flat, n))
        flat_files = set(scan_directory(flat))
        rec_files = set(scan_directory_recursive(flat))
        assert flat_files == rec_files
    print("PASS: scan_recursive matches scan_directory at leaf")


if __name__ == "__main__":
    test_scan_recursive_flat_dir()
    test_scan_recursive_nested_verse_layout()
    test_scan_recursive_skips_hidden_and_pycache()
    test_scan_recursive_mixed_extensions()
    test_scan_recursive_empty_dir()
    test_scan_recursive_max_files_cap()
    test_scan_recursive_preserves_scan_directory_semantics_at_leaf()
    print("\n=== Phase 16 ALL TESTS PASSED ===")
