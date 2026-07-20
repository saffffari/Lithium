"""Phase 9 verification: 4D timeline and sequence caching."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plyfile import PlyData, PlyElement

from src.data.sequence import PointCloudSequence, detect_sequence, _numeric_sort_key


def make_numbered_ply(path: str, n_points: int = 100, seed: int = 0):
    """Create a minimal PLY file with random points."""
    rng = np.random.default_rng(seed)
    positions = rng.random((n_points, 3)).astype(np.float32)
    vertex = np.array(
        [(p[0], p[1], p[2]) for p in positions],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')],
    )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(path)


def test_numeric_sort():
    keys = [
        _numeric_sort_key('frame_10.ply'),
        _numeric_sort_key('frame_2.ply'),
        _numeric_sort_key('frame_1.ply'),
    ]
    sorted_keys = sorted(keys)
    # First should be frame_1, then frame_2, then frame_10
    assert sorted_keys[0][1] == 1
    assert sorted_keys[1][1] == 2
    assert sorted_keys[2][1] == 10
    print("PASS: Numeric sort handles multi-digit")


def test_detect_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(5):
            make_numbered_ply(os.path.join(tmp, f'frame_{i:03d}.ply'), seed=i)
        files = detect_sequence(tmp)
        assert files is not None
        assert len(files) == 5
        # Sorted order
        for i in range(4):
            assert 'frame_00' in files[i] or 'frame_0' in files[i]
    print("PASS: Detect sequence")


def test_detect_no_sequence():
    with tempfile.TemporaryDirectory() as tmp:
        make_numbered_ply(os.path.join(tmp, 'cloud.ply'))
        make_numbered_ply(os.path.join(tmp, 'another.ply'))
        files = detect_sequence(tmp)
        # No numbers -> not a sequence
        assert files is None
    print("PASS: Non-sequence correctly rejected")


def test_sequence_lru_cache():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(5):
            p = os.path.join(tmp, f'frame_{i:03d}.ply')
            make_numbered_ply(p, seed=i)
            paths.append(p)

        seq = PointCloudSequence(paths, cache_size=3)

        # Load frames 0, 1, 2
        c0 = seq.get_frame(0)
        c1 = seq.get_frame(1)
        c2 = seq.get_frame(2)
        assert len(seq._cache) == 3

        # Load frame 3 -> should evict frame 0
        c3 = seq.get_frame(3)
        assert 0 not in seq._cache
        assert 3 in seq._cache

        # Access frame 1 -> should not evict it (LRU bump)
        _ = seq.get_frame(1)
        # Load frame 4 -> should evict frame 2 (oldest after bump)
        c4 = seq.get_frame(4)
        assert 2 not in seq._cache
        assert 1 in seq._cache
    print("PASS: LRU cache eviction")


def test_sequence_seek_step():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(5):
            p = os.path.join(tmp, f'frame_{i:03d}.ply')
            make_numbered_ply(p, seed=i)
            paths.append(p)

        seq = PointCloudSequence(paths)
        assert seq.current_index == 0

        seq.step(2)
        assert seq.current_index == 2

        seq.step(-1)
        assert seq.current_index == 1

        # Clamping
        seq.seek(100)
        assert seq.current_index == 4
        seq.seek(-100)
        assert seq.current_index == 0

        assert seq.frame_count == 5
    print("PASS: Seek and step")


def test_labels_persist_per_frame():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i in range(3):
            p = os.path.join(tmp, f'frame_{i:03d}.ply')
            make_numbered_ply(p, seed=i, n_points=50)
            paths.append(p)

        seq = PointCloudSequence(paths, cache_size=3)

        # Label frame 0
        c0 = seq.get_frame(0)
        c0.labels[0:20] = 7
        assert seq.has_labels(0)

        # Visit frame 1 (not labeled)
        c1 = seq.get_frame(1)
        assert not seq.has_labels(1)
        assert (c1.labels == 0).all()

        # Back to frame 0 - labels should still be there (in cache)
        c0_again = seq.get_frame(0)
        assert (c0_again.labels[0:20] == 7).all()
    print("PASS: Per-frame label persistence (cached)")


if __name__ == '__main__':
    test_numeric_sort()
    test_detect_sequence()
    test_detect_no_sequence()
    test_sequence_lru_cache()
    test_sequence_seek_step()
    test_labels_persist_per_frame()
    print("\n=== Phase 9 ALL TESTS PASSED ===")
