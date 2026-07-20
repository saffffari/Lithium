"""Phase 10 verification: 4D label propagation."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.core.propagation import propagate_labels, estimate_radius


def test_estimate_radius():
    # Grid of points spaced 1.0 apart
    n = 20
    x = np.linspace(0, n - 1, n)
    xx, yy, zz = np.meshgrid(x, x, x, indexing='ij')
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    radius = estimate_radius(positions, factor=2.0)
    # Expected: nearest neighbor distance is ~1.0, factor 2 -> radius ~2.0
    assert 1.5 <= radius <= 3.0, f"expected ~2.0, got {radius}"
    print(f"PASS: Estimate radius ({radius:.3f})")


def test_propagate_basic():
    """Source has labels 1 and 2 in separated regions. Target should inherit."""
    rng = np.random.default_rng(42)
    # Source: 100 points at y=0 (label 1) and 100 points at y=10 (label 2)
    source_pos = np.concatenate([
        np.column_stack([rng.random(100), np.zeros(100), rng.random(100)]),
        np.column_stack([rng.random(100), np.full(100, 10.0), rng.random(100)]),
    ]).astype(np.float32)
    source_labels = np.concatenate([
        np.ones(100, dtype=np.int32),
        np.full(100, 2, dtype=np.int32),
    ])

    # Target: mirror positions with tiny noise
    target_pos = source_pos + rng.normal(0, 0.05, source_pos.shape).astype(np.float32)

    propagated, confidences = propagate_labels(
        source_pos, source_labels, target_pos, radius=0.5, k=5,
    )

    # First 100 should be mostly label 1, next 100 mostly label 2
    correct_1 = (propagated[0:100] == 1).sum()
    correct_2 = (propagated[100:200] == 2).sum()
    assert correct_1 >= 95, f"expected ~100 label 1, got {correct_1}"
    assert correct_2 >= 95, f"expected ~100 label 2, got {correct_2}"

    # Confidences should all be positive for labeled points
    labeled_mask = propagated != 0
    assert (confidences[labeled_mask] > 0).all()

    mean_conf = confidences[labeled_mask].mean()
    print(f"PASS: Basic propagation (acc: {(correct_1 + correct_2) / 2}%, conf: {mean_conf:.2f})")


def test_propagate_out_of_range():
    """Target points far from any source should remain unlabeled."""
    source_pos = np.array([[0, 0, 0]], dtype=np.float32)
    source_labels = np.array([5], dtype=np.int32)

    target_pos = np.array([
        [0.1, 0, 0],   # close
        [100, 100, 100],  # far
    ], dtype=np.float32)

    propagated, confidences = propagate_labels(
        source_pos, source_labels, target_pos, radius=1.0, k=5,
    )

    assert propagated[0] == 5, f"close point should be labeled 5, got {propagated[0]}"
    assert propagated[1] == 0, f"far point should be unlabeled, got {propagated[1]}"
    assert confidences[0] > 0
    assert confidences[1] == 0
    print("PASS: Out-of-range points stay unlabeled")


def test_propagate_weighted_majority():
    """Multiple neighbors with different labels — weighted majority wins."""
    # Target point surrounded by 3 'label 1' points close and 1 'label 2' point far
    source_pos = np.array([
        [0.1, 0, 0],
        [0, 0.1, 0],
        [0, 0, 0.1],
        [0.8, 0.8, 0.8],
    ], dtype=np.float32)
    source_labels = np.array([1, 1, 1, 2], dtype=np.int32)
    target_pos = np.array([[0, 0, 0]], dtype=np.float32)

    propagated, _ = propagate_labels(
        source_pos, source_labels, target_pos, radius=2.0, k=5,
    )
    assert propagated[0] == 1, f"majority should win, got {propagated[0]}"
    print("PASS: Weighted majority vote")


def test_propagate_confidence_scale():
    """Points near source should have high confidence, points near radius edge low."""
    source_pos = np.array([[0, 0, 0]], dtype=np.float32)
    source_labels = np.array([1], dtype=np.int32)
    target_pos = np.array([
        [0.01, 0, 0],  # very close
        [0.9, 0, 0],   # near radius edge
    ], dtype=np.float32)

    _, conf = propagate_labels(
        source_pos, source_labels, target_pos, radius=1.0, k=1,
    )
    assert conf[0] > 0.95, f"near point should have high conf, got {conf[0]}"
    assert conf[1] < 0.2, f"far point should have low conf, got {conf[1]}"
    print(f"PASS: Confidence scaling (near={conf[0]:.2f}, far={conf[1]:.2f})")


def test_vectorized_matches_loop():
    """Vectorized propagate_labels must agree with a hand-rolled loop baseline."""
    rng = np.random.default_rng(123)
    n_source = 5000
    n_target = 5000
    source_pos = rng.random((n_source, 3)).astype(np.float32) * 10
    source_labels = rng.integers(1, 8, size=n_source).astype(np.int32)
    # Sprinkle 20% unlabeled
    source_labels[rng.random(n_source) < 0.2] = 0
    target_pos = rng.random((n_target, 3)).astype(np.float32) * 10

    from scipy.spatial import cKDTree
    labeled_mask = source_labels != 0
    labeled_pos = source_pos[labeled_mask]
    labeled_lab = source_labels[labeled_mask]
    tree = cKDTree(labeled_pos)
    dists, idx = tree.query(target_pos, k=5, distance_upper_bound=0.7)
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)
    valid = np.isfinite(dists)
    expected = np.zeros(n_target, dtype=np.int32)
    for i in np.nonzero(valid.any(axis=1))[0]:
        mask_i = valid[i]
        nd = dists[i, mask_i]
        nl = labeled_lab[idx[i, mask_i]]
        w = 1.0 / (nd + 1e-8)
        uniq, inv = np.unique(nl, return_inverse=True)
        scores = np.zeros(len(uniq))
        for j, wj in zip(inv, w):
            scores[j] += wj
        expected[i] = uniq[int(np.argmax(scores))]

    propagated, _ = propagate_labels(
        source_pos, source_labels, target_pos, radius=0.7, k=5,
    )
    mismatches = int((propagated != expected).sum())
    assert mismatches == 0, f"vectorized diverged in {mismatches} positions"
    print(f"PASS: Vectorized equivalence ({n_target} pts, 0 mismatches)")


if __name__ == '__main__':
    test_estimate_radius()
    test_propagate_basic()
    test_propagate_out_of_range()
    test_propagate_weighted_majority()
    test_propagate_confidence_scale()
    test_vectorized_matches_loop()
    print("\n=== Phase 10 ALL TESTS PASSED ===")
