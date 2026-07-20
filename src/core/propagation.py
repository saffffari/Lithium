"""4D label propagation from source frame to target frame.

Uses spatial proximity (KD-tree radius search) to transfer labels
from a labeled source to an unlabeled target. Returns propagated
labels and per-point confidences.
"""

import numpy as np
from scipy.spatial import cKDTree


def estimate_radius(positions: np.ndarray, sample_size: int = 10_000,
                    factor: float = 2.0) -> float:
    """Estimate a reasonable propagation radius from nearest-neighbor distances.

    Computes the median nearest-neighbor distance on a random sample and
    multiplies by factor (default 2x).
    """
    n = len(positions)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, min(sample_size, n), replace=False)
    sample = positions[sample_idx]
    tree = cKDTree(sample)
    # Query each sample point for its 2nd nearest (1st is itself)
    dists, _ = tree.query(sample, k=2)
    median_nn = float(np.median(dists[:, 1]))
    return median_nn * factor


_VECTORIZED_L_CAP = 64  # max label-id range for vectorized path


def propagate_labels(source_positions: np.ndarray,
                     source_labels: np.ndarray,
                     target_positions: np.ndarray,
                     radius: float,
                     k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Propagate labels from source to target via KD-tree proximity.

    For each target point, find up to k labeled source points within
    radius. Majority vote weighted by inverse distance wins. Confidence
    is computed from the nearest-neighbor distance relative to radius.

    Returns:
        propagated: (N_target,) int32 labels
        confidences: (N_target,) float32 in [0, 1]
    """
    n_target = len(target_positions)
    propagated = np.zeros(n_target, dtype=np.int32)
    confidences = np.zeros(n_target, dtype=np.float32)

    # Filter to only labeled source points
    labeled_mask = source_labels != 0
    if not labeled_mask.any():
        # Nothing to propagate from. Warn so the caller's undo entry
        # doesn't silently overwrite the target with all-unlabeled.
        import warnings
        warnings.warn(
            "propagate_labels: source frame has no labeled points — "
            "returning no-op; caller should skip the apply step.",
            RuntimeWarning,
        )
        return propagated, confidences

    labeled_positions = source_positions[labeled_mask]
    labeled_labels = source_labels[labeled_mask]

    tree = cKDTree(labeled_positions)
    # Query all target points for up to k neighbors within radius
    dists, idx = tree.query(
        target_positions,
        k=k,
        distance_upper_bound=radius,
    )

    # Ensure 2D shape even when k=1
    if dists.ndim == 1:
        dists = dists.reshape(-1, 1)
        idx = idx.reshape(-1, 1)

    valid = np.isfinite(dists)
    has_any = valid.any(axis=1)
    if not has_any.any():
        return propagated, confidences

    max_label = int(labeled_labels.max())

    if max_label < _VECTORIZED_L_CAP:
        # Vectorized scatter-vote path. Build an (N, L+1) score matrix where
        # votes[i, j] = sum of inverse distances from target i to labeled
        # sources carrying label j. argmax picks the winning label.
        L = max_label + 1
        # Pad invalid neighbors with label 0 + weight 0 so they don't affect
        # the sum. Clip indices so the gather never addresses out-of-bounds.
        safe_idx = np.where(valid, idx, 0)
        inv_d = np.where(valid, 1.0 / (dists + 1e-8), 0.0).astype(np.float32)
        neigh_labels = labeled_labels[safe_idx]  # (N, k) int32

        votes = np.zeros((n_target, L), dtype=np.float32)
        rows = np.repeat(np.arange(n_target), inv_d.shape[1])
        cols = neigh_labels.ravel().astype(np.intp)
        np.add.at(votes, (rows, cols), inv_d.ravel())
        votes[:, 0] = -np.inf  # never pick "unlabeled"

        # Only write winners where we actually had neighbors
        winners = votes.argmax(axis=1).astype(np.int32)
        propagated[has_any] = winners[has_any]

        min_dist = np.where(valid, dists, np.inf).min(axis=1)
        conf = 1.0 - (min_dist / max(radius, 1e-8))
        confidences[has_any] = conf[has_any].astype(np.float32)
    else:
        # Fall back to the legacy loop when the label space is so wide that
        # the score matrix would blow up memory. Rare in practice.
        import warnings
        warnings.warn(
            f"propagate_labels: max label id = {max_label} exceeds "
            f"vectorized cap ({_VECTORIZED_L_CAP}); using slower loop.")
        for i in np.nonzero(has_any)[0]:
            valid_i = valid[i]
            neighbor_dists = dists[i, valid_i]
            neighbor_ids = idx[i, valid_i]
            neighbor_labels = labeled_labels[neighbor_ids]
            weights = 1.0 / (neighbor_dists + 1e-8)
            unique_labels, inverse = np.unique(neighbor_labels, return_inverse=True)
            scores = np.zeros(len(unique_labels))
            for j, w in zip(inverse, weights):
                scores[j] += w
            winner_idx = int(np.argmax(scores))
            propagated[i] = unique_labels[winner_idx]
            confidences[i] = 1.0 - (neighbor_dists.min() / max(radius, 1e-8))

    return propagated, confidences
