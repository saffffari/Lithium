"""Freeform lasso selection tool using point-in-polygon (winding number)."""

import numpy as np

from src.core.tools.pick_tool import _project_to_screen


def _point_in_polygon(sx: np.ndarray, sy: np.ndarray,
                      poly: np.ndarray) -> np.ndarray:
    """Vectorized winding-number point-in-polygon test.

    sx, sy: (N,) screen coordinates of points
    poly:   (M, 2) polygon vertices (open polygon, auto-closed)

    Returns (N,) bool mask.
    """
    if len(poly) < 3:
        return np.zeros(len(sx), dtype=bool)

    x = sx
    y = sy
    inside = np.zeros(len(x), dtype=bool)

    # Ray casting: count edge crossings
    m = len(poly)
    for i in range(m):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % m]
        # Edge crosses horizontal ray from point to +x
        cond1 = (y0 > y) != (y1 > y)
        # Compute x of intersection
        denom = (y1 - y0)
        denom_safe = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        x_intersect = x0 + (y - y0) * (x1 - x0) / denom_safe
        cond2 = x < x_intersect
        inside ^= (cond1 & cond2)

    return inside


def lasso_select(positions: np.ndarray, mvp: np.ndarray,
                 polygon: list[tuple[float, float]],
                 width: int, height: int,
                 max_depth: float = float('inf'),
                 view_matrix: np.ndarray = None) -> np.ndarray:
    """Select points inside a polygon drawn in screen space.

    polygon: list of (x, y) screen coordinates forming the lasso path.
    """
    if len(positions) == 0 or len(polygon) < 3:
        return np.array([], dtype=np.int32)

    poly = np.array(polygon, dtype=np.float32)

    screen, depth_ndc = _project_to_screen(positions, mvp, width, height)
    # First filter by polygon bounding box for speed
    bmin = poly.min(axis=0)
    bmax = poly.max(axis=0)
    in_bbox = ((screen[:, 0] >= bmin[0]) & (screen[:, 0] <= bmax[0]) &
               (screen[:, 1] >= bmin[1]) & (screen[:, 1] <= bmax[1]) &
               (depth_ndc > -1.0) & (depth_ndc < 1.0))

    if np.isfinite(max_depth) and view_matrix is not None:
        n = len(positions)
        homo = np.ones((n, 4), dtype=np.float32)
        homo[:, :3] = positions
        view_pos = homo @ view_matrix.T
        view_z = -view_pos[:, 2]
        in_bbox &= view_z <= max_depth

    # Winding-number test on candidates only
    candidate_idx = np.nonzero(in_bbox)[0]
    if len(candidate_idx) == 0:
        return np.array([], dtype=np.int32)

    sx = screen[candidate_idx, 0]
    sy = screen[candidate_idx, 1]
    inside = _point_in_polygon(sx, sy, poly)
    return candidate_idx[inside].astype(np.int32)
