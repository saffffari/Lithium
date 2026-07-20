"""Curve selection tool — polyline with perpendicular distance threshold.

Select all points whose perpendicular distance to any segment of a
user-drawn polyline is within threshold. Good for edges, ridges, and
boundary annotation.
"""

import numpy as np

from src.core.tools.pick_tool import _project_to_screen


def _point_to_segment_distance_2d(px: np.ndarray, py: np.ndarray,
                                   x0: float, y0: float,
                                   x1: float, y1: float) -> np.ndarray:
    """Vectorized 2D point-to-segment distance.

    px, py: (N,) arrays of point coordinates
    (x0, y0) -> (x1, y1): segment endpoints
    Returns (N,) distances.
    """
    dx = x1 - x0
    dy = y1 - y0
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-10:
        # Degenerate segment: distance to point
        return np.sqrt((px - x0) ** 2 + (py - y0) ** 2)
    # Parametric t along segment
    t = ((px - x0) * dx + (py - y0) * dy) / length_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x = x0 + t * dx
    proj_y = y0 + t * dy
    return np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)


def curve_select(positions: np.ndarray, mvp: np.ndarray,
                 polyline: list[tuple[float, float]],
                 width: int, height: int,
                 threshold_px: float = 10.0,
                 max_depth: float = float('inf'),
                 view_matrix: np.ndarray = None) -> np.ndarray:
    """Select points within threshold distance of any polyline segment.

    polyline: list of (x, y) screen coordinates
    threshold_px: max perpendicular distance in pixels
    """
    if len(positions) == 0 or len(polyline) < 2:
        return np.array([], dtype=np.int32)

    screen, depth_ndc = _project_to_screen(positions, mvp, width, height)
    in_frustum = (depth_ndc > -1.0) & (depth_ndc < 1.0)

    if np.isfinite(max_depth) and view_matrix is not None:
        n = len(positions)
        homo = np.ones((n, 4), dtype=np.float32)
        homo[:, :3] = positions
        view_pos = homo @ view_matrix.T
        view_z = -view_pos[:, 2]
        in_frustum &= view_z <= max_depth

    # Compute min distance across all segments
    min_dist = np.full(len(positions), np.inf, dtype=np.float32)
    for i in range(len(polyline) - 1):
        x0, y0 = polyline[i]
        x1, y1 = polyline[i + 1]
        d = _point_to_segment_distance_2d(
            screen[:, 0], screen[:, 1], x0, y0, x1, y1
        )
        min_dist = np.minimum(min_dist, d)

    mask = in_frustum & (min_dist <= threshold_px)
    return np.nonzero(mask)[0].astype(np.int32)
