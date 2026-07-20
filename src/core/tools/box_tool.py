"""Box (screen-space rectangle) selection tool."""

import numpy as np

from src.core.tools.pick_tool import _project_to_screen


def box_select(positions: np.ndarray, mvp: np.ndarray,
               x0: float, y0: float, x1: float, y1: float,
               width: int, height: int,
               max_depth: float = float('inf'),
               view_matrix: np.ndarray = None) -> np.ndarray:
    """Select all points within a screen-space rectangle.

    Returns int32 array of selected indices.

    max_depth: if set, exclude points farther than this from the camera in view space.
    view_matrix: required if max_depth is finite; used to compute view-space depth.
    """
    if len(positions) == 0:
        return np.array([], dtype=np.int32)

    lo_x, hi_x = min(x0, x1), max(x0, x1)
    lo_y, hi_y = min(y0, y1), max(y0, y1)

    screen, depth_ndc = _project_to_screen(positions, mvp, width, height)
    mask = ((screen[:, 0] >= lo_x) & (screen[:, 0] <= hi_x) &
            (screen[:, 1] >= lo_y) & (screen[:, 1] <= hi_y) &
            (depth_ndc > -1.0) & (depth_ndc < 1.0))

    if np.isfinite(max_depth) and view_matrix is not None:
        n = len(positions)
        homo = np.ones((n, 4), dtype=np.float32)
        homo[:, :3] = positions
        view_pos = homo @ view_matrix.T
        view_z = -view_pos[:, 2]  # positive = in front of camera
        mask &= view_z <= max_depth

    return np.nonzero(mask)[0].astype(np.int32)
