"""3D sphere brush selection tool."""

import numpy as np


def brush_select(positions: np.ndarray,
                 center: np.ndarray, radius: float,
                 envelope=None, weight_threshold: float = 0.05) -> np.ndarray:
    """Select all points within radius of a 3D center.

    If an envelope is provided, each candidate point is weighted by
    envelope.sample(normalized_distance), and only points whose weight
    is >= weight_threshold are selected. This lets the user shape the
    brush falloff curve (hard edge, soft edge, bullseye, etc.).

    positions: (N, 3) float32
    center: (3,) float32
    radius: float (world units)
    envelope: optional src.core.envelope.Envelope instance

    Returns int32 array of selected indices.
    """
    if len(positions) == 0:
        return np.array([], dtype=np.int32)
    diff = positions - center
    dist_sq = np.einsum('ij,ij->i', diff, diff)
    in_radius_mask = dist_sq <= radius * radius
    if envelope is None:
        return np.nonzero(in_radius_mask)[0].astype(np.int32)

    # Normalize distances to [0, 1], sample envelope, threshold
    in_radius_idx = np.nonzero(in_radius_mask)[0]
    if len(in_radius_idx) == 0:
        return np.array([], dtype=np.int32)
    dist = np.sqrt(dist_sq[in_radius_idx])
    norm_dist = dist / max(radius, 1e-8)
    weights = envelope.to_lut(256)  # 256 samples
    # Index into the LUT by normalized distance
    lut_idx = np.clip((norm_dist * 255).astype(np.int32), 0, 255)
    point_weights = weights[lut_idx]
    keep = point_weights >= weight_threshold
    return in_radius_idx[keep].astype(np.int32)


def screen_to_world(positions: np.ndarray, mvp: np.ndarray,
                    mouse_x: float, mouse_y: float,
                    width: int, height: int,
                    view: np.ndarray | None = None,
                    projection: np.ndarray | None = None,
                    camera_target: np.ndarray | None = None,
                    camera_distance: float = 1.0,
                    radius_px: float | None = None) -> np.ndarray | None:
    """Find a 3D brush center near the mouse.

    First tries to snap to the nearest visible point within `radius_px`
    (so the brush follows the surface). If no point is nearby and
    `view`/`projection` are provided, falls back to unprojecting the
    mouse into a world ray and sampling it at `camera_distance` from
    the camera — this lets the brush work over empty sky without
    silently dropping strokes.
    """
    from src.core.tools.pick_tool import pick_point, BRUSH_SNAP_RADIUS_PX
    if radius_px is None:
        radius_px = BRUSH_SNAP_RADIUS_PX
    idx = pick_point(positions, mvp, mouse_x, mouse_y, width, height,
                     radius_px=radius_px)
    if idx >= 0:
        return positions[idx].copy()

    # Fallback: world-space ray through the mouse at camera distance
    if view is None or projection is None:
        return None
    from src.rendering.gizmo_renderer import unproject_mouse
    ray_o, ray_d = unproject_mouse(mouse_x, mouse_y, width, height, view, projection)
    t = float(camera_distance)
    hit = ray_o + ray_d * t
    return hit.astype(np.float32)
