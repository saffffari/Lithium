"""Single-point pick tool via screen-space projection.

Finds the point nearest to the mouse cursor in screen space, within a
pixel radius, with the closest-to-camera point winning when multiple
points overlap.
"""

import numpy as np

# Shared pixel radii for selection tools. Kept here so the pick tool,
# brush snap, and main app all agree.
PICK_RADIUS_PX: float = 15.0        # click-to-pick tolerance
BRUSH_SNAP_RADIUS_PX: float = 40.0  # brush "snap to nearest point" radius
# Cursor / anchor placement (Space, Alt+LMB, double-click, measure click)
# uses a tight radius so the snap can't grab back-facing points that
# happen to fall inside a generous search. The depth tiebreak in
# pick_point still picks closest-to-camera among in-range candidates,
# but a small radius is what guarantees the candidate set itself is
# constrained to the surface the user is actually pointing at.
CURSOR_SNAP_RADIUS_PX: float = 5.0


def _project_to_screen(positions: np.ndarray, mvp: np.ndarray,
                       width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D positions to 2D screen coordinates.

    Returns (screen_xy, view_depth) where:
      screen_xy: (N, 2) float32, pixel coordinates with origin top-left
      view_depth: (N,) float32, NDC z used for camera-depth comparison
    """
    n = len(positions)
    homo = np.ones((n, 4), dtype=np.float32)
    homo[:, :3] = positions
    clip = homo @ mvp.T
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, :3] / w
    # Convert NDC [-1,1] to screen (origin top-left)
    sx = (ndc[:, 0] + 1.0) * 0.5 * width
    sy = (1.0 - ndc[:, 1]) * 0.5 * height
    screen = np.stack([sx, sy], axis=1)
    return screen, ndc[:, 2]


def pick_point(positions: np.ndarray, mvp: np.ndarray,
               mouse_x: float, mouse_y: float,
               width: int, height: int,
               radius_px: float = 10.0) -> int:
    """Find the closest point to the mouse cursor.

    Returns the index of the picked point, or -1 if no point within radius.
    """
    if len(positions) == 0:
        return -1
    screen, depth = _project_to_screen(positions, mvp, width, height)
    dx = screen[:, 0] - mouse_x
    dy = screen[:, 1] - mouse_y
    dist_sq = dx * dx + dy * dy
    radius_sq = radius_px * radius_px
    in_range = dist_sq < radius_sq
    # Also require point to be in front of camera
    in_range &= depth > -1.0
    in_range &= depth < 1.0
    if not in_range.any():
        return -1
    # Among candidates, pick closest to camera
    candidate_depths = np.where(in_range, depth, np.inf)
    best = int(np.argmin(candidate_depths))
    return best


def pick_visible_disk(positions: np.ndarray,
                      model: np.ndarray, view: np.ndarray, proj: np.ndarray,
                      mouse_x: float, mouse_y: float,
                      width: int, height: int,
                      point_size: float,
                      dof_strength: float = 0.0,
                      dof_focus_dist: float = 1.0) -> int:
    """Pick the front-most point whose *rendered disk* contains the cursor.

    Mirrors the point-cloud shader's size formula
    ``gl_PointSize = u_point_size * (300/dist) * (1 + dof * coc)`` so
    what the user clicks is what they see — picking a back-facing point
    just because its centre happens to be a pixel closer to the cursor
    than a front-facing point whose disk visually contains the click
    is the bug this function fixes (especially noticeable with FOCUS
    cranked up — points balloon and their disks overlap).

    Falls back to nearest-centre semantics if *no* rendered disk
    contains the click — preserves the old behaviour for clicks in
    empty space between points.
    """
    if len(positions) == 0:
        return -1

    # Screen-space positions + NDC depth (for in-view filter + tiebreak).
    mvp = proj @ view @ model
    screen, depth = _project_to_screen(positions, mvp, width, height)
    in_view = (depth > -1.0) & (depth < 1.0)
    if not in_view.any():
        return -1

    # View-space distance per point — exactly what the shader's
    # ``length(view_pos.xyz)`` returns. We need this for the size
    # formula, not the NDC depth.
    view_xform = view @ model
    homo = np.empty((len(positions), 4), dtype=np.float32)
    homo[:, :3] = positions
    homo[:, 3] = 1.0
    view_pos = homo @ view_xform.T
    dist = np.maximum(np.linalg.norm(view_pos[:, :3], axis=1), 0.1)

    # Replicate the shader's gl_PointSize formula. The result is point
    # *diameter* in pixels, so halve for the radius.
    radius_px = point_size * (300.0 / dist)
    if dof_strength > 0.0:
        coc = np.abs(dist - dof_focus_dist) / max(dof_focus_dist, 1e-3)
        radius_px = radius_px * (1.0 + dof_strength * coc)
    radius_px *= 0.5

    dx = screen[:, 0] - mouse_x
    dy = screen[:, 1] - mouse_y
    dist_to_mouse_sq = dx * dx + dy * dy
    dist_px = np.sqrt(np.maximum(dist_to_mouse_sq, 0.0))

    # Unified depth-weighted score. Each point's penalty is:
    #
    #   penalty = miss_distance + DEPTH_BIAS_PX * normalized_depth
    #
    # where ``miss_distance = max(0, pixel_dist - rendered_disk_radius)``
    # — zero when the click is inside the rendered disk, otherwise the
    # gap between the click and the disk edge. Depth bias adds a
    # 0..DEPTH_BIAS_PX penalty for far points (NDC depth +1.0) and
    # zero penalty for near points (NDC depth -1.0).
    #
    # Why this scoring replaced the older two-pass (disk-overlap-
    # first, fallback-tie-band-second): the old logic preferred a far
    # point whose 2-pixel disk happened to sit under the cursor over a
    # small near point whose 7-pixel disk sat 10 px to the side. The
    # user reads this as "I misclicked by 10 px and it jumped to the
    # back of the head." The unified score gives near-camera points a
    # 0..40 px depth-bias credit so a 10–25 px miss on a near point
    # still beats a centered hit on a far point — but a 100 px miss
    # doesn't, because the miss distance has fully overwhelmed the
    # depth bias by then.
    #
    # Eligibility: stay inside the disk OR be within MAX_MISS_PX of
    # the disk edge. A click in genuinely empty space (no point within
    # 60 px of its rendered disk) returns -1 so the caller can decide.
    miss_px = np.maximum(0.0, dist_px - radius_px)
    DEPTH_BIAS_PX = 40.0
    MAX_MISS_PX = 60.0
    # Use view-space distance (which we already computed for the disk
    # radius formula) for the depth bias, not NDC z. NDC depth is
    # non-linear in perspective and saturates near 1.0 for most points
    # in a typical scene — two points at view-distances 2mm and 7mm
    # end up with nearly identical NDC z, defeating the bias. View
    # distance is monotonic, so it cleanly separates near-camera and
    # far-camera points. We normalize relative to the points-in-view
    # range: the front-most in-view point pays no depth bias, the
    # back-most pays the full DEPTH_BIAS_PX.
    if in_view.any():
        dist_in_view = dist[in_view]
        dist_min = float(dist_in_view.min())
        dist_max = float(dist_in_view.max())
    else:
        return -1
    if dist_max - dist_min > 1e-6:
        norm_dist = np.clip((dist - dist_min) / (dist_max - dist_min),
                            0.0, 1.0)
    else:
        norm_dist = np.zeros_like(dist)
    penalty = miss_px + DEPTH_BIAS_PX * norm_dist
    eligible = in_view & (miss_px < MAX_MISS_PX)
    if not eligible.any():
        return -1
    scored = np.where(eligible, penalty, np.inf)
    return int(np.argmin(scored))


def pick_nearest_point(positions: np.ndarray, mvp: np.ndarray,
                       mouse_x: float, mouse_y: float,
                       width: int, height: int) -> int:
    """Find the screen-space-nearest *visible* point to the mouse.

    Unlike :func:`pick_point` this never returns ``-1`` for "outside the
    radius" — there is no radius. Used by the orbit-pivot cursor and any
    other "what point did the user click on" caller where falling back
    to a synthetic plane intersection (the old behaviour) would just
    place the result somewhere visibly offset from the click.

    Behaviour:
      * filters to points in front of the camera and inside the depth
        range, then picks the one closest to the mouse in pixels.
      * if multiple points overlap on top of the winner (within a few
        pixels), prefers the one closest to the camera so overlapping
        front/back shells don't accidentally snap to a back-facing
        point.
      * returns -1 only when *no* point of the cloud is on screen at
        all — empty viewport, cloud entirely off-frustum, etc.
    """
    if len(positions) == 0:
        return -1
    screen, depth = _project_to_screen(positions, mvp, width, height)
    in_view = (depth > -1.0) & (depth < 1.0)
    if not in_view.any():
        return -1
    dx = screen[:, 0] - mouse_x
    dy = screen[:, 1] - mouse_y
    dist_sq = np.where(in_view, dx * dx + dy * dy, np.inf)
    nearest_dist_sq = float(dist_sq.min())
    # Tiebreak band: any point whose screen distance is within ~3 px of
    # the winner counts as a co-located candidate, and among those the
    # front-most wins. The band is added to the radius (not the squared
    # radius) so the geometry is intuitive: "3 px wider in pixel space".
    TIE_BAND_PX = 3.0
    tie_radius = float(np.sqrt(nearest_dist_sq)) + TIE_BAND_PX
    tie_radius_sq = tie_radius * tie_radius
    candidates = dist_sq <= tie_radius_sq
    candidate_depths = np.where(candidates, depth, np.inf)
    return int(np.argmin(candidate_depths))
