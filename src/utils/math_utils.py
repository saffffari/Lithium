"""MVP matrix construction and math utilities."""

import numpy as np


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Create a perspective projection matrix."""
    f = 1.0 / np.tan(fov_y / 2.0)
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = f / aspect
    mat[1, 1] = f
    mat[2, 2] = (far + near) / (near - far)
    mat[2, 3] = (2.0 * far * near) / (near - far)
    mat[3, 2] = -1.0
    return mat


def orthographic(left: float, right: float, bottom: float, top: float,
                 near: float, far: float) -> np.ndarray:
    """Standard orthographic projection matrix."""
    mat = np.zeros((4, 4), dtype=np.float32)
    mat[0, 0] = 2.0 / (right - left)
    mat[1, 1] = 2.0 / (top - bottom)
    mat[2, 2] = -2.0 / (far - near)
    mat[0, 3] = -(right + left) / (right - left)
    mat[1, 3] = -(top + bottom) / (top - bottom)
    mat[2, 3] = -(far + near) / (far - near)
    mat[3, 3] = 1.0
    return mat


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a view matrix looking from eye toward target.

    Degenerate inputs fall back to a sane default orientation instead of
    returning NaNs: an eye-on-target (zero forward) lands on identity view;
    an ``up`` vector parallel to ``forward`` picks a perpendicular axis so
    the right vector isn't zero-length.
    """
    f = target - eye
    f_norm = float(np.linalg.norm(f))
    if f_norm < 1e-8:
        return np.eye(4, dtype=np.float32)
    f = f / f_norm
    s = np.cross(f, up)
    s_norm = float(np.linalg.norm(s))
    if s_norm < 1e-6:
        # up is (nearly) parallel to forward — pick any world axis that
        # isn't collinear with f to rebuild a stable orthonormal basis.
        fallback = np.array([1.0, 0.0, 0.0], dtype=f.dtype)
        if abs(float(f[0])) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=f.dtype)
        s = np.cross(f, fallback)
        s_norm = float(np.linalg.norm(s))
        if s_norm < 1e-6:
            return np.eye(4, dtype=np.float32)
    s = s / s_norm
    u = np.cross(s, f)

    mat = np.eye(4, dtype=np.float32)
    mat[0, 0:3] = s
    mat[1, 0:3] = u
    mat[2, 0:3] = -f
    mat[0, 3] = -np.dot(s, eye)
    mat[1, 3] = -np.dot(u, eye)
    mat[2, 3] = np.dot(f, eye)
    return mat


def project_points_batch(mvp: np.ndarray, positions, width: int, height: int,
                          ) -> list:
    """Project world-space positions to screen space in a single matmul.

    ``positions`` is an iterable of (x, y, z) tuples / ndarrays. Returns
    a parallel list where each entry is either ``(sx, sy)`` (inside the
    NDC z-range) or ``None`` (clipped / behind the camera).

    Avoids the per-anchor ``np.array([x,y,z,1])`` allocation + matmul
    that measure overlays used to do — critical when a scene has many
    measurements and the overlay draws every frame.
    """
    if not positions:
        return []
    n = len(positions)
    homo = np.empty((n, 4), dtype=np.float32)
    for i, p in enumerate(positions):
        homo[i, 0] = float(p[0])
        homo[i, 1] = float(p[1])
        homo[i, 2] = float(p[2])
    homo[:, 3] = 1.0
    clip = homo @ mvp.T
    w = clip[:, 3]
    out: list = [None] * n
    safe = np.abs(w) > 1e-8
    if not safe.any():
        return out
    inv_w = np.zeros_like(w)
    inv_w[safe] = 1.0 / w[safe]
    nx = clip[:, 0] * inv_w
    ny = clip[:, 1] * inv_w
    nz = clip[:, 2] * inv_w
    visible = safe & (nz >= -1.0) & (nz <= 1.0)
    sx = (nx + 1.0) * 0.5 * width
    sy = (1.0 - ny) * 0.5 * height
    for i in range(n):
        if visible[i]:
            out[i] = (float(sx[i]), float(sy[i]))
    return out


def orbit_camera(target: np.ndarray, distance: float, azimuth: float,
                 elevation: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute eye position and view matrix from orbit parameters.

    Returns (eye_position, view_matrix).
    """
    # Clamp elevation to avoid gimbal lock. Use plain min/max (scalar)
    # to avoid np.clip overhead on a hot per-frame path.
    _pole = 1.5607963  # pi/2 - 0.01
    elevation = max(-_pole, min(_pole, float(elevation)))

    eye = np.array([
        target[0] + distance * np.cos(elevation) * np.sin(azimuth),
        target[1] + distance * np.sin(elevation),
        target[2] + distance * np.cos(elevation) * np.cos(azimuth),
    ], dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = look_at(eye, target, up)
    return eye, view
