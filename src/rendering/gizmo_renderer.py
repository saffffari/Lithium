"""3D translation/rotation gizmo for adjusting point cloud origin."""

import math
import numpy as np
import moderngl

from src.rendering.shaders import load_shader
from src.core.colors import (
    AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR, GIZMO_HIGHLIGHT_COLOR,
)


# Axis definitions
AXIS_X = 0
AXIS_Y = 1
AXIS_Z = 2
AXIS_NONE = -1

# Axis colors must match the clip rack and other axis-colored widgets.
# Canonical RGB lives in src.core.colors; gui.theme keeps named aliases
# (CORAL / OP1_GREEN / OP1_BLUE) for ImGui foreground use.
AXIS_COLORS = {
    AXIS_X: (*AXIS_X_COLOR, 0.9),
    AXIS_Y: (*AXIS_Y_COLOR, 0.9),
    AXIS_Z: (*AXIS_Z_COLOR, 0.9),
}
HIGHLIGHT_COLOR = GIZMO_HIGHLIGHT_COLOR


def _make_cone(base: np.ndarray, tip: np.ndarray, color: tuple,
               radius: float = 0.08, segments: int = 12) -> list:
    """Generate vertices for a cone (arrow head)."""
    direction = tip - base
    length = np.linalg.norm(direction)
    if length < 1e-8:
        return []
    d = direction / length

    # Find perpendicular vectors
    if abs(d[1]) < 0.9:
        perp1 = np.cross(d, np.array([0, 1, 0], dtype=np.float32))
    else:
        perp1 = np.cross(d, np.array([1, 0, 0], dtype=np.float32))
    perp1_norm = float(np.linalg.norm(perp1))
    if perp1_norm < 1e-6:
        return []
    perp1 /= perp1_norm
    perp2 = np.cross(d, perp1)

    verts = []
    for i in range(segments):
        a1 = 2 * math.pi * i / segments
        a2 = 2 * math.pi * (i + 1) / segments
        p1 = base + radius * (math.cos(a1) * perp1 + math.sin(a1) * perp2)
        p2 = base + radius * (math.cos(a2) * perp1 + math.sin(a2) * perp2)
        # Triangle: tip, p1, p2
        verts.extend([*tip, *color, *p1, *color, *p2, *color])
    return verts


def _make_circle(center: np.ndarray, normal: np.ndarray, radius: float,
                 color: tuple, segments: int = 64) -> list:
    """Generate line vertices for a circle (rotation ring)."""
    if abs(normal[1]) < 0.9:
        perp1 = np.cross(normal, np.array([0, 1, 0], dtype=np.float32))
    else:
        perp1 = np.cross(normal, np.array([1, 0, 0], dtype=np.float32))
    perp1_norm = float(np.linalg.norm(perp1))
    if perp1_norm < 1e-6:
        return []
    perp1 /= perp1_norm
    perp2 = np.cross(normal, perp1)

    verts = []
    for i in range(segments):
        a1 = 2 * math.pi * i / segments
        a2 = 2 * math.pi * (i + 1) / segments
        p1 = center + radius * (math.cos(a1) * perp1 + math.sin(a1) * perp2)
        p2 = center + radius * (math.cos(a2) * perp1 + math.sin(a2) * perp2)
        verts.extend([*p1, *color, *p2, *color])
    return verts


class GizmoRenderer:
    """Interactive translation/rotation gizmo."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=load_shader('gizmo.vert'),
            fragment_shader=load_shader('gizmo.frag'),
        )
        # Gizmo is hidden by default — only visible while the MOVE/ROTATE
        # tool is active. App._toggle_tool sets this.
        self.visible = False
        self.mode = 'translate'  # 'translate' or 'rotate'
        self.hovered_axis = AXIS_NONE
        self.active_axis = AXIS_NONE
        self.position = np.zeros(3, dtype=np.float32)
        self.scale = 1.0
        # When True the gizmo is drawn at reduced opacity so the user
        # knows it's not interactive (e.g. while a selection tool is active).
        self.dimmed = False
        self._last_dimmed = False

        # Interaction state
        self._dragging = False
        self._drag_start_pos = np.zeros(3, dtype=np.float32)
        self._drag_start_transform = np.eye(4, dtype=np.float32)

        # GPU resources
        self._translate_vao = None
        self._translate_vbo = None
        self._translate_vbo_cap = 0
        self._translate_tri_count = 0
        self._translate_line_count = 0
        self._rotate_vao = None
        self._rotate_vbo = None
        self._rotate_vbo_cap = 0
        self._rotate_line_count = 0
        # Dirty-check cache to skip rebuild when nothing changed
        self._build_cache_key = None

    def build(self, center: np.ndarray, cam_distance: float, fov: float = math.radians(60)):
        """Rebuild gizmo geometry at the given position and screen-constant scale.

        Skips rebuild when center, camera distance, hovered axis, and dim
        state are all unchanged from the last call.
        """
        cache_key = (
            tuple(float(v) for v in center),
            round(cam_distance, 4),
            round(fov, 4),
            self.hovered_axis,
            self.dimmed,
        )
        if cache_key == self._build_cache_key:
            return
        self._build_cache_key = cache_key

        self.position = center.copy()

        # Constant screen size: gizmo is ~15% of viewport height
        self.scale = cam_distance * math.tan(fov / 2) * 0.15
        s = self.scale

        self._build_translate_gizmo(center, s)
        self._build_rotate_gizmo(center, s)
        self._last_dimmed = self.dimmed

    def _axis_color(self, axis_id: int) -> tuple:
        """Color for an axis, scaled by the dim factor when applicable."""
        base = HIGHLIGHT_COLOR if axis_id == self.hovered_axis else AXIS_COLORS[axis_id]
        if self.dimmed:
            return (base[0], base[1], base[2], base[3] * 0.25)
        return base

    def _build_translate_gizmo(self, center: np.ndarray, s: float):
        """Build translation arrows (3 lines + 3 cones).

        Reuses the existing VBO when the vertex count is unchanged
        (the common case — the vertex count only depends on the cone
        segments, which is a constant).
        """
        lines = []
        tris = []
        for axis_id, axis_dir in [(AXIS_X, [1, 0, 0]), (AXIS_Y, [0, 1, 0]), (AXIS_Z, [0, 0, 1])]:
            color = self._axis_color(axis_id)
            d = np.array(axis_dir, dtype=np.float32)
            start = center
            end = center + d * s
            cone_base = center + d * s * 0.8

            # Line segment
            lines.extend([*start, *color, *end, *color])
            # Cone
            tris.extend(_make_cone(cone_base, end, color, radius=s * 0.06))

        # Pack lines then triangles
        line_data = np.array(lines, dtype=np.float32) if lines else np.array([], dtype=np.float32)
        tri_data = np.array(tris, dtype=np.float32) if tris else np.array([], dtype=np.float32)

        self._translate_line_count = len(line_data) // 7 if len(line_data) > 0 else 0
        self._translate_tri_count = len(tri_data) // 7 if len(tri_data) > 0 else 0

        all_data = np.concatenate([line_data, tri_data]) if len(line_data) + len(tri_data) > 0 else np.array([], dtype=np.float32)

        if len(all_data) == 0:
            return
        data = all_data.tobytes()
        if self._translate_vbo is not None and len(data) == self._translate_vbo_cap:
            self._translate_vbo.write(data)
            return
        if self._translate_vbo is not None:
            self._translate_vbo.release()
            self._translate_vao.release()
        self._translate_vbo = self.ctx.buffer(data)
        self._translate_vbo_cap = len(data)
        self._translate_vao = self.ctx.vertex_array(
            self.program,
            [(self._translate_vbo, '3f 4f', 'in_position', 'in_color')],
        )

    def _build_rotate_gizmo(self, center: np.ndarray, s: float):
        """Build rotation rings (3 circles).

        Circle segment count is a constant so the VBO size is stable —
        reuse the existing buffer after the first build.
        """
        verts = []
        for axis_id, normal in [(AXIS_X, [1, 0, 0]), (AXIS_Y, [0, 1, 0]), (AXIS_Z, [0, 0, 1])]:
            color = self._axis_color(axis_id)
            n = np.array(normal, dtype=np.float32)
            verts.extend(_make_circle(center, n, s * 1.1, color))

        if not verts:
            return
        arr = np.array(verts, dtype=np.float32)
        self._rotate_line_count = len(arr) // 7
        data = arr.tobytes()
        if self._rotate_vbo is not None and len(data) == self._rotate_vbo_cap:
            self._rotate_vbo.write(data)
            return
        if self._rotate_vbo is not None:
            self._rotate_vbo.release()
            self._rotate_vao.release()
        self._rotate_vbo = self.ctx.buffer(data)
        self._rotate_vbo_cap = len(data)
        self._rotate_vao = self.ctx.vertex_array(
            self.program,
            [(self._rotate_vbo, '3f 4f', 'in_position', 'in_color')],
        )

    def draw(self, view: np.ndarray, projection: np.ndarray):
        """Render the gizmo."""
        if not self.visible:
            return

        mvp = (projection @ view).astype(np.float32)
        self.program['u_mvp'].write(np.ascontiguousarray(mvp.T).tobytes())

        # Disable depth test so gizmo is always visible
        self.ctx.disable(moderngl.DEPTH_TEST)

        if self.mode == 'translate' and self._translate_vao:
            # Draw lines
            if self._translate_line_count > 0:
                self._translate_vao.render(moderngl.LINES, vertices=self._translate_line_count)
            # Draw triangles (cones)
            if self._translate_tri_count > 0:
                self._translate_vao.render(
                    moderngl.TRIANGLES,
                    first=self._translate_line_count,
                    vertices=self._translate_tri_count,
                )
        elif self.mode == 'rotate' and self._rotate_vao:
            self._rotate_vao.render(moderngl.LINES, vertices=self._rotate_line_count)

        self.ctx.enable(moderngl.DEPTH_TEST)

    def hit_test(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
                 threshold_world: float = None) -> int:
        """Test which axis the ray is closest to. Returns AXIS_X/Y/Z or AXIS_NONE."""
        if threshold_world is None:
            threshold_world = self.scale * 0.15

        best_axis = AXIS_NONE
        best_dist = threshold_world

        axes = [
            (AXIS_X, np.array([1, 0, 0], dtype=np.float32)),
            (AXIS_Y, np.array([0, 1, 0], dtype=np.float32)),
            (AXIS_Z, np.array([0, 0, 1], dtype=np.float32)),
        ]

        for axis_id, axis_dir in axes:
            # Line segment from position to position + axis * scale
            seg_start = self.position
            seg_end = self.position + axis_dir * self.scale

            dist = _ray_segment_distance(ray_origin, ray_dir, seg_start, seg_end)
            if dist < best_dist:
                best_dist = dist
                best_axis = axis_id

        return best_axis

    def start_drag(self, axis: int, model_transform: np.ndarray):
        """Begin dragging on an axis."""
        self.active_axis = axis
        self._dragging = True
        self._drag_start_transform = model_transform.copy()

    def update_drag(self, delta_world: np.ndarray, model_transform: np.ndarray):
        """Update the model transform during a drag."""
        if not self._dragging or self.active_axis == AXIS_NONE:
            return model_transform

        axis_dirs = {
            AXIS_X: np.array([1, 0, 0], dtype=np.float32),
            AXIS_Y: np.array([0, 1, 0], dtype=np.float32),
            AXIS_Z: np.array([0, 0, 1], dtype=np.float32),
        }
        axis = axis_dirs[self.active_axis]

        if self.mode == 'translate':
            # Project delta onto axis
            displacement = np.dot(delta_world, axis) * axis
            result = self._drag_start_transform.copy()
            result[0:3, 3] += displacement
            return result
        elif self.mode == 'rotate':
            # Rotation angle proportional to delta projected onto perpendicular
            angle = np.dot(delta_world, np.roll(axis, 1)) * 2.0
            c, s = np.cos(angle), np.sin(angle)
            rot = np.eye(4, dtype=np.float32)
            i, j = [(1, 2), (2, 0), (0, 1)][self.active_axis]
            rot[i, i] = c
            rot[i, j] = -s
            rot[j, i] = s
            rot[j, j] = c
            return rot @ self._drag_start_transform

        return model_transform

    def end_drag(self):
        """End a drag operation."""
        self._dragging = False
        self.active_axis = AXIS_NONE

    def release(self):
        if self._translate_vbo:
            self._translate_vbo.release()
            self._translate_vao.release()
        if self._rotate_vbo:
            self._rotate_vbo.release()
            self._rotate_vao.release()
        if self.program:
            self.program.release()


def _ray_segment_distance(ray_origin, ray_dir, seg_start, seg_end):
    """Compute minimum distance between a ray and a line segment."""
    u = ray_dir
    v = seg_end - seg_start
    w = ray_origin - seg_start

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    denom = a * c - b * b
    if abs(denom) < 1e-10:
        # Parallel
        t_seg = 0.0
        t_ray = d / max(a, 1e-10)
    else:
        t_ray = (b * e - c * d) / denom
        t_seg = (a * e - b * d) / denom

    t_ray = max(t_ray, 0.0)
    t_seg = np.clip(t_seg, 0.0, 1.0)

    closest_ray = ray_origin + u * t_ray
    closest_seg = seg_start + v * t_seg

    return np.linalg.norm(closest_ray - closest_seg)


def unproject_mouse(mx: float, my: float, width: int, height: int,
                    view: np.ndarray, projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert mouse position to a world-space ray (origin, direction).

    Returns a degenerate ray (origin at world zero, direction +Z) if the
    view/projection matrices are singular — callers should treat this as a
    miss rather than crashing the pick path.
    """
    # Avoid divide-by-zero if the framebuffer collapsed mid-frame.
    if width <= 0 or height <= 0:
        return (np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Normalized device coordinates
    ndc_x = (2.0 * mx / width) - 1.0
    ndc_y = 1.0 - (2.0 * my / height)

    try:
        inv_proj = np.linalg.inv(projection)
        inv_view = np.linalg.inv(view)
    except np.linalg.LinAlgError:
        return (np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32))

    # Near point in clip space
    clip_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    clip_far = np.array([ndc_x, ndc_y, 1.0, 1.0], dtype=np.float32)

    eye_near = inv_proj @ clip_near
    if abs(eye_near[3]) < 1e-8:
        return (np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32))
    eye_near /= eye_near[3]
    eye_far = inv_proj @ clip_far
    if abs(eye_far[3]) < 1e-8:
        return (np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32))
    eye_far /= eye_far[3]

    world_near = inv_view @ eye_near
    world_far = inv_view @ eye_far

    origin = world_near[:3]
    direction = world_far[:3] - world_near[:3]
    dir_norm = float(np.linalg.norm(direction))
    if dir_norm < 1e-8:
        return (np.zeros(3, dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32))
    direction = direction / dir_norm

    return origin, direction
