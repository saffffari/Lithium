"""3D cursor — solid camera-facing triangle that marks the orbit pivot point.

A single flat triangle (apex down) billboarded toward the camera every
frame, so the silhouette stays a clean triangle regardless of viewing
angle. Depth test is disabled during draw so the cursor is always
visible on top of the cloud — useful when the pivot has been placed on
a back-facing surface.
"""

import math
import numpy as np
import moderngl

from src.rendering.shaders import load_shader


# Equilateral triangle pointing down (apex at the bottom), inscribed in
# the unit circle in the local XY plane. Built in CPU once; the
# billboard transform in draw() rotates it to face the camera each
# frame. Apex points down so the tip indicates the picked point from
# above, like a map pin.
_SQRT3_OVER_2 = math.sqrt(3.0) / 2.0
_TRI_VERTS = np.array([
    [        0.0, -1.0, 0.0],   # bottom apex
    [-_SQRT3_OVER_2,  0.5, 0.0],   # top-left
    [ _SQRT3_OVER_2,  0.5, 0.0],   # top-right
], dtype=np.float32)


class Cursor3D:
    """A solid red triangle, camera-facing, that marks the orbit pivot.

    Depth test is off during draw so the cursor is always visible.
    Geometry is built once; the model matrix in ``draw`` orients the
    local XY frame to camera-right / camera-up each frame so the
    triangle stays planar to the screen.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=load_shader('overlay.vert'),
            fragment_shader=load_shader('overlay.frag'),
        )
        self.position = np.zeros(3, dtype=np.float32)
        self.visible = False

        self._build_geometry()

    def _build_geometry(self):
        # Body fill — deep saturated red. Slightly above 1.0 on the red
        # channel survives ACES tonemapping cleanly on pale clouds.
        body_color = (1.30, 0.00, 0.03, 1.0)

        verts: list[float] = []
        for v in _TRI_VERTS:
            verts.extend([*v, *body_color])

        data = np.array(verts, dtype=np.float32)
        self._vertex_count = len(data) // 7
        self._vbo = self.ctx.buffer(data.tobytes())
        self._vao = self.ctx.vertex_array(
            self.program,
            [(self._vbo, '3f 4f', 'in_position', 'in_color')],
        )

    def draw(self, view: np.ndarray, projection: np.ndarray, cam_distance: float,
             fov: float = math.radians(60.0),
             dof_strength: float = 0.0,
             dof_focus_dist: float = 1.0,
             point_size: float = 5.0):
        """Render the cursor at its position.

        Size tracks the cloud's ``point_size`` so the cursor visually
        scales with the points as the POINT SIZE knob is adjusted.
        Default ``point_size=5`` reproduces the historical 1.2%-of-
        viewport apparent size.

        When ``dof_strength > 0``, the cursor also blooms with the same
        circle-of-confusion formula the point-cloud shader uses — so a
        cursor placed *out* of the focal plane scales up (and dims)
        alongside the points around it.
        """
        if not self.visible:
            return

        # Base size: ~1.2% of viewport height at point_size=5. Linear in
        # point_size so doubling the knob doubles the cursor.
        ps_factor = max(point_size / 5.0, 0.2)  # clamp so small points still leave a clickable cursor
        scale = cam_distance * math.tan(fov / 2.0) * 0.012 * ps_factor

        # Match the point-cloud shader's circle-of-confusion factor:
        #   coc = |dist_to_cursor - focus_dist| / focus_dist
        #   factor = 1 + dof_strength * coc
        # We need view-space distance of the cursor itself, not the camera
        # orbit distance (the cursor can sit anywhere in the scene).
        coc_alpha = 1.0
        if dof_strength > 0.0:
            cursor_h = np.array([self.position[0], self.position[1],
                                 self.position[2], 1.0], dtype=np.float32)
            view_pos = view @ cursor_h
            cursor_dist = float(np.linalg.norm(view_pos[:3]))
            coc = abs(cursor_dist - dof_focus_dist) / max(dof_focus_dist, 1e-3)
            coc_factor = 1.0 + dof_strength * coc
            scale *= coc_factor
            # Match the shader's energy-conservation dim: alpha /= coc^2.
            coc_alpha = 1.0 / max(coc_factor * coc_factor, 1.0)

        # Billboard: build a model matrix whose local X axis maps to the
        # camera's right direction (world), local Y to camera-up, local
        # Z to camera-forward. With Z=0 triangle vertices, the third
        # column doesn't affect the silhouette, but we set it anyway to
        # keep the matrix well-formed.
        right = view[0, :3]
        up = view[1, :3]
        forward = -view[2, :3]  # view's third row is camera-back

        model = np.eye(4, dtype=np.float32)
        model[:3, 0] = right * scale
        model[:3, 1] = up * scale
        model[:3, 2] = forward * scale
        model[:3, 3] = self.position

        mvp = (projection @ view @ model).astype(np.float32)
        self.program['u_mvp'].write(np.ascontiguousarray(mvp.T).tobytes())

        # When defocused, blend so the bloomed cursor isn't a harsh
        # opaque disc — matches what defocused points look like.
        depth_was_on = True
        if coc_alpha < 1.0:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA,
                                   moderngl.ONE_MINUS_SRC_ALPHA)
            if hasattr(self.program, 'get') and 'u_alpha' in self.program:
                self.program['u_alpha'].value = coc_alpha
            else:
                # overlay shader doesn't currently expose per-draw alpha,
                # so the geometry's vertex-colour alpha (1.0) is what
                # drives the result. We at least bloom the size.
                pass

        # Draw on top of everything so the cursor is always visible.
        self.ctx.disable(moderngl.DEPTH_TEST)
        self._vao.render(moderngl.TRIANGLES, vertices=self._vertex_count)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def place(self, world_pos: np.ndarray):
        """Place the cursor center at a world position."""
        self.position = world_pos.astype(np.float32).copy()
        self.visible = True

    def clear(self):
        """Reset cursor to origin."""
        self.position = np.zeros(3, dtype=np.float32)
        self.visible = False

    def release(self):
        self._vbo.release()
        self._vao.release()
        if self.program:
            self.program.release()
