"""Overlay rendering: bounding box wireframe and grid floor."""

import numpy as np
import moderngl

from src.rendering.shaders import load_shader


# Reusable column-major MVP buffer. ``mvp.T.tobytes()`` used to allocate
# a fresh 64-byte array on every overlay draw — multiple times per frame
# across bbox, grid, clip planes, depth plane, etc. Writing into a
# persistent scratch buffer drops those allocations without changing the
# wire format.
_MVP_SCRATCH = np.empty((4, 4), dtype=np.float32)


def _mvp_bytes(mat: np.ndarray) -> bytes:
    np.copyto(_MVP_SCRATCH, np.asarray(mat, dtype=np.float32).T)
    return _MVP_SCRATCH.tobytes()


class OverlayRenderer:
    """Renders bounding box and grid overlays."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=load_shader('overlay.vert'),
            fragment_shader=load_shader('overlay.frag'),
        )
        # Infinite adaptive grid — renders a large XZ quad centered on
        # the camera's ground projection and computes the grid pattern
        # from world coords in the fragment shader.
        self.grid_program = ctx.program(
            vertex_shader=load_shader('grid_infinite.vert'),
            fragment_shader=load_shader('grid_infinite.frag'),
        )
        self._grid_quad_vbo = None
        self._grid_quad_vao = None
        self._grid_y_plane = 0.0
        self._grid_extent = 1.0e4
        self._bbox_vao = None
        self._bbox_vbo = None
        self._bbox_vbo_cap = 0  # current VBO capacity in bytes
        self._grid_vao = None
        self._grid_vbo = None
        self._grid_vbo_cap = 0
        self._grid_vertex_count = 0
        self._depth_plane_vao = None
        self._depth_plane_vbo = None
        self._depth_plane_vbo_cap = 0
        # Dirty-check cache for clip planes — avoids VBO rebuild every frame
        self._clip_cache_key = None

        # Build the static unit quad once
        self._build_grid_quad()

    def _build_grid_quad(self):
        """Build the single unit quad used by the infinite grid shader."""
        if self._grid_quad_vbo is not None:
            return
        quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
             1.0,  1.0,
            -1.0, -1.0,
             1.0,  1.0,
            -1.0,  1.0,
        ], dtype=np.float32)
        self._grid_quad_vbo = self.ctx.buffer(quad.tobytes())
        self._grid_quad_vao = self.ctx.vertex_array(
            self.grid_program,
            [(self._grid_quad_vbo, '2f', 'in_pos')],
        )

    def build_bounding_box(self, bounds_min: np.ndarray, bounds_max: np.ndarray,
                           color: tuple = (0.5, 0.8, 1.0, 0.6)):
        """Build VBO for a bounding box wireframe.

        Reuses the VBO when the byte size doesn't change (the common
        case — only vertex positions move, not the vertex count), so
        the per-selection / per-camera-fit overlay rebuild avoids a
        GL buffer allocate + VAO recreate churn.
        """
        x0, y0, z0 = bounds_min
        x1, y1, z1 = bounds_max
        c = color

        # 12 edges of a box = 24 vertices
        verts = np.array([
            # Bottom face
            x0, y0, z0, *c,  x1, y0, z0, *c,
            x1, y0, z0, *c,  x1, y0, z1, *c,
            x1, y0, z1, *c,  x0, y0, z1, *c,
            x0, y0, z1, *c,  x0, y0, z0, *c,
            # Top face
            x0, y1, z0, *c,  x1, y1, z0, *c,
            x1, y1, z0, *c,  x1, y1, z1, *c,
            x1, y1, z1, *c,  x0, y1, z1, *c,
            x0, y1, z1, *c,  x0, y1, z0, *c,
            # Vertical edges
            x0, y0, z0, *c,  x0, y1, z0, *c,
            x1, y0, z0, *c,  x1, y1, z0, *c,
            x1, y0, z1, *c,  x1, y1, z1, *c,
            x0, y0, z1, *c,  x0, y1, z1, *c,
        ], dtype=np.float32)

        data = verts.tobytes()
        if self._bbox_vbo is not None and len(data) == self._bbox_vbo_cap:
            # Size matches → just rewrite the existing VBO and keep the VAO.
            self._bbox_vbo.write(data)
            return
        if self._bbox_vbo is not None:
            self._bbox_vbo.release()
            self._bbox_vao.release()
        self._bbox_vbo = self.ctx.buffer(data)
        self._bbox_vbo_cap = len(data)
        self._bbox_vao = self.ctx.vertex_array(
            self.program,
            [(self._bbox_vbo, '3f 4f', 'in_position', 'in_color')],
        )

    def build_grid(self, bounds_min: np.ndarray, bounds_max: np.ndarray,
                   color: tuple = (0.3, 0.3, 0.3, 0.4)):
        """Remember the Y level for the infinite grid plane.

        ``color`` and ``bounds_max`` are kept for API compatibility but
        are no longer used — the grid is now a shader-based infinite
        plane (see ``grid_infinite.frag``).
        """
        self._grid_y_plane = float(bounds_min[1])
        # Keep a very large extent so the quad always reaches the horizon
        # at any realistic zoom level. The fragment shader's distance fade
        # hides the edge.
        extent = bounds_max - bounds_min
        max_extent = float(max(extent[0], extent[1], extent[2], 1.0))
        self._grid_extent = max(max_extent * 1000.0, 1.0e4)

    def build_depth_plane(self, center: np.ndarray, right: np.ndarray,
                          up: np.ndarray, half_size: float,
                          color: tuple = (0.95, 0.6, 0.15, 0.85)):
        """Build a wireframe rectangle centered on `center` spanning
        `half_size` along `right` and `up`.

        Used to visualize the selection depth cutoff plane — lets the
        user actually see where their Ctrl+scroll depth limit lives.
        """
        right = right * half_size
        up = up * half_size
        p0 = center - right - up
        p1 = center + right - up
        p2 = center + right + up
        p3 = center - right + up
        c = color
        verts = np.array([
            *p0, *c,  *p1, *c,
            *p1, *c,  *p2, *c,
            *p2, *c,  *p3, *c,
            *p3, *c,  *p0, *c,
            # Diagonal cross-hairs make the orientation clear
            *p0, *c,  *p2, *c,
            *p1, *c,  *p3, *c,
        ], dtype=np.float32)
        data = verts.tobytes()
        if self._depth_plane_vbo is not None and len(data) == self._depth_plane_vbo_cap:
            self._depth_plane_vbo.write(data)
            return
        if self._depth_plane_vbo is not None:
            self._depth_plane_vbo.release()
            self._depth_plane_vao.release()
        self._depth_plane_vbo = self.ctx.buffer(data)
        self._depth_plane_vbo_cap = len(data)
        self._depth_plane_vao = self.ctx.vertex_array(
            self.program,
            [(self._depth_plane_vbo, '3f 4f', 'in_position', 'in_color')],
        )

    def draw_depth_plane(self, mvp: np.ndarray):
        if not self._depth_plane_vao:
            return
        self.program['u_mvp'].write(_mvp_bytes(mvp))
        self._depth_plane_vao.render(moderngl.LINES)

    def _release_clip_planes(self):
        """Drop any existing clip-plane VBO/VAO. Safe to call multiple times."""
        vbo = getattr(self, '_clip_planes_vbo', None)
        vao = getattr(self, '_clip_planes_vao', None)
        if vbo is not None:
            try:
                vbo.release()
            except Exception:
                pass
        if vao is not None:
            try:
                vao.release()
            except Exception:
                pass
        self._clip_planes_vbo = None
        self._clip_planes_vao = None
        for attr in ('_clip_fill_vbo', '_clip_fill_vao'):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
                setattr(self, attr, None)

    def build_clip_planes(self, app):
        """Build three axis-aligned rectangles at the current clip cut
        faces for the selected cloud, so the user can see where each
        clip plane lives. Skips rebuild when clip state is unchanged.
        """
        if not (0 <= app.selected_index < len(app.entries)):
            self._release_clip_planes()
            self._clip_cache_key = None
            return
        entry = app.entries[app.selected_index]
        if entry.bounds_min is None:
            self._release_clip_planes()
            self._clip_cache_key = None
            return

        axis_enabled = tuple(bool(v) for v in
                             getattr(app, 'clip_axis_enabled',
                                     [True, True, True]))
        # Dirty check — skip rebuild if clip state hasn't changed
        cache_key = (
            app.selected_index,
            tuple(float(f) for f in app.clip_fraction_lo),
            tuple(float(f) for f in app.clip_fraction_hi),
            axis_enabled,
            tuple(float(v) for v in entry.bounds_min),
            tuple(float(v) for v in entry.bounds_max),
        )
        if cache_key == self._clip_cache_key:
            return
        self._clip_cache_key = cache_key

        # State changed — release old buffers and rebuild
        self._release_clip_planes()

        bmin = np.asarray(entry.bounds_min, dtype=np.float32)
        bmax = np.asarray(entry.bounds_max, dtype=np.float32)

        axis_colors = [
            (0.95, 0.35, 0.35, 0.55),  # X = coral
            (0.35, 0.90, 0.45, 0.55),  # Y = green
            (0.45, 0.65, 0.98, 0.55),  # Z = blue
        ]
        verts = []
        fill_tris = []
        for axis in range(3):
            if not axis_enabled[axis]:
                continue
            frac_lo = float(app.clip_fraction_lo[axis])
            frac_hi = float(app.clip_fraction_hi[axis])

            lo = float(bmin[axis])
            hi = float(bmax[axis])
            span = hi - lo

            u_axis = (axis + 1) % 3
            v_axis = (axis + 2) % 3
            u0 = float(bmin[u_axis])
            u1 = float(bmax[u_axis])
            v0 = float(bmin[v_axis])
            v1 = float(bmax[v_axis])

            # Draw a plane for each active side (lo and/or hi)
            for frac, pos in [(frac_lo, lo + span * frac_lo),
                              (frac_hi, hi - span * frac_hi)]:
                if frac <= 1e-4:
                    continue

                def _point(uu, vv, _pos=pos, _axis=axis, _u=u_axis, _v=v_axis):
                    p = [0.0, 0.0, 0.0]
                    p[_axis] = _pos
                    p[_u] = uu
                    p[_v] = vv
                    return p

                c = axis_colors[axis]
                p00 = _point(u0, v0)
                p10 = _point(u1, v0)
                p11 = _point(u1, v1)
                p01 = _point(u0, v1)
                # Four edges of the rectangle (LINES)
                for a, b in ((p00, p10), (p10, p11), (p11, p01), (p01, p00)):
                    verts.extend([*a, *c, *b, *c])

                # Filled quad (two TRIANGLES) — very light opacity
                cf = (c[0], c[1], c[2], 0.08)
                fill_tris.extend([*p00, *cf, *p10, *cf, *p11, *cf])
                fill_tris.extend([*p00, *cf, *p11, *cf, *p01, *cf])

        if not verts and not fill_tris:
            return

        # Lines VBO/VAO
        if verts:
            data = np.array(verts, dtype=np.float32)
            self._clip_planes_vbo = self.ctx.buffer(data.tobytes())
            self._clip_planes_vao = self.ctx.vertex_array(
                self.program,
                [(self._clip_planes_vbo, '3f 4f', 'in_position', 'in_color')],
            )
        # Filled triangles VBO/VAO
        if fill_tris:
            fdata = np.array(fill_tris, dtype=np.float32)
            self._clip_fill_vbo = self.ctx.buffer(fdata.tobytes())
            self._clip_fill_vao = self.ctx.vertex_array(
                self.program,
                [(self._clip_fill_vbo, '3f 4f', 'in_position', 'in_color')],
            )

    def draw_clip_planes(self, mvp: np.ndarray, show_fill: bool = False):
        """Draw clip plane overlays.

        With ``show_fill=False`` (default) only the wireframe edges are
        drawn — a persistent subtle reference. With ``show_fill=True``
        the semi-transparent filled quads are also rendered (used while
        the user is actively dragging a clip slider).
        """
        self.program['u_mvp'].write(_mvp_bytes(mvp))
        # Filled quads (only while dragging)
        if show_fill:
            fill_vao = getattr(self, '_clip_fill_vao', None)
            if fill_vao is not None:
                self.ctx.enable(moderngl.BLEND)
                fill_vao.render(moderngl.TRIANGLES)
        # Wireframe edges (always)
        vao = getattr(self, '_clip_planes_vao', None)
        if vao is None:
            return
        vao.render(moderngl.LINES)

    def draw_bbox(self, mvp: np.ndarray):
        """Draw the bounding box wireframe."""
        if not self._bbox_vao:
            return
        self.program['u_mvp'].write(_mvp_bytes(mvp))
        self._bbox_vao.render(moderngl.LINES)

    def draw_grid(self, mvp: np.ndarray):
        """Compatibility shim: legacy callers that only have the MVP
        can no longer reach the new infinite grid, which needs view +
        proj + camera position separately. No-op here; call
        ``draw_grid_infinite`` instead.
        """
        return

    def draw_grid_infinite(self, view: np.ndarray, proj: np.ndarray,
                           camera_pos: np.ndarray):
        """Draw the shader-based infinite adaptive grid.

        View and projection are uploaded separately so the shader can
        compute world-space derivatives for anti-aliasing and find
        the quad-plane intersection without losing precision.
        """
        if self._grid_quad_vao is None:
            return
        prog = self.grid_program
        prog['u_view'].write(_mvp_bytes(view))
        prog['u_proj'].write(_mvp_bytes(proj))
        prog['u_y_plane'].value = float(self._grid_y_plane)
        cam_pos = np.asarray(camera_pos, dtype=np.float32).reshape(3)
        prog['u_camera_pos'].value = (float(cam_pos[0]),
                                       float(cam_pos[1]),
                                       float(cam_pos[2]))
        prog['u_extent'].value = float(self._grid_extent)
        self._grid_quad_vao.render(moderngl.TRIANGLES)

    def release(self):
        if self._bbox_vbo:
            self._bbox_vbo.release()
            self._bbox_vao.release()
        if self._grid_vbo:
            self._grid_vbo.release()
            self._grid_vao.release()
        if self._grid_quad_vbo:
            self._grid_quad_vbo.release()
            self._grid_quad_vao.release()
        if self._depth_plane_vbo:
            self._depth_plane_vbo.release()
            self._depth_plane_vao.release()
        self._release_clip_planes()
        if self.program:
            self.program.release()
        if getattr(self, 'grid_program', None):
            self.grid_program.release()
