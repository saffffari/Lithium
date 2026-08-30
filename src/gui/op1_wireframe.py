"""OP-1 inspired 3D wireframe primitive widgets.

Small ImGui draw-list widgets that render 3D primitives (cube, sphere,
pyramid, torus) as wireframes using pure numpy projection. No moderngl
involved — these are UI chrome, not scene geometry.

Projection: each widget defines a local model matrix, applies a fixed
isometric view, and projects to the widget's screen rect. All vertices
are computed per-frame in numpy (fast for these small primitives).
"""

import math
import time
import numpy as np
import imgui

from src.gui.theme import *


# ------------------------------------------------------------------
# Shared projection helper
# ------------------------------------------------------------------

def _make_iso_view(azimuth: float = math.pi * 0.25,
                   elevation: float = math.pi * 0.18) -> np.ndarray:
    """Return a 3x3 rotation matrix for an isometric-ish view."""
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    # Rotate around Y, then around X
    ry = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, ce, se], [0, -se, ce]], dtype=np.float32)
    return rx @ ry


def _project(verts_3d: np.ndarray, cx: float, cy: float,
             scale: float, view: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply view rotation and orthographic projection to a (N,3) array.

    Returns (screen_xy, depths) where screen_xy is (N,2) and depths is (N,)
    in view-space Z (higher = closer to camera for painter sorting).
    """
    rotated = verts_3d @ view.T  # (N,3)
    screen = np.empty((len(rotated), 2), dtype=np.float32)
    screen[:, 0] = cx + rotated[:, 0] * scale
    screen[:, 1] = cy - rotated[:, 1] * scale  # flip Y for screen space
    depths = rotated[:, 2]
    return screen, depths


# ------------------------------------------------------------------
# Cube geometry
# ------------------------------------------------------------------

CUBE_VERTS = np.array([
    [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],  # back face (z=-1)
    [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1],  # front face (z=+1)
], dtype=np.float32)

CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # back
    (4, 5), (5, 6), (6, 7), (7, 4),  # front
    (0, 4), (1, 5), (2, 6), (3, 7),  # connectors
]

# Faces as (name, vertex_indices, normal).
# Winding: counter-clockwise when looking down the +normal direction.
CUBE_FACES = [
    ('front',  [4, 5, 6, 7], np.array([0, 0,  1], dtype=np.float32)),
    ('back',   [1, 0, 3, 2], np.array([0, 0, -1], dtype=np.float32)),
    ('right',  [5, 1, 2, 6], np.array([1, 0, 0], dtype=np.float32)),
    ('left',   [0, 4, 7, 3], np.array([-1, 0, 0], dtype=np.float32)),
    ('top',    [7, 6, 2, 3], np.array([0, 1, 0], dtype=np.float32)),
    ('bottom', [0, 1, 5, 4], np.array([0, -1, 0], dtype=np.float32)),
]


def _point_in_quad(px: float, py: float, poly) -> bool:
    """Ray-cast point-in-polygon test for a 4-vertex quad."""
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
                (px < (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi):
            inside = not inside
        j = i
    return inside


# ------------------------------------------------------------------
# Interactive camera cube
# ------------------------------------------------------------------

def op1_camera_cube(camera, active_preset: str | None = None,
                    size_units: float = 6.0,
                    width: float | None = None) -> str | None:
    """Draw a wireframe cube. Returns a preset name if a face was clicked.

    Faces:
      - top face    -> 'top'
      - front face  -> 'front'
      - right face  -> 'right'
      - iso diag    -> 'iso' (handled by clicking the center/edge vertex)

    The face corresponding to `active_preset` is filled with a subtle
    OP-1 blue tint so the user can see which orthographic view they're in.

    ``width`` optionally overrides the horizontal span used to center
    the cube (defaults to the remaining content-region width). Use this
    when placing multiple cubes side-by-side within the same panel.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = width if width is not None else imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    # Never wider than the column it sits in: the panel hands us the
    # leftover sidebar *height*, which on a tall window can be several
    # times the sidebar width and made the cube balloon past the rail.
    size = min(th * size_units, w * 0.9)
    # Center the cube horizontally; leave some vertical padding
    cx = wx + w * 0.5
    cy = wy + size * 0.55
    scale = size * 0.28

    # Spin the cube to match the live viewport camera so its pose
    # always mirrors what the user sees in the 3D canvas.
    cam_az = float(getattr(camera, 'azimuth', math.radians(35)))
    cam_el = float(getattr(camera, 'elevation', math.radians(25)))
    view = _make_iso_view(azimuth=cam_az, elevation=cam_el)
    screen, depths = _project(CUBE_VERTS, cx, cy, scale, view)

    # Determine which faces are visible (facing camera) via face normal dot +z
    # In our view, camera looks along -z after view rotation, so a face is
    # visible if its rotated normal has z > 0.
    visible_faces = []
    for name, idx, normal in CUBE_FACES:
        rn = view @ normal
        if rn[2] > 0.01:
            # Compute average depth for sorting
            avg_z = depths[idx].mean()
            visible_faces.append((avg_z, name, idx))
    visible_faces.sort(key=lambda t: t[0])  # back-to-front

    # Determine mouse hover
    mxp, myp = imgui.get_mouse_pos()
    hovered_face: str | None = None
    in_widget = (wx <= mxp <= wx + w and wy <= myp <= wy + size * 1.2)

    if in_widget:
        # Check visible faces in front-to-back order for accurate hover
        for _, name, idx in reversed(visible_faces):
            poly = [tuple(screen[i]) for i in idx]
            if _point_in_quad(mxp, myp, poly):
                hovered_face = name
                break

    # Map faces to camera presets
    face_to_preset = {
        'top': 'top',
        'front': 'front',
        'right': 'right',
        'left': 'left',
        'back': 'back',
        'bottom': 'bottom',
    }

    # Draw filled faces for hovered face only — the active preset face
    # no longer gets a fill (the alien inside the cube is the indicator).
    for _, name, idx in visible_faces:
        poly_pts = [tuple(screen[i]) for i in idx]
        fill_col = None
        if name == hovered_face and active_preset != face_to_preset.get(name):
            fill_col = col32((OP1_WHITE[0], OP1_WHITE[1], OP1_WHITE[2], 0.15))
        if fill_col is not None:
            # Split the quad into two triangles for add_triangle_filled
            dl.add_triangle_filled(
                poly_pts[0][0], poly_pts[0][1],
                poly_pts[1][0], poly_pts[1][1],
                poly_pts[2][0], poly_pts[2][1],
                fill_col,
            )
            dl.add_triangle_filled(
                poly_pts[0][0], poly_pts[0][1],
                poly_pts[2][0], poly_pts[2][1],
                poly_pts[3][0], poly_pts[3][1],
                fill_col,
            )

    # Grid subdivision on visible faces — light midlines so the cube
    # reads as a spatial reference grid without being heavy.
    grid_col = col32((OP1_BLUE[0], OP1_BLUE[1], OP1_BLUE[2], 0.18))
    for _, name, idx in visible_faces:
        p = [screen[i] for i in idx]
        # Two midlines per face (2×2 subdivision)
        for t in (0.5,):
            # Midline along edge 0→1 / 3→2
            mx0 = p[0][0] + (p[1][0] - p[0][0]) * t
            my0 = p[0][1] + (p[1][1] - p[0][1]) * t
            mx1 = p[3][0] + (p[2][0] - p[3][0]) * t
            my1 = p[3][1] + (p[2][1] - p[3][1]) * t
            dl.add_line(mx0, my0, mx1, my1, grid_col, 1.0)
            # Midline along edge 0→3 / 1→2
            mx0 = p[0][0] + (p[3][0] - p[0][0]) * t
            my0 = p[0][1] + (p[3][1] - p[0][1]) * t
            mx1 = p[1][0] + (p[2][0] - p[1][0]) * t
            my1 = p[1][1] + (p[2][1] - p[1][1]) * t
            dl.add_line(mx0, my0, mx1, my1, grid_col, 1.0)

    # Draw wireframe edges. Edges adjacent to at least one front-facing
    # face draw solid; edges whose adjacent faces are both back-facing
    # (hidden) draw as a dotted line at reduced alpha so the cube reads
    # as a real 3D object instead of a flat wire octagon.
    edge_col = col32(OP1_BLUE)
    hidden_col = col32((OP1_BLUE[0], OP1_BLUE[1], OP1_BLUE[2], 0.45))
    edge_thickness = 2.0
    # Build a map: edge -> list of adjacent face indices based on
    # shared vertex pairs. Cheap because we only have 12 edges / 6 faces.
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, (_name, idx, _n) in enumerate(CUBE_FACES):
        for k in range(4):
            a, b = idx[k], idx[(k + 1) % 4]
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append(fi)
    # Precompute per-face visibility once.
    face_visible = []
    for _name, _idx, normal in CUBE_FACES:
        rn = view @ normal
        face_visible.append(rn[2] > 0.0)

    def _dash(x0, y0, x1, y1, col, thick):
        # Roughly 4px-on / 3px-off dash pattern driven by segment length.
        dx = x1 - x0
        dy = y1 - y0
        length = math.hypot(dx, dy)
        if length <= 0.5:
            return
        on_len = 4.0
        off_len = 3.0
        step = on_len + off_len
        n = max(1, int(length / step))
        inv = 1.0 / length
        ux = dx * inv
        uy = dy * inv
        for i in range(n + 1):
            s0 = i * step
            s1 = min(s0 + on_len, length)
            if s0 >= length:
                break
            dl.add_line(x0 + ux * s0, y0 + uy * s0,
                        x0 + ux * s1, y0 + uy * s1, col, thick)

    for a, b in CUBE_EDGES:
        x0, y0 = screen[a]
        x1, y1 = screen[b]
        adj = edge_faces.get((min(a, b), max(a, b)), [])
        visible = any(face_visible[fi] for fi in adj)
        if visible:
            dl.add_line(x0, y0, x1, y1, edge_col, edge_thickness)
        else:
            _dash(x0, y0, x1, y1, hidden_col, edge_thickness)

    # Draw face labels on the three visible faces
    label_col = col32(OP1_GRAY)
    for _, name, idx in visible_faces:
        if name not in ('top', 'front', 'right'):
            continue
        pts = screen[idx]
        lx = pts[:, 0].mean()
        ly = pts[:, 1].mean()
        label_str = {'top': 'T', 'front': 'F', 'right': 'R'}[name]
        lbl_size = imgui.calc_text_size(label_str)
        dl.add_text(lx - lbl_size[0] * 0.5, ly - th * 0.5,
                    label_col, label_str)

    # Reserve the space
    imgui.dummy(w, size * 1.2)

    # Handle click — single click jumps to face, double click jumps
    # to opposite, RMB click *anywhere in the widget* jumps back to
    # the ISO home view (so the user always has a one-click escape
    # from any plane back to free perspective).
    OPPOSITE = {
        'top': 'bottom', 'bottom': 'top',
        'front': 'back', 'back': 'front',
        'left': 'right', 'right': 'left',
    }
    clicked_preset: str | None = None
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
        clicked_preset = 'iso'
    elif hovered_face is not None and imgui.is_item_hovered():
        face_preset = face_to_preset.get(hovered_face)
        if imgui.is_mouse_double_clicked(0):
            clicked_preset = OPPOSITE.get(face_preset, face_preset)
        elif imgui.is_mouse_clicked(0):
            clicked_preset = face_preset

    # Project the bottom-face center (0, -1, 0) so the caller can anchor
    # decorations (e.g. the alien) to the correct rotated position.
    bottom_pt = np.array([[0.0, -1.0, 0.0]], dtype=np.float32)
    bottom_screen, _ = _project(bottom_pt, cx, cy, scale, view)
    bottom_center = (float(bottom_screen[0, 0]), float(bottom_screen[0, 1]))

    return clicked_preset, bottom_center


def op1_camera_planes(camera, active_preset: str | None = None,
                      size_units: float = 6.0) -> str | None:
    """Three perpendicular intersecting planes (XY, YZ, XZ) as a view picker.

    Returns a preset name if a plane is clicked:
      XY plane  -> 'top'   (click near the horizontal plane)
      YZ plane  -> 'right' (click near the right-facing plane)
      XZ plane  -> 'front' (click near the front-facing plane)
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    size = th * size_units
    cx = wx + w * 0.5
    cy = wy + size * 0.55
    scale = size * 0.32

    view = _make_iso_view(azimuth=math.radians(35), elevation=math.radians(25))

    # Three unit squares lying on the XY, YZ, and XZ planes, each
    # centered at the origin. They intersect along the three axes.
    plane_verts_xy = np.array([
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],
    ], dtype=np.float32)
    plane_verts_yz = np.array([
        [0, -1, -1], [0, 1, -1], [0, 1, 1], [0, -1, 1],
    ], dtype=np.float32)
    plane_verts_xz = np.array([
        [-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1],
    ], dtype=np.float32)

    planes = [
        # (name, verts, preset, accent color, normal)
        ('xy', plane_verts_xy, 'top',   OP1_GREEN,
         np.array([0, 0, 1], dtype=np.float32)),
        ('yz', plane_verts_yz, 'right', OP1_RED,
         np.array([1, 0, 0], dtype=np.float32)),
        ('xz', plane_verts_xz, 'front', OP1_BLUE,
         np.array([0, 1, 0], dtype=np.float32)),
    ]

    # Project + painter-sort back to front by average depth.
    projected = []
    for name, verts, preset, color, normal in planes:
        screen, depths = _project(verts, cx, cy, scale, view)
        avg_z = float(depths.mean())
        projected.append((avg_z, name, screen, preset, color))
    projected.sort(key=lambda t: t[0])

    # Mouse detection — check in reverse (front-most first).
    mxp, myp = imgui.get_mouse_pos()
    in_widget = (wx <= mxp <= wx + w and wy <= myp <= wy + size * 1.2)
    hovered_preset: str | None = None
    if in_widget:
        for _, _name, screen, preset, _color in reversed(projected):
            poly = [tuple(screen[i]) for i in range(4)]
            if _point_in_quad(mxp, myp, poly):
                hovered_preset = preset
                break

    # Draw each plane: translucent fill + outline.
    for _z, name, screen, preset, color in projected:
        is_active = (active_preset == preset)
        is_hovered = (hovered_preset == preset)
        fill_alpha = 0.55 if is_active else (0.28 if is_hovered else 0.16)
        fill_col = col32((color[0], color[1], color[2], fill_alpha))
        dl.add_triangle_filled(
            screen[0, 0], screen[0, 1],
            screen[1, 0], screen[1, 1],
            screen[2, 0], screen[2, 1], fill_col)
        dl.add_triangle_filled(
            screen[0, 0], screen[0, 1],
            screen[2, 0], screen[2, 1],
            screen[3, 0], screen[3, 1], fill_col)
        stroke_col = col32((color[0], color[1], color[2],
                            1.0 if is_active else 0.85))
        for i in range(4):
            j = (i + 1) % 4
            dl.add_line(screen[i, 0], screen[i, 1],
                        screen[j, 0], screen[j, 1], stroke_col, 1.4)

    imgui.dummy(w, size * 1.2)

    clicked_preset: str | None = None
    if hovered_preset is not None and imgui.is_item_hovered():
        if imgui.is_mouse_clicked(0):
            clicked_preset = hovered_preset
    return clicked_preset


# ------------------------------------------------------------------
# Clip plane cube (axis selector + slider + invert)
# ------------------------------------------------------------------

# Module-level drag flag for the clip slider, mirrors the _gauge_drag_key
# idiom in op1_widgets.py.
_clip_slider_drag: bool = False


def op1_clip_cube(app) -> bool:
    """Axis selector cube + horizontal slider + invert button.

    Click one of the three visible cube faces (X=right, Y=top, Z=front) to
    select which axis to clip. The slider on the right shows / adjusts
    ``app.clip_fraction[axis]`` for the selected axis. The INVERT button
    flips ``app.clip_invert[axis]``. All mutations call
    ``app._sync_clip_box()`` so the shader uniforms stay in sync.

    Stores the currently-selected axis on ``app._clip_active_axis``
    (0=X, 1=Y, 2=Z). Defaults to 0 if the attribute is missing.
    """
    global _clip_slider_drag
    from src.gui.scale import s as _s

    changed = False
    if not hasattr(app, '_clip_active_axis'):
        app._clip_active_axis = 0

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    panel_h = th * 6.0
    # Flat 95%-black panel — no outline.
    dl.add_rect_filled(wx, wy, wx + w, wy + panel_h,
                       col32((0.05, 0.05, 0.05, 1.0)), 6.0)

    margin = th * 0.4
    cube_w = th * 5.2
    cube_h = panel_h - margin * 2
    cube_x = wx + margin
    cube_y = wy + margin

    cx = cube_x + cube_w * 0.5
    cy = cube_y + cube_h * 0.5
    scale = min(cube_w, cube_h) * 0.32

    view = _make_iso_view(azimuth=math.radians(35), elevation=math.radians(25))
    screen, depths = _project(CUBE_VERTS, cx, cy, scale, view)

    # Visible faces (facing camera).
    visible_faces = []
    for name, idx, normal in CUBE_FACES:
        rn = view @ normal
        if rn[2] > 0.01:
            avg_z = depths[idx].mean()
            visible_faces.append((avg_z, name, idx))
    visible_faces.sort(key=lambda t: t[0])  # back-to-front

    # Map cube face → clip axis. right=R(X), top=S(Y), front=A(Z).
    face_to_axis = {'right': 0, 'top': 1, 'front': 2}
    axis_colors = [CORAL, OP1_GREEN, OP1_BLUE]
    axis_labels = ['R', 'S', 'A']

    mxp, myp = imgui.get_mouse_pos()
    in_cube = (cube_x <= mxp <= cube_x + cube_w and
               cube_y <= myp <= cube_y + cube_h)
    hovered_axis: int | None = None
    if in_cube:
        # Front-to-back for accurate top-face hover.
        for _, name, idx in reversed(visible_faces):
            if name not in face_to_axis:
                continue
            poly = [tuple(screen[i]) for i in idx]
            if _point_in_quad(mxp, myp, poly):
                hovered_axis = face_to_axis[name]
                break

    active_axis = app._clip_active_axis

    # Tint the active face (strong) and hovered face (subtle).
    for _, name, idx in visible_faces:
        poly = [tuple(screen[i]) for i in idx]
        axis = face_to_axis.get(name)
        fill = None
        if axis is not None and axis == active_axis:
            c = axis_colors[axis]
            fill = col32((c[0], c[1], c[2], 0.55))
        elif axis is not None and axis == hovered_axis:
            fill = col32((1.0, 1.0, 1.0, 0.14))
        if fill is not None:
            dl.add_triangle_filled(
                poly[0][0], poly[0][1],
                poly[1][0], poly[1][1],
                poly[2][0], poly[2][1],
                fill,
            )
            dl.add_triangle_filled(
                poly[0][0], poly[0][1],
                poly[2][0], poly[2][1],
                poly[3][0], poly[3][1],
                fill,
            )

    # Wireframe edges.
    edge_col = col32(OP1_BLUE)
    for a, b in CUBE_EDGES:
        x0, y0 = screen[a]
        x1, y1 = screen[b]
        dl.add_line(x0, y0, x1, y1, edge_col, 1.6)

    # Face labels X / Y / Z on the three visible axis faces.
    for _, name, idx in visible_faces:
        axis = face_to_axis.get(name)
        if axis is None:
            continue
        pts = screen[idx]
        lx = pts[:, 0].mean()
        ly = pts[:, 1].mean()
        lbl = axis_labels[axis]
        lbl_size = imgui.calc_text_size(lbl)
        lbl_col = OP1_WHITE if axis == active_axis else axis_colors[axis]
        dl.add_text(lx - lbl_size[0] * 0.5, ly - th * 0.5, col32(lbl_col), lbl)

    # Click cube to select axis.
    if in_cube and hovered_axis is not None and imgui.is_mouse_clicked(0):
        app._clip_active_axis = hovered_axis
        active_axis = hovered_axis
        changed = True

    # ---- Right side: slider + invert button ----
    ctrl_x = cube_x + cube_w + th * 0.6
    ctrl_w = (wx + w - margin) - ctrl_x
    if ctrl_w < th * 4:
        # Sidebar too narrow to fit controls — just reserve space.
        imgui.dummy(w, panel_h + _s(4))
        return changed

    color = axis_colors[active_axis]
    ax = active_axis

    # Dual range slider bar — lo from left, hi from right.
    slider_y = cube_y + th * 0.4
    slider_h = th * 1.9
    frac_lo = max(0.0, min(1.0, float(app.clip_fraction_lo[ax])))
    frac_hi = max(0.0, min(1.0, float(app.clip_fraction_hi[ax])))
    dl.add_rect_filled(ctrl_x, slider_y, ctrl_x + ctrl_w, slider_y + slider_h,
                       col32((0.04, 0.04, 0.04, 1.0)), 3.0)
    # Dim clipped regions
    dim_col = col32((color[0] * 0.3, color[1] * 0.3, color[2] * 0.3, 0.4))
    lo_px = ctrl_x + ctrl_w * frac_lo
    hi_px = ctrl_x + ctrl_w * (1.0 - frac_hi)
    if frac_lo > 0.001:
        dl.add_rect_filled(ctrl_x, slider_y, lo_px, slider_y + slider_h,
                           dim_col, 3.0)
    if frac_hi > 0.001:
        dl.add_rect_filled(hi_px, slider_y, ctrl_x + ctrl_w, slider_y + slider_h,
                           dim_col, 3.0)
    # Bright kept slab
    dl.add_rect_filled(lo_px, slider_y, hi_px, slider_y + slider_h,
                       col32((color[0], color[1], color[2], 0.55)), 3.0)
    dl.add_rect(ctrl_x, slider_y, ctrl_x + ctrl_w, slider_y + slider_h,
                col32(color), 3.0, 0, 1.4)
    # Label: "X  lo% — hi%"
    lbl = f"{axis_labels[ax]}  {frac_lo * 100:.0f}–{frac_hi * 100:.0f}%"
    lbl_size = imgui.calc_text_size(lbl)
    dl.add_text(ctrl_x + (ctrl_w - lbl_size[0]) * 0.5,
                slider_y + (slider_h - th) * 0.5, col32(OP1_WHITE), lbl)

    in_slider = (ctrl_x <= mxp <= ctrl_x + ctrl_w and
                 slider_y <= myp <= slider_y + slider_h)
    if not _clip_slider_drag and in_slider and imgui.is_mouse_clicked(0):
        dist_lo = abs(mxp - lo_px)
        dist_hi = abs(mxp - hi_px)
        app._clip_cube_side = 'lo' if dist_lo <= dist_hi else 'hi'
        _clip_slider_drag = True
    if _clip_slider_drag:
        if imgui.is_mouse_down(0):
            side = getattr(app, '_clip_cube_side', 'lo')
            if side == 'lo':
                new_frac = (mxp - ctrl_x) / max(ctrl_w, 1.0)
                new_frac = max(0.0, min(1.0, new_frac))
                if abs(new_frac - frac_lo) > 1e-4:
                    app.clip_fraction_lo[ax] = new_frac
                    app._sync_clip_box()
                    changed = True
            else:
                new_frac = (ctrl_x + ctrl_w - mxp) / max(ctrl_w, 1.0)
                new_frac = max(0.0, min(1.0, new_frac))
                if abs(new_frac - frac_hi) > 1e-4:
                    app.clip_fraction_hi[ax] = new_frac
                    app._sync_clip_box()
                    changed = True
        else:
            _clip_slider_drag = False

    # Double-click resets both sides
    if in_slider and imgui.is_mouse_double_clicked(0):
        app.clip_fraction_lo[ax] = 0.0
        app.clip_fraction_hi[ax] = 0.0
        app._sync_clip_box()
        changed = True

    imgui.dummy(w, panel_h + _s(4))
    return changed


# ------------------------------------------------------------------
# EQ-style clipping panel — kitbashed from Figma node 2302:312.
# ------------------------------------------------------------------

_clip_fader_drag_axis: int | None = None


def op1_clip_fader_panel(app) -> bool:
    """Horizontal clipping panel — 3 horizontal faders with per-axis
    toggle buttons + isometric plane picker on the right.

    Layout:
      [R] [====== horizontal fader ======]  [         ]
      [S] [====== horizontal fader ======]  [ planes  ]
      [A] [====== horizontal fader ======]  [         ]

      Sliders occupy the left 2/3, plane picker the right 1/3.

    Each R/S/A label button toggles that axis clip on/off (filled
    background = active). The horizontal fader has lo/hi thumbs.
    Double-click a fader to reset that axis.
    """
    global _clip_fader_drag_axis
    from src.gui.scale import s as _s
    from src.gui.op1_widgets import _KNOB_TOP_MARK_COLOR

    changed = False

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    # Ensure per-axis enable state exists
    if not hasattr(app, 'clip_axis_enabled'):
        app.clip_axis_enabled = [True, True, True]

    # Layout constants
    margin_x = th * 0.8
    margin_y = th * 0.5
    row_h = th * 2.0
    row_gap = th * 0.25
    label_w = th * 2.0
    label_gap = th * 0.4
    section_gap = th * 0.6  # gap between sliders and plane picker

    fader_rows_h = row_h * 3 + row_gap * 2
    panel_h = margin_y * 2 + fader_rows_h

    # Flat 95%-black panel
    dl.add_rect_filled(wx, wy, wx + w, wy + panel_h,
                       col32((0.05, 0.05, 0.05, 1.0)), 10.0)

    inner_x = wx + margin_x
    inner_w = w - margin_x * 2

    # Split: sliders left 2/3, plane picker right 1/3
    planes_section_w = inner_w * (1.0 / 3.0)
    slider_section_w = inner_w - planes_section_w - section_gap

    track_x0 = inner_x + label_w + label_gap
    track_w = slider_section_w - label_w - label_gap
    thumb_half_w = th * 0.4

    axis_colors = [OP1_RED, OP1_GREEN, OP1_BLUE]
    axis_labels = ['R', 'S', 'A']
    _fader_track_col = col32((0.32, 0.32, 0.32, 1.0))
    _fader_n_ticks = 8
    _fader_track_thick = 2.0
    _fader_fill_thick = 2.6

    mxp, myp = imgui.get_mouse_pos()
    clip_on = bool(getattr(app, 'clip_enabled', True))

    if not hasattr(app, '_clip_active_axis'):
        app._clip_active_axis = 0
    active_axis = int(app._clip_active_axis)

    # ---- 3 horizontal fader rows ----
    for axis in range(3):
        row_y = wy + margin_y + axis * (row_h + row_gap)
        color = axis_colors[axis]
        color_u32 = col32(color)
        dim_u32 = col32((color[0] * 0.5, color[1] * 0.5, color[2] * 0.5, 0.7))
        lbl = axis_labels[axis]
        axis_on = app.clip_axis_enabled[axis]

        # --- Label toggle button ---
        lbl_x = inner_x
        lbl_y = row_y
        lbl_rounding = 4.0
        if axis_on:
            # Filled background in axis color
            dl.add_rect_filled(lbl_x, lbl_y, lbl_x + label_w, lbl_y + row_h,
                               col32((color[0], color[1], color[2], 0.85)),
                               lbl_rounding)
            lbl_text_col = col32((0.05, 0.05, 0.05, 1.0))
        else:
            # Dim grey background
            dl.add_rect_filled(lbl_x, lbl_y, lbl_x + label_w, lbl_y + row_h,
                               col32((0.15, 0.15, 0.15, 1.0)), lbl_rounding)
            dl.add_rect(lbl_x, lbl_y, lbl_x + label_w, lbl_y + row_h,
                        col32((0.3, 0.3, 0.3, 1.0)), lbl_rounding, 0, 1.0)
            lbl_text_col = col32((0.5, 0.5, 0.5, 1.0))

        lbl_tw = imgui.calc_text_size(lbl)[0]
        dl.add_text(lbl_x + (label_w - lbl_tw) * 0.5,
                    lbl_y + (row_h - th) * 0.5, lbl_text_col, lbl)

        # Label button hit test
        if (lbl_x <= mxp <= lbl_x + label_w and lbl_y <= myp <= lbl_y + row_h
                and imgui.is_mouse_clicked(0)):
            app.clip_axis_enabled[axis] = not axis_on
            app._clip_active_axis = axis
            changed = True

        # --- Horizontal fader ---
        track_cy = row_y + row_h * 0.5

        # Dark background track
        dl.add_line(track_x0, track_cy, track_x0 + track_w, track_cy,
                    _fader_track_col, _fader_track_thick)

        frac_lo = max(0.0, min(1.0, float(app.clip_fraction_lo[axis])))
        frac_hi = max(0.0, min(1.0, float(app.clip_fraction_hi[axis])))
        lo_x = track_x0 + track_w * frac_lo       # lo thumb (from left)
        hi_x = track_x0 + track_w * (1.0 - frac_hi)  # hi thumb (from right)

        # Dim alpha when axis disabled
        draw_alpha = 1.0 if axis_on else 0.35

        if frac_lo > 0.001 or frac_hi > 0.001:
            # Clipped regions (dim)
            if frac_lo > 0.001:
                dl.add_line(track_x0, track_cy, lo_x, track_cy,
                            dim_u32, _fader_fill_thick)
            if frac_hi > 0.001:
                dl.add_line(hi_x, track_cy, track_x0 + track_w, track_cy,
                            dim_u32, _fader_fill_thick)
            # Kept slab (bright)
            dl.add_line(lo_x, track_cy, hi_x, track_cy,
                        color_u32, _fader_fill_thick)

        # Thumb bars (vertical)
        _bar_hw = th * 0.08
        _bar_hh = row_h * 0.35
        dl.add_rect_filled(lo_x - _bar_hw, track_cy - _bar_hh,
                           lo_x + _bar_hw, track_cy + _bar_hh,
                           color_u32, 1.0)
        dl.add_rect_filled(hi_x - _bar_hw, track_cy - _bar_hh,
                           hi_x + _bar_hw, track_cy + _bar_hh,
                           color_u32, 1.0)

        # Tick marks
        tick_hh = row_h * 0.12
        for i in range(_fader_n_ticks + 1):
            t = i / _fader_n_ticks
            tx = track_x0 + track_w * t
            dl.add_line(tx, track_cy - tick_hh, tx, track_cy + tick_hh,
                        _fader_track_col, 1.2)

        # Hit box for fader
        hit_x0 = track_x0 - thumb_half_w
        hit_x1 = track_x0 + track_w + thumb_half_w
        hit_y0 = row_y
        hit_y1 = row_y + row_h
        in_hit = (hit_x0 <= mxp <= hit_x1 and hit_y0 <= myp <= hit_y1)

        # Drag detection — grab whichever thumb is closer
        if _clip_fader_drag_axis is None and in_hit and imgui.is_mouse_clicked(0):
            # Don't start drag if clicking the label button area
            if mxp > lbl_x + label_w:
                dist_lo = abs(mxp - lo_x)
                dist_hi = abs(mxp - hi_x)
                _clip_fader_drag_axis = axis
                app._clip_fader_side = 'lo' if dist_lo <= dist_hi else 'hi'

        if _clip_fader_drag_axis == axis:
            if imgui.is_mouse_down(0):
                side = getattr(app, '_clip_fader_side', 'lo')
                if side == 'lo':
                    new_frac = (mxp - track_x0) / track_w
                    new_frac = max(0.0, min(1.0, new_frac))
                    if abs(new_frac - frac_lo) > 1e-5:
                        app.clip_fraction_lo[axis] = new_frac
                        changed = True
                else:
                    new_frac = (track_x0 + track_w - mxp) / track_w
                    new_frac = max(0.0, min(1.0, new_frac))
                    if abs(new_frac - frac_hi) > 1e-5:
                        app.clip_fraction_hi[axis] = new_frac
                        changed = True
            else:
                _clip_fader_drag_axis = None

        if in_hit and mxp > lbl_x + label_w and imgui.is_mouse_double_clicked(0):
            app.clip_fraction_lo[axis] = 0.0
            app.clip_fraction_hi[axis] = 0.0
            changed = True

    # ---- Plane picker (right of faders) ----
    planes_x0 = inner_x + slider_section_w + section_gap
    planes_cx = planes_x0 + planes_section_w * 0.5
    planes_cy = wy + margin_y + fader_rows_h * 0.5
    planes_r = min(planes_section_w * 0.45, fader_rows_h * 0.40)

    if planes_r > 2.0:
        cam_az = float(getattr(app.camera, 'azimuth', math.radians(35)))
        cam_el = float(getattr(app.camera, 'elevation', math.radians(25)))
        iso_view = _make_iso_view(azimuth=cam_az, elevation=cam_el)

        def _make_plane(axis_idx, offset):
            if axis_idx == 0:
                return np.array([[offset, -1, -1], [offset, 1, -1],
                                 [offset, 1, 1], [offset, -1, 1]], dtype=np.float32)
            elif axis_idx == 1:
                return np.array([[-1, offset, -1], [1, offset, -1],
                                 [1, offset, 1], [-1, offset, 1]], dtype=np.float32)
            else:
                return np.array([[-1, -1, offset], [1, -1, offset],
                                 [1, 1, offset], [-1, 1, offset]], dtype=np.float32)

        plane_specs = []
        for axis_idx in range(3):
            frac_lo = float(app.clip_fraction_lo[axis_idx])
            frac_hi = float(app.clip_fraction_hi[axis_idx])
            if frac_lo > 1e-4 or frac_hi > 1e-4:
                if frac_lo > 1e-4:
                    pos = -1.0 + 2.0 * frac_lo
                    plane_specs.append((axis_idx, _make_plane(axis_idx, pos)))
                if frac_hi > 1e-4:
                    pos = 1.0 - 2.0 * frac_hi
                    plane_specs.append((axis_idx, _make_plane(axis_idx, pos)))
            else:
                plane_specs.append((axis_idx, _make_plane(axis_idx, 0.0)))

        projected = []
        for axis, verts in plane_specs:
            screen, depths = _project(verts, planes_cx, planes_cy,
                                      planes_r, iso_view)
            avg_z = float(depths.mean())
            projected.append((avg_z, axis, screen))
        projected.sort(key=lambda t: t[0])

        hit_box = (planes_cx - planes_r * 1.1, planes_cy - planes_r * 1.1,
                   planes_cx + planes_r * 1.1, planes_cy + planes_r * 1.1)
        in_planes = (hit_box[0] <= mxp <= hit_box[2]
                     and hit_box[1] <= myp <= hit_box[3])

        hovered_axis: int | None = None
        if in_planes:
            for _z, axis, screen in reversed(projected):
                poly = [tuple(screen[i]) for i in range(4)]
                if _point_in_quad(mxp, myp, poly):
                    hovered_axis = axis
                    break

        for _z, axis, screen in projected:
            color = axis_colors[axis]
            frac_light = max(float(app.clip_fraction_lo[axis]),
                             float(app.clip_fraction_hi[axis]))
            base_alpha = 0.18 + 0.60 * frac_light
            if not clip_on:
                base_alpha *= 0.35
            if axis == hovered_axis:
                base_alpha = min(1.0, base_alpha + 0.15)
            if axis == active_axis:
                base_alpha = min(1.0, base_alpha + 0.15)
            fill = col32((color[0], color[1], color[2], base_alpha))
            dl.add_triangle_filled(
                screen[0, 0], screen[0, 1],
                screen[1, 0], screen[1, 1],
                screen[2, 0], screen[2, 1], fill)
            dl.add_triangle_filled(
                screen[0, 0], screen[0, 1],
                screen[2, 0], screen[2, 1],
                screen[3, 0], screen[3, 1], fill)
            stroke_alpha = 1.0 if axis == active_axis else 0.85
            stroke_w = 2.0 if axis == active_axis else 1.4
            stroke_col = col32((color[0], color[1], color[2], stroke_alpha))
            for i in range(4):
                j = (i + 1) % 4
                dl.add_line(screen[i, 0], screen[i, 1],
                            screen[j, 0], screen[j, 1], stroke_col, stroke_w)

        if (in_planes and hovered_axis is not None
                and imgui.is_mouse_clicked(0)):
            app._clip_active_axis = hovered_axis
            changed = True

    # Dragging a fader implicitly sets that axis as active
    if _clip_fader_drag_axis is not None:
        app._clip_active_axis = _clip_fader_drag_axis

    # Tell the viewport whether a clip slider is being dragged
    app._clip_dragging = (_clip_slider_drag or _clip_fader_drag_axis is not None)

    if changed:
        app._sync_clip_box()

    imgui.dummy(w, panel_h + _s(4))
    return changed


# ------------------------------------------------------------------
# Scaling wireframe sphere
# ------------------------------------------------------------------

def _build_three_ring_sphere(segments: int = 48):
    """Build three perpendicular rings (XY, XZ, YZ planes).

    Returns (verts, edges_list) where edges_list is a list of three
    edge arrays, one per ring.
    """
    verts = []
    rings = []

    # Ring A — XZ plane (equator, y = 0)
    start = len(verts)
    for s in range(segments):
        angle = 2.0 * math.pi * s / segments
        verts.append([math.cos(angle), 0.0, math.sin(angle)])
    rings.append([(start + s, start + (s + 1) % segments) for s in range(segments)])

    # Ring B — XY plane (meridian, z = 0)
    start = len(verts)
    for s in range(segments):
        angle = 2.0 * math.pi * s / segments
        verts.append([math.cos(angle), math.sin(angle), 0.0])
    rings.append([(start + s, start + (s + 1) % segments) for s in range(segments)])

    # Ring C — YZ plane (meridian, x = 0)
    start = len(verts)
    for s in range(segments):
        angle = 2.0 * math.pi * s / segments
        verts.append([0.0, math.cos(angle), math.sin(angle)])
    rings.append([(start + s, start + (s + 1) % segments) for s in range(segments)])

    return np.array(verts, dtype=np.float32), rings


_SPHERE_VERTS, _SPHERE_RINGS = _build_three_ring_sphere()


_sphere_drag_active: bool = False
# Idle-state sphere rotation memory — only advances when the user is
# actively interacting with the POINT SIZE panel.
_sphere_last_az: float | None = None
_sphere_last_time: float = 0.0


def op1_sphere_size_gauge(app, point_size: float, min_size: float = 0.005,
                           max_size: float = 10.0,
                           height_units: float = 5.4) -> tuple[bool, float]:
    """Wireframe sphere gauge with an OP-1 knob controller.

    Layout (left → right):
      1. Three-ring wireframe sphere that scales with point_size (display only)
      2. Circular knob (shared 270° style via op1_knob_draw) — the drag target
      3. Big orange numeric readout

    Only the knob is draggable. Double-click the knob or right-click
    anywhere on the panel to reset to 3.0 (the default point size).
    """
    global _sphere_drag_active
    changed = False

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    panel_h = th * height_units
    from src.gui.op1_widgets import (
        op1_number, draw_styled_rect, PANEL_ROUNDING, _rclick_in_rect,
        op1_knob_draw,
    )

    # Orange-tinted fill — POINT SIZE keeps its warm background.
    draw_styled_rect(dl, wx, wy, w, panel_h, OP1_ORANGE,
                     rounding=PANEL_ROUNDING, thickness=1.2,
                     colored_fill=True)

    # ------------------------------------------------------------------
    # Layout: sphere on far left, knob in the middle, number on the right.
    # Sized with generous panel-interior margins so nothing can graze the
    # outer border regardless of sidebar width.
    # ------------------------------------------------------------------
    # Sphere on left third, knob on right two-thirds
    sphere_cx = wx + w * 0.22
    sphere_cy = wy + panel_h * 0.5
    sphere_max_r = min(w * 0.16, panel_h * 0.36)

    knob_cx = wx + w * 0.68
    knob_cy = wy + panel_h * 0.5
    knob_radius = min(w * 0.12, panel_h * 0.33)

    # ------------------------------------------------------------------
    # Knob interaction — only the knob is draggable.
    # ------------------------------------------------------------------
    mxp, myp = imgui.get_mouse_pos()
    knob_hit_r = knob_radius * 1.6
    in_knob = ((mxp - knob_cx) ** 2 + (myp - knob_cy) ** 2) <= knob_hit_r ** 2

    if not _sphere_drag_active and in_knob and imgui.is_mouse_clicked(0):
        _sphere_drag_active = True

    if _sphere_drag_active:
        if imgui.is_mouse_down(0):
            dy = imgui.get_io().mouse_delta[1]
            if abs(dy) > 0.001:
                span = max(max_size - min_size, 1e-6)
                point_size = point_size - dy * 0.008 * span
                point_size = max(min_size, min(max_size, point_size))
                changed = True
        else:
            _sphere_drag_active = False

    # Double-click knob or right-click anywhere on the panel resets.
    if in_knob and imgui.is_mouse_double_clicked(0):
        point_size = max(min_size, min(max_size, 3.0))
        changed = True
    if _rclick_in_rect(wx, wy, w, panel_h):
        point_size = max(min_size, min(max_size, 3.0))
        changed = True

    # ------------------------------------------------------------------
    # Sphere visual — scales with the current value but not interactive.
    # ------------------------------------------------------------------
    span = max(max_size - min_size, 1e-6)
    frac = max(0.0, min(1.0, (point_size - min_size) / span))
    frac_eased = frac ** 0.6
    sphere_r = sphere_max_r * (0.22 + 0.75 * frac_eased)

    # Rotation only ticks while the user is interacting (hover over the
    # panel or actively dragging the knob). Idle-state frames reuse the
    # last-seen angle — we don't need an animated sphere 60 times a
    # second just for decorative polish when the user isn't looking.
    global _sphere_last_az, _sphere_last_time
    panel_hover = (wx <= mxp <= wx + w) and (wy <= myp <= wy + panel_h)
    now = time.time()
    if _sphere_drag_active or panel_hover or changed:
        _sphere_last_az = math.radians(35) + now * 0.15
        _sphere_last_time = now
    _sphere_az = _sphere_last_az if _sphere_last_az is not None else math.radians(35)
    view = _make_iso_view(azimuth=_sphere_az, elevation=math.radians(25))
    screen, _ = _project(_SPHERE_VERTS, sphere_cx, sphere_cy, sphere_r, view)

    ring_col = col32(OP1_ORANGE)
    for ring_edges in _SPHERE_RINGS:
        for a, b in ring_edges:
            dl.add_line(screen[a, 0], screen[a, 1],
                        screen[b, 0], screen[b, 1], ring_col, 1.8)

    dl.add_circle_filled(sphere_cx, sphere_cy,
                         max(2.0, sphere_r * 0.09), ring_col, 10)

    # ------------------------------------------------------------------
    # Knob — unified 270° style shared with every other OP-1 widget.
    # ------------------------------------------------------------------
    op1_knob_draw(dl, knob_cx, knob_cy, knob_radius, frac, OP1_ORANGE)

    # No numeric readout — knob + sphere only.

    imgui.dummy(w, panel_h + imgui.get_style().item_spacing[1])
    return changed, point_size
