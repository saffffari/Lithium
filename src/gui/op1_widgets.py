"""OP-1 inspired widget library for 3Photon.

All widgets follow the same visual language established by the RGB knob
panel: dark rounded panels on OP-1 BG color, thin arc sweeps, tick marks,
geometric icons, big display-font numerals, and channel-specific accent
colors from the Figma OP-1 Screen Graphics kit.

Design primitives:
    _op1_panel  — rounded dark panel with subtle border
    _op1_arc    — semicircular arc with tick marks and hand
    _op1_keycap — dark rounded button with colored border when active
    _op1_number — big display-font numeral centered in a rect
    _op1_tick_scale — horizontal tick scale for value sliders

Everything is drawn through the ImGui foreground/background draw lists and
positioned using font-metric units so it scales with DPI.
"""

import math
import random
import time
import imgui

from src.gui.theme import *


def _th():
    return imgui.get_text_line_height()


def _pad():
    return _th() * 0.15


def _rclick_in_rect(x: float, y: float, w: float, h: float) -> bool:
    """Return True if the user right-clicked inside the given screen rect."""
    if not imgui.is_mouse_clicked(1):
        return False
    mx, my = imgui.get_mouse_pos()
    return (x <= mx <= x + w and y <= my <= y + h)


def _tooltip_if_hovered(x: float, y: float, w: float, h: float, text: str):
    """Show a tooltip when the mouse is inside the given screen rect.

    Custom-drawn widgets don't participate in ImGui's normal hover
    tracking, so we test the rect ourselves and call set_tooltip.
    """
    if not text:
        return
    mx, my = imgui.get_mouse_pos()
    if x <= mx <= x + w and y <= my <= y + h:
        imgui.set_tooltip(text)


# ============================================================================
# Primitive drawing helpers
# ============================================================================

def op1_arc(dl, cx, cy, radius, a_min, a_max, color_u32, thickness, segments=56):
    """Draw an arc as a polyline."""
    if a_max < a_min:
        a_min, a_max = a_max, a_min
    steps = max(2, int(segments * (a_max - a_min) / (2.0 * math.pi)))
    prev_x = cx + radius * math.cos(a_min)
    prev_y = cy + radius * math.sin(a_min)
    for i in range(1, steps + 1):
        t = i / steps
        a = a_min + (a_max - a_min) * t
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        dl.add_line(prev_x, prev_y, x, y, color_u32, thickness)
        prev_x, prev_y = x, y


# The canonical knob geometry for every widget in the app. A 270° arc
# sweeps from 135° (bottom-left) clockwise to 405° (bottom-right).
# Visuals match the Figma GS Op-1 "vol" component (node 54:295):
# thin dark track, accent fill, cursor dot, outer radial ticks, blue
# 12-o'clock reference mark, short inner pointer nub. Every knob in
# the program routes through op1_knob_draw so the style is unified.
_KNOB_ARC_START = math.radians(135.0)
_KNOB_ARC_END = math.radians(405.0)

# Design tokens from the Figma file.
_KNOB_TRACK_COLOR = (0.32, 0.32, 0.32, 1.0)       # flat grey track (brighter)
_KNOB_TOP_MARK_COLOR = (0.368, 0.514, 1.0, 1.0)  # #5E83FF — top reference mark (keep blue)
_KNOB_INNER_PTR_COLOR = (0.30, 0.30, 0.30, 1.0)  # flat grey inner pointer

# Geometric ratios (all relative to the knob radius so the knob scales
# cleanly with DPI and cell size).
_KNOB_TICK_INNER = 1.08
_KNOB_TICK_OUTER = 1.15
_KNOB_TOP_MARK_INNER = 1.35
_KNOB_TOP_MARK_OUTER = 1.52
_KNOB_INNER_PTR_INNER = 0.68
_KNOB_INNER_PTR_OUTER = 0.86
_KNOB_CURSOR_DOT_R = 0.09        # cursor dot radius as fraction of knob radius
_KNOB_TRACK_THICKNESS = 2.0      # px, scaled implicitly by ImGui draw list
_KNOB_FILL_THICKNESS = 2.6       # px — slightly thicker so fill reads on top


def op1_knob_draw(dl, cx: float, cy: float, radius: float,
                   value_frac: float, color: tuple) -> None:
    """Draw the unified OP-1 knob — Figma "vol" component style.

    Every knob in the program routes through this function, so updating
    the visuals here updates every knob in the app. The caller is
    responsible for layout, labels, numeric readouts, and drag input.

    Elements (outer → inner):
      * 12-o'clock blue reference mark (outside the tick ring)
      * Radial tick marks at 8 evenly-spaced positions across the 270° span
      * Dark violet 270° track arc
      * Accent-color fill arc from the start to the current value
      * Accent-color cursor dot at the head of the fill arc
      * Short violet inner pointer nub at the top of the knob interior
    """
    value_frac = max(0.0, min(1.0, value_frac))

    track_u32 = col32(_KNOB_TRACK_COLOR)
    fill_u32 = col32(color)

    # -- 1. Background track (thin dark violet arc, 270° span).
    op1_arc(dl, cx, cy, radius, _KNOB_ARC_START, _KNOB_ARC_END,
            track_u32, _KNOB_TRACK_THICKNESS, segments=64)

    # -- 2. Accent fill arc from the start of the sweep to the current value.
    cur_angle = _KNOB_ARC_START + (_KNOB_ARC_END - _KNOB_ARC_START) * value_frac
    if value_frac > 0.001:
        op1_arc(dl, cx, cy, radius, _KNOB_ARC_START, cur_angle,
                fill_u32, _KNOB_FILL_THICKNESS, segments=64)

    # -- 3. Cursor dot at the head of the fill arc.
    dot_r = max(1.5, radius * _KNOB_CURSOR_DOT_R)
    cdx = cx + radius * math.cos(cur_angle)
    cdy = cy + radius * math.sin(cur_angle)
    dl.add_circle_filled(cdx, cdy, dot_r, fill_u32, 12)

    # -- 4. Outer radial tick marks — 8 ticks + 1 closing tick evenly spaced
    #       across the 270° sweep (so the ticks visually bracket both ends
    #       of the track and mark the 8 intermediate positions).
    num_ticks = 8
    for i in range(num_ticks + 1):
        t = i / num_ticks
        a = _KNOB_ARC_START + (_KNOB_ARC_END - _KNOB_ARC_START) * t
        ca, sa = math.cos(a), math.sin(a)
        tx0 = cx + radius * _KNOB_TICK_INNER * ca
        ty0 = cy + radius * _KNOB_TICK_INNER * sa
        tx1 = cx + radius * _KNOB_TICK_OUTER * ca
        ty1 = cy + radius * _KNOB_TICK_OUTER * sa
        dl.add_line(tx0, ty0, tx1, ty1, track_u32, 1.2)

    # -- 5. Blue 12-o'clock reference mark sitting above the tick ring.
    tm_y0 = cy - radius * _KNOB_TOP_MARK_INNER
    tm_y1 = cy - radius * _KNOB_TOP_MARK_OUTER
    dl.add_line(cx, tm_y0, cx, tm_y1, col32(_KNOB_TOP_MARK_COLOR), 2.0)

    # -- 6. Short violet inner pointer nub at the top of the knob interior.
    ip_y0 = cy - radius * _KNOB_INNER_PTR_OUTER
    ip_y1 = cy - radius * _KNOB_INNER_PTR_INNER
    dl.add_line(cx, ip_y0, cx, ip_y1, col32(_KNOB_INNER_PTR_COLOR), 2.0)


def _iso_project(px: float, py: float, pz: float,
                 cx: float, cy: float) -> tuple[float, float]:
    """Standard isometric projection — 30° axonometric with screen-y
    flipped (world +y = up, screen +y = down). Returns screen (sx, sy)
    for a world-space point (px, py, pz) centered at (cx, cy)."""
    COS30 = 0.8660254  # math.cos(math.radians(30))
    SIN30 = 0.5
    sx = cx + (px - pz) * COS30
    sy = cy - py + (px + pz) * SIN30
    return sx, sy


def op1_tone_polyhedron(dl, cx: float, cy: float,
                         area_w: float, area_h: float,
                         shape: str, value_frac: float,
                         color_u32: int,
                         thickness: float = 2.2,
                         rotation: float = 0.0) -> None:
    """Draw a wireframe polyhedron centered at (cx, cy) whose vertical
    height encodes ``value_frac`` (0..1).

    Kitbashed from the Figma EQ kit (node 154:2146) — three isometric
    shapes (cylinder / box / pyramid) standing on an invisible floor,
    stretching vertically as the parameter rises. Used as the TONE
    panel knob replacement so the value is communicated by the shape's
    height rather than a knob pointer.

    ``shape`` is one of 'cylinder', 'box', 'pyramid', 'tetrahedron',
    'sphere', 'icosahedron', 'lens_tube'.
    ``rotation`` rotates the shape around the vertical (Y) axis in radians.
    """
    value_frac = max(0.0, min(1.0, value_frac))
    # Base radius — fill the cell generously
    base_r = min(area_w * 0.36, area_h * 0.20)
    max_h = area_h * 0.72
    min_h = area_h * 0.16
    full_h = min_h + (max_h - min_h) * value_frac
    # World y=0 is the floor; apex at y=full_h. Shift downward so the
    # base sits in the lower third of the cell.
    base_y = area_h * 0.30  # screen-space offset from (cx, cy) down
    floor = cy + base_y
    top = floor - full_h

    # Y-axis rotation applied to world-space xz before iso projection
    cos_r = math.cos(rotation)
    sin_r = math.sin(rotation)

    def P(px, py, pz):
        rx = px * cos_r - pz * sin_r
        rz = px * sin_r + pz * cos_r
        return _iso_project(rx, py, rz, cx, floor)

    if shape == 'cylinder':
        # Top circle at py=full_h, bottom circle at py=0 (the floor).
        N = 32
        top_pts = []
        bot_pts = []
        for i in range(N):
            a = 2.0 * math.pi * i / N
            px = math.cos(a) * base_r
            pz = math.sin(a) * base_r
            top_pts.append(P(px, full_h, pz))
            bot_pts.append(P(px, 0.0, pz))
        for i in range(N):
            a0 = top_pts[i]
            a1 = top_pts[(i + 1) % N]
            dl.add_line(a0[0], a0[1], a1[0], a1[1], color_u32, thickness)
            b0 = bot_pts[i]
            b1 = bot_pts[(i + 1) % N]
            dl.add_line(b0[0], b0[1], b1[0], b1[1], color_u32, thickness)
        # Silhouette verticals at 90° and 270° (fixed relative to rotation)
        for a_deg in (90, 270):
            a = math.radians(a_deg)
            px = math.cos(a) * base_r
            pz = math.sin(a) * base_r
            t = P(px, full_h, pz)
            b = P(px, 0.0, pz)
            dl.add_line(t[0], t[1], b[0], b[1], color_u32, thickness)

    elif shape == 'box':
        w2 = base_r
        d2 = base_r
        verts = [
            (-w2, 0.0, -d2), (w2, 0.0, -d2),
            (w2, 0.0,  d2), (-w2, 0.0,  d2),
            (-w2, full_h, -d2), (w2, full_h, -d2),
            (w2, full_h,  d2), (-w2, full_h,  d2),
        ]
        pts = [P(x, y, z) for (x, y, z) in verts]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),   # top
            (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
        ]
        for a, b in edges:
            dl.add_line(pts[a][0], pts[a][1],
                        pts[b][0], pts[b][1], color_u32, thickness)

    elif shape == 'pyramid':
        w2 = base_r
        d2 = base_r
        verts = [
            (-w2, 0.0, -d2), (w2, 0.0, -d2),
            (w2, 0.0,  d2), (-w2, 0.0,  d2),
            (0.0, full_h, 0.0),                       # apex
        ]
        pts = [P(x, y, z) for (x, y, z) in verts]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),   # base
            (0, 4), (1, 4), (2, 4), (3, 4),   # slant
        ]
        for a, b in edges:
            dl.add_line(pts[a][0], pts[a][1],
                        pts[b][0], pts[b][1], color_u32, thickness)

    elif shape == 'sphere':
        # 3-ring wireframe sphere matching the Contact Sheets sphere gauge.
        # Centered in the cell, scales uniformly outward in all directions.
        max_r = min(area_w * 0.36, area_h * 0.20)
        sphere_r = max_r * (0.25 + 0.75 * (value_frac ** 0.6))
        # Fixed center point (mid-height of the shape area, not dependent on value)
        center_y = max_h * 0.5
        N = 32
        for ring in range(3):
            pts = []
            for i in range(N):
                a = 2.0 * math.pi * i / N
                if ring == 0:
                    # Equator (XZ plane)
                    px = math.cos(a) * sphere_r
                    py = center_y
                    pz = math.sin(a) * sphere_r
                elif ring == 1:
                    # Meridian (XY plane)
                    px = math.cos(a) * sphere_r
                    py = center_y + math.sin(a) * sphere_r
                    pz = 0.0
                else:
                    # Meridian (YZ plane)
                    px = 0.0
                    py = center_y + math.sin(a) * sphere_r
                    pz = math.cos(a) * sphere_r
                pts.append(P(px, py, pz))
            for i in range(N):
                dl.add_line(pts[i][0], pts[i][1],
                            pts[(i + 1) % N][0], pts[(i + 1) % N][1],
                            color_u32, thickness)
        # Center dot
        cp = P(0.0, center_y, 0.0)
        dl.add_circle_filled(cp[0], cp[1], max(2.0, sphere_r * 0.08),
                             color_u32, 10)

    elif shape == 'tetrahedron':
        # Octahedron — 6 vertices, 12 edges. Tips stretch vertically
        # away from center as value increases; nearly flat at zero.
        s = base_r * 1.1
        center_y = max_h * 0.5
        tip_h = s * (0.08 + 0.92 * value_frac)  # almost flat at 0, full at 1
        verts = [
            (s, center_y, 0.0),               # +X
            (-s, center_y, 0.0),              # -X
            (0.0, center_y, s),               # +Z
            (0.0, center_y, -s),              # -Z
            (0.0, center_y + tip_h, 0.0),     # top tip
            (0.0, center_y - tip_h, 0.0),     # bottom tip
        ]
        pts = [P(x, y, z) for (x, y, z) in verts]
        edges = [
            # Equator square
            (0, 2), (2, 1), (1, 3), (3, 0),
            # Top tip to equator
            (4, 0), (4, 1), (4, 2), (4, 3),
            # Bottom tip to equator
            (5, 0), (5, 1), (5, 2), (5, 3),
        ]
        for a, b in edges:
            dl.add_line(pts[a][0], pts[a][1],
                        pts[b][0], pts[b][1], color_u32, thickness)

    elif shape == 'icosahedron' or shape == 'torus':
        # Torus — two concentric rings (major circle in XZ, tube cross-section).
        # Major radius fixed, tube radius scales with value.
        center_y = max_h * 0.5
        major_r = base_r * 0.85
        minor_r = base_r * (0.12 + 0.35 * value_frac)
        N = 32  # segments around major circle
        M = 16  # segments around tube cross-section
        for j in range(N):
            a0 = 2.0 * math.pi * j / N
            a1 = 2.0 * math.pi * ((j + 1) % N) / N
            ca0, sa0 = math.cos(a0), math.sin(a0)
            ca1, sa1 = math.cos(a1), math.sin(a1)
            for k in range(M):
                b0 = 2.0 * math.pi * k / M
                b1 = 2.0 * math.pi * ((k + 1) % M) / M
                # Point on torus at (j, k)
                r0 = major_r + minor_r * math.cos(b0)
                px0 = r0 * ca0
                py0 = center_y + minor_r * math.sin(b0)
                pz0 = r0 * sa0
                # Next along major circle (j+1, k)
                r1 = major_r + minor_r * math.cos(b0)
                px1 = r1 * ca1
                py1 = center_y + minor_r * math.sin(b0)
                pz1 = r1 * sa1
                # Next along tube (j, k+1)
                r2 = major_r + minor_r * math.cos(b1)
                px2 = r2 * ca0
                py2 = center_y + minor_r * math.sin(b1)
                pz2 = r2 * sa0
                p0 = P(px0, py0, pz0)
                p1 = P(px1, py1, pz1)
                p2 = P(px2, py2, pz2)
                # Draw only every 4th ring line + every 4th cross line for wireframe look
                if k % 4 == 0:
                    dl.add_line(p0[0], p0[1], p1[0], p1[1], color_u32, thickness)
                if j % 4 == 0:
                    dl.add_line(p0[0], p0[1], p2[0], p2[1], color_u32, thickness)

    elif shape == 'lens_tube':
        # Cylindrical tube with an inner aperture ring at the midpoint.
        # The outer cylinder is fixed; the inner ring's radius shrinks
        # as value_frac increases — like a camera iris closing for
        # stronger DOF. value=0 → inner ring matches outer (no DOF),
        # value=1 → inner ring is a tiny pinhole.
        tube_r = base_r * 1.1
        tube_h = full_h
        inner_frac = 1.0 - value_frac  # 1=wide open, 0=pinhole
        inner_r = tube_r * max(inner_frac, 0.08)
        mid_y = tube_h * 0.5  # midpoint height for the iris ring
        N = 32

        # Outer cylinder — top and bottom circles + tangent verticals
        top_pts = []
        bot_pts = []
        for i in range(N):
            a = 2.0 * math.pi * i / N
            px = math.cos(a) * tube_r
            pz = math.sin(a) * tube_r
            top_pts.append(P(px, tube_h, pz))
            bot_pts.append(P(px, 0.0, pz))
        for i in range(N):
            dl.add_line(top_pts[i][0], top_pts[i][1],
                        top_pts[(i + 1) % N][0], top_pts[(i + 1) % N][1],
                        color_u32, thickness)
            dl.add_line(bot_pts[i][0], bot_pts[i][1],
                        bot_pts[(i + 1) % N][0], bot_pts[(i + 1) % N][1],
                        color_u32, thickness)
        # Silhouette verticals at 90° and 270°
        for a_deg in (90, 270):
            a = math.radians(a_deg)
            px = math.cos(a) * tube_r
            pz = math.sin(a) * tube_r
            t = P(px, tube_h, pz)
            b = P(px, 0.0, pz)
            dl.add_line(t[0], t[1], b[0], b[1], color_u32, thickness)

        # Inner aperture ring at the midpoint
        inner_pts = []
        for i in range(N):
            a = 2.0 * math.pi * i / N
            px = math.cos(a) * inner_r
            pz = math.sin(a) * inner_r
            inner_pts.append(P(px, mid_y, pz))
        for i in range(N):
            dl.add_line(inner_pts[i][0], inner_pts[i][1],
                        inner_pts[(i + 1) % N][0], inner_pts[(i + 1) % N][1],
                        color_u32, thickness)


def op1_tone_cell(app, cell_x: float, cell_y: float,
                   cell_w: float, cell_h: float,
                   uid: str, label: str, value: float,
                   min_val: float, max_val: float, color: tuple,
                   shape: str,
                   default_val: float | None = None) -> tuple[bool, float]:
    """Interactive TONE cell — the knob is replaced with a wireframe
    polyhedron whose height encodes the parameter value. Click-and-
    drag vertically to scrub (0.004 span/px, same as op1_knob_cell);
    double-click or right-click resets to ``default_val``. Returns
    ``(changed, new_value)``.
    """
    global _gauge_drag_key
    changed = False
    dl = imgui.get_window_draw_list()
    th = _th()

    span = max(max_val - min_val, 1e-6)
    reset_val = default_val if default_val is not None else (min_val + max_val) * 0.5
    value_frac = max(0.0, min(1.0, (value - min_val) / span))

    mx, my = imgui.get_mouse_pos()
    in_rect = (cell_x <= mx <= cell_x + cell_w
               and cell_y <= my <= cell_y + cell_h)

    if _gauge_drag_key is None and in_rect and imgui.is_mouse_clicked(0):
        _gauge_drag_key = uid
    if _gauge_drag_key == uid:
        if imgui.is_mouse_down(0):
            dy = imgui.get_io().mouse_delta[1]
            if abs(dy) > 0.001:
                value = value - dy * 0.004 * span
                value = max(min_val, min(max_val, value))
                changed = True
        else:
            _gauge_drag_key = None
    if in_rect and imgui.is_mouse_double_clicked(0):
        value = reset_val
        changed = True
    if _rclick_in_rect(cell_x, cell_y, cell_w, cell_h):
        value = reset_val
        changed = True

    # Shape occupies the upper ~82% of the cell; label strip below.
    shape_area_h = cell_h * 0.82
    shape_cx = cell_x + cell_w * 0.5
    shape_cy = cell_y + shape_area_h * 0.5
    # Slow continuous rotation for visual polish
    rot = time.time() * 0.03
    op1_tone_polyhedron(
        dl, shape_cx, shape_cy,
        cell_w, shape_area_h,
        shape, value_frac, col32(color),
        thickness=2.4,
        rotation=rot,
    )

    # Label centered under the shape.
    if label:
        lbl_size = imgui.calc_text_size(label)
        ly = cell_y + cell_h - th * 1.1
        dl.add_text(cell_x + (cell_w - lbl_size[0]) * 0.5, ly,
                    col32(color), label)

    return changed, value


def op1_knob_cell(app, cell_x: float, cell_y: float,
                   cell_w: float, cell_h: float,
                   uid: str, label: str, value: float,
                   min_val: float, max_val: float, color: tuple,
                   decimals: int = 2,
                   default_val: float | None = None,
                   display_scale: float = 1.0) -> tuple[bool, float]:
    """Interactive knob cell inside a caller-provided panel rect.

    Layout inside the cell:
        top 55% — knob (arc + body + pointer)
        middle  — big numeric readout
        bottom  — small accent-colored label

    Click-and-drag vertically to change value; double-click or
    right-click inside the cell to reset to ``default_val`` (or the
    midpoint when None). Returns ``(changed, new_value)``.
    """
    global _gauge_drag_key
    changed = False
    dl = imgui.get_window_draw_list()
    th = _th()

    # Layout anchors inside the cell. Radius is sized so the outer tick
    # ring (_KNOB_TICK_OUTER = 1.22 × radius) and the blue top reference
    # mark (up to 1.52 × radius above center) stay comfortably inside
    # the cell and clear of the numeric readout below.
    knob_cy = cell_y + cell_h * 0.33
    knob_radius = min(cell_w * 0.30, cell_h * 0.22)
    knob_cx = cell_x + cell_w * 0.5

    num_y = cell_y + cell_h * 0.54
    num_h = cell_h * 0.32

    label_y = cell_y + cell_h - th * 1.15

    span = max(max_val - min_val, 1e-6)
    reset_val = default_val if default_val is not None else (min_val + max_val) * 0.5

    # Hit region = full cell rect.
    mx, my = imgui.get_mouse_pos()
    in_rect = (cell_x <= mx <= cell_x + cell_w
               and cell_y <= my <= cell_y + cell_h)

    if _gauge_drag_key is None and in_rect and imgui.is_mouse_clicked(0):
        _gauge_drag_key = uid

    if _gauge_drag_key == uid:
        if imgui.is_mouse_down(0):
            dy = imgui.get_io().mouse_delta[1]
            if abs(dy) > 0.001:
                # Sensitivity: 0.004 span/pixel — half the old 0.008
                # so fine adjustments take more travel.
                value = value - dy * 0.004 * span
                value = max(min_val, min(max_val, value))
                changed = True
        else:
            _gauge_drag_key = None

    if in_rect and imgui.is_mouse_double_clicked(0):
        value = reset_val
        changed = True
    if _rclick_in_rect(cell_x, cell_y, cell_w, cell_h):
        value = reset_val
        changed = True

    value_frac = max(0.0, min(1.0, (value - min_val) / span))
    op1_knob_draw(dl, knob_cx, knob_cy, knob_radius, value_frac, color)

    # Big numeric readout below the knob.
    if decimals == 0:
        value_str = f"{int(round(value * display_scale))}"
    else:
        value_str = f"{value * display_scale:.{decimals}f}"
    op1_number(app, cell_x + th * 0.2, num_y,
               cell_w - th * 0.4, num_h, value_str, color, align='center')

    # Tiny label at the bottom of the cell.
    if label:
        lbl_size = imgui.calc_text_size(label)
        dl.add_text(cell_x + (cell_w - lbl_size[0]) * 0.5, label_y,
                    col32(color), label)

    return changed, value


# ---------------------------------------------------------------------------
# Unified panel / button styling
# ---------------------------------------------------------------------------
# Every panel and button shares the same corner radius and the same
# fill + border pairing rule:
#   * outlined shapes always get a matching dark tinted fill
#   * filled shapes always get a matching border
# PANEL_ROUNDING is used for section panels and bigger surfaces;
# BUTTON_ROUNDING is used for individual tap targets. Both values live
# here so that bumping the radius bumps it everywhere.
PANEL_ROUNDING = 6.0
BUTTON_ROUNDING = 5.0

# Flat 95%-black fill for every panel/button surface so they sit one
# step darker than the 0.10 window bg, matching the OP-1 screen kit.
_PANEL_FILL   = (0.05, 0.05, 0.05, 1.0)
_PANEL_BORDER = (0.22, 0.22, 0.22, 1.0)    # legacy — kept for unused references
_BUTTON_FILL  = (0.05, 0.05, 0.05, 1.0)
_BUTTON_BORDER = (0.22, 0.22, 0.22, 1.0)

# Keep these aliases so callers that pass fill_scale= still work;
# the scale arg is ignored — we return fixed greys.
_PANEL_FILL_SCALE = None
_PANEL_BORDER_SCALE = None
_BUTTON_FILL_SCALE = None
_BUTTON_BORDER_SCALE = None


def tinted_fill(color, scale=None):
    """Return the standard panel fill (flat grey, ignores accent color)."""
    return _PANEL_FILL


def tinted_border(color, scale=None):
    """Return the standard panel border (flat grey, ignores accent color)."""
    return _PANEL_BORDER


def _scaled_rounding(r: float) -> float:
    """Scale a rounding constant by the current DPI factor so visuals
    keep a consistent *apparent* radius on HiDPI displays.

    Called lazily so the module can still be imported before scale has
    been initialised.
    """
    from src.gui.scale import s as _s
    return _s(r)


def draw_styled_rect(dl, x, y, w, h, color,
                     rounding=BUTTON_ROUNDING, thickness=0.7,
                     fill_scale=None, border_scale=None,
                     colored_fill=False, border=None):
    """Draw a cell rect — flat grey fill by default, accent-colored border.

    Pass colored_fill=True to tint the fill with the accent color (used
    only for the POINT SIZE panel which keeps its warm orange background).
    Borders are drawn by default for buttons (BUTTON_ROUNDING) but skipped
    for panel-style containers (PANEL_ROUNDING) so panels read as flat
    fills against the window background, matching the OP-1 screen kit.
    Pass border=True/False to override the auto-detection.
    """
    r = _scaled_rounding(rounding)
    if colored_fill:
        # Deep dark tinted fill that preserves the accent's hue
        # proportionally. Works for warm (orange) and cool (blue) alike.
        fill = (color[0] * 0.12, color[1] * 0.12, color[2] * 0.12, 1.0)
    else:
        fill = _BUTTON_FILL
    dl.add_rect_filled(x, y, x + w, y + h, col32(fill), r)
    if border is None:
        border = (rounding != PANEL_ROUNDING)
    if border:
        dl.add_rect(x, y, x + w, y + h, col32(color),
                    r, 0, thickness)


def draw_styled_rect_active(dl, x, y, w, h, color,
                            rounding=BUTTON_ROUNDING, thickness=1.0,
                            border=None):
    """Active/selected cell — slightly lighter fill, full accent border."""
    r = _scaled_rounding(rounding)
    dl.add_rect_filled(x, y, x + w, y + h, col32((0.18, 0.18, 0.18, 1.0)), r)
    if border is None:
        border = (rounding != PANEL_ROUNDING)
    if border:
        dl.add_rect(x, y, x + w, y + h, col32(color),
                    r, 0, thickness)


# Classic pixel-art alien glyph (11x8) — kitbashed from Figma node 45:1638.
_ALIEN_GRID = [
    "..X.....X..",
    "...X...X...",
    "..XXXXXXX..",
    ".XX.XXX.XX.",
    "XXXXXXXXXXX",
    "X.XXXXXXX.X",
    "X.X.....X.X",
    "...XX.XX...",
]

# Back view: same as front but with the two eye gaps (row 5) filled in.
_ALIEN_BACK_GRID = [
    "..X.....X..",
    "...X...X...",
    "..XXXXXXX..",
    ".XX.XXX.XX.",
    "XXXXXXXXXXX",
    "XXXXXXXXXXX",
    "X.X.....X.X",
    "...XX.XX...",
]

# Side view: 1px wide, 8 tall — seen from left or right, just a vertical line.
_ALIEN_SIDE_GRID = [
    "X",
    "X",
    "X",
    "X",
    "X",
    "X",
    "X",
    "X",
]

# Top view: 11px wide, 1 tall — seen from above, just a horizontal line.
_ALIEN_TOP_GRID = [
    "XXXXXXXXXXX",
]

# Bottom view: same as top (looking up from below).
_ALIEN_BOTTOM_GRID = [
    "XXXXXXXXXXX",
]

_ALIEN_GRIDS = {
    'front':  _ALIEN_GRID,
    'back':   _ALIEN_BACK_GRID,
    'side':   _ALIEN_SIDE_GRID,
    'top':    _ALIEN_TOP_GRID,
    'bottom': _ALIEN_BOTTOM_GRID,
}


def op1_folder_icon(dl, cx: float, cy: float, size: float, color_u32: int,
                    thickness: float = 1.0):
    """Folder glyph (outline) — kitbashed from Figma node 45:337.

    Drawn centered at (cx, cy), ``size`` is the icon bounding box edge.
    Shape: rectangular body with a small tab on the top-left, matching
    the classic folder silhouette.
    """
    # Native viewBox is 18x12 for just the folder portion (y 3.5 → 14.5
    # in the 18x36 SVG). We scale proportionally and center.
    vw, vh = 18.0, 12.0
    scale = size / vw
    ox = cx - (vw * scale) * 0.5
    oy = cy - (vh * scale) * 0.5

    def p(x, y):
        return (ox + x * scale, oy + (y - 3.0) * scale)

    # Outline points (converted from the SVG path):
    # (7.25,3.5) (8.6,5.3) (8.75,5.5) (16.5,5.5) (16.5,14.5)
    # (1.5,14.5) (1.5,3.5) (7.25,3.5)
    pts = [
        p(7.25, 3.5), p(8.6, 5.3), p(8.75, 5.5),
        p(16.5, 5.5), p(16.5, 14.5),
        p(1.5, 14.5), p(1.5, 3.5),
    ]
    for i in range(len(pts)):
        a = pts[i]
        b = pts[(i + 1) % len(pts)]
        dl.add_line(a[0], a[1], b[0], b[1], color_u32, thickness)
    # Small tab line on the top edge (right side): x=10..16, y=3
    t0 = p(10.0, 3.0)
    t1 = p(16.0, 3.0)
    dl.add_line(t0[0], t0[1], t1[0], t1[1], color_u32, thickness)


def op1_alien(dl, cx: float, cy: float, pixel_size: float,
              color_u32: int, angle: float = 0.0, profile: str = 'front'):
    """Draw the space-invader alien centered at (cx, cy).

    Each filled cell is a ``pixel_size``×``pixel_size`` square.
    ``angle`` is a continuous clockwise rotation in radians.
    ``profile`` selects the anatomical view: 'front', 'back', 'side',
    'top', or 'bottom' — so the alien acts as a patient-orientation guide.
    """
    import math
    grid = _ALIEN_GRIDS.get(profile, _ALIEN_GRID)
    gh = len(grid)
    gw = len(grid[0]) if grid else 1
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for gy, row in enumerate(grid):
        for gx, ch in enumerate(row):
            if ch != 'X':
                continue
            # Local offset from the center of the glyph.
            lx = (gx - (gw - 1) * 0.5) * pixel_size
            ly = (gy - (gh - 1) * 0.5) * pixel_size
            # Clockwise rotation by `angle`
            rx = lx * cos_a + ly * sin_a
            ry = -lx * sin_a + ly * cos_a
            px = cx + rx - pixel_size * 0.5
            py = cy + ry - pixel_size * 0.5
            dl.add_rect_filled(px, py, px + pixel_size, py + pixel_size,
                               color_u32)


def op1_panel(dl, x, y, w, h, rounding=PANEL_ROUNDING):
    """Section panel — flat fill, no outline."""
    r = _scaled_rounding(rounding)
    dl.add_rect_filled(x, y, x + w, y + h, col32(_PANEL_FILL), r)


def op1_inner_panel(dl, x, y, w, h, rounding=BUTTON_ROUNDING):
    """Inset panel — flat fill, no outline."""
    r = _scaled_rounding(rounding)
    dl.add_rect_filled(x, y, x + w, y + h, col32(_PANEL_FILL), r)


def op1_number(app, x, y, w, h, value_str, color, align='center'):
    """Draw a big display-font numeral inside a rect.

    Uses imgui.text_colored so the pushed display font is respected. The
    layout cursor is saved and restored around the draw so the parent
    widget's dummy/layout is unaffected.
    """
    display_font = getattr(app.gui, 'font_display', None) if getattr(app, 'gui', None) else None

    saved_cursor = imgui.get_cursor_screen_pos()

    if display_font is not None:
        imgui.push_font(display_font)
        v_size = imgui.calc_text_size(value_str)
        if align == 'center':
            tx = x + (w - v_size[0]) * 0.5
        elif align == 'right':
            tx = x + w - v_size[0] - _th() * 0.3
        else:
            tx = x + _th() * 0.3
        ty = y + (h - v_size[1]) * 0.5
        imgui.set_cursor_screen_pos((tx, ty))
        imgui.text_colored(value_str, *color[:3])
        imgui.pop_font()
    else:
        dl = imgui.get_window_draw_list()
        v_size = imgui.calc_text_size(value_str)
        if align == 'center':
            tx = x + (w - v_size[0]) * 0.5
        elif align == 'right':
            tx = x + w - v_size[0] - _th() * 0.3
        else:
            tx = x + _th() * 0.3
        ty = y + (h - v_size[1]) * 0.5
        dl.add_text(tx, ty, col32(color), value_str)

    imgui.set_cursor_screen_pos(saved_cursor)


def op1_tick_scale(dl, x, y, w, h, value_frac, color):
    """Horizontal tick scale with an indicator at value_frac (0..1)."""
    tick_count = 21
    tick_col = col32(OP1_DIM)
    active_col = col32(color)
    y_top = y + h * 0.4
    y_bot = y + h * 0.8
    for i in range(tick_count):
        t = i / (tick_count - 1)
        tx = x + t * w
        # Major ticks at 0/50/100
        is_major = (i == 0 or i == tick_count // 2 or i == tick_count - 1)
        tick_y_top = y_top if not is_major else y + h * 0.25
        dl.add_line(tx, tick_y_top, tx, y_bot, tick_col, 1.0)

    # Indicator
    ix = x + value_frac * w
    dl.add_line(ix, y + h * 0.1, ix, y + h * 0.95, active_col, 2.0)
    dl.add_triangle_filled(
        ix - 4, y + h * 0.1,
        ix + 4, y + h * 0.1,
        ix, y + h * 0.3,
        active_col,
    )


# ============================================================================
# Section header
# ============================================================================

_section_collapsed: dict[str, bool] = {}


def op1_section(text: str, color=OP1_ORANGE, fill_color=None,
                collapsible: bool = True) -> bool:
    """Section header: title sits inside the panel fill.

    Draws a full-width panel strip that continues the fill of the
    content panel below, with the bullet square + label text placed
    directly on that strip (no surrounding pill). Content that follows
    butts up flush with the bottom edge so the title reads as the first
    line of the same panel.

    ``fill_color`` lets callers override the default 0.05 grey strip
    with the tinted fill their content panel uses (e.g. the POINT SIZE
    and TONE panels have warm/cool colored_fill rects — the title
    strip should match so the section reads as one block).

    Returns True if the section is open (content should be drawn),
    False if collapsed. Click the title to toggle.
    """
    # Small vertical gap before each section header
    imgui.dummy(0, imgui.get_text_line_height() * 0.25)

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    sq_size = th * 0.5
    gap = th * 0.4
    pad_x = th * 0.7
    pad_y = th * 0.3

    strip_h = th + pad_y * 2
    rounding = _scaled_rounding(PANEL_ROUNDING)

    collapsed = _section_collapsed.get(text, False) if collapsible else False

    strip_fill = fill_color if fill_color is not None else _PANEL_FILL
    corners = (imgui.DRAW_ROUND_CORNERS_TOP if not collapsed
               else imgui.DRAW_ROUND_CORNERS_ALL)
    bottom_ext = rounding if not collapsed else 0.0
    dl.add_rect_filled(wx, wy,
                       wx + w, wy + strip_h + bottom_ext,
                       col32(strip_fill), rounding, corners)

    # Bullet square
    sq_x = wx + pad_x
    sq_y = wy + (strip_h - sq_size) * 0.5
    dl.add_rect_filled(sq_x, sq_y, sq_x + sq_size, sq_y + sq_size, col32(color))

    # Label text
    text_x = sq_x + sq_size + gap
    text_y = wy + (strip_h - th) * 0.5
    dl.add_text(text_x, text_y, col32(color), text)

    # Reserve just the strip height so the following draw call lands
    # inside the strip's extended fill area and merges seamlessly.
    imgui.dummy(w, strip_h)

    # Click to toggle
    if collapsible:
        mx, my = imgui.get_mouse_pos()
        if (wx <= mx <= wx + w and wy <= my <= wy + strip_h
                and imgui.is_mouse_clicked(0)):
            _section_collapsed[text] = not collapsed
            collapsed = not collapsed

    return not collapsed


# ============================================================================
# Gauge widgets (arcs + display numerals)
# ============================================================================

_gauge_drag_key: str | None = None
_gauge_editing_uid: str | None = None  # Ctrl+double-click to type a value


def op1_gauge(app, uid: str, label: str, value: float,
              min_val: float, max_val: float, color: tuple,
              unit: str = "", display_scale: float = 1.0,
              height_units: float = 4.2,
              decimals: int = 1,
              default_val: float | None = None,
              tooltip: str = "") -> tuple[bool, float]:
    """A single gauge widget: half-moon arc + big display numeral.

    Click+drag vertically to change the value. Double-click or right-click
    to reset to default_val (or midpoint if None).

    Returns (changed, new_value).
    """
    global _gauge_drag_key, _gauge_editing_uid
    changed = False

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    panel_h = th * height_units
    draw_styled_rect(dl, wx, wy, w, panel_h, color,
                     rounding=PANEL_ROUNDING, thickness=1.2)

    reset_val = default_val if default_val is not None else (min_val + max_val) * 0.5

    # Knob on the left, number on the right
    knob_size = panel_h - th * 0.8
    knob_x = wx + th * 0.6
    knob_y = wy + th * 0.4
    cx = knob_x + knob_size * 0.5
    cy = knob_y + knob_size * 0.55
    radius = knob_size * 0.38

    # Map value to fraction
    span = max(max_val - min_val, 1e-6)
    value_frac = max(0.0, min(1.0, (value - min_val) / span))
    current_angle = math.pi + math.pi * value_frac

    # Interaction region: whole panel
    mx, my = imgui.get_mouse_pos()
    in_rect = (wx <= mx <= wx + w and wy <= my <= wy + panel_h)
    io = imgui.get_io()

    # Don't drag/reset while typing into another gauge's input field
    if _gauge_editing_uid is None:
        if _gauge_drag_key is None and in_rect and imgui.is_mouse_clicked(0):
            _gauge_drag_key = uid

        if _gauge_drag_key == uid:
            if imgui.is_mouse_down(0):
                dy = io.mouse_delta[1]
                if abs(dy) > 0.001:
                    # Halved sensitivity (was 0.006) for finer control.
                    value = value - dy * 0.003 * span
                    value = max(min_val, min(max_val, value))
                    changed = True
            else:
                _gauge_drag_key = None

        if in_rect and imgui.is_mouse_double_clicked(0):
            if io.key_ctrl:
                # Ctrl+double-click: enter numeric typing mode
                _gauge_editing_uid = uid
            else:
                value = reset_val
                changed = True
        if _rclick_in_rect(wx, wy, w, panel_h):
            value = reset_val
            changed = True

    # Recompute after potential change
    value_frac = max(0.0, min(1.0, (value - min_val) / span))
    current_angle = math.pi + math.pi * value_frac

    # Background arc
    op1_arc(dl, cx, cy, radius, math.pi, 2.0 * math.pi, col32(OP1_DIM), 2.2)
    # Fill arc
    if value_frac > 0.001:
        op1_arc(dl, cx, cy, radius, math.pi, current_angle, col32(color), 2.8)
    # Tick marks
    for a_deg in (180, 225, 270, 315, 360):
        a = math.radians(a_deg)
        tx = cx + (radius + 3) * math.cos(a)
        ty = cy + (radius + 3) * math.sin(a)
        tx2 = cx + (radius + 7) * math.cos(a)
        ty2 = cy + (radius + 7) * math.sin(a)
        dl.add_line(tx, ty, tx2, ty2, col32(OP1_DIM), 1.2)
    # Hand
    hand_len = radius - 2
    hx = cx + hand_len * math.cos(current_angle)
    hy = cy + hand_len * math.sin(current_angle)
    dl.add_line(cx, cy, hx, hy, col32(OP1_WHITE), 2.2)
    # Center dot
    dl.add_circle_filled(cx, cy, max(2.5, radius * 0.12), col32(OP1_WHITE), 14)

    # Label under the knob
    lbl_size = imgui.calc_text_size(label)
    dl.add_text(cx - lbl_size[0] * 0.5, knob_y + knob_size - th * 0.1,
                col32(color), label)

    # Big numeric value on the right
    num_x = knob_x + knob_size + th * 0.4
    num_w = wx + w - num_x - th * 0.4
    num_y = wy + th * 0.2
    num_h = panel_h - th * 0.4

    if decimals == 0:
        value_str = f"{int(round(value * display_scale))}"
    else:
        value_str = f"{value * display_scale:.{decimals}f}"

    if _gauge_editing_uid == uid:
        # Host an imgui.input_float inside an overlapping no-background window
        imgui.set_next_window_position(num_x, num_y + num_h * 0.18)
        imgui.set_next_window_size(num_w, num_h * 0.75)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.10, 0.10, 0.10, 1.0)
        imgui.begin(f"##gauge_edit_{uid}", flags=(
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR |
            imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BACKGROUND))
        imgui.push_item_width(num_w - 10)
        imgui.set_keyboard_focus_here()
        fmt = "%.0f" if decimals == 0 else f"%.{decimals}f"
        entered, new_v = imgui.input_float(
            f"##gauge_in_{uid}", float(value),
            step=0.0, step_fast=0.0, format=fmt,
            flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        imgui.pop_item_width()
        imgui.end()
        imgui.pop_style_color(2)
        if entered:
            value = max(min_val, min(max_val, float(new_v)))
            changed = True
            _gauge_editing_uid = None
        elif imgui.is_mouse_clicked(0) and not in_rect:
            # Clicking away cancels
            _gauge_editing_uid = None
    else:
        op1_number(app, num_x, num_y, num_w, num_h, value_str, color, align='center')

    # Unit in small grey text at bottom right
    if unit:
        u_size = imgui.calc_text_size(unit)
        dl.add_text(wx + w - u_size[0] - th * 0.4,
                    wy + panel_h - th * 1.1,
                    col32(OP1_GRAY), unit)

    imgui.dummy(w, panel_h + _pad())
    _tooltip_if_hovered(wx, wy, w, panel_h, tooltip)
    return changed, value


def op1_dual_gauge(app, uid_a: str, label_a: str, value_a: float, min_a: float, max_a: float, color_a: tuple,
                   uid_b: str, label_b: str, value_b: float, min_b: float, max_b: float, color_b: tuple,
                   height_units: float = 8.0, decimals: int = 2,
                   default_a: float | None = None,
                   default_b: float | None = None,
                   tooltip: str = "") -> tuple[bool, float, float]:
    """Two unified knob cells side-by-side in a single tinted panel.

    Uses the same 270° knob style as the POINT SIZE and RGB widgets
    (via :func:`op1_knob_cell`), so every knob in the sidebar looks
    and feels identical. An optional hover tooltip may be provided.
    """
    changed = False
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    panel_h = th * height_units
    # Dual-knob panel tinted with a blend of both accent colors.
    blend = tuple((color_a[i] * 0.5 + color_b[i] * 0.5) for i in range(4))
    draw_styled_rect(dl, wx, wy, w, panel_h, blend,
                     rounding=PANEL_ROUNDING, thickness=1.2)

    margin = th * 0.4
    cell_gap = th * 0.3
    cell_w = (w - margin * 2 - cell_gap) * 0.5
    cell_y = wy + th * 0.3
    cell_h = panel_h - th * 0.6

    cell_a_x = wx + margin
    cell_b_x = cell_a_x + cell_w + cell_gap

    # Subtle divider between the two cells to reinforce the split.
    dl.add_line(cell_a_x + cell_w + cell_gap * 0.5,
                wy + th * 0.5,
                cell_a_x + cell_w + cell_gap * 0.5,
                wy + panel_h - th * 0.5,
                col32((0.14, 0.14, 0.14, 1.0)), 1.0)

    c1, new_a = op1_knob_cell(
        app, cell_a_x, cell_y, cell_w, cell_h,
        uid=uid_a, label=label_a, value=value_a,
        min_val=min_a, max_val=max_a, color=color_a,
        decimals=decimals, default_val=default_a,
    )
    c2, new_b = op1_knob_cell(
        app, cell_b_x, cell_y, cell_w, cell_h,
        uid=uid_b, label=label_b, value=value_b,
        min_val=min_b, max_val=max_b, color=color_b,
        decimals=decimals, default_val=default_b,
    )

    imgui.dummy(w, panel_h + _pad())
    _tooltip_if_hovered(wx, wy, w, panel_h, tooltip)
    return (c1 or c2), new_a, new_b


# ============================================================================
# Stats strip
# ============================================================================

def op1_stats_strip(app, cloud_count: int, total_points: int,
                    pending: int, mode_str: str, fps: float,
                    selection_count: int = 0, active_name: str = None):
    """OP-1 stats strip.

    Top row: filename pill (left) + fps (right).
    Bottom row: cloud count (left) + point count (right).
    active_name: stem of the current cloud file; shown in the pill when set.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    # Measure the actual display-font height so the strip sizes itself
    # correctly at any DPI instead of guessing with th multipliers.
    display_font = (getattr(app.gui, 'font_display', None)
                    if getattr(app, 'gui', None) else None)
    if display_font is not None:
        imgui.push_font(display_font)
        num_h = imgui.calc_text_size("0")[1]
        imgui.pop_font()
    else:
        num_h = th * 2.0

    # Vertical layout (top → bottom):
    #   pad_top | pill row | gap | number (num_h) | gap | unit label | pad_bot
    pad_top = th * 0.45
    pill_h  = th * 1.1
    gap1    = th * 0.15
    gap2    = th * 0.05
    pad_bot = th * 0.35

    h = pad_top + pill_h + gap1 + num_h + gap2 + th + pad_bot
    op1_panel(dl, wx, wy, w, h)

    # --- Top row: pill (left) + fps (right) ---
    pill_y = wy + pad_top
    pill_x = wx + th * 0.6

    fps_str = f"{fps:.0f}"
    fps_color = OP1_GREEN if fps >= 45 else (OP1_RED if fps < 25 else OP1_WHITE)
    fps_size = imgui.calc_text_size(fps_str)
    unit_size = imgui.calc_text_size("fps")
    fps_x = wx + w - fps_size[0] - unit_size[0] - th * 0.5
    fps_ty = pill_y + (pill_h - th) * 0.5
    dl.add_text(fps_x, fps_ty, col32(fps_color), fps_str)
    dl.add_text(fps_x + fps_size[0] + th * 0.15, fps_ty, col32(OP1_GRAY), "fps")

    if pending > 0:
        pulse = (math.sin(time.time() * 4.0) + 1.0) * 0.5
        dl.add_circle_filled(fps_x - th * 0.6, fps_ty + th * 0.5,
                             th * 0.28, col32((1.0, 0.35 * pulse, 0.2, 1.0)), 14)

    mode_color = OP1_GREEN if mode_str == "VIEW" else OP1_ORANGE
    if mode_str == "TRAIN":
        mode_color = TEAL

    pill_label = active_name if active_name else mode_str
    max_pill_w = fps_x - pill_x - th * 0.8
    # Cache the elided label + its measured size keyed on
    # (label, mode, max width). The while-loop is O(len(name)) text
    # measurements per frame; memoising drops it to ~zero cost on the
    # steady-state frames where nothing changed.
    cache_key = (pill_label, mode_str, round(max_pill_w))
    cache = getattr(op1_stats_strip, '_pill_cache', None)
    if cache is None or cache[0] != cache_key:
        full_size = imgui.calc_text_size(pill_label)
        if full_size[0] > max_pill_w - th * 0.8:
            trunc = pill_label
            while len(trunc) > 1 and imgui.calc_text_size(trunc + "…")[0] > max_pill_w - th * 0.8:
                trunc = trunc[:-1]
            pill_label = trunc + "…"
        lbl_size = imgui.calc_text_size(pill_label)
        op1_stats_strip._pill_cache = (cache_key, pill_label, lbl_size)
    else:
        pill_label = cache[1]
        lbl_size = cache[2]
    pill_w = max(lbl_size[0] + th * 1.0, th * 3.5)

    dl.add_rect_filled(pill_x, pill_y, pill_x + pill_w, pill_y + pill_h,
                       col32(mode_color), pill_h * 0.5)
    dl.add_text(pill_x + (pill_w - lbl_size[0]) * 0.5,
                pill_y + (pill_h - th) * 0.5,
                col32(OP1_BG), pill_label)

    # --- Bottom row: big numbers + unit labels ---
    def _fmt(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}", "M"
        if n >= 1_000:
            return f"{n // 1000}", "K"
        return str(n), ""

    cell_y = pill_y + pill_h + gap1
    cell_h = num_h
    unit_label_y = cell_y + num_h + gap2

    has_sel = selection_count > 0
    pad_x = th * 0.6

    if has_sel:
        col_w = (w - pad_x * 2) / 3.0
        cols = [
            (wx + pad_x,            str(cloud_count), "",    OP1_BLUE,  "CLOUDS", 'left'),
            (wx + pad_x + col_w,    *_fmt(total_points),     OP1_GREEN, "PTS",    'center'),
            (wx + pad_x + col_w*2,  *_fmt(selection_count),  OP1_ORANGE, "SEL",   'right'),
        ]
    else:
        col_w = (w - pad_x * 2) / 2.0
        cols = [
            (wx + pad_x,           str(cloud_count), "",   OP1_BLUE,  "CLOUDS", 'left'),
            (wx + pad_x + col_w,   *_fmt(total_points),    OP1_GREEN, "PTS",    'right'),
        ]

    for cx, val_str, scale, color, unit_lbl, align in cols:
        unit_full = f"{scale} {unit_lbl}" if scale else unit_lbl
        op1_number(app, cx, cell_y, col_w, cell_h, val_str, color, align=align)
        lbl_sz = imgui.calc_text_size(unit_full)
        if align == 'left':
            lx = cx
        elif align == 'right':
            lx = cx + col_w - lbl_sz[0]
        else:
            lx = cx + (col_w - lbl_sz[0]) * 0.5
        dl.add_text(lx, unit_label_y, col32(OP1_GRAY), unit_full)

    imgui.dummy(w, h + _pad())


# ============================================================================
# Tool palette — OP-1 keycaps
# ============================================================================

_TOOL_TOOLTIPS = {
    'pick': "PICK (P)\nClick a single point. Shift-click to add, Ctrl-click to remove.",
    'box': "BOX (B)\nDrag a screen-space rectangle. Ctrl+scroll to set depth limit.",
    'lasso': "LASSO (O)\nDraw a freeform shape around points.",
    'brush': "BRUSH (K)\nClick+drag to paint a 3D sphere. Ctrl+scroll to resize.",
    'curve': "CURVE (U)\nDraw a polyline; points within threshold px are selected.\nCtrl+scroll to adjust threshold.",
    'move': "MOVE\nTranslate the cloud origin via the 3D gizmo.",
    'rotate': "ROTATE\nRotate the cloud via the 3D gizmo.",
    'measure_line': "DISTANCE\nClick two points to measure the distance between them.\nSnaps to the nearest point cloud vertex.",
    'measure_angle': "ANGLE\nClick three points: arm — vertex — arm.\nShows the angle at the middle (vertex) point.",
    'measure_landmark': "LANDMARK\nClick a single point to mark a named 3D position.\nSnaps to the nearest point cloud vertex.",
}


def op1_tool_palette(active_tool: str | None) -> str | None:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    tools = [
        ("PCK", "pick",         OP1_RED,                _icon_pick),
        ("BOX", "box",          OP1_BLUE,               _icon_box),
        ("LSO", "lasso",        OP1_GREEN,              _icon_lasso),
        ("BRU", "brush",        OP1_WHITE,              _icon_brush),
        ("CRV", "curve",        (0.7, 0.5, 0.95, 1.0), _icon_curve),
        ("MOV", "move",         OP1_ORANGE,             _icon_move),
        ("ROT", "rotate",       (1.0, 0.75, 0.2, 1.0), _icon_rotate),
        ("DST", "measure_line", (1.0, 0.2, 0.2, 1.0),  _icon_measure_line),
        ("ANG", "measure_angle",(1.0, 0.2, 0.2, 1.0),  _icon_measure_angle),
    ]

    n = len(tools)
    gap = th * 0.12
    btn_w = (w - gap * (n - 1)) / n
    btn_h = th * 2.6

    clicked = None
    for i, (label, name, color, icon_fn) in enumerate(tools):
        bx = wx + i * (btn_w + gap)
        is_active = active_tool == name

        # Keycap background — active gets the accent fill+border,
        # inactive gets a dimmed generic styled rect
        if is_active:
            draw_styled_rect_active(dl, bx, wy, btn_w, btn_h, color,
                                    rounding=BUTTON_ROUNDING)
        else:
            draw_styled_rect(dl, bx, wy, btn_w, btn_h, OP1_DIM,
                             rounding=BUTTON_ROUNDING, thickness=1.0)

        # Icon area
        icon_cx = bx + btn_w * 0.5
        icon_cy = wy + btn_h * 0.35
        icon_size = btn_h * 0.25
        icon_col = col32(color if is_active else OP1_GRAY)
        icon_fn(dl, icon_cx, icon_cy, icon_size, icon_col)

        # Label
        lbl_col = color if is_active else OP1_GRAY
        lbl_size = imgui.calc_text_size(label)
        dl.add_text(bx + (btn_w - lbl_size[0]) * 0.5,
                    wy + btn_h - th * 1.1,
                    col32(lbl_col), label)

    imgui.dummy(w, btn_h + _pad())
    # Per-tool tooltips (checks bounds against each keycap individually)
    mx_h, my_h = imgui.get_mouse_pos()
    for i, (_, name, _, _) in enumerate(tools):
        bx = wx + i * (btn_w + gap)
        if bx <= mx_h <= bx + btn_w and wy <= my_h <= wy + btn_h:
            tip = _TOOL_TOOLTIPS.get(name, "")
            if tip:
                imgui.set_tooltip(tip)
            break
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        for i, (_, name, _, _) in enumerate(tools):
            bx = wx + i * (btn_w + gap)
            if bx <= mx <= bx + btn_w:
                clicked = name
                break
    return clicked


def _icon_pick(dl, cx, cy, size, col):
    # Crosshair
    dl.add_line(cx - size, cy, cx + size, cy, col, 1.6)
    dl.add_line(cx, cy - size, cx, cy + size, col, 1.6)
    dl.add_circle(cx, cy, size * 0.5, col, 16, 1.4)


def _icon_box(dl, cx, cy, size, col):
    dl.add_rect(cx - size, cy - size * 0.7, cx + size, cy + size * 0.7,
                col, 0.0, 0, 1.6)


def _icon_grid(dl, cx, cy, size, col):
    """Tic-tac-toe grid icon."""
    a = size * 0.9
    # Outer square
    dl.add_rect(cx - a, cy - a, cx + a, cy + a, col, 0.0, 0, 1.4)
    # Two horizontal + two vertical inner lines
    third = a * 2.0 / 3.0
    dl.add_line(cx - a, cy - a + third, cx + a, cy - a + third, col, 1.2)
    dl.add_line(cx - a, cy - a + 2 * third, cx + a, cy - a + 2 * third, col, 1.2)
    dl.add_line(cx - a + third, cy - a, cx - a + third, cy + a, col, 1.2)
    dl.add_line(cx - a + 2 * third, cy - a, cx - a + 2 * third, cy + a, col, 1.2)


def _icon_lasso(dl, cx, cy, size, col):
    # Wavy loop
    pts = []
    for i in range(20):
        a = (i / 20.0) * 2.0 * math.pi
        r = size * (0.9 + 0.15 * math.sin(a * 3))
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        dl.add_line(x0, y0, x1, y1, col, 1.4)


def _icon_polygon(dl, cx, cy, size, col):
    # Pentagon outline with a small dot at each vertex — visually
    # distinguishes from the wavy LSO loop while reading as
    # "discrete-vertex polygon".
    n = 5
    pts = []
    for i in range(n):
        a = -math.pi / 2 + (i / n) * 2.0 * math.pi
        pts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        dl.add_line(x0, y0, x1, y1, col, 1.4)
    for x, y in pts:
        dl.add_circle_filled(x, y, max(1.5, size * 0.12), col, 8)


def _icon_brush(dl, cx, cy, size, col):
    # Filled circle with dotted ring
    dl.add_circle_filled(cx, cy, size * 0.45, col, 16)
    dl.add_circle(cx, cy, size, col, 20, 1.4)


def _icon_curve(dl, cx, cy, size, col):
    # Sinusoid
    prev_x = cx - size
    prev_y = cy
    for i in range(1, 20):
        t = i / 19.0
        x = cx - size + 2 * size * t
        y = cy + math.sin(t * math.pi * 2) * size * 0.6
        dl.add_line(prev_x, prev_y, x, y, col, 1.6)
        prev_x, prev_y = x, y


def _icon_measure_line(dl, cx, cy, size, col):
    """Ruler icon: horizontal line with endpoint dots and tick marks."""
    x0 = cx - size
    x1 = cx + size
    # Main line
    dl.add_line(x0, cy, x1, cy, col, 1.8)
    # Endpoint dots
    dl.add_circle_filled(x0, cy, size * 0.22, col, 8)
    dl.add_circle_filled(x1, cy, size * 0.22, col, 8)
    # Three tick marks (top and bottom)
    tick = size * 0.45
    for tx in (x0, cx, x1):
        dl.add_line(tx, cy - tick, tx, cy + tick, col, 1.2)


def _icon_measure_angle(dl, cx, cy, size, col):
    """Angle icon: two lines meeting at a vertex with a small arc."""
    # Vertex at bottom-left of icon
    vx = cx - size * 0.4
    vy = cy + size * 0.55
    # Arm 1: goes right
    a1x = cx + size * 0.9
    a1y = vy
    # Arm 2: goes up-right at ~50 deg
    a2x = cx + size * 0.3
    a2y = cy - size * 0.65
    dl.add_line(vx, vy, a1x, a1y, col, 1.8)
    dl.add_line(vx, vy, a2x, a2y, col, 1.8)
    # Small arc between arms (approximate with line segments)
    import math as _math
    arc_r = size * 0.48
    ang1 = 0.0           # arm 1 direction (right)
    ang2 = _math.radians(50)  # arm 2 direction
    steps = 10
    for i in range(steps):
        t0 = ang1 + (ang2 - ang1) * i / steps
        t1 = ang1 + (ang2 - ang1) * (i + 1) / steps
        dl.add_line(
            vx + arc_r * _math.cos(t0), vy - arc_r * _math.sin(t0),
            vx + arc_r * _math.cos(t1), vy - arc_r * _math.sin(t1),
            col, 1.2,
        )
    # Endpoint dots on each arm
    dl.add_circle_filled(a1x, a1y, size * 0.18, col, 8)
    dl.add_circle_filled(a2x, a2y, size * 0.18, col, 8)


def _icon_measure_landmark(dl, cx, cy, size, col):
    """Landmark icon: pin/crosshair target with a central filled dot."""
    # Outer ring
    dl.add_circle(cx, cy, size * 0.82, col, 16, 1.4)
    # Central filled dot
    dl.add_circle_filled(cx, cy, size * 0.32, col, 10)
    # Crosshair lines extending beyond the ring
    ext = size * 1.05
    gap = size * 0.50
    dl.add_line(cx - ext, cy, cx - gap, cy, col, 1.3)
    dl.add_line(cx + gap, cy, cx + ext, cy, col, 1.3)
    dl.add_line(cx, cy - ext, cx, cy - gap, col, 1.3)
    dl.add_line(cx, cy + gap, cx, cy + ext, col, 1.3)


# ============================================================================
# Cloud list row — tape deck style
# ============================================================================

def op1_cloud_row(index: int, name: str, point_count: int,
                  is_selected: bool, has_preview: bool,
                  accent_color: tuple = None,
                  label_colors: list = None,
                  is_in_multi_set: bool = False) -> bool:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    h = th * 1.6  # compact, matching label row height
    ac = accent_color or OP1_GREEN

    # Background:
    #   selected (single)  → strong accent fill
    #   in multi-set       → softer accent fill (lighter than the primary
    #                        selection so the user can distinguish "this is
    #                        the focused cloud" from "this is in the batch")
    #   neither            → standard dark
    r = BUTTON_ROUNDING
    if is_selected:
        tint = (ac[0] * 0.18, ac[1] * 0.18, ac[2] * 0.18, 1.0)
        dl.add_rect_filled(wx, wy, wx + w, wy + h,
                           imgui.get_color_u32_rgba(*tint), r)
    elif is_in_multi_set:
        tint = (ac[0] * 0.10, ac[1] * 0.10, ac[2] * 0.10, 1.0)
        dl.add_rect_filled(wx, wy, wx + w, wy + h,
                           imgui.get_color_u32_rgba(*tint), r)
    else:
        dl.add_rect_filled(wx, wy, wx + w, wy + h,
                           imgui.get_color_u32_rgba(0.11, 0.11, 0.11, 1.0), r)

    # LED status dot
    led_x = wx + th * 0.5
    led_y = wy + h * 0.5
    led_r = th * 0.18
    if not has_preview:
        pulse = (math.sin(time.time() * 4.0 + index) + 1.0) * 0.5
        dl.add_circle_filled(led_x, led_y, led_r,
                             col32((1.0, 0.6 * pulse, 0.15, 1.0)), 10)
    else:
        dl.add_circle_filled(led_x, led_y, led_r,
                             col32(ac if is_selected else OP1_GRAY), 10)

    # Name on left
    text_x = led_x + th * 0.5
    name_col = OP1_WHITE if is_selected else OP1_GRAY
    dl.add_text(text_x, wy + (h - th) * 0.5, col32(name_col), name[:24])

    # Right-side: [label swatches] [point count]
    pts_str = None
    pts_w = 0.0
    if point_count > 0:
        if point_count >= 1_000_000:
            pts_str = f"{point_count / 1_000_000:.1f}M"
        elif point_count >= 1_000:
            pts_str = f"{point_count // 1000}K"
        else:
            pts_str = str(point_count)
        pts_w = imgui.calc_text_size(pts_str)[0]

    # Label color swatches — drawn left of the point count, right-anchored
    if label_colors:
        max_swatches = 6
        n = min(len(label_colors), max_swatches)
        sw_w = th * 0.42
        sw_h = th * 0.85
        sw_gap = th * 0.12
        sw_total = n * sw_w + (n - 1) * sw_gap
        right_anchor = wx + w - (pts_w + th * 0.6 if pts_str else th * 0.3)
        sw_x0 = right_anchor - sw_total
        sw_y = wy + (h - sw_h) * 0.5
        sw_r = max(1.0, BUTTON_ROUNDING * 0.5)
        for i, color in enumerate(label_colors[:n]):
            cx = sw_x0 + i * (sw_w + sw_gap)
            dl.add_rect_filled(cx, sw_y, cx + sw_w, sw_y + sw_h,
                               col32(color), sw_r)
        if len(label_colors) > n:
            plus = f"+{len(label_colors) - n}"
            plus_w = imgui.calc_text_size(plus)[0]
            dl.add_text(sw_x0 - plus_w - th * 0.15,
                        wy + (h - th) * 0.5,
                        col32(OP1_DIM), plus)

    if pts_str is not None:
        dl.add_text(wx + w - pts_w - th * 0.4, wy + (h - th) * 0.5,
                    col32(OP1_GRAY), pts_str)

    imgui.dummy(w, h + 2)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        return True
    return False


# ============================================================================
# Overlay toggles — OP-1 switch style
# ============================================================================

def op1_toggles(show_bbox: bool, show_grid: bool) -> tuple[bool, bool]:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    new_bbox = show_bbox
    new_grid = show_grid

    toggles = [
        ("BBOX", show_bbox, OP1_BLUE),
        ("GRID", show_grid, OP1_GREEN),
    ]
    n = len(toggles)
    gap = th * 0.3
    btn_w = (w - gap * (n - 1)) / n
    btn_h = th * 2.0

    mx, my = imgui.get_mouse_pos()

    for i, (label, active, color) in enumerate(toggles):
        bx = wx + i * (btn_w + gap)

        # Base panel (always filled+bordered)
        if active:
            draw_styled_rect_active(dl, bx, wy, btn_w, btn_h, color,
                                    rounding=BUTTON_ROUNDING)
        else:
            draw_styled_rect(dl, bx, wy, btn_w, btn_h, OP1_DIM,
                             rounding=BUTTON_ROUNDING, thickness=1.0)

        # Switch track on the right
        track_w = th * 2.0
        track_h = th * 0.8
        track_x = bx + btn_w - track_w - th * 0.4
        track_y = wy + (btn_h - track_h) * 0.5
        dl.add_rect_filled(track_x, track_y, track_x + track_w, track_y + track_h,
                           col32((0.08, 0.08, 0.08, 1.0)), track_h * 0.5)
        dl.add_rect(track_x, track_y, track_x + track_w, track_y + track_h,
                    col32(color if active else OP1_DIM), track_h * 0.5, 0, 1.0)

        # Knob
        knob_r = track_h * 0.42
        knob_x = (track_x + track_w - knob_r - 2) if active else (track_x + knob_r + 2)
        knob_y = track_y + track_h * 0.5
        dl.add_circle_filled(knob_x, knob_y, knob_r,
                             col32(color if active else OP1_GRAY), 16)

        # Label on the left
        dl.add_text(bx + th * 0.5, wy + (btn_h - th) * 0.5,
                    col32(color if active else OP1_GRAY), label)

        # State text
        state = "ON" if active else "OFF"
        dl.add_text(bx + btn_w * 0.45, wy + (btn_h - th) * 0.5,
                    col32(OP1_GRAY), state)

    imgui.dummy(w, btn_h + _pad())

    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        for i in range(n):
            bx = wx + i * (btn_w + gap)
            if bx <= mx <= bx + btn_w:
                if i == 0:
                    new_bbox = not show_bbox
                elif i == 1:
                    new_grid = not show_grid

    # Right-click anywhere on the row resets both toggles to OFF
    if _rclick_in_rect(wx, wy, w, btn_h):
        new_bbox = False
        new_grid = False

    return new_bbox, new_grid


def op1_overlay_squares(show_bbox: bool, show_grid: bool) -> tuple[bool, bool]:
    """Compact overlay toggles — square icon cells with a label below.

    Tighter than op1_toggles: each cell is a square with the icon centered
    in the upper area and the label tucked along the bottom edge.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    cells = [
        ('BBOX', show_bbox, OP1_BLUE, _icon_box),
        ('GRID', show_grid, OP1_GREEN, _icon_grid),
    ]
    cell_size = th * 3.0
    gap = _th() * 0.3

    new_states = [show_bbox, show_grid]
    mx, my = imgui.get_mouse_pos()
    for i, (label, active, color, icon_fn) in enumerate(cells):
        cx0 = wx + i * (cell_size + gap)
        cy0 = wy
        if active:
            draw_styled_rect_active(dl, cx0, cy0, cell_size, cell_size, color,
                                    rounding=BUTTON_ROUNDING)
        else:
            draw_styled_rect(dl, cx0, cy0, cell_size, cell_size, OP1_DIM,
                             rounding=BUTTON_ROUNDING, thickness=1.0)
        # Icon centered in upper 65% of cell
        icon_fn(dl, cx0 + cell_size * 0.5, cy0 + cell_size * 0.38,
                cell_size * 0.20, col32(color if active else OP1_GRAY))
        # Label centered in lower band
        lbl_w = imgui.calc_text_size(label)[0]
        dl.add_text(cx0 + (cell_size - lbl_w) * 0.5,
                    cy0 + cell_size - th * 1.05,
                    col32(color if active else OP1_GRAY), label)
        # Click detection
        if (cx0 <= mx <= cx0 + cell_size and cy0 <= my <= cy0 + cell_size
                and imgui.is_mouse_clicked(0)):
            new_states[i] = not active

    imgui.dummy(w, cell_size + th * 0.3)
    return new_states[0], new_states[1]


# ============================================================================
# Gizmo mode selector (MOVE / ROT)
# ============================================================================

def op1_gizmo_mode(current_mode: str, visible: bool) -> tuple[str, bool]:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    new_mode = current_mode
    new_visible = visible
    btn_h = th * 2.2
    gap = th * 0.3
    half_w = (w - gap) * 0.5

    mx, my = imgui.get_mouse_pos()

    def _keycap(x, label, active, color, icon_fn):
        if active:
            draw_styled_rect_active(dl, x, wy, half_w, btn_h, color,
                                    rounding=BUTTON_ROUNDING)
        else:
            draw_styled_rect(dl, x, wy, half_w, btn_h, OP1_DIM,
                             rounding=BUTTON_ROUNDING, thickness=1.0)
        icon_cx = x + th * 1.0
        icon_cy = wy + btn_h * 0.5
        icon_fn(dl, icon_cx, icon_cy, th * 0.6,
                col32(color if active else OP1_GRAY))
        dl.add_text(x + th * 2.2, wy + (btn_h - th) * 0.5,
                    col32(color if active else OP1_GRAY), label)

    _keycap(wx, "MOVE", current_mode == 'translate' and visible,
            OP1_BLUE, _icon_move)
    _keycap(wx + half_w + gap, "ROT", current_mode == 'rotate' and visible,
            OP1_GREEN, _icon_rotate)

    imgui.dummy(w, btn_h + _pad())
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        if wx <= mx <= wx + half_w:
            new_mode = 'translate'
            new_visible = True
        elif wx + half_w + gap <= mx <= wx + w:
            new_mode = 'rotate'
            new_visible = True
    return new_mode, new_visible


def _icon_move(dl, cx, cy, size, col):
    # Four-arrow cross
    dl.add_line(cx - size, cy, cx + size, cy, col, 1.6)
    dl.add_line(cx, cy - size, cx, cy + size, col, 1.6)
    # Arrowheads
    for (dx, dy) in [(size, 0), (-size, 0), (0, size), (0, -size)]:
        # tiny perpendicular offset
        if dx != 0:
            dl.add_line(cx + dx, cy, cx + dx - dx * 0.3, cy - size * 0.3, col, 1.4)
            dl.add_line(cx + dx, cy, cx + dx - dx * 0.3, cy + size * 0.3, col, 1.4)
        else:
            dl.add_line(cx, cy + dy, cx - size * 0.3, cy + dy - dy * 0.3, col, 1.4)
            dl.add_line(cx, cy + dy, cx + size * 0.3, cy + dy - dy * 0.3, col, 1.4)


def _icon_rotate(dl, cx, cy, size, col):
    # 3/4 circle arrow
    for i in range(20):
        a0 = -math.pi * 0.75 + (i / 20) * math.pi * 1.5
        a1 = -math.pi * 0.75 + ((i + 1) / 20) * math.pi * 1.5
        x0 = cx + size * math.cos(a0)
        y0 = cy + size * math.sin(a0)
        x1 = cx + size * math.cos(a1)
        y1 = cy + size * math.sin(a1)
        dl.add_line(x0, y0, x1, y1, col, 1.4)


# ============================================================================
# Import / Export / Apply buttons — OP-1 key caps
# ============================================================================

def op1_import_buttons() -> tuple[bool, bool]:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    gap = th * 0.3
    half_w = (w - gap) * 0.5
    btn_h = th * 2.4

    x0 = wx
    x1 = wx + half_w + gap

    _keycap_with_icon(dl, x0, wy, half_w, btn_h, "FILE", OP1_BLUE, _icon_file)
    _keycap_with_icon(dl, x1, wy, half_w, btn_h, "FOLDER",
                      OP1_GREEN, _icon_folder)

    imgui.dummy(w, btn_h + _pad())
    file_clicked = False
    folder_clicked = False
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        if x0 <= mx <= x0 + half_w:
            file_clicked = True
        elif x1 <= mx <= x1 + half_w:
            folder_clicked = True
    return file_clicked, folder_clicked


def op1_export_buttons(app) -> tuple[bool, bool]:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    gap = th * 0.3
    half_w = (w - gap) * 0.5
    btn_h = th * 2.4

    _keycap_with_icon(dl, wx, wy, half_w, btn_h, "STILL", OP1_WHITE,
                      _icon_camera)
    _keycap_with_icon(dl, wx + half_w + gap, wy, half_w, btn_h, "SPIN",
                      OP1_GREEN, _icon_spin)

    imgui.dummy(w, btn_h + _pad())
    still_clicked = False
    spin_clicked = False
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        if wx <= mx <= wx + half_w:
            still_clicked = True
        elif wx + half_w + gap <= mx <= wx + w:
            spin_clicked = True
    return still_clicked, spin_clicked


def _keycap_with_icon(dl, x, y, w, h, label, color, icon_fn):
    draw_styled_rect(dl, x, y, w, h, color,
                     rounding=BUTTON_ROUNDING, thickness=1.4)
    icon_cx = x + w * 0.5
    icon_cy = y + h * 0.4
    icon_fn(dl, icon_cx, icon_cy, h * 0.2, col32(color))
    lbl_size = imgui.calc_text_size(label)
    th = _th()
    dl.add_text(x + (w - lbl_size[0]) * 0.5, y + h - th * 1.05,
                col32(color), label)


def _icon_file(dl, cx, cy, size, col):
    dl.add_rect(cx - size * 0.7, cy - size, cx + size * 0.7, cy + size,
                col, 0.0, 0, 1.4)
    dl.add_line(cx - size * 0.3, cy - size * 0.4, cx + size * 0.3, cy - size * 0.4,
                col, 1.2)
    dl.add_line(cx - size * 0.3, cy, cx + size * 0.3, cy, col, 1.2)
    dl.add_line(cx - size * 0.3, cy + size * 0.4, cx + size * 0.3, cy + size * 0.4,
                col, 1.2)


def _icon_folder(dl, cx, cy, size, col):
    # Folder tab + body
    dl.add_rect(cx - size, cy - size * 0.7, cx - size * 0.2, cy - size * 0.3,
                col, 0.0, 0, 1.4)
    dl.add_rect(cx - size, cy - size * 0.4, cx + size, cy + size * 0.8,
                col, 0.0, 0, 1.4)


def _icon_camera(dl, cx, cy, size, col):
    # Camera rectangle with lens circle
    dl.add_rect(cx - size, cy - size * 0.6, cx + size, cy + size * 0.6,
                col, size * 0.15, 0, 1.4)
    dl.add_circle(cx, cy, size * 0.35, col, 16, 1.4)


def _icon_spin(dl, cx, cy, size, col):
    # 3/4 circle with arrowhead
    for i in range(18):
        a0 = (i / 18) * math.pi * 1.6
        a1 = ((i + 1) / 18) * math.pi * 1.6
        x0 = cx + size * math.cos(a0)
        y0 = cy + size * math.sin(a0)
        x1 = cx + size * math.cos(a1)
        y1 = cy + size * math.sin(a1)
        dl.add_line(x0, y0, x1, y1, col, 1.4)


def _icon_volume(dl, cx, cy, size, col):
    # Stacked layers / z-stack icon (three offset horizontal planes)
    s = size
    for dy in (-s * 0.55, 0.0, s * 0.55):
        y = cy + dy
        dl.add_line(cx - s * 0.8, y, cx + s * 0.8, y, col, 1.4)
        # Small edge ticks to suggest depth
        dl.add_line(cx - s * 0.8, y, cx - s * 0.6, y - s * 0.2, col, 1.0)
        dl.add_line(cx + s * 0.8, y, cx + s * 0.6, y - s * 0.2, col, 1.0)


# ============================================================================
# Apply label button — big OP-1 display
# ============================================================================

def op1_apply_button(app, selection_count: int, label_name: str) -> bool:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    h = th * 2.8
    enabled = selection_count > 0
    color = OP1_RED if enabled else OP1_DIM

    if enabled:
        draw_styled_rect_active(dl, wx, wy, w, h, color,
                                rounding=PANEL_ROUNDING, thickness=2.0)
    else:
        op1_panel(dl, wx, wy, w, h)

    if enabled:
        # Two-line display: "APPLY label" / "N points"
        big_line = label_name.upper()[:14]
        op1_number(app, wx, wy, w, h * 0.7, big_line, color, align='center')
        pts_str = f"{selection_count:,} pts"
        ps_size = imgui.calc_text_size(pts_str)
        dl.add_text(wx + (w - ps_size[0]) * 0.5,
                    wy + h - th * 1.1, col32(color), pts_str)
    else:
        msg = "SELECT POINTS TO LABEL"
        ms_size = imgui.calc_text_size(msg)
        dl.add_text(wx + (w - ms_size[0]) * 0.5,
                    wy + (h - th) * 0.5, col32(color), msg)

    imgui.dummy(w, h + _pad())
    if not enabled:
        return False
    return imgui.is_item_hovered() and imgui.is_mouse_clicked(0)


# ============================================================================
# Background color swatches — OP-1 styled
# ============================================================================

def op1_bg_swatches(bg_color: tuple) -> tuple:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    presets = [
        ("BLK", (0.0, 0.0, 0.0)),
        ("DRK", (0.08, 0.08, 0.08)),
        ("GRY", (0.25, 0.25, 0.25)),
        ("LGT", (0.85, 0.85, 0.85)),
    ]
    n = len(presets)
    gap = th * 0.25
    h = th * 1.7
    sw_w = (w - gap * (n - 1)) / n

    new_color = bg_color
    for i, (label, color) in enumerate(presets):
        sx = wx + i * (sw_w + gap)
        is_active = (abs(bg_color[0] - color[0]) < 0.02 and
                     abs(bg_color[1] - color[1]) < 0.02)
        # Swatch fill shows the actual background color; border hints at
        # selection state. Rounding matches other buttons.
        dl.add_rect_filled(sx, wy, sx + sw_w, wy + h,
                           col32((*color, 1.0)), BUTTON_ROUNDING)
        if is_active:
            dl.add_rect(sx, wy, sx + sw_w, wy + h,
                        col32(OP1_BLUE), BUTTON_ROUNDING, 0, 2.0)
        else:
            dl.add_rect(sx, wy, sx + sw_w, wy + h,
                        col32(tinted_border(OP1_DIM)),
                        BUTTON_ROUNDING, 0, 1.0)
        lbl_col = OP1_BG if color[0] > 0.5 else OP1_GRAY
        lbl_size = imgui.calc_text_size(label)
        dl.add_text(sx + (sw_w - lbl_size[0]) * 0.5,
                    wy + (h - th) * 0.5, col32(lbl_col), label)

    imgui.dummy(w, h + _pad())
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        for i, (_, color) in enumerate(presets):
            sx = wx + i * (sw_w + gap)
            if sx <= mx <= sx + sw_w:
                new_color = color
                break
    # Right-click resets to black
    if _rclick_in_rect(wx, wy, w, h):
        new_color = (0.0, 0.0, 0.0)
    return new_color


# ============================================================================
# Resample control — OP-1 segment buttons
# ============================================================================

def op1_resample(app) -> bool:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    options = [("50", 0.5), ("25", 0.25), ("10", 0.10)]
    n = len(options)
    gap = th * 0.25
    btn_w = (w - gap * (n - 1)) / n
    btn_h = th * 2.2

    for i, (label, frac) in enumerate(options):
        bx = wx + i * (btn_w + gap)
        draw_styled_rect(dl, bx, wy, btn_w, btn_h, OP1_GRAY,
                         rounding=BUTTON_ROUNDING, thickness=1.0)
        lbl = f"{label}%"
        lbl_size = imgui.calc_text_size(lbl)
        dl.add_text(bx + (btn_w - lbl_size[0]) * 0.5,
                    wy + (btn_h - th) * 0.5, col32(OP1_GRAY), lbl)

    imgui.dummy(w, btn_h + _pad())
    # Resample is mostly a visual indicator right now; real implementation
    # is attached via draw_resample_control elsewhere. We return False.
    return False


# ============================================================================
# Camera preset buttons — geometric face icons
# ============================================================================

# ============================================================================
# OP-1 tape slider (vertical channel with tick marks + colored cap)
# ============================================================================

_tape_drag_key: str | None = None


def op1_tape_slider(app, uid: str, label: str, value: float,
                    min_val: float, max_val: float, color: tuple,
                    width_units: float = 2.4,
                    height_units: float = 12.0,
                    default_val: float | None = None) -> tuple[bool, float]:
    """Single vertical tape slider, modeled on the OP-1 Figma reference.

    Visual:
      - Vertical channel with ~18 short tick marks at 14px spacing
      - Center tick is wider (extends past the channel) — neutral/default mark
      - Colored slider cap (70x32 @ reference scale) with a white highlight bar
      - 2-char label above the slider in the channel color

    Interaction:
      - Click+drag vertically on the cap to move it
      - Double-click or right-click to reset to default_val (midpoint if None)
    """
    reset_val = default_val if default_val is not None else (min_val + max_val) * 0.5
    global _tape_drag_key
    changed = False

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    th = _th()

    w = th * width_units
    h = th * height_units

    # Label at top in channel color
    lbl_size = imgui.calc_text_size(label)
    dl.add_text(wx + (w - lbl_size[0]) * 0.5,
                wy + th * 0.15, col32(color), label)

    # Channel area starts below the label
    channel_top = wy + th * 1.6
    channel_bot = wy + h - th * 0.6
    channel_h = channel_bot - channel_top
    channel_cx = wx + w * 0.5

    # Tick marks — 18 evenly spaced short lines, center tick extra wide
    tick_count = 18
    tick_short = w * 0.55
    tick_long = w * 0.95
    tick_col = col32((0.24, 0.24, 0.24, 1.0))
    for i in range(tick_count):
        ty = channel_top + (i / (tick_count - 1)) * channel_h
        is_center = (i == (tick_count - 1) // 2 or i == tick_count // 2)
        tlen = tick_long if is_center else tick_short
        dl.add_line(channel_cx - tlen * 0.5, ty,
                    channel_cx + tlen * 0.5, ty, tick_col, 2.0)

    # Cap position from current value (inverted: max at top, min at bottom)
    span = max(max_val - min_val, 1e-6)
    frac = max(0.0, min(1.0, (value - min_val) / span))
    cap_cy = channel_bot - frac * channel_h

    # Cap dimensions
    cap_w = w * 1.05
    cap_h = th * 1.6
    cap_x0 = channel_cx - cap_w * 0.5
    cap_y0 = cap_cy - cap_h * 0.5

    # Cap fill
    dl.add_rect_filled(cap_x0, cap_y0, cap_x0 + cap_w, cap_y0 + cap_h,
                       col32(color), 2.0)
    # Subtle darker border
    dl.add_rect(cap_x0, cap_y0, cap_x0 + cap_w, cap_y0 + cap_h,
                col32((color[0] * 0.4, color[1] * 0.4, color[2] * 0.4, 1.0)),
                2.0, 0, 1.0)
    # White highlight bar across the middle of the cap
    hl_y = cap_y0 + cap_h * 0.4
    dl.add_rect_filled(cap_x0, hl_y, cap_x0 + cap_w, hl_y + cap_h * 0.18,
                       col32((0.80, 0.82, 0.76, 1.0)))

    # Interaction: drag the cap to change the value
    imgui.dummy(w, h)
    if imgui.is_item_hovered():
        mxp, myp = imgui.get_mouse_pos()
        in_cap = (cap_x0 <= mxp <= cap_x0 + cap_w and
                  cap_y0 - 4 <= myp <= cap_y0 + cap_h + 4)
        in_channel = (wx <= mxp <= wx + w and
                      channel_top - 4 <= myp <= channel_bot + 4)

        if imgui.is_mouse_clicked(0) and (in_cap or in_channel):
            _tape_drag_key = uid
            if in_channel and not in_cap:
                new_frac = 1.0 - (myp - channel_top) / channel_h
                value = min_val + max(0.0, min(1.0, new_frac)) * span
                changed = True

        if in_cap and imgui.is_mouse_double_clicked(0):
            value = reset_val
            changed = True

    if _rclick_in_rect(wx, wy, w, h):
        value = reset_val
        changed = True

    if _tape_drag_key == uid:
        if imgui.is_mouse_down(0):
            dy = imgui.get_io().mouse_delta[1]
            if abs(dy) > 0.001:
                value = value - (dy / channel_h) * span
                value = max(min_val, min(max_val, value))
                changed = True
        else:
            _tape_drag_key = None

    return changed, value


def op1_clip_rack(app) -> bool:
    """Six-slider clip AABB rack: X/Y/Z min and max.

    Uses the currently selected cloud's bounds as the slider ranges. If no
    cloud is selected, the rack is drawn disabled.
    """
    changed = False

    if not (0 <= app.selected_index < len(app.entries)):
        return False
    entry = app.entries[app.selected_index]
    if entry.bounds_min is None:
        return False

    import numpy as _np
    bmin = entry.bounds_min.astype(_np.float32)
    bmax = entry.bounds_max.astype(_np.float32)

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    panel_h = th * 9.0
    op1_panel(dl, wx, wy, w, panel_h)

    # Six sliders: X min, X max, Y min, Y max, Z min, Z max
    # Colors pulled from existing axis conventions: X=CORAL, Y=GREEN, Z=BLUE
    x_color = CORAL
    y_color = OP1_GREEN
    z_color = OP1_BLUE

    # Each slider's "default" (reset target) is the extent on that side:
    # the min sliders reset to bounds.min, the max sliders reset to bounds.max.
    sliders = [
        ("R-", "clip_x_min", float(app.clip_min[0]), float(bmin[0]), float(bmax[0]), x_color, 0, 'min', float(bmin[0])),
        ("R+", "clip_x_max", float(app.clip_max[0]), float(bmin[0]), float(bmax[0]), x_color, 0, 'max', float(bmax[0])),
        ("S-", "clip_y_min", float(app.clip_min[1]), float(bmin[1]), float(bmax[1]), y_color, 1, 'min', float(bmin[1])),
        ("S+", "clip_y_max", float(app.clip_max[1]), float(bmin[1]), float(bmax[1]), y_color, 1, 'max', float(bmax[1])),
        ("A-", "clip_z_min", float(app.clip_min[2]), float(bmin[2]), float(bmax[2]), z_color, 2, 'min', float(bmin[2])),
        ("A+", "clip_z_max", float(app.clip_max[2]), float(bmin[2]), float(bmax[2]), z_color, 2, 'max', float(bmax[2])),
    ]

    n = len(sliders)
    margin = th * 0.5
    avail = w - margin * 2
    gap = th * 0.18
    slider_area_w = (avail - gap * (n - 1)) / n

    # Render each slider into its own sub-column
    cursor_screen_x = wx + margin
    cursor_screen_y = wy + th * 0.3

    current_values = [s[2] for s in sliders]

    slider_h_units = max(1.5, (panel_h - th * 0.6) / th)

    for i, (label, uid, value, lo, hi, color, axis, which, default_val) in enumerate(sliders):
        # Clamp the current value to the slider range (in case the cloud
        # changed size under us)
        value = max(lo, min(hi, value))
        # Position this slider
        imgui.set_cursor_screen_pos((cursor_screen_x + i * (slider_area_w + gap),
                                     cursor_screen_y))
        # Shrink the tape slider to fit the allotted column
        width_units = max(1.6, slider_area_w / th)
        c, new_val = op1_tape_slider(
            app, uid, label, value, lo, hi, color,
            width_units=width_units, height_units=slider_h_units,
            default_val=default_val,
        )
        if c:
            if which == 'min':
                other = app.clip_max[axis]
                new_val = min(new_val, other - (hi - lo) * 0.001)
                app.clip_min[axis] = new_val
            else:
                other = app.clip_min[axis]
                new_val = max(new_val, other + (hi - lo) * 0.001)
                app.clip_max[axis] = new_val
            changed = True

    # Reset the cursor below the rack and reserve the full panel height so
    # the next widget lays out below it (imgui.dummy with w, h actually
    # advances the layout cursor; imgui.dummy(w, 0) does not).
    imgui.set_cursor_screen_pos((wx, wy))
    imgui.dummy(w, panel_h + _pad())

    return changed


def op1_clip_h_rack(app) -> bool:
    """Three horizontal range sliders (X/Y/Z) for dual-sided clipping.

    Each slider has two thumbs: left = lo fraction (clips from min side),
    right = hi fraction (clips from max side). The visible slab is the
    region between the two thumbs.
    """
    from src.gui.scale import s as _s
    changed = False
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    rows = [
        ('R', CORAL, 0),
        ('S', OP1_GREEN, 1),
        ('A', OP1_BLUE, 2),
    ]

    row_h = th * 1.9
    row_gap = _s(4)
    panel_h = len(rows) * row_h + (len(rows) - 1) * row_gap

    mx, my = imgui.get_mouse_pos()

    for label, color, axis in rows:
        ry = wy + axis * (row_h + row_gap)

        lbl_w = th * 1.2
        gap = _s(6)
        slider_x = wx + lbl_w + gap
        slider_w = w - lbl_w - gap

        # Axis label
        dl.add_text(wx + _s(2), ry + (row_h - th) * 0.5,
                    col32(color), label)

        # Slider track background
        track_h = _s(6)
        track_y = ry + row_h * 0.5 - track_h * 0.5
        dl.add_rect_filled(slider_x, track_y, slider_x + slider_w, track_y + track_h,
                           col32((0.10, 0.10, 0.10, 1.0)), _s(2))

        frac_lo = max(0.0, min(1.0, float(app.clip_fraction_lo[axis])))
        frac_hi = max(0.0, min(1.0, float(app.clip_fraction_hi[axis])))

        # Lo thumb at left side, hi thumb at right side
        lo_x = slider_x + slider_w * frac_lo
        hi_x = slider_x + slider_w * (1.0 - frac_hi)

        # Dim clipped regions, bright kept region
        dim_col = col32((color[0] * 0.3, color[1] * 0.3, color[2] * 0.3, 0.5))
        if frac_lo > 0.001:
            dl.add_rect_filled(slider_x, track_y, lo_x, track_y + track_h,
                               dim_col, _s(2))
        if frac_hi > 0.001:
            dl.add_rect_filled(hi_x, track_y, slider_x + slider_w, track_y + track_h,
                               dim_col, _s(2))
        # Bright kept slab
        dl.add_rect_filled(lo_x, track_y, hi_x, track_y + track_h,
                           col32(color), _s(2))

        # Two knob bars
        _knob_hw = _s(4)
        _knob_hh = _s(6)
        knob_y = track_y + track_h * 0.5
        dl.add_rect_filled(lo_x - _knob_hw, knob_y - _knob_hh,
                           lo_x + _knob_hw, knob_y + _knob_hh,
                           col32(color), _s(1.5))
        dl.add_rect_filled(hi_x - _knob_hw, knob_y - _knob_hh,
                           hi_x + _knob_hw, knob_y + _knob_hh,
                           col32(color), _s(1.5))

        # Hit detection — range slider
        in_slider = (slider_x <= mx <= slider_x + slider_w
                     and track_y - _s(8) <= my <= track_y + track_h + _s(8))

        drag_key = f'_clip_hrack_drag_{axis}'
        if in_slider and imgui.is_mouse_clicked(0):
            # Grab whichever thumb is closer
            dist_lo = abs(mx - lo_x)
            dist_hi = abs(mx - hi_x)
            setattr(app, drag_key, 'lo' if dist_lo <= dist_hi else 'hi')

        side = getattr(app, drag_key, None)
        if side is not None:
            if imgui.is_mouse_down(0):
                if side == 'lo':
                    new_frac = (mx - slider_x) / max(slider_w, 1.0)
                    new_frac = max(0.0, min(1.0, new_frac))
                    if abs(new_frac - frac_lo) > 1e-6:
                        app.clip_fraction_lo[axis] = float(new_frac)
                        changed = True
                else:
                    new_frac = (slider_x + slider_w - mx) / max(slider_w, 1.0)
                    new_frac = max(0.0, min(1.0, new_frac))
                    if abs(new_frac - frac_hi) > 1e-6:
                        app.clip_fraction_hi[axis] = float(new_frac)
                        changed = True
            else:
                setattr(app, drag_key, None)

        if in_slider and imgui.is_mouse_double_clicked(0):
            app.clip_fraction_lo[axis] = 0.0
            app.clip_fraction_hi[axis] = 0.0
            changed = True

    imgui.dummy(w, panel_h + row_gap)
    if changed:
        app._sync_clip_box()
    return changed


_envelope_drag: tuple[str, int] | None = None  # (widget_uid, point_index)


def op1_envelope(app, uid: str, envelope, color_primary: tuple = OP1_BLUE,
                 color_secondary: tuple = OP1_GREEN,
                 height_units: float = 5.0,
                 x_label: str = "", y_label: str = "",
                 default_points: list[tuple[float, float]] | None = None) -> bool:
    """Draggable envelope editor modeled on the OP-1 Figma reference.

    - Panel background (OP-1 BG), rounded
    - Dim baseline axis across the bottom
    - Vertical tick ruler at each control point position (dashed dim blue)
    - Curve segments drawn in primary color, one segment per interior point
      is drawn in secondary color for visual rhythm
    - Circle handles at each control point (last handle gets secondary color)
    - Click an empty area to ADD a control point
    - Drag a handle to move it (endpoints can only move vertically)
    - Right-click a handle to REMOVE it (non-endpoint only)
    - Right-click empty area to RESET the entire envelope to default_points

    envelope: the src.core.envelope.Envelope instance (modified in place)
    default_points: baseline curve to reset to on empty-area right-click
    Returns True if the envelope was changed this frame.
    """
    from src.core.envelope import Envelope
    global _envelope_drag
    changed = False

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    panel_h = th * height_units

    op1_panel(dl, wx, wy, w, panel_h)

    # Curve area — leave a small interior margin
    mx_ = th * 1.0
    my_ = th * 0.8
    plot_x = wx + mx_
    plot_y = wy + my_
    plot_w = w - mx_ * 2
    plot_h = panel_h - my_ * 2

    def _pt_to_screen(tp, vp):
        px = plot_x + tp * plot_w
        py = plot_y + (1.0 - vp) * plot_h
        return px, py

    # Baseline (horizontal axis) at y=0
    baseline_y = plot_y + plot_h
    dl.add_line(plot_x, baseline_y, plot_x + plot_w, baseline_y,
                col32((0.22, 0.33, 0.65, 1.0)), 2.0)

    # Top line
    top_y = plot_y
    dl.add_line(plot_x, top_y, plot_x + plot_w, top_y,
                col32((0.12, 0.12, 0.12, 1.0)), 1.0)

    points = envelope.points

    # Vertical tick rulers at each interior control point (skip endpoints)
    tick_color = col32((0.22, 0.33, 0.65, 1.0))
    for i in range(1, len(points) - 1):
        tpt, vpt = points[i]
        tx = plot_x + tpt * plot_w
        # Dashed ticks from bottom to top
        segment = 5.0
        gap = 3.0
        cy = baseline_y
        while cy > top_y:
            y0 = cy
            y1 = max(top_y, cy - segment)
            dl.add_line(tx, y0, tx, y1, tick_color, 1.5)
            cy = y1 - gap

    # Curve segments between control points
    for i in range(len(points) - 1):
        t0, v0 = points[i]
        t1, v1 = points[i + 1]
        x0, y0 = _pt_to_screen(t0, v0)
        x1, y1 = _pt_to_screen(t1, v1)
        # Alternate colors: even segments primary, odd secondary (mimics Figma)
        seg_col = color_primary if i % 2 == 0 else color_secondary
        dl.add_line(x0, y0, x1, y1, col32(seg_col), 2.5)

    # Control point handles (last one gets secondary color)
    handle_r = max(5.0, th * 0.38)
    for i, (tpt, vpt) in enumerate(points):
        px, py = _pt_to_screen(tpt, vpt)
        is_last = (i == len(points) - 1)
        handle_col = color_secondary if is_last else color_primary
        dl.add_circle_filled(px, py, handle_r, col32(handle_col), 20)

    # X / Y labels — drawn inside the plot area with enough inset that
    # they can't graze the rounded panel border.
    if x_label:
        lbl_size = imgui.calc_text_size(x_label)
        dl.add_text(plot_x + plot_w - lbl_size[0] - th * 0.2,
                    plot_y + plot_h - lbl_size[1] - th * 0.15,
                    col32(OP1_GRAY), x_label)
    if y_label:
        dl.add_text(plot_x + th * 0.2,
                    plot_y + th * 0.15,
                    col32(OP1_GRAY), y_label)

    # Interaction
    imgui.dummy(w, panel_h + _pad())
    if imgui.is_item_hovered():
        mxp, myp = imgui.get_mouse_pos()

        # Find hovered point
        hover_idx = -1
        for i, (tpt, vpt) in enumerate(points):
            px, py = _pt_to_screen(tpt, vpt)
            if (mxp - px) ** 2 + (myp - py) ** 2 <= (handle_r * 1.6) ** 2:
                hover_idx = i
                break

        if imgui.is_mouse_clicked(0):
            if hover_idx >= 0:
                _envelope_drag = (uid, hover_idx)
            else:
                # Add a new point at mouse position (only within plot area)
                if plot_x <= mxp <= plot_x + plot_w and plot_y <= myp <= plot_y + plot_h:
                    new_t = (mxp - plot_x) / plot_w
                    new_v = 1.0 - (myp - plot_y) / plot_h
                    new_idx = envelope.add_point(new_t, new_v)
                    if new_idx >= 0:
                        _envelope_drag = (uid, new_idx)
                        changed = True

        # Right-click: on a non-endpoint handle = remove it, on empty area = reset envelope
        if imgui.is_mouse_clicked(1):
            if hover_idx > 0 and hover_idx < len(points) - 1:
                if envelope.remove_point(hover_idx):
                    changed = True
            elif hover_idx < 0 and default_points is not None:
                # Full reset
                replacement = Envelope(default_points)
                # Replace in place so upstream references stay valid
                envelope._points = [list(p) for p in replacement._points]
                changed = True

    # Apply drag if this envelope owns it
    if _envelope_drag is not None and _envelope_drag[0] == uid:
        _, drag_idx = _envelope_drag
        if imgui.is_mouse_down(0):
            mxp, myp = imgui.get_mouse_pos()
            new_t = (mxp - plot_x) / plot_w
            new_v = 1.0 - (myp - plot_y) / plot_h
            if drag_idx < len(envelope):
                # Endpoints are anchored at t=0/t=1 — only let the user
                # change their value, never their position
                is_endpoint = (drag_idx == 0 or drag_idx == len(envelope) - 1)
                envelope.set_point(drag_idx, new_t, new_v, lock_t=is_endpoint)
                changed = True
        else:
            _envelope_drag = None

    return changed


def op1_camera_presets(camera) -> bool:
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    presets = [
        ("TOP", "top", OP1_BLUE),
        ("FRT", "front", OP1_GREEN),
        ("RGT", "right", OP1_RED),
        ("ISO", "iso", OP1_WHITE),
    ]
    n = len(presets)
    gap = th * 0.25
    btn_w = (w - gap * (n - 1)) / n
    btn_h = th * 2.0

    changed = False
    mx, my = imgui.get_mouse_pos()
    active_preset = getattr(camera, 'active_preset', None)
    for i, (label, preset, color) in enumerate(presets):
        bx = wx + i * (btn_w + gap)
        is_active = (active_preset == preset)
        if is_active:
            draw_styled_rect_active(dl, bx, wy, btn_w, btn_h, color,
                                    rounding=BUTTON_ROUNDING)
        else:
            draw_styled_rect(dl, bx, wy, btn_w, btn_h, color,
                             rounding=BUTTON_ROUNDING, thickness=1.2)
        lbl_size = imgui.calc_text_size(label)
        dl.add_text(bx + (btn_w - lbl_size[0]) * 0.5,
                    wy + (btn_h - th) * 0.5, col32(color), label)

    imgui.dummy(w, btn_h + _pad())
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        for i, (_, preset, _) in enumerate(presets):
            bx = wx + i * (btn_w + gap)
            if bx <= mx <= bx + btn_w:
                camera.set_preset(preset)
                changed = True
                break
    return changed


# ---------------------------------------------------------------------------
# LED matrix / pixel-art renderer
# ---------------------------------------------------------------------------
#
# Space Invader bitmap kitbashed from Figma:
#   https://www.figma.com/design/oFspkLoatxBNuXyS6x95IU/Space-Invader-Community
#   node 3:8 — animation
#
# 24 columns × 20 rows. '#' = LED on (bright glowing green), any other
# character is a faded background dot. The Figma design uses #00f327
# for the lit LEDs with a small #1fff43 glow halo and 20% opacity for
# the dim background dots; ``op1_led_grid`` reproduces both layers.
SPACE_INVADER_BITMAP = (
    "........................",  # row  0
    "........................",  # row  1
    "......##..........##....",  # row  2 — antennae top
    "......##..........##....",  # row  3 — antennae base
    "........##......##......",  # row  4 — eye stalks
    "........##......##......",  # row  5
    "......##############....",  # row  6 — head top
    "......##############....",  # row  7
    "....#####..####..#####..",  # row  8 — wings + body, eye gaps
    "....#####..####..#####..",  # row  9
    "..######################",  # row 10 — body widest
    "..######################",  # row 11
    "..##..##############..##",  # row 12 — body lower (with mouth)
    "..##..##############..##",  # row 13
    "..##..##..........##..##",  # row 14 — outer feet, mouth open
    "..##..##..........##..##",  # row 15
    "........####..####......",  # row 16 — center legs
    "........####..####......",  # row 17
    "........................",  # row 18
    "........................",  # row 19
)


# Figma palette for the Space Invader LEDs.
LED_GREEN_ON = (0.00, 0.953, 0.153, 1.00)   # #00f327
LED_GREEN_OFF = (0.00, 0.953, 0.153, 1.00)  # same hue, alpha multiplied below
LED_GREEN_GLOW = (0.122, 1.00, 0.263, 1.00)  # #1fff43


def op1_led_grid(
    dl,
    cx: float,
    cy: float,
    cell_size: float,
    bitmap=SPACE_INVADER_BITMAP,
    on_color: tuple = LED_GREEN_ON,
    off_color: tuple = LED_GREEN_OFF,
    cell_gap: float = 4.0,
    glow: bool = True,
    bg_color: tuple | None = None,
) -> tuple[float, float]:
    """Draw a Figma-style LED matrix centered at (cx, cy).

    The bitmap is a sequence of equal-length strings where '#' is an
    illuminated LED and any other character is a faded background dot.
    Returns the (width, height) of the rendered grid in pixels so the
    caller can position other elements relative to it.

    When ``glow`` is True each lit LED gets a soft inner-bloom rect
    drawn underneath it so the panel reads like a real LED matrix
    rather than flat coloured squares.

    When ``bg_color`` is non-None, a black-ish backdrop rect is drawn
    around the entire grid (matches the Figma design's 'bg-black').
    """
    if not bitmap:
        return 0.0, 0.0
    rows = len(bitmap)
    cols = max(len(row) for row in bitmap)
    step = cell_size + cell_gap
    total_w = cols * cell_size + (cols - 1) * cell_gap
    total_h = rows * cell_size + (rows - 1) * cell_gap
    x0 = cx - total_w * 0.5
    y0 = cy - total_h * 0.5

    if bg_color is not None:
        pad = max(8.0, cell_size * 1.0)
        dl.add_rect_filled(
            x0 - pad, y0 - pad, x0 + total_w + pad, y0 + total_h + pad,
            imgui.get_color_u32_rgba(*bg_color),
            max(4.0, cell_size * 0.5),
        )

    on_u32 = imgui.get_color_u32_rgba(*on_color)
    off_u32 = imgui.get_color_u32_rgba(
        off_color[0], off_color[1], off_color[2], off_color[3] * 0.20
    )
    glow_u32 = imgui.get_color_u32_rgba(
        LED_GREEN_GLOW[0], LED_GREEN_GLOW[1], LED_GREEN_GLOW[2], 0.45
    )
    rounding = max(1.0, cell_size * 0.25)

    for r in range(rows):
        row = bitmap[r]
        y = y0 + r * step
        for c in range(cols):
            x = x0 + c * step
            ch = row[c] if c < len(row) else '.'
            if ch == '#':
                if glow:
                    pad = max(2.0, cell_size * 0.40)
                    dl.add_rect_filled(
                        x - pad, y - pad,
                        x + cell_size + pad, y + cell_size + pad,
                        glow_u32, rounding + pad,
                    )
                dl.add_rect_filled(
                    x, y, x + cell_size, y + cell_size,
                    on_u32, rounding,
                )
            else:
                dl.add_rect_filled(
                    x, y, x + cell_size, y + cell_size,
                    off_u32, rounding,
                )
    return total_w, total_h


# ---------------------------------------------------------------------------
# 3D Geometric Wireframes — perspective grid panels
# ---------------------------------------------------------------------------
#
# Kitbashed from Figma 'Geometric Wireframes Community' node 1:14427:
#   https://www.figma.com/design/0JfdYNtYsafbuGTLZJx83B
#
# Eight rectangular wireframe panels arranged in a horizontal row, each
# rotated about its own vertical axis by an angle that depends on its
# position. Outermost panels are face-on; innermost are nearly edge-on.
# Same vibe as a row of OP-1 album covers spinning toward edge view.

def op1_wireframe_depth_panels(
    dl,
    cx: float,
    cy: float,
    area_w: float,
    area_h: float,
    color: tuple = OP1_ORANGE,
    line_thickness: float = 1.0,
    n_panels: int = 8,
    n_cols: int = 6,
    n_rows: int = 12,
    spread: float = 1.0,
    vertical: bool = False,
) -> None:
    """Render the 8-panel perspective wireframe row centered at (cx, cy).

    Horizontal mode (default) places panels along world-x rotated about
    the y-axis; vertical mode places them along world-y rotated about
    the x-axis so the fan stacks top-to-bottom inside a narrow column.
    ``spread`` (0..1) interpolates the rotation: 0 = all panels nearly
    edge-on (collapsed), 1 = the Figma-default fan. Hook ``spread``
    into a depth-falloff value for live feedback that ties the visual
    to a real graphics control.
    """
    col_u32 = imgui.get_color_u32_rgba(*color)

    if vertical:
        # Panels are horizontal quads stacked along world-y.
        panel_w = area_w * 0.85
        panel_h = panel_w * 0.55
        half_axis = area_h * 0.42
    else:
        # Panels are vertical quads in a row along world-x.
        panel_h = area_h * 0.85
        panel_w = panel_h * 0.55
        half_axis = area_w * 0.42

    focal = max(area_w, area_h) * 0.9
    camera_z = focal

    spread = max(0.0, min(1.0, float(spread)))
    theta_max = math.radians(85.0)

    for i in range(n_panels):
        # Position along the stacking axis: evenly spaced across the row/column
        if n_panels > 1:
            u = (i / (n_panels - 1)) * 2.0 - 1.0  # u in [-1, 1]
        else:
            u = 0.0

        # Rotation magnitude grows toward the center of the stack.
        # Outer panels (|u|→1) face the camera (theta→0); inner panels
        # (|u|→0) are nearly edge-on (|theta|→theta_max). The `spread`
        # parameter dampens the swing.
        theta_mag = (1.0 - abs(u)) * theta_max
        theta = theta_mag * (-1.0 if u >= 0 else 1.0)
        if spread < 1.0:
            theta = theta_mag * spread * (-1.0 if u >= 0 else 1.0) + \
                math.copysign(theta_max, theta) * (1.0 - spread)

        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        w2 = panel_w * 0.5
        h2 = panel_h * 0.5
        local = [
            (-w2, -h2),
            ( w2, -h2),
            ( w2,  h2),
            (-w2,  h2),
        ]

        screen = []
        if vertical:
            world_y = u * half_axis
            for lx, ly in local:
                # Rotate about x-axis: y' = y*cos, z' = -y*sin
                ry = ly * cos_t
                rz = -ly * sin_t
                wxw = lx
                wyw = ry + world_y
                wz = rz + camera_z
                sx = cx + (wxw * focal) / wz
                sy = cy + (wyw * focal) / wz
                screen.append((sx, sy))
        else:
            world_x = u * half_axis
            for lx, ly in local:
                # Rotate about y-axis: x' = x*cos, z' = -x*sin
                rx = lx * cos_t
                rz = -lx * sin_t
                wxw = rx + world_x
                wyw = ly
                wz = rz + camera_z
                sx = cx + (wxw * focal) / wz
                sy = cy + (wyw * focal) / wz
                screen.append((sx, sy))

        # Outline
        for j in range(4):
            x0, y0 = screen[j]
            x1, y1 = screen[(j + 1) % 4]
            dl.add_line(x0, y0, x1, y1, col_u32, line_thickness)

        # Interior vertical grid lines (interp between top edge and bottom edge)
        for c in range(1, n_cols):
            f = c / n_cols
            top = (screen[0][0] * (1 - f) + screen[1][0] * f,
                   screen[0][1] * (1 - f) + screen[1][1] * f)
            bot = (screen[3][0] * (1 - f) + screen[2][0] * f,
                   screen[3][1] * (1 - f) + screen[2][1] * f)
            dl.add_line(top[0], top[1], bot[0], bot[1], col_u32, line_thickness)

        # Interior horizontal grid lines (interp between left edge and right edge)
        for r in range(1, n_rows):
            f = r / n_rows
            lft = (screen[0][0] * (1 - f) + screen[3][0] * f,
                   screen[0][1] * (1 - f) + screen[3][1] * f)
            rgt = (screen[1][0] * (1 - f) + screen[2][0] * f,
                   screen[1][1] * (1 - f) + screen[2][1] * f)
            dl.add_line(lft[0], lft[1], rgt[0], rgt[1], col_u32, line_thickness)


# ---------------------------------------------------------------------------
# Waterman polyhedra — wireframe with hidden-line dashing
# ---------------------------------------------------------------------------
#
# Kitbashed from Figma 'Waterman Polyhedra 1-8' node 1:2:
#   https://www.figma.com/design/o3vrUYIVQHkjveR0LBJ8vn
#
# Each polyhedron is defined by vertices, edges, and faces. The renderer
# applies a free rotation, computes per-face camera-facing tests, and
# draws front-facing edges as solid lines and back-facing edges as
# dashed lines (the Figma "Hidden Curves" / "Visible Curves" pair).

# Golden ratio for icosahedron / dodecahedron.
_PHI = (1.0 + math.sqrt(5.0)) * 0.5
_INV_PHI = 1.0 / _PHI


def _normalize_polyhedron(verts):
    """Scale a vertex set so the longest radius is exactly 1.0."""
    r = max(math.sqrt(x * x + y * y + z * z) for x, y, z in verts)
    if r <= 0:
        return list(verts)
    inv = 1.0 / r
    return [(x * inv, y * inv, z * inv) for x, y, z in verts]


def _make_octahedron():
    verts = [
        (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
    ]
    faces = [
        [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
        [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
    ]
    return _normalize_polyhedron(verts), faces


def _make_cube():
    verts = [
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
    ]
    faces = [
        [0, 1, 2, 3],  # back
        [4, 5, 6, 7],  # front (will be reversed for normal)
        [0, 1, 5, 4],  # bottom
        [2, 3, 7, 6],  # top
        [1, 2, 6, 5],  # right
        [0, 3, 7, 4],  # left
    ]
    # Cube needs consistent CCW winding from outside; flip the ones above
    # whose normal would point inward. Easier: just fix here:
    faces = [
        [0, 3, 2, 1],  # back  (-z)
        [4, 5, 6, 7],  # front (+z)
        [0, 1, 5, 4],  # bottom (-y)
        [3, 7, 6, 2],  # top   (+y)
        [1, 2, 6, 5],  # right (+x)
        [0, 4, 7, 3],  # left  (-x)
    ]
    return _normalize_polyhedron(verts), faces


def _make_icosahedron():
    # Standard icosahedron coordinates.
    p = _PHI
    verts = [
        (-1,  p, 0), ( 1,  p, 0), (-1, -p, 0), ( 1, -p, 0),
        (0, -1,  p), (0,  1,  p), (0, -1, -p), (0,  1, -p),
        ( p, 0, -1), ( p, 0,  1), (-p, 0, -1), (-p, 0,  1),
    ]
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]
    return _normalize_polyhedron(verts), faces


def _make_dodecahedron():
    # Dodecahedron via the (±1, ±1, ±1) ∪ (0, ±1/φ, ±φ) ∪ ... vertex set.
    p = _PHI
    q = _INV_PHI
    verts = [
        (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
        (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
        (0, -q, -p), (0,  q, -p), (0, -q,  p), (0,  q,  p),
        (-q, -p, 0), ( q, -p, 0), (-q,  p, 0), ( q,  p, 0),
        (-p, 0, -q), ( p, 0, -q), (-p, 0,  q), ( p, 0,  q),
    ]
    faces = [
        [0, 8, 9, 3, 16],
        [0, 16, 18, 4, 12],
        [0, 12, 13, 1, 8],
        [1, 13, 5, 19, 17],
        [1, 17, 2, 9, 8],
        [2, 17, 19, 6, 15],
        [2, 15, 14, 3, 9],
        [3, 14, 7, 18, 16],
        [4, 18, 7, 11, 10],
        [4, 10, 5, 13, 12],
        [5, 10, 11, 6, 19],
        [6, 11, 7, 14, 15],
    ]
    return _normalize_polyhedron(verts), faces


# Built-in polyhedron registry. Hand-rolled standard solids — close
# enough in spirit to the Waterman 1-8 set without needing the actual
# lattice convex-hull computation. Names are lowercase.
_POLYHEDRA: dict[str, tuple[list, list]] = {
    "octahedron": _make_octahedron(),
    "cube": _make_cube(),
    "icosahedron": _make_icosahedron(),
    "dodecahedron": _make_dodecahedron(),
}

# ---------------------------------------------------------------------------
# Axes cross — 3D coordinate axes with dots along each arm. Used as the
# training-progress target shape instead of a platonic solid so the widget
# reads as an anatomical orientation guide.
# ---------------------------------------------------------------------------
_AXES_N_ARM_DOTS = 7   # dots per arm (not counting center)

def _make_axes_cross_verts() -> list:
    """Return list of (x,y,z) tuples: center at index 0, then 6 arms × N dots."""
    arms = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    verts = [(0.0, 0.0, 0.0)]
    for ax, ay, az in arms:
        for i in range(1, _AXES_N_ARM_DOTS + 1):
            t = i / _AXES_N_ARM_DOTS
            verts.append((ax * t, ay * t, az * t))
    return verts

_AXES_CROSS_VERTS = _make_axes_cross_verts()
# Register with empty faces so the polyhedra projection helper still works
# for Phase-2 dot migration; edges are drawn manually.
_POLYHEDRA["axes_cross"] = (_AXES_CROSS_VERTS, [])

# Default rotation order so the polyhedra read three-quarter-front
# instead of edge-on at rotation 0.
_DEFAULT_TILT_X = math.radians(20.0)


def _edges_from_faces(faces):
    """Return a {(min,max): [face_idx, ...]} edge → adjacent-faces map."""
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, face in enumerate(faces):
        n = len(face)
        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            key = (min(a, b), max(a, b))
            edge_faces.setdefault(key, []).append(fi)
    return edge_faces


def _rotate_y(v, c, s):
    x, y, z = v
    return (x * c + z * s, y, -x * s + z * c)


def _rotate_x(v, c, s):
    x, y, z = v
    return (x, y * c - z * s, y * s + z * c)


def op1_polyhedron(
    dl,
    cx: float,
    cy: float,
    size: float,
    kind: str = "icosahedron",
    rot_y: float = 0.0,
    rot_x: float = _DEFAULT_TILT_X,
    color: tuple = OP1_WHITE,
    line_thickness: float = 1.4,
    dashed_hidden: bool = True,
    hidden_alpha: float = 0.45,
) -> None:
    """Draw a 3D wireframe polyhedron centered at (cx, cy) with hidden-line
    dashing.

    Convex polyhedra only — uses face-normal back-face culling to
    classify each edge as visible (both adjacent faces front-facing
    is enough) or hidden (both back-facing). Visible edges draw as
    solid lines; hidden edges draw dashed at reduced alpha.

    ``kind`` selects from the built-in registry: "octahedron", "cube",
    "icosahedron", "dodecahedron". Falls back to icosahedron silently
    on an unknown name.
    """
    poly = _POLYHEDRA.get(kind, _POLYHEDRA["icosahedron"])
    verts3d, faces = poly

    # Apply rotation
    cy_y, sy_y = math.cos(rot_y), math.sin(rot_y)
    cx_x, sx_x = math.cos(rot_x), math.sin(rot_x)
    rotated = []
    for v in verts3d:
        v = _rotate_y(v, cy_y, sy_y)
        v = _rotate_x(v, cx_x, sx_x)
        rotated.append(v)

    # Project to screen (orthographic — keeps the silhouette tidy)
    screen = [
        (cx + v[0] * size, cy - v[1] * size) for v in rotated
    ]

    # Compute camera-facing for each face. Camera looks down -z (we use
    # +z toward camera so a face is front-facing iff its normal.z > 0).
    front_face = [False] * len(faces)
    for fi, face in enumerate(faces):
        if len(face) < 3:
            continue
        v0 = rotated[face[0]]
        v1 = rotated[face[1]]
        v2 = rotated[face[2]]
        ux, uy, uz = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
        wx, wy, wz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
        nz = ux * wy - uy * wx  # z component of cross(u, w)
        front_face[fi] = (nz > 0.0)

    edge_faces = _edges_from_faces(faces)

    visible_u32 = imgui.get_color_u32_rgba(*color)
    hidden_u32 = imgui.get_color_u32_rgba(
        color[0], color[1], color[2], color[3] * hidden_alpha
    )

    for (a, b), adj in edge_faces.items():
        x0, y0 = screen[a]
        x1, y1 = screen[b]
        is_visible = any(front_face[fi] for fi in adj)
        if is_visible:
            dl.add_line(x0, y0, x1, y1, visible_u32, line_thickness)
        elif dashed_hidden:
            # Manual dashing — ImGui draw list has no native dash, so
            # split the segment into ~5 evenly-spaced dashes.
            dashes = 6
            for k in range(dashes):
                if k % 2 == 1:
                    continue  # gap
                t0 = k / dashes
                t1 = (k + 1) / dashes
                dx0 = x0 + (x1 - x0) * t0
                dy0 = y0 + (y1 - y0) * t0
                dx1 = x0 + (x1 - x0) * t1
                dy1 = y0 + (y1 - y0) * t1
                dl.add_line(dx0, dy0, dx1, dy1, hidden_u32, line_thickness * 0.85)
        else:
            dl.add_line(x0, y0, x1, y1, hidden_u32, line_thickness * 0.85)


# ============================================================================
# OP-1 kitbash primitives — five new visual helpers lifted from the
# OP-1 Screen Graphics Community kit in Figma:
#   Cluster  -> op1_clock_dial      (clock-face gauge with cross ticks)
#   Pulse    -> op1_pulse_grid      (dotted grid + square-wave signal)
#   Sketch   -> op1_envelope_graph  (dotted grid + shaped curve)
#   FM       -> op1_cube_numeral    (wireframe cube with a numeral inside)
#   Digital  -> op1_node_graph      (colored nodes connected by edges)
#
# Each is a pure draw-list helper — the caller controls layout (cursor
# advance) and has no ImGui state dependencies.
# ============================================================================


def op1_clock_dial(
    dl,
    cx: float,
    cy: float,
    radius: float,
    value_frac: float,
    color: tuple = OP1_WHITE,
) -> None:
    """Clock-face dial with cardinal ticks and a needle pointing at value.

    value_frac in [0, 1] maps to a full rotation from 12 o'clock clockwise.
    Kitbashed from the OP-1 Cluster dials row.
    """
    value_frac = max(0.0, min(1.0, value_frac))

    dim_u32 = imgui.get_color_u32_rgba(0.22, 0.22, 0.22, 1.0)
    color_u32 = imgui.get_color_u32_rgba(*color)

    # Cardinal ticks (N/E/S/W) sitting just outside the face
    tick_inner = radius * 0.85
    tick_outer = radius * 1.02
    for k in range(4):
        a = k * (math.pi * 0.5) - math.pi * 0.5
        ca, sa = math.cos(a), math.sin(a)
        dl.add_line(
            cx + ca * tick_inner, cy + sa * tick_inner,
            cx + ca * tick_outer, cy + sa * tick_outer,
            dim_u32, 1.2,
        )

    # Outer faint ring
    ring_u32 = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 1.0)
    dl.add_circle(cx, cy, radius * 0.95, ring_u32, 32, 1.0)

    # Needle — straight line from the centre to the edge, topped
    # by a solid dot in the accent color.
    theta = value_frac * math.pi * 2.0 - math.pi * 0.5
    nx = cx + math.cos(theta) * radius * 0.75
    ny = cy + math.sin(theta) * radius * 0.75
    dl.add_line(cx, cy, nx, ny, color_u32, 1.6)
    dl.add_circle_filled(nx, ny, radius * 0.18, color_u32, 16)
    # Small centre hub
    dl.add_circle_filled(cx, cy, radius * 0.10, dim_u32, 12)


def op1_pulse_grid(
    dl,
    x: float,
    y: float,
    w: float,
    h: float,
    signal: list[int] | tuple[int, ...],
    color: tuple = OP1_BLUE,
    grid_color: tuple = (0.0, 0.28, 0.36, 1.0),
    divisions: int = 10,
) -> None:
    """Square-wave signal traced across a dashed reference grid.

    ``signal`` is a sequence of ints in {0, 1} defining the state of
    each column. Columns are spaced across the width; the waveform
    line draws at ~1/3 height when low and ~2/3 when high.
    Kitbashed from the OP-1 Pulse grid panel.
    """
    if not signal:
        return

    grid_u32 = imgui.get_color_u32_rgba(*grid_color)
    sig_u32 = imgui.get_color_u32_rgba(*color)

    # Dashed vertical + horizontal reference grid
    for i in range(divisions + 1):
        gx = x + (w * i) / divisions
        dash_h = 3.0
        gap_h = 4.0
        yy = y
        while yy < y + h:
            yy2 = min(yy + dash_h, y + h)
            dl.add_line(gx, yy, gx, yy2, grid_u32, 1.0)
            yy += dash_h + gap_h

    for i in range(5 + 1):
        gy = y + (h * i) / 5
        dash_w = 3.0
        gap_w = 4.0
        xx = x
        while xx < x + w:
            xx2 = min(xx + dash_w, x + w)
            dl.add_line(xx, gy, xx2, gy, grid_u32, 1.0)
            xx += dash_w + gap_w

    # Square-wave signal
    n = len(signal)
    if n < 2:
        return
    y_low = y + h * 0.72
    y_high = y + h * 0.24
    col_w = w / n
    prev_y = y_high if signal[0] else y_low
    for i in range(n):
        cx_a = x + i * col_w
        cx_b = x + (i + 1) * col_w
        cur_y = y_high if signal[i] else y_low
        # Vertical riser on state change
        if cur_y != prev_y and i > 0:
            dl.add_line(cx_a, prev_y, cx_a, cur_y, sig_u32, 1.8)
        # Horizontal segment
        dl.add_line(cx_a, cur_y, cx_b, cur_y, sig_u32, 1.8)
        prev_y = cur_y


def op1_envelope_graph(
    dl,
    x: float,
    y: float,
    w: float,
    h: float,
    curve_fn,
    color: tuple = OP1_BLUE,
    segments: int = 48,
    dot_spacing: float = 8.0,
) -> None:
    """Dotted grid background with a shaped curve drawn through it.

    ``curve_fn`` takes t in [0,1] and returns y in [0,1] where 0 is the
    bottom of the panel and 1 is the top. Kitbashed from the OP-1 Sketch
    envelope display.
    """
    dot_u32 = imgui.get_color_u32_rgba(0.35, 0.35, 0.35, 0.75)
    curve_u32 = imgui.get_color_u32_rgba(*color)

    # Dotted grid
    n_cols = max(2, int(w / dot_spacing))
    n_rows = max(2, int(h / dot_spacing))
    step_x = w / n_cols
    step_y = h / n_rows
    for r in range(n_rows + 1):
        for c in range(n_cols + 1):
            dx = x + c * step_x
            dy = y + r * step_y
            dl.add_rect_filled(dx, dy, dx + 1.2, dy + 1.2, dot_u32)

    # Curve polyline
    pts = []
    for i in range(segments + 1):
        t = i / segments
        try:
            v = curve_fn(t)
        except Exception:
            v = 0.0
        v = max(0.0, min(1.0, v))
        px = x + t * w
        py = y + (1.0 - v) * h
        pts.append((px, py))

    for i in range(len(pts) - 1):
        dl.add_line(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1],
                    curve_u32, 1.8)

    # End-of-curve dot like the Figma asset
    if pts:
        ex, ey = pts[-1]
        dl.add_circle_filled(ex, ey, 3.0, curve_u32, 12)


def op1_cube_numeral(
    dl,
    app,
    cx: float,
    cy: float,
    size: float,
    value: str,
    color: tuple = OP1_BLUE,
    rot_y: float | None = None,
    rot_x: float | None = None,
) -> None:
    """Isometric wireframe cube with a large numeral rendered inside.

    Kitbashed from the OP-1 FM cube stack. Uses op1_polyhedron for the
    wireframe (full hidden-line dashing) and op1_number for the
    display-font numeral centered in the cube.
    """
    if rot_y is None:
        rot_y = math.radians(35.0)
    if rot_x is None:
        rot_x = math.radians(25.0)

    op1_polyhedron(
        dl, cx, cy, size,
        kind="cube",
        rot_y=rot_y,
        rot_x=rot_x,
        color=color,
        line_thickness=1.3,
        dashed_hidden=True,
        hidden_alpha=0.38,
    )

    if not value:
        return

    # Display-font numeral centered in the cube. op1_number uses the
    # display font if available and falls back to the UI font otherwise.
    num_size = size * 1.4
    op1_number(
        app,
        cx - num_size * 0.5,
        cy - num_size * 0.5,
        num_size, num_size,
        value, color, align='center',
    )


def op1_node_graph(
    dl,
    x: float,
    y: float,
    w: float,
    h: float,
    nodes: list[tuple[float, float, tuple]],
    edges: list[tuple[int, int]],
    edge_color: tuple = (0.643, 0.388, 1.0, 1.0),  # OP-1 purple2
    node_radius: float = 5.0,
) -> None:
    """Constellation graph — colored dots connected by lines.

    ``nodes`` is a list of (nx, ny, rgba) tuples where nx/ny are in
    [0, 1] normalized panel coordinates. Kitbashed from the OP-1
    Digital constellation panel.
    """
    if not nodes:
        return

    edge_u32 = imgui.get_color_u32_rgba(*edge_color)

    # Edges first so nodes sit on top
    for a, b in edges:
        if a < 0 or b < 0 or a >= len(nodes) or b >= len(nodes):
            continue
        ax = x + nodes[a][0] * w
        ay = y + nodes[a][1] * h
        bx = x + nodes[b][0] * w
        by = y + nodes[b][1] * h
        dl.add_line(ax, ay, bx, by, edge_u32, 1.8)

    # Nodes
    for (nx, ny, col) in nodes:
        px = x + nx * w
        py = y + ny * h
        col_u32 = imgui.get_color_u32_rgba(*col)
        dl.add_circle_filled(px, py, node_radius, col_u32, 16)


# ============================================================================
# Training progress net — the "loading bar" for training runs. A dot grid
# forms a random network during Phase 1 (CONNECT), the random edges dissolve
# into an emerging icosahedron wireframe during Phase 2 (DENSIFY), and the
# shape resolves into a fully-rotating hidden-line wireframe during Phase 3
# (RESOLVE). Kitbashed from Figma 154:2138 (Sketch panel) with the grid
# double-height so the panel reads square rather than 2:1 landscape.
# ============================================================================


def _project_polyhedron_vertices(
    kind: str,
    cx: float,
    cy: float,
    size: float,
    rot_y: float,
    rot_x: float,
) -> list[tuple[float, float]]:
    """Project a polyhedron's vertices to screen coordinates.

    Same rotation + orthographic projection math as ``op1_polyhedron`` but
    returns just the vertex screen positions without drawing any edges.
    Useful for animations that need to interpolate arbitrary points toward
    a polyhedron's vertices.
    """
    poly = _POLYHEDRA.get(kind, _POLYHEDRA["icosahedron"])
    verts3d, _faces = poly
    cy_y, sy_y = math.cos(rot_y), math.sin(rot_y)
    cx_x, sx_x = math.cos(rot_x), math.sin(rot_x)
    screen: list[tuple[float, float]] = []
    for v in verts3d:
        v = _rotate_y(v, cy_y, sy_y)
        v = _rotate_x(v, cx_x, sx_x)
        screen.append((cx + v[0] * size, cy - v[1] * size))
    return screen


# Deterministic caches keyed on (run_seed, n_cols, n_rows). These let the
# random phase 1 edges and the 12 anchor indices stay stable across frames
# for a given run + panel geometry, so the visual doesn't flicker.
_net_anchor_cache: dict[tuple[int, int, int], list[int]] = {}
_net_rand_edges_cache: dict[tuple[int, int, int], list[tuple[int, int]]] = {}


def _net_smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _get_net_anchors(seed: int, n_cols: int, n_rows: int, count: int) -> list[int]:
    key = (int(seed), int(n_cols), int(n_rows))
    cached = _net_anchor_cache.get(key)
    if cached is not None and len(cached) >= count:
        return cached[:count]
    n_dots = n_cols * n_rows
    if n_dots <= 0:
        return []
    rng = random.Random(int(seed) & 0xFFFFFFFF)
    indices = list(range(n_dots))
    rng.shuffle(indices)
    picked = indices[:count]
    _net_anchor_cache[key] = picked
    return picked


def _get_net_random_edges(
    seed: int, n_cols: int, n_rows: int, max_edges: int
) -> list[tuple[int, int]]:
    key = (int(seed), int(n_cols), int(n_rows))
    cached = _net_rand_edges_cache.get(key)
    if cached is not None and len(cached) >= max_edges:
        return cached[:max_edges]
    n_dots = n_cols * n_rows
    if n_dots < 2:
        _net_rand_edges_cache[key] = []
        return []
    rng = random.Random((int(seed) ^ 0x5A5A5A5A) & 0xFFFFFFFF)
    edges: list[tuple[int, int]] = []
    attempts = 0
    while len(edges) < max_edges and attempts < max_edges * 4:
        attempts += 1
        a = rng.randrange(n_dots)
        b = rng.randrange(n_dots)
        if a == b:
            continue
        edges.append((a, b))
    _net_rand_edges_cache[key] = edges
    return edges


def _draw_axes_cross_resolve(
    dl, cx: float, cy: float, size: float,
    rot_y: float, rot_x: float, t_now: float,
) -> None:
    """Phase-3 RESOLVE draw for the axes cross training animation.

    Draws 6 coloured arms (±R coral, ±S green, ±A blue) radiating from a
    white center dot. Each arm is a dashed line with pulsing dots that stream
    outward from the center, like a neural signal propagating along an axis.
    """
    cy_y = math.cos(rot_y)
    sy_y = math.sin(rot_y)
    cx_x = math.cos(rot_x)
    sx_x = math.sin(rot_x)

    def proj(v):
        v = _rotate_y(v, cy_y, sy_y)
        v = _rotate_x(v, cx_x, sx_x)
        return cx + v[0] * size, cy - v[1] * size, v[2]

    N = _AXES_N_ARM_DOTS
    # arm_dir, color, label
    axes = [
        ([1, 0, 0], [-1, 0, 0], CORAL,     'R'),
        ([0, 1, 0], [0, -1, 0], OP1_GREEN, 'S'),
        ([0, 0, 1], [0, 0, -1], OP1_BLUE,  'A'),
    ]

    # Pulse wave that travels outward: phase increases with arm distance.
    pulse_speed = 1.8
    th = imgui.get_text_line_height()

    for pos_dir, neg_dir, color, label in axes:
        for direction in (pos_dir, neg_dir):
            tip = [direction[0], direction[1], direction[2]]
            tip_sx, tip_sy, tip_sz = proj(tip)
            ctr_sx, ctr_sy, _ = proj([0, 0, 0])

            # Depth alpha: arms pointing toward camera are brighter.
            depth_a = 0.40 + 0.60 * max(0.0, (tip_sz + 1.0) * 0.5)

            # Dashed arm line (faint)
            line_col = col32((color[0], color[1], color[2], 0.22 * depth_a))
            dl.add_line(ctr_sx, ctr_sy, tip_sx, tip_sy, line_col, 1.0)

            # Dots along arm — pulse wave travels outward
            for i in range(1, N + 1):
                t = i / N
                v = [direction[0] * t, direction[1] * t, direction[2] * t]
                sx, sy, sz = proj(v)

                # Outward pulse: brightness peaks when the wave front passes
                wave_phase = (t_now * pulse_speed - t * 1.6) % 2.0
                pulse = 0.5 + 0.5 * math.sin(wave_phase * math.pi)
                dot_a = depth_a * (0.45 + 0.55 * pulse)
                dot_r = 2.0 + 1.5 * pulse
                dot_col = col32((color[0], color[1], color[2], dot_a))
                dl.add_circle_filled(sx, sy, dot_r, dot_col, 10)

        # Axis label at positive tip only (when near front)
        tip_sx, tip_sy, tip_sz = proj(pos_dir)
        if tip_sz > -0.3:
            label_a = 0.55 + 0.45 * max(0.0, tip_sz)
            lbl_col = col32((color[0], color[1], color[2], label_a))
            lw = imgui.calc_text_size(label)[0]
            dl.add_text(tip_sx - lw * 0.5, tip_sy - th * 0.55, lbl_col, label)

    # Center dot
    ctr_sx, ctr_sy, _ = proj([0, 0, 0])
    dl.add_circle_filled(ctr_sx, ctr_sy, 3.5,
                         imgui.get_color_u32_rgba(0.9, 0.9, 0.9, 0.95), 14)


def op1_training_progress_net(
    dl,
    app,
    x: float,
    y: float,
    w: float,
    h: float,
    progress: float = 0.0,
    running: bool = False,
    finished: bool = False,
    run_seed: int = 0,
    shape_kind: str = "icosahedron",
    dot_spacing: float = 14.0,
) -> None:
    """Draw the training loading panel — a dot grid that forms a network
    and resolves into a wireframe platonic shape.

    The widget runs three phases driven by ``progress`` (smoothed 0..1):
      0.00-0.30  CONNECT  — random edges grow between grid dots
      0.30-0.70  DENSIFY  — icosahedron edges emerge, anchor dots migrate
      0.70-1.00  RESOLVE  — clean rotating wireframe with hidden-line dashing

    ``running`` and ``finished`` drive the idle / active / done state text.
    When neither is true the panel shows only a breathing dot grid.

    The caller reserves layout space with ``imgui.dummy(w, h + ...)`` after
    the call — this function only draws into ``dl`` and does not advance
    ImGui's cursor.
    """
    # --- Panel background (well color) ---
    bg_u32 = imgui.get_color_u32_rgba(0.065, 0.065, 0.065, 1.0)
    border_u32 = imgui.get_color_u32_rgba(0.20, 0.20, 0.20, 0.9)
    dl.add_rect_filled(x, y, x + w, y + h, bg_u32, PANEL_ROUNDING)
    dl.add_rect(x, y, x + w, y + h, border_u32, PANEL_ROUNDING, 0, 1.0)

    pad = max(8.0, dot_spacing * 0.6)
    ix, iy = x + pad, y + pad
    iw, ih = w - pad * 2.0, h - pad * 2.0
    if iw <= 0.0 or ih <= 0.0:
        return

    # Grid layout — fit as many dots as we can at the given spacing, then
    # tile them exactly to the inner rect so both edges touch the border.
    n_cols = max(2, int(iw / dot_spacing))
    n_rows = max(2, int(ih / dot_spacing))
    step_x = iw / max(n_cols - 1, 1)
    step_y = ih / max(n_rows - 1, 1)

    t_now = time.time()
    p = max(0.0, min(1.0, progress))
    P1 = 0.30
    P2 = 0.70

    # Grid alpha: breathing when idle, steady when running, fades in phase 3
    breath = 0.30 + 0.15 * (0.5 + 0.5 * math.sin(t_now * 2.0))
    grid_alpha_base = 0.40 if running else breath
    if p >= P2:
        t_p3 = (p - P2) / max(1.0 - P2, 1e-6)
        grid_alpha = grid_alpha_base * (1.0 - 0.78 * _net_smoothstep(t_p3))
    else:
        grid_alpha = grid_alpha_base
    grid_alpha = max(0.05, grid_alpha)
    dot_u32 = imgui.get_color_u32_rgba(0.55, 0.55, 0.55, grid_alpha)

    # Compute grid dot positions (flat list, index = r * n_cols + c)
    grid_pts: list[tuple[float, float]] = []
    for r in range(n_rows):
        for c in range(n_cols):
            gx = ix + c * step_x
            gy = iy + r * step_y
            grid_pts.append((gx, gy))

    # Draw dot grid
    for (gx, gy) in grid_pts:
        dl.add_rect_filled(gx - 1.1, gy - 1.1, gx + 1.1, gy + 1.1, dot_u32)

    th = imgui.get_text_line_height()

    # --- Idle state — breathing grid only ---
    if not running and not finished:
        label = "IDLE"
        lw = imgui.calc_text_size(label)[0]
        idle_u32 = imgui.get_color_u32_rgba(0.35, 0.35, 0.35, 0.9)
        dl.add_text(x + (w - lw) * 0.5, y + h - th - 4.0, idle_u32, label)
        return

    # --- Polyhedron geometry (shared by phases 2 and 3) ---
    cx = x + w * 0.5
    cy = y + h * 0.5
    size = min(w, h) * 0.34
    rest_rot_y = 0.0
    rest_rot_x = _DEFAULT_TILT_X
    target_verts = _project_polyhedron_vertices(
        shape_kind, cx, cy, size, rest_rot_y, rest_rot_x
    )
    n_target = len(target_verts)

    anchors = _get_net_anchors(run_seed, n_cols, n_rows, n_target)
    max_rand_edges = 18

    # --- Phase 1: CONNECT (random edges grow between random dots) ---
    if p < P1:
        t1 = p / P1
        n_edges = int(max_rand_edges * _net_smoothstep(t1))
        rand_edges = _get_net_random_edges(run_seed, n_cols, n_rows, max_rand_edges)
        edge_u32 = imgui.get_color_u32_rgba(
            OP1_ORANGE[0], OP1_ORANGE[1], OP1_ORANGE[2], 0.85
        )
        glow_u32 = imgui.get_color_u32_rgba(
            OP1_ORANGE[0], OP1_ORANGE[1], OP1_ORANGE[2], 1.0
        )
        active: set[int] = set()
        for i in range(min(n_edges, len(rand_edges))):
            a, b = rand_edges[i]
            if a >= len(grid_pts) or b >= len(grid_pts):
                continue
            ax, ay = grid_pts[a]
            bx, by = grid_pts[b]
            dl.add_line(ax, ay, bx, by, edge_u32, 1.5)
            active.add(a)
            active.add(b)
        for idx in active:
            if idx < len(grid_pts):
                gx, gy = grid_pts[idx]
                dl.add_circle_filled(gx, gy, 2.8, glow_u32, 10)
        return

    # --- Phase 2: DENSIFY (random fades out, icosahedron edges fade in, ---
    # --- anchors migrate from grid slots to target vertex positions)    ---
    if p < P2:
        t2 = (p - P1) / max(P2 - P1, 1e-6)
        st2 = _net_smoothstep(t2)

        # Fading random edges
        rand_alpha = 0.85 * (1.0 - st2)
        if rand_alpha > 0.015:
            rand_u32 = imgui.get_color_u32_rgba(
                OP1_ORANGE[0], OP1_ORANGE[1], OP1_ORANGE[2], rand_alpha
            )
            rand_edges = _get_net_random_edges(
                run_seed, n_cols, n_rows, max_rand_edges
            )
            for (a, b) in rand_edges:
                if a >= len(grid_pts) or b >= len(grid_pts):
                    continue
                ax, ay = grid_pts[a]
                bx, by = grid_pts[b]
                dl.add_line(ax, ay, bx, by, rand_u32, 1.3)

        # Migrating anchor positions
        anchor_positions: list[tuple[float, float]] = []
        for i in range(n_target):
            if i >= len(anchors):
                break
            ai = anchors[i]
            if ai >= len(grid_pts):
                continue
            gx, gy = grid_pts[ai]
            tx, ty = target_verts[i]
            px = gx + (tx - gx) * st2
            py = gy + (ty - gy) * st2
            anchor_positions.append((px, py))

        if len(anchor_positions) == n_target and n_target > 0:
            emerge_alpha = 0.95 * st2
            emerge_u32 = imgui.get_color_u32_rgba(
                OP1_WHITE[0], OP1_WHITE[1], OP1_WHITE[2], emerge_alpha
            )
            if shape_kind == "axes_cross":
                # Draw lines from screen center to each migrating dot
                # (arms emerge from origin rather than edges between verts).
                scx, scy = anchor_positions[0]  # index 0 = center
                N = _AXES_N_ARM_DOTS
                _ARM_COLORS = [CORAL, CORAL, OP1_GREEN, OP1_GREEN, OP1_BLUE, OP1_BLUE]
                for arm_i in range(6):
                    base = 1 + arm_i * N
                    acol = _ARM_COLORS[arm_i]
                    arm_u32 = imgui.get_color_u32_rgba(
                        acol[0], acol[1], acol[2], emerge_alpha * 0.7
                    )
                    for j in range(N):
                        vi = base + j
                        if vi >= len(anchor_positions):
                            continue
                        ax, ay = anchor_positions[vi]
                        # Connect to center for first dot, previous dot otherwise
                        if j == 0:
                            dl.add_line(scx, scy, ax, ay, arm_u32, 1.3)
                        else:
                            prev_ax, prev_ay = anchor_positions[base + j - 1]
                            dl.add_line(prev_ax, prev_ay, ax, ay, arm_u32, 1.3)
                        dot_col = imgui.get_color_u32_rgba(
                            acol[0], acol[1], acol[2], emerge_alpha * 0.9
                        )
                        dl.add_circle_filled(ax, ay, 2.4, dot_col, 10)
                # Center dot
                dl.add_circle_filled(scx, scy, 3.2, emerge_u32, 12)
            else:
                poly = _POLYHEDRA.get(shape_kind, _POLYHEDRA["icosahedron"])
                _verts3d, faces = poly
                edge_faces = _edges_from_faces(faces)
                for (a, b) in edge_faces.keys():
                    if a >= n_target or b >= n_target:
                        continue
                    ax, ay = anchor_positions[a]
                    bx, by = anchor_positions[b]
                    dl.add_line(ax, ay, bx, by, emerge_u32, 1.5)

                vert_u32 = imgui.get_color_u32_rgba(
                    OP1_WHITE[0], OP1_WHITE[1], OP1_WHITE[2], 0.9
                )
                for (px, py) in anchor_positions:
                    dl.add_circle_filled(px, py, 2.6, vert_u32, 12)
        return

    # --- Phase 3: RESOLVE (clean rotating wireframe) ---
    # Rotation eases up from 0 as phase 3 begins, so the phase 2 → 3 handoff
    # is continuous (both draw at rest rotation when t3 = 0).
    t3 = (p - P2) / max(1.0 - P2, 1e-6)
    st3 = _net_smoothstep(t3)
    rot_y = rest_rot_y + (t_now * 0.35) * st3
    if shape_kind == "axes_cross":
        _draw_axes_cross_resolve(dl, cx, cy, size, rot_y, rest_rot_x, t_now)
    else:
        op1_polyhedron(
            dl, cx, cy, size,
            kind=shape_kind,
            rot_y=rot_y,
            rot_x=rest_rot_x,
            color=OP1_WHITE,
            line_thickness=1.6,
            dashed_hidden=True,
            hidden_alpha=0.45,
        )

    if finished and not running:
        done_u32 = imgui.get_color_u32_rgba(
            OP1_GREEN[0], OP1_GREEN[1], OP1_GREEN[2], 0.95
        )
        label = "RESOLVED"
        lw = imgui.calc_text_size(label)[0]
        dl.add_text(x + (w - lw) * 0.5, y + h - th - 4.0, done_u32, label)


