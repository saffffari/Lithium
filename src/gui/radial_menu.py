"""Radial preset menu — activated by RMB long-press, commits on release.

Four cardinal slices (right/top/left/bottom) arranged around a circle.
The dead zone in the middle is a "cancel" target — releasing there
leaves the camera untouched. Moving into a slice and releasing fires
the associated preset through ``Camera.set_preset``, which rides the
existing smooth angle ease.
"""

import math

import imgui


# Slice labels and the camera-preset names they fire. Indexed by slice.
RADIAL_LABELS = ["RIGHT", "TOP", "LEFT", "BOTTOM"]
RADIAL_PRESETS = ["right", "top", "left", "bottom"]

# Center angle of each slice, in standard math convention (0 = +x right,
# π/2 = +y up). One slice per 90°.
_SLICE_COUNT = 4
_SLICE_SIZE = 2.0 * math.pi / _SLICE_COUNT
_SLICE_ANGLES = [i * _SLICE_SIZE for i in range(_SLICE_COUNT)]

# Visual geometry (pixels). The dead zone is where releases count as
# "cancel". The label ring sits between the dead zone and outer edge.
DEAD_ZONE_RADIUS = 44.0
LABEL_RADIUS = 96.0
OUTER_RADIUS = 142.0


def slice_from_cursor(cx: float, cy: float, mx: float, my: float) -> int:
    """Return the slice index under the cursor, or -1 inside the dead zone.

    Inputs are screen coordinates (Y grows downward, as GLFW reports them).
    The returned index is compatible with ``RADIAL_LABELS`` and
    ``RADIAL_PRESETS``.
    """
    dx = mx - cx
    dy = -(my - cy)  # flip to math convention (up is positive)
    dist_sq = dx * dx + dy * dy
    if dist_sq < DEAD_ZONE_RADIUS * DEAD_ZONE_RADIUS:
        return -1
    angle = math.atan2(dy, dx)
    if angle < 0.0:
        angle += 2.0 * math.pi
    # Shift so the slice boundaries sit halfway between the center angles,
    # then bucket.
    shifted = (angle + _SLICE_SIZE * 0.5) % (2.0 * math.pi)
    return int(shifted / _SLICE_SIZE)


def draw_radial_menu(center: tuple[float, float], selected: int) -> None:
    """Render the menu via ImGui's foreground draw list.

    ``center`` is the screen-space origin of the menu (where RMB was
    pressed) and ``selected`` is the index returned by
    :py:func:`slice_from_cursor` — -1 renders the menu with nothing
    highlighted (cursor in the dead zone).
    """
    dl = imgui.get_foreground_draw_list()
    cx, cy = center

    bg_color = imgui.get_color_u32_rgba(0.04, 0.04, 0.04, 0.92)
    ring_color = imgui.get_color_u32_rgba(0.98, 0.55, 0.18, 0.85)
    dead_color = imgui.get_color_u32_rgba(0.55, 0.35, 0.15, 0.55)
    divider_color = imgui.get_color_u32_rgba(0.98, 0.55, 0.18, 0.35)
    label_color = imgui.get_color_u32_rgba(0.85, 0.85, 0.85, 1.0)
    highlight_bg = imgui.get_color_u32_rgba(0.98, 0.55, 0.18, 0.28)
    highlight_color = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)

    # Outer disk
    dl.add_circle_filled(cx, cy, OUTER_RADIUS, bg_color, 64)
    dl.add_circle(cx, cy, OUTER_RADIUS, ring_color, 64, 2.0)

    # Highlight the selected slice as a soft wedge fill.
    if 0 <= selected < _SLICE_COUNT:
        center_angle = _SLICE_ANGLES[selected]
        a0 = center_angle - _SLICE_SIZE * 0.5
        a1 = center_angle + _SLICE_SIZE * 0.5
        steps = 24
        dl.path_clear()
        dl.path_line_to(cx, cy)
        for i in range(steps + 1):
            t = a0 + (a1 - a0) * (i / steps)
            px = cx + math.cos(t) * OUTER_RADIUS
            py = cy - math.sin(t) * OUTER_RADIUS
            dl.path_line_to(px, py)
        dl.path_fill_convex(highlight_bg)

    # Dead zone outline
    dl.add_circle(cx, cy, DEAD_ZONE_RADIUS, dead_color, 40, 1.5)

    # Slice dividers — lines from the dead zone out to the outer edge
    # at each boundary.
    for i in range(_SLICE_COUNT):
        boundary = _SLICE_ANGLES[i] + _SLICE_SIZE * 0.5
        cos_b = math.cos(boundary)
        sin_b = math.sin(boundary)
        x0 = cx + cos_b * DEAD_ZONE_RADIUS
        y0 = cy - sin_b * DEAD_ZONE_RADIUS
        x1 = cx + cos_b * OUTER_RADIUS
        y1 = cy - sin_b * OUTER_RADIUS
        dl.add_line(x0, y0, x1, y1, divider_color, 1.0)

    # Labels
    for i, (label, angle) in enumerate(zip(RADIAL_LABELS, _SLICE_ANGLES)):
        lx = cx + math.cos(angle) * LABEL_RADIUS
        ly = cy - math.sin(angle) * LABEL_RADIUS
        size = imgui.calc_text_size(label)
        tx = lx - size[0] * 0.5
        ty = ly - size[1] * 0.5
        color = highlight_color if i == selected else label_color
        dl.add_text(tx, ty, color, label)

    # Hint text in the dead zone
    hint = "RELEASE TO SET"
    hint_size = imgui.calc_text_size(hint)
    dl.add_text(cx - hint_size[0] * 0.5,
                cy - hint_size[1] * 0.5,
                dead_color, hint)
