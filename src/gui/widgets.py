"""Custom-drawn OP-1 style widgets for Lithium — font-metric based layout."""

import math
import time
import imgui
from src.core.modes import MODE_INDIVIDUAL
from src.gui.theme import *


def _th():
    """Get current text line height — the base unit for all layout."""
    return imgui.get_text_line_height()


def _pad():
    """Standard padding."""
    return _th() * 0.3


def draw_section_label(text: str, color=COLOR_SECTION):
    """Section header with border accent."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    h = _th() + _pad() * 2

    dl.add_rect(wx, wy, wx + w, wy + h, col32(color), 0.0, 0, 1.0)
    dl.add_text(wx + _pad() * 2, wy + _pad(), col32(color), text)
    imgui.dummy(w, h + _pad())


def draw_point_size_control(current_size: float, min_val=0.1, max_val=20.0) -> tuple:
    """Point size control with live preview dot."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    preview_h = th * 3.5
    dl.add_rect_filled(wx, wy, wx + w, wy + preview_h, col32(BG_DARK))
    dl.add_rect(wx, wy, wx + w, wy + preview_h, col32(FAINT), 0.0, 0, 1.0)

    center_x = wx + w * 0.35
    center_y = wy + preview_h * 0.5
    radius = max(current_size * th * 0.1, th * 0.15)
    dl.add_circle(center_x, center_y, radius + th * 0.2, col32((0.9, 0.35, 0.15, 0.3)), 32, th * 0.1)
    dl.add_circle_filled(center_x, center_y, radius, col32(ORANGE), 32)

    dl.add_text(wx + w * 0.65, center_y - th * 0.5, col32(WHITE), f"{current_size:.1f}")
    dl.add_text(wx + w * 0.65, center_y + th * 0.5, col32(DIM), "pt")

    imgui.dummy(w, preview_h + _pad())

    imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *ORANGE)
    imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CORAL)
    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *BG_DARK)
    changed, new_val = imgui.slider_float("##ptsize", current_size, min_val, max_val, "")
    imgui.pop_style_color(3)

    return changed, new_val


def draw_sharpness_control(current_sharpness: float) -> tuple:
    """Sharpness control with gradient preview circles."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    preview_h = th * 3.5
    dl.add_rect_filled(wx, wy, wx + w, wy + preview_h, col32(BG_DARK))
    dl.add_rect(wx, wy, wx + w, wy + preview_h, col32(FAINT), 0.0, 0, 1.0)

    num_dots = 5
    dot_spacing = w / (num_dots + 1)
    dot_r = th * 0.6
    for i in range(num_dots):
        cx = wx + dot_spacing * (i + 1)
        cy = wy + preview_h * 0.5
        t = i / (num_dots - 1)

        ring_alpha = 0.15 + 0.5 * t
        dl.add_circle(cx, cy, dot_r + th * 0.15, col32((0.95, 0.5, 0.15, ring_alpha)), 32, 1.0)
        inner_alpha = 1.0 - t * 0.7
        dl.add_circle_filled(cx, cy, dot_r, col32((0.95, 0.55, 0.2, inner_alpha)), 32)

    indicator_x = wx + dot_spacing + current_sharpness * dot_spacing * (num_dots - 1)
    tri = th * 0.3
    dl.add_triangle_filled(indicator_x - tri, wy + preview_h - tri,
                           indicator_x + tri, wy + preview_h - tri,
                           indicator_x, wy + preview_h - tri * 2.5,
                           col32(ORANGE))

    dl.add_text(wx + _pad() * 2, wy + _pad(), col32(DIM), "HARD")
    dl.add_text(wx + w - th * 3, wy + _pad(), col32(DIM), "SOFT")

    imgui.dummy(w, preview_h + _pad())

    imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *ORANGE)
    imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CORAL)
    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *BG_DARK)
    changed, new_val = imgui.slider_float("##sharp", current_sharpness, 0.0, 1.0, "")
    imgui.pop_style_color(3)

    return changed, new_val


def draw_color_adjust(brightness: float, contrast: float, saturation: float) -> tuple:
    """Color adjustment sliders. Returns (changed, brightness, contrast, saturation)."""
    th = _th()
    changed = False

    imgui.push_style_color(imgui.COLOR_SLIDER_GRAB, *BLUE)
    imgui.push_style_color(imgui.COLOR_SLIDER_GRAB_ACTIVE, *CYAN)
    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *BG_DARK)

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()

    # Brightness
    dl.add_text(wx, wy, col32(DIM), "BRT")
    dl.add_text(wx + w - th * 3, wy, col32(WHITE), f"{brightness:+.2f}")
    imgui.dummy(w, th + _pad() * 0.5)
    c, brightness = imgui.slider_float("##brt", brightness, -0.5, 0.5, "")
    changed |= c

    wx2, wy2 = imgui.get_cursor_screen_pos()
    dl.add_text(wx2, wy2, col32(DIM), "CON")
    dl.add_text(wx2 + w - th * 3, wy2, col32(WHITE), f"{contrast:.2f}")
    imgui.dummy(w, th + _pad() * 0.5)
    c, contrast = imgui.slider_float("##con", contrast, 0.2, 3.0, "")
    changed |= c

    wx3, wy3 = imgui.get_cursor_screen_pos()
    dl.add_text(wx3, wy3, col32(DIM), "SAT")
    dl.add_text(wx3 + w - th * 3, wy3, col32(WHITE), f"{saturation:.2f}")
    imgui.dummy(w, th + _pad() * 0.5)
    c, saturation = imgui.slider_float("##sat", saturation, 0.0, 3.0, "")
    changed |= c

    imgui.pop_style_color(3)
    return changed, brightness, contrast, saturation


def draw_resample_control(app) -> bool:
    """Resample control — decimate the current cloud. Returns True if resampled."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    if app.mode != MODE_INDIVIDUAL or app.selected_index < 0 or app.selected_index >= len(app.entries):
        dl.add_text(wx, wy, col32(DIM), "Select a cloud first")
        imgui.dummy(w, th + _pad())
        return False

    entry = app.entries[app.selected_index]
    gpu = entry.full_gpu or entry.preview_gpu
    if not gpu:
        return False

    pts = entry.point_count

    # Show current count
    dl.add_text(wx, wy, col32(WHITE), f"{pts:,} pts")
    imgui.dummy(w, th + _pad() * 0.5)

    # Decimate buttons
    h = th * 2
    gap = th * 0.3
    btn_w = (w - gap * 2) / 3

    labels = [("50%", 0.5), ("25%", 0.25), ("10%", 0.1)]
    resampled = False
    for i, (label, factor) in enumerate(labels):
        bx = wx + i * (btn_w + gap)
        target = max(int(pts * factor), 100)
        dl.add_rect_filled(bx, wy + th + _pad(), bx + btn_w, wy + th + _pad() + h, col32(BG_DARK))
        dl.add_rect(bx, wy + th + _pad(), bx + btn_w, wy + th + _pad() + h, col32(LAVENDER), 0.0, 0, 1.0)
        dl.add_text(bx + btn_w * 0.5 - th * 0.8, wy + th + _pad() + h * 0.5 - th * 0.5,
                    col32(LAVENDER), label)

    imgui.dummy(w, th + _pad() + h + _pad())
    mx, my = imgui.get_mouse_pos()
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        for i, (label, factor) in enumerate(labels):
            bx = wx + i * (btn_w + gap)
            by = wy + th + _pad()
            if bx <= mx <= bx + btn_w and by <= my <= by + h:
                target = max(int(pts * factor), 100)
                _do_resample(app, app.selected_index, target)
                resampled = True
                break

    return resampled


def _do_resample(app, index: int, target_points: int):
    """Resample a cloud entry to target point count and re-upload."""
    from src.data.resampler import voxel_downsample
    from src.rendering.point_cloud_renderer import upload_cloud

    entry = app.entries[index]
    gpu = entry.full_gpu or entry.preview_gpu
    if not gpu or not gpu.cloud_data:
        return

    resampled = voxel_downsample(gpu.cloud_data, target_points)
    if resampled.point_count >= entry.point_count:
        return  # no reduction achieved

    new_gpu = upload_cloud(resampled, app.ctx, app.program)

    # Release old GPU resources (preview and full may be same object)
    old_full = entry.full_gpu
    old_preview = entry.preview_gpu
    entry.full_gpu = new_gpu
    entry.preview_gpu = new_gpu
    entry.point_count = resampled.point_count
    entry.bounds_min = resampled.bounds_min.copy()
    entry.bounds_max = resampled.bounds_max.copy()

    if old_full:
        old_full.release()
    if old_preview and old_preview is not old_full:
        old_preview.release()

    # Update legacy list
    if gpu in app.gpu_clouds:
        idx = app.gpu_clouds.index(gpu)
        app.gpu_clouds[idx] = new_gpu

    app._overlays_dirty = True
    print(f"Resampled {entry.name}: {resampled.point_count:,} points")


def draw_camera_cube(preset_name: str = 'iso'):
    """Isometric camera cube indicator."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    th = _th()

    size = th * 2
    cx, cy = wx + size + th * 0.3, wy + size

    sz = size * 0.4
    dx, dy = sz * 0.866, sz * 0.5

    bl = (cx - dx, cy + dy)
    br = (cx + dx, cy + dy)
    fc = (cx, cy)
    tf = (cx, cy - sz)
    tl = (cx - dx, cy - sz + dy)
    tr = (cx + dx, cy - sz + dy)

    edge_color = col32(ORANGE)
    lw = max(th * 0.08, 1.0)

    dl.add_quad_filled(fc[0], fc[1], bl[0], bl[1], cx, cy + dy * 2, br[0], br[1], col32(FAINT))
    dl.add_quad_filled(tf[0], tf[1], tl[0], tl[1], fc[0], fc[1], bl[0], bl[1],
                       col32((0.25, 0.15, 0.08, 0.5)))
    dl.add_quad_filled(tf[0], tf[1], tr[0], tr[1], br[0], br[1], fc[0], fc[1],
                       col32((0.2, 0.12, 0.06, 0.5)))

    dl.add_line(*tf, *tl, edge_color, lw)
    dl.add_line(*tf, *tr, edge_color, lw)
    dl.add_line(*tl, *bl, edge_color, lw)
    dl.add_line(*tr, *br, edge_color, lw)
    dl.add_line(*tf, *fc, edge_color, lw)
    dl.add_line(*bl, *fc, edge_color, lw)
    dl.add_line(*br, *fc, edge_color, lw)

    imgui.dummy(size * 2 + th, size * 2 + th * 0.3)


def draw_camera_preset_buttons(camera) -> bool:
    """Camera preset buttons."""
    changed = False
    th = _th()
    btn_w = th * 3.2
    btn_h = th * 2

    presets = [
        ('TOP', 'top', CORAL),
        ('FRT', 'front', ORANGE),
        ('RGT', 'right', AMBER),
        ('ISO', 'iso', TEAL),
    ]

    for i, (label, name, color) in enumerate(presets):
        if i > 0:
            imgui.same_line()
        imgui.push_style_color(imgui.COLOR_BUTTON, *BG_DARK)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, color[0]*0.3, color[1]*0.3, color[2]*0.3, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *color)
        imgui.push_style_color(imgui.COLOR_TEXT, *color)
        if imgui.button(label, width=btn_w, height=btn_h):
            camera.set_preset(name)
            changed = True
        imgui.pop_style_color(4)

    return changed


def draw_cloud_list_item(index: int, name: str, point_count: int,
                         is_selected: bool, has_preview: bool) -> bool:
    """Cloud list entry. Returns True if clicked."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    h = th * 2.4

    if is_selected:
        dl.add_rect_filled(wx, wy, wx + w, wy + h, col32((0.12, 0.12, 0.12, 1.0)))
        dl.add_rect(wx, wy, wx + w, wy + h, col32(CORAL), 0.0, 0, 1.0)
    else:
        dl.add_rect_filled(wx, wy, wx + w, wy + h, col32(BG_DARK))

    led_x = wx + th * 0.7
    led_y = wy + h * 0.5
    led_r = th * 0.2
    if not has_preview:
        pulse = (math.sin(time.time() * 4.0 + index) + 1.0) * 0.5
        dl.add_circle_filled(led_x, led_y, led_r, col32((0.9 * pulse, 0.5 * pulse, 0.1, 1.0)))
    elif is_selected:
        dl.add_circle_filled(led_x, led_y, led_r, col32(CORAL))
    else:
        dl.add_circle_filled(led_x, led_y, led_r, col32(FAINT))

    text_color = CORAL if is_selected else WHITE
    text_x = wx + th * 1.5
    dl.add_text(text_x, wy + th * 0.2, col32(text_color), name[:24])

    if point_count > 0:
        pts_str = f"{point_count // 1000}K" if point_count >= 1000 else str(point_count)
        dl.add_text(text_x, wy + th * 1.2, col32(DIM), pts_str)

    imgui.dummy(w, h + _pad() * 0.5)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        return True
    return False


def draw_gizmo_mode_selector(current_mode: str, visible: bool) -> tuple:
    """Translate/rotate mode selector."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    new_mode = current_mode
    new_visible = visible
    h = th * 2.5
    gap = th * 0.5
    half_w = (w - gap) * 0.5

    # Translate
    t_active = current_mode == 'translate' and visible
    t_color = TEAL if t_active else FAINT
    dl.add_rect_filled(wx, wy, wx + half_w, wy + h, col32(BG_DARK))
    dl.add_rect(wx, wy, wx + half_w, wy + h, col32(t_color), 0.0, 0, 1.0)

    ic = (wx + half_w * 0.3, wy + h * 0.5)
    al = th * 0.6
    lw = max(th * 0.1, 1.0)
    dl.add_line(ic[0] - al, ic[1], ic[0] + al, ic[1], col32(t_color), lw)
    dl.add_line(ic[0], ic[1] - al, ic[0], ic[1] + al, col32(t_color), lw)
    dl.add_text(wx + half_w * 0.55, wy + h * 0.5 - th * 0.5, col32(t_color), "MOVE")

    imgui.dummy(half_w, h)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        new_mode = 'translate'
        new_visible = True

    imgui.same_line()

    # Rotate
    r_active = current_mode == 'rotate' and visible
    r_color = CORAL if r_active else FAINT
    rx = wx + half_w + gap
    dl.add_rect_filled(rx, wy, rx + half_w, wy + h, col32(BG_DARK))
    dl.add_rect(rx, wy, rx + half_w, wy + h, col32(r_color), 0.0, 0, 1.0)

    rc = (rx + half_w * 0.3, wy + h * 0.5)
    arc_r = th * 0.5
    for a in range(0, 270, 15):
        a1, a2 = math.radians(a), math.radians(a + 15)
        dl.add_line(rc[0] + arc_r * math.cos(a1), rc[1] + arc_r * math.sin(a1),
                    rc[0] + arc_r * math.cos(a2), rc[1] + arc_r * math.sin(a2),
                    col32(r_color), lw)
    dl.add_text(rx + half_w * 0.55, wy + h * 0.5 - th * 0.5, col32(r_color), "ROT")

    imgui.dummy(half_w, h)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        new_mode = 'rotate'
        new_visible = True

    return new_mode, new_visible


def draw_export_buttons(app):
    """Export buttons."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    gap = th * 0.5
    half_w = (w - gap) * 0.5
    h = th * 3

    # Screenshot
    dl.add_rect_filled(wx, wy, wx + half_w, wy + h, col32(BG_DARK))
    dl.add_rect(wx, wy, wx + half_w, wy + h, col32(CORAL), 0.0, 0, 1.0)
    ic = (wx + half_w * 0.5, wy + th)
    r = th * 0.5
    dl.add_rect(ic[0] - r, ic[1] - r * 0.6, ic[0] + r, ic[1] + r * 0.6,
                col32(CORAL), th * 0.15, 0, max(th * 0.08, 1.0))
    dl.add_circle(ic[0], ic[1], r * 0.35, col32(CORAL), 8, max(th * 0.08, 1.0))
    dl.add_text(wx + half_w * 0.5 - th * 1.2, wy + th * 1.8, col32(CORAL), "STILL")

    imgui.dummy(half_w, h)
    still_clicked = imgui.is_item_hovered() and imgui.is_mouse_clicked(0)

    imgui.same_line()

    # Spin
    sx = wx + half_w + gap
    dl.add_rect_filled(sx, wy, sx + half_w, wy + h, col32(BG_DARK))
    dl.add_rect(sx, wy, sx + half_w, wy + h, col32(GREEN), 0.0, 0, 1.0)
    sc_pos = (sx + half_w * 0.5, wy + th)
    arc_r = th * 0.45
    for a in range(0, 330, 20):
        a1, a2 = math.radians(a), math.radians(a + 20)
        dl.add_line(sc_pos[0] + arc_r * math.cos(a1), sc_pos[1] + arc_r * math.sin(a1),
                    sc_pos[0] + arc_r * math.cos(a2), sc_pos[1] + arc_r * math.sin(a2),
                    col32(GREEN), max(th * 0.1, 1.0))
    dl.add_text(sx + half_w * 0.5 - th * 0.8, wy + th * 1.8, col32(GREEN), "SPIN")

    imgui.dummy(half_w, h)
    spin_clicked = imgui.is_item_hovered() and imgui.is_mouse_clicked(0)

    return still_clicked, spin_clicked


def draw_stats_bar(cloud_count: int, total_points: int, pending: int,
                   mode_str: str, fps: float = 0.0):
    """Status bar with live stats."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    h = th * 2.6

    dl.add_rect_filled(wx, wy, wx + w, wy + h, col32(BG_DARK))
    dl.add_rect(wx, wy, wx + w, wy + h, col32(FAINT), 0.0, 0, 1.0)

    # Mode badge
    mode_color = AMBER if mode_str == "GALLERY" else CORAL
    badge_w = th * 4.5 if mode_str == "GALLERY" else th * 3
    dl.add_rect_filled(wx + th * 0.4, wy + th * 0.3,
                       wx + th * 0.4 + badge_w, wy + th * 1.3, col32(mode_color))
    dl.add_text(wx + th * 0.7, wy + th * 0.3, col32(BG_DARK), mode_str)

    # Clouds
    dl.add_text(wx + th * 0.4, wy + th * 1.5, col32(WHITE), f"{cloud_count}")
    dl.add_text(wx + th * 0.4 + len(str(cloud_count)) * th * 0.6 + th * 0.3,
                wy + th * 1.5, col32(DIM), "clouds")

    # Points
    pts_str = _format_points(total_points)
    dl.add_text(wx + w * 0.45, wy + th * 0.4, col32(WHITE), pts_str)
    dl.add_text(wx + w * 0.45, wy + th * 1.5, col32(DIM), "points")

    # FPS
    if fps > 0:
        fps_str = f"{fps:.0f}"
        dl.add_text(wx + w - th * 3, wy + th * 0.4,
                    col32(GREEN if fps >= 30 else ORANGE), fps_str)
        dl.add_text(wx + w - th * 3, wy + th * 1.5, col32(DIM), "fps")

    # Loading indicator
    if pending > 0:
        pulse = (math.sin(time.time() * 4.0) + 1.0) * 0.5
        dl.add_circle_filled(wx + w - th * 0.7, wy + h * 0.5, th * 0.25,
                             col32((0.95, 0.5 * pulse, 0.1, 1.0)))

    imgui.dummy(w, h + _pad())


def draw_bg_color_control(bg_color: tuple) -> tuple:
    """Background color control with preset swatches."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    new_color = bg_color
    h = th * 2

    presets = [
        ("BLK", (0.0, 0.0, 0.0)),
        ("DRK", (0.08, 0.08, 0.08)),
        ("GRY", (0.18, 0.18, 0.18)),
        ("LGT", (0.85, 0.85, 0.85)),
    ]

    gap = th * 0.5
    swatch_w = (w - gap * (len(presets) - 1)) / len(presets)

    for i, (label, color) in enumerate(presets):
        sx = wx + i * (swatch_w + gap)
        is_active = (abs(bg_color[0] - color[0]) < 0.02 and
                     abs(bg_color[1] - color[1]) < 0.02)

        dl.add_rect_filled(sx, wy, sx + swatch_w, wy + h, col32((*color, 1.0)))
        border_color = CORAL if is_active else FAINT
        dl.add_rect(sx, wy, sx + swatch_w, wy + h,
                    col32(border_color), 0.0, 0, max(th * 0.1, 1.0) if is_active else 1.0)
        label_color = BG_DARK if color[0] > 0.5 else DIM
        dl.add_text(sx + th * 0.3, wy + h * 0.5 - th * 0.5, col32(label_color), label)

    imgui.dummy(w, h + _pad())

    mx, my = imgui.get_mouse_pos()
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        for i, (label, color) in enumerate(presets):
            sx = wx + i * (swatch_w + gap)
            if sx <= mx <= sx + swatch_w and wy <= my <= wy + h:
                new_color = color
                break

    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, *BG_DARK)
    changed, new_color = imgui.color_edit3("##bg_custom", *new_color,
                                            flags=imgui.COLOR_EDIT_NO_INPUTS | imgui.COLOR_EDIT_NO_LABEL)
    imgui.pop_style_color(1)

    return new_color


def draw_overlay_toggles(show_bbox: bool, show_grid: bool) -> tuple:
    """Overlay toggle switches."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    new_bbox = show_bbox
    new_grid = show_grid
    h = th * 1.8
    gap = th * 0.5
    half_w = (w - gap) * 0.5

    toggles = [
        ("BBOX", show_bbox, ORANGE),
        ("GRID", show_grid, TEAL),
    ]

    for i, (label, active, color) in enumerate(toggles):
        tx = wx + i * (half_w + gap)
        track_color = color if active else FAINT
        dl.add_rect_filled(tx, wy, tx + half_w, wy + h, col32(BG_DARK))
        dl.add_rect(tx, wy, tx + half_w, wy + h, col32(track_color), 0.0, 0, 1.0)
        dl.add_text(tx + th * 0.4, wy + h * 0.5 - th * 0.5, col32(track_color), label)
        state_text = "ON" if active else "OFF"
        dl.add_text(tx + half_w - th * 2, wy + h * 0.5 - th * 0.5, col32(track_color), state_text)

    imgui.dummy(w, h + _pad())

    mx, my = imgui.get_mouse_pos()
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        for i in range(len(toggles)):
            tx = wx + i * (half_w + gap)
            if tx <= mx <= tx + half_w:
                if i == 0:
                    new_bbox = not show_bbox
                elif i == 1:
                    new_grid = not show_grid

    return new_bbox, new_grid


def draw_import_buttons() -> tuple[bool, bool]:
    """Import buttons — FILE and FOLDER. Returns (file_clicked, folder_clicked)."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    gap = th * 0.5
    half_w = (w - gap) * 0.5
    h = th * 2.5

    # FILE button
    dl.add_rect_filled(wx, wy, wx + half_w, wy + h, col32(BG_DARK))
    dl.add_rect(wx, wy, wx + half_w, wy + h, col32(TEAL), 0.0, 0, 1.0)
    dl.add_text(wx + half_w * 0.5 - th * 1.0, wy + h * 0.5 - th * 0.5,
                col32(TEAL), "FILE")

    imgui.dummy(half_w, h)
    file_clicked = imgui.is_item_hovered() and imgui.is_mouse_clicked(0)

    imgui.same_line()

    # FOLDER button
    fx = wx + half_w + gap
    dl.add_rect_filled(fx, wy, fx + half_w, wy + h, col32(BG_DARK))
    dl.add_rect(fx, wy, fx + half_w, wy + h, col32(BLUE), 0.0, 0, 1.0)
    dl.add_text(fx + half_w * 0.5 - th * 1.5, wy + h * 0.5 - th * 0.5,
                col32(BLUE), "FOLDER")

    imgui.dummy(half_w, h)
    folder_clicked = imgui.is_item_hovered() and imgui.is_mouse_clicked(0)

    return file_clicked, folder_clicked


def _format_points(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def draw_tool_palette(active_tool: str | None) -> str | None:
    """Tool palette with PICK / BOX / LASSO / BRUSH / CURVE buttons.

    Returns a tool name if one was clicked, or None.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    tools = [
        ("PICK", "pick", CORAL),
        ("BOX", "box", ORANGE),
        ("LASSO", "lasso", AMBER),
        ("BRUSH", "brush", TEAL),
        ("CURVE", "curve", LAVENDER),
    ]

    n = len(tools)
    gap = th * 0.3
    btn_w = (w - gap * (n - 1)) / n
    h = th * 2.2

    clicked = None
    for i, (label, name, color) in enumerate(tools):
        bx = wx + i * (btn_w + gap)
        is_active = active_tool == name

        if is_active:
            dl.add_rect_filled(bx, wy, bx + btn_w, wy + h, col32(BG_DARK))
            dl.add_rect(bx, wy, bx + btn_w, wy + h, col32(color), 0.0, 0, 1.5)
            text_col = col32(color)
        else:
            dl.add_rect_filled(bx, wy, bx + btn_w, wy + h, col32((0.04, 0.04, 0.04, 1.0)))
            dl.add_rect(bx, wy, bx + btn_w, wy + h, col32(FAINT), 0.0, 0, 1.0)
            text_col = col32(DIM)

        tw = imgui.calc_text_size(label)[0]
        dl.add_text(bx + (btn_w - tw) * 0.5, wy + (h - th) * 0.5, text_col, label)

    imgui.dummy(w, h)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        for i, (_, name, _) in enumerate(tools):
            bx = wx + i * (btn_w + gap)
            if bx <= mx <= bx + btn_w:
                clicked = name
                break
    return clicked


def draw_op1_color_knobs(app) -> bool:
    """OP-1 inspired 4-knob color adjustment panel: R, G, B, Saturation.

    Channels map to:
      R -> app.r_gain     (red channel multiplier, 0-2, default 1)
      G -> app.g_gain     (green channel multiplier)
      B -> app.b_gain     (blue channel multiplier)
      S -> app.saturation (saturation, existing state)

    Four unified 270-degree knob cells inside a single tinted panel.
    Click+drag vertically to change; double-click / right-click to
    reset to 1.0.
    """
    from src.gui.op1_widgets import (
        draw_styled_rect, op1_knob_cell, PANEL_ROUNDING,
    )

    changed = False

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()

    panel_h = th * 8.0

    # Tinted panel background — matches every other OP-1 section card.
    draw_styled_rect(dl, wx, wy, w, panel_h, OP1_WHITE,
                     rounding=PANEL_ROUNDING, thickness=1.2)

    knobs = [
        ("R", "r_gain", OP1_RED),
        ("G", "g_gain", OP1_GREEN),
        ("B", "b_gain", OP1_BLUE),
        ("S", "saturation", OP1_WHITE),
    ]

    n = len(knobs)
    margin = th * 0.4
    cell_gap = th * 0.15
    cell_w = (w - margin * 2 - cell_gap * (n - 1)) / n
    cell_y = wy + th * 0.3
    cell_h = panel_h - th * 0.6

    for i, (label, attr, color) in enumerate(knobs):
        cell_x = wx + margin + i * (cell_w + cell_gap)
        value = getattr(app, attr)
        c, new_value = op1_knob_cell(
            app, cell_x, cell_y, cell_w, cell_h,
            uid=f"rgb_{attr}",
            label=label, value=value,
            min_val=0.0, max_val=2.0, color=color,
            decimals=0, default_val=1.0,
            display_scale=100.0,
        )
        if c:
            setattr(app, attr, new_value)
            changed = True

    imgui.dummy(w, panel_h + _pad())
    return changed


def draw_op1_knob_row(app, specs: list, colored_fill: bool = False,
                      panel_accent: tuple = None) -> bool:
    """Generic tinted 4-knob panel matching the RGB panel layout.

    specs is a list of dicts, one per knob:
      label, uid, value, min, max, color, decimals, default_val,
      display_scale (optional), on_change(new_value) callback.
    Returns True if any knob changed this frame.
    """
    from src.gui.op1_widgets import (
        draw_styled_rect, op1_knob_cell, PANEL_ROUNDING,
    )

    changed = False
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    panel_h = th * 8.0

    if panel_accent is not None:
        border_color = panel_accent
    elif specs:
        border_color = (
            sum(spec['color'][0] for spec in specs) / len(specs),
            sum(spec['color'][1] for spec in specs) / len(specs),
            sum(spec['color'][2] for spec in specs) / len(specs),
            1.0,
        )
    else:
        border_color = (0.5, 0.5, 0.5, 1.0)
    draw_styled_rect(dl, wx, wy, w, panel_h, border_color,
                     rounding=PANEL_ROUNDING, thickness=1.2,
                     colored_fill=colored_fill)

    n = max(len(specs), 1)
    margin = th * 0.4
    cell_gap = th * 0.15
    cell_w = (w - margin * 2 - cell_gap * (n - 1)) / n
    cell_y = wy + th * 0.3
    cell_h = panel_h - th * 0.6

    for i, spec in enumerate(specs):
        cell_x = wx + margin + i * (cell_w + cell_gap)
        c, new_value = op1_knob_cell(
            app, cell_x, cell_y, cell_w, cell_h,
            uid=spec['uid'],
            label=spec['label'],
            value=spec['value'],
            min_val=spec['min'],
            max_val=spec['max'],
            color=spec['color'],
            decimals=spec.get('decimals', 2),
            default_val=spec.get('default_val', 0.0),
            display_scale=spec.get('display_scale', 1.0),
        )
        if c:
            cb = spec.get('on_change')
            if cb is not None:
                cb(new_value)
            changed = True

    imgui.dummy(w, panel_h + _pad())
    return changed


def draw_apply_label_button(selection_count: int, label_name: str) -> bool:
    """Big ORANGE apply button. Returns True if clicked."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = _th()
    h = th * 2.8

    enabled = selection_count > 0
    bg = BG_DARK if enabled else (0.04, 0.04, 0.04, 1.0)
    border = ORANGE if enabled else FAINT
    text_color = ORANGE if enabled else FAINT

    dl.add_rect_filled(wx, wy, wx + w, wy + h, col32(bg))
    dl.add_rect(wx, wy, wx + w, wy + h, col32(border), 0.0, 0, 2.0)

    if selection_count > 0:
        label = f"APPLY '{label_name[:12]}' TO {_format_points(selection_count)}"
    else:
        label = "SELECT POINTS TO LABEL"
    tw = imgui.calc_text_size(label)[0]
    dl.add_text(wx + (w - tw) * 0.5, wy + (h - th) * 0.5, col32(text_color), label)

    imgui.dummy(w, h)
    if not enabled:
        return False
    return imgui.is_item_hovered() and imgui.is_mouse_clicked(0)
