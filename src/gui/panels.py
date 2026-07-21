"""ImGui panels for 3Photon — tabbed sidebar with Contact Sheets, Light Table, Automation."""

import math
import time
import os
import json
import imgui
import numpy as np

from src.core.modes import (
    MODE_CONTACT_SHEETS, MODE_LIGHT_TABLE, MODE_AUTOMATION,
)
from src.gui.theme import *
from src.gui.scale import sidebar_width, s
from src.gui.widgets import (
    draw_camera_cube, draw_resample_control, draw_op1_color_knobs,
    draw_op1_knob_row,
)
from src.gui.op1_widgets import (
    op1_section, op1_gauge, op1_dual_gauge, op1_stats_strip,
    op1_tool_palette, op1_cloud_row, op1_toggles, op1_gizmo_mode,
    op1_import_buttons, op1_export_buttons,
    op1_camera_presets, op1_envelope, op1_clip_rack,
    op1_node_graph, op1_pulse_grid, op1_envelope_graph,
    op1_clock_dial, op1_training_progress_net,
    draw_styled_rect, draw_styled_rect_active,
    PANEL_ROUNDING, BUTTON_ROUNDING,
)
from src.gui.op1_wireframe import op1_camera_cube, op1_sphere_size_gauge, op1_clip_fader_panel
from src.gui.label_panel import draw_label_panel


# 3-mode tab order (browse → label → train). Names short enough to fit
# at sidebar widths down to ~280px logical.
_TAB_NAMES = ["SHEETS", "LIGHT", "TRAIN"]
_TAB_COLORS = [OP1_BLUE, OP1_ORANGE, OP1_RED]

# Multi-select state for the Train tab mini-gallery. Stores cloud
# indices the user has checked for export. Kept at module scope
# because ImGui is immediate-mode.
_train_selected: set[int] = set()
# Label filter for export/train — None means "all labels allowed";
# an empty set means "only unlabeled" (effectively no classes).
_train_label_sel: set[int] | None = None

# Training progress net state — smoothed 0..1 progress so the loading
# animation eases between epoch ticks, plus a run seed that re-randomizes
# the grid layout each time training is re-launched.
_net_smooth_progress: float = 0.0
_net_last_running: bool = False
_net_run_seed: int = 0


# Tiny path-stat cache to kill per-frame `os.path.isdir`/`isfile` calls
# from the Train tab. A short TTL (1s) means users dragging things in
# and out still get near-instant feedback.
_PATH_CHECK_TTL = 1.0
_path_check_cache: dict[tuple[str, str], tuple[float, bool]] = {}


def _path_ok(kind: str, path: str) -> bool:
    """Cached ``os.path.isdir``/``isfile``. ``kind`` is 'dir' or 'file'."""
    if not path:
        return False
    key = (kind, path)
    now = time.time()
    hit = _path_check_cache.get(key)
    if hit is not None and (now - hit[0]) < _PATH_CHECK_TTL:
        return hit[1]
    ok = os.path.isdir(path) if kind == 'dir' else os.path.isfile(path)
    _path_check_cache[key] = (now, ok)
    return ok


def _training_progress_fraction(runner) -> float:
    """0..1 progress across a training run. Handles both TrainingRunner
    and PointceptRunner status shapes, plus the pre-launch / finished
    edge cases."""
    if runner is None:
        return 0.0
    status = runner.status
    total = max(int(getattr(status, 'total_epochs', 0) or 0), 1)
    if not getattr(status, 'running', False):
        return 1.0 if getattr(status, 'finished', False) else 0.0
    cur_ep = int(getattr(status, 'current_epoch', 0) or 0)
    if hasattr(status, 'current_step') and hasattr(status, 'total_steps'):
        total_steps = max(int(getattr(status, 'total_steps', 0) or 0), 1)
        cur_step = int(getattr(status, 'current_step', 0) or 0)
        step_frac = cur_step / total_steps
        frac = (max(cur_ep - 1, 0) + step_frac) / total
    else:
        frac = cur_ep / total
    return max(0.0, min(1.0, frac))

# Module-level state for the LIBRARY panel. ImGui's immediate-mode API
# doesn't keep per-widget state for us, so we cache folder tree
# expansion and the "new project" input buffer here.
_library_expanded: set[str] = set()
_library_section_open: dict[str, bool] = {
    "projects": True,
    "folders": True,
}
_new_project_buf: str = ""
_show_new_project: bool = False


def draw_main_menu_bar(app) -> float:
    """Top-level File/Edit/View/Help menu bar. Returns its height in px.

    The bar is drawn via ImGui's `main_menu_bar` so it sits at screen y=0
    across the full window width. Other overlays (sidebar, gallery,
    timeline) must offset by its height so they don't overlap.
    """
    bar_h = 0.0
    if imgui.begin_main_menu_bar():
        bar_h = imgui.get_window_height()

        if imgui.begin_menu("File", True):
            if imgui.menu_item("Import File...", "", False, True)[0]:
                app.import_file_dialog()
            if imgui.menu_item("Import Folder...", "", False, True)[0]:
                app.import_folder_dialog()
            imgui.separator()
            if imgui.menu_item("Save Points", "", False, True)[0]:
                _do_save_points(app)
            if imgui.menu_item("Predict Current Cloud", "", False, True)[0]:
                _do_predict_current_cloud(app)
            imgui.separator()
            if imgui.menu_item("Export Screenshot", "Ctrl+E", False, True)[0]:
                _do_screenshot(app)
            if imgui.menu_item("Spin Render", "Ctrl+Shift+E", False, True)[0]:
                _do_spin(app)
            imgui.separator()
            if imgui.menu_item("Quit", "Esc", False, True)[0]:
                import glfw
                glfw.set_window_should_close(app.window, True)
            imgui.end_menu()

        if imgui.begin_menu("Edit", True):
            if imgui.menu_item("Undo", "Ctrl+Z", False, True)[0]:
                app._do_undo()
            if imgui.menu_item("Redo", "Ctrl+Shift+Z", False, True)[0]:
                app._do_redo()
            imgui.separator()
            if imgui.menu_item("Clear 3D Cursor + Fit", "C", False, True)[0]:
                app.cursor3d.clear()
                app._fit_camera()
            imgui.end_menu()

        if imgui.begin_menu("View", True):
            if imgui.menu_item("Contact Sheets", "1",
                               app.mode == MODE_CONTACT_SHEETS, True)[0]:
                app.mode = MODE_CONTACT_SHEETS
            if imgui.menu_item("Light Table", "2",
                               app.mode == MODE_LIGHT_TABLE, True)[0]:
                app.mode = MODE_LIGHT_TABLE
            if imgui.menu_item("Train", "3",
                               app.mode == MODE_AUTOMATION, True)[0]:
                app.mode = MODE_AUTOMATION
            imgui.separator()
            if imgui.menu_item("Toggle Sidebar", "H", app.gui_visible, True)[0]:
                app.gui_visible = not app.gui_visible
            if imgui.menu_item("Bounding Box", "Shift+B",
                               app.show_bbox, True)[0]:
                app.show_bbox = not app.show_bbox
            if imgui.menu_item("Ground Grid", "Shift+G",
                               app.show_grid, True)[0]:
                app.show_grid = not app.show_grid
            mp_open = getattr(app, 'measure_panel_open', False)
            mp_enabled = mp_open or bool(getattr(app, 'measure_registry', None) and app.measure_registry.items)
            if imgui.menu_item("Measurements", "M", mp_open, mp_enabled)[0]:
                app.measure_panel_open = not mp_open
            imgui.separator()
            if imgui.menu_item("Top",   "T", False, True)[0]:
                app.camera.set_preset('top')
            if imgui.menu_item("Right", "R", False, True)[0]:
                app.camera.set_preset('right')
            if imgui.menu_item("Iso",   "I", False, True)[0]:
                app.camera.set_preset('iso')
            if imgui.menu_item("Frame View", "F / Space", False, True)[0]:
                app._fit_camera()
            imgui.end_menu()

        if imgui.begin_menu("Help", True):
            if imgui.menu_item("Keyboard Shortcuts", "?", app.show_shortcuts, True)[0]:
                app.show_shortcuts = not app.show_shortcuts
            imgui.end_menu()

        imgui.end_main_menu_bar()
    return bar_h


def draw_settings_panel(app) -> bool:
    """Draw the main settings sidebar with tabbed navigation."""
    changed = False
    sw = sidebar_width()

    # Selection hygiene: a view switch invalidates every index-based
    # selection set (indices point into the OLD entries list — carrying
    # them over used to export / infer on the wrong clouds after
    # switching to a smaller project).
    global _train_selected, _train_label_sel
    _cur_view = getattr(app, 'active_view', None)
    if _cur_view != getattr(app, '_panels_last_view', ('__unset__',)):
        app._panels_last_view = _cur_view
        _train_selected.clear()
        _train_label_sel = None
        cs_sel = getattr(app, 'contact_sheets_selected', None)
        if cs_sel:
            cs_sel.clear()

    menu_h = getattr(app, '_menu_bar_height', 0.0)
    imgui.set_next_window_position(0, menu_h, imgui.ALWAYS)
    imgui.set_next_window_size(sw, max(1, app.height - int(menu_h)), imgui.ALWAYS)

    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE |
             imgui.WINDOW_NO_SCROLLBAR)

    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.06, 0.06, 0.06, 0.97)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (s(10), s(6)))
    imgui.begin("##sidebar", flags=flags)

    # Record the sidebar's actual right edge in FRAMEBUFFER pixels so the
    # GL gallery (which renders in framebuffer space) starts exactly where
    # the sidebar ends. imgui lays out in logical/display units, so convert
    # via display_fb_scale — at fractional DPI the logical width differs
    # from the framebuffer width the gallery viewport uses.
    try:
        _fbx = imgui.get_io().display_fb_scale[0] or 1.0
        _right = imgui.get_window_position()[0] + imgui.get_window_size()[0]
        app._sidebar_right_fb = int(round(_right * _fbx))
    except Exception:
        app._sidebar_right_fb = 0

    # --- STATS STRIP ---
    mode_labels = {
        MODE_CONTACT_SHEETS: "SHEETS",
        MODE_LIGHT_TABLE: "VIEW",
        MODE_AUTOMATION: "TRAIN",
    }
    mode_str = mode_labels.get(app.mode, "VIEW")
    active_name = None
    if app.mode == MODE_LIGHT_TABLE and 0 <= app.selected_index < len(app.entries):
        points = app.entries[app.selected_index].point_count
        cloud_count = 1
        import os
        active_name = os.path.splitext(
            os.path.basename(app.entries[app.selected_index].file_path)
        )[0]
    else:
        points = sum(e.point_count for e in app.entries)
        cloud_count = len(app.entries)
    pending = app.catalog.pending_count if app.catalog else 0
    op1_stats_strip(app, cloud_count, points, pending, mode_str, app.fps,
                    active_name=active_name)
    imgui.spacing()

    # --- TAB BAR ---
    _draw_tab_bar(app)
    imgui.spacing()

    # --- Tab content ---
    if app.mode == MODE_CONTACT_SHEETS:
        changed = _draw_contact_sheets_tab(app)
    elif app.mode == MODE_LIGHT_TABLE:
        changed = _draw_light_table_tab(app)
    elif app.mode == MODE_AUTOMATION:
        _draw_train_tab(app)

    imgui.end()
    imgui.pop_style_var(1)
    imgui.pop_style_color(1)

    # Drag-to-resize handle on the sidebar's right edge.
    _handle_sidebar_resize(app)

    # Gallery right-click context menu + rename modal — rendered at top
    # level so the popups aren't clipped by the sidebar child windows.
    _draw_gallery_ctx_popup(app)
    _draw_gallery_rename_modal(app)

    return changed


def _draw_view_panel(app, show_bg: bool = True,
                      show_camera: bool = False) -> bool:
    """Single VIEW section replacing the legacy BG / OVERLAYS / CAMERA
    sections. Lays out one compact row of equal-width cells (BG knob +
    BBOX + GRID, or BBOX + GRID only) and optionally places the
    interactive camera cube below the row.

    Scaled down relative to the old individual panels.
    """
    from src.gui.op1_widgets import (
        op1_knob_cell, draw_styled_rect, draw_styled_rect_active,
        BUTTON_ROUNDING,
    )
    from src.gui.theme import (
        OP1_BLUE, OP1_GREEN, OP1_GRAY, OP1_WHITE, OP1_DIM, col32,
    )

    changed = False
    dl = imgui.get_window_draw_list()
    th = imgui.get_text_line_height()

    op1_section("CAMERA", collapsible=False)

    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    mxp, myp = imgui.get_mouse_pos()

    # Presets that lock projection to orthographic (planar views).
    # 'iso' and no-preset free-orbit remain togglable.
    PLANAR_PRESETS = {'top', 'bottom', 'front', 'back', 'left', 'right'}

    # --- Icon draw helpers ---
    def _icon_bbox(cx, cy, sz, color_u32):
        """Axonometric wireframe cube icon."""
        d = sz * 0.35
        x0 = cx - sz * 0.5
        y0 = cy - sz * 0.5 + d * 0.3
        x1 = cx + sz * 0.5 - d
        y1 = cy + sz * 0.5
        # front face
        dl.add_rect(x0, y0, x1, y1, color_u32, 0.0, 0, s(1.2))
        # offset back face
        dl.add_line(x0 + d, y0 - d, x1 + d, y0 - d, color_u32, s(1.2))
        dl.add_line(x1 + d, y0 - d, x1 + d, y1 - d, color_u32, s(1.2))
        # connectors
        dl.add_line(x0, y0, x0 + d, y0 - d, color_u32, s(1.2))
        dl.add_line(x1, y0, x1 + d, y0 - d, color_u32, s(1.2))
        dl.add_line(x1, y1, x1 + d, y1 - d, color_u32, s(1.2))

    def _icon_grid(cx, cy, sz, color_u32):
        """3x3 grid lines."""
        half = sz * 0.45
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        dl.add_rect(x0, y0, x1, y1, color_u32, 0.0, 0, s(1.2))
        for i in (1, 2):
            t = i / 3.0
            dl.add_line(x0 + (x1 - x0) * t, y0,
                        x0 + (x1 - x0) * t, y1, color_u32, s(1.0))
            dl.add_line(x0, y0 + (y1 - y0) * t,
                        x1, y0 + (y1 - y0) * t, color_u32, s(1.0))

    def _icon_proj(cx, cy, sz, color_u32):
        """Viewing frustum icon — trapezoid silhouette (wider at base
        than top) that reads as a camera frustum. Used for the
        parallel/perspective toggle button."""
        half = sz * 0.55
        top_half = half * 0.55
        bot_half = half * 0.95
        y0 = cy - half * 0.75
        y1 = cy + half * 0.75
        # Outline
        dl.add_line(cx - top_half, y0, cx + top_half, y0, color_u32, s(1.4))
        dl.add_line(cx + top_half, y0, cx + bot_half, y1, color_u32, s(1.4))
        dl.add_line(cx + bot_half, y1, cx - bot_half, y1, color_u32, s(1.4))
        dl.add_line(cx - bot_half, y1, cx - top_half, y0, color_u32, s(1.4))
        # Pupil dot at the apex (top) to ground it as a lens / eye
        dl.add_circle_filled(cx, y0 + half * 0.15,
                             s(1.6), color_u32, 8)

    def _icon_home(cx, cy, sz, color_u32):
        """House silhouette."""
        half = sz * 0.45
        body_top = cy - half * 0.1
        body_bot = cy + half * 0.7
        body_l = cx - half * 0.7
        body_r = cx + half * 0.7
        roof_top_y = cy - half * 0.8
        # Roof triangle
        dl.add_line(body_l - half * 0.1, body_top,
                    cx, roof_top_y, color_u32, s(1.4))
        dl.add_line(cx, roof_top_y,
                    body_r + half * 0.1, body_top, color_u32, s(1.4))
        # Body rectangle (left, right, bottom)
        dl.add_line(body_l, body_top, body_l, body_bot, color_u32, s(1.4))
        dl.add_line(body_r, body_top, body_r, body_bot, color_u32, s(1.4))
        dl.add_line(body_l, body_bot, body_r, body_bot, color_u32, s(1.4))
        # Door
        door_w = half * 0.22
        door_h = half * 0.35
        dl.add_rect(cx - door_w, body_bot - door_h,
                    cx + door_w, body_bot, color_u32, 0.0, 0, s(1.2))

    def _icon_dof(cx, cy, sz, color_u32):
        """Concentric circles — bokeh / depth of field."""
        dl.add_circle(cx, cy, sz * 0.45, color_u32, 16, s(1.2))
        dl.add_circle(cx, cy, sz * 0.25, color_u32, 12, s(1.2))
        dl.add_circle_filled(cx, cy, sz * 0.08, color_u32, 8)

    def _toggle_cell(cx, cy, cw, ch, icon_fn, active, accent):
        if active:
            draw_styled_rect_active(dl, cx, cy, cw, ch, accent,
                                    rounding=BUTTON_ROUNDING, thickness=1.0)
        else:
            draw_styled_rect(dl, cx, cy, cw, ch, OP1_DIM,
                             rounding=BUTTON_ROUNDING, thickness=1.0)
        icon_col = col32(accent if active else OP1_GRAY)
        icon_sz = min(cw, ch) * 0.45
        icon_fn(cx + cw * 0.5, cy + ch * 0.5, icon_sz, icon_col)
        return (cx <= mxp <= cx + cw and cy <= myp <= cy + ch
                and imgui.is_mouse_clicked(0))

    def _action_cell(cx, cy, cw, ch, icon_fn, accent):
        draw_styled_rect(dl, cx, cy, cw, ch, accent,
                         rounding=BUTTON_ROUNDING, thickness=1.0)
        icon_sz = min(cw, ch) * 0.45
        icon_fn(cx + cw * 0.5, cy + ch * 0.5, icon_sz, col32(accent))
        return (cx <= mxp <= cx + cw and cy <= myp <= cy + ch
                and imgui.is_mouse_clicked(0))

    if show_camera:
        # Camera panel extends to the bottom of the sidebar.
        from src.gui.op1_widgets import op1_alien
        from src.gui.op1_wireframe import op1_camera_cube
        import math

        avail_h = imgui.get_content_region_available()[1]
        panel_h = max(th * 10.0, avail_h)

        preset = app.camera.active_preset
        is_planar = preset in PLANAR_PRESETS

        # Planar presets lock projection to orthographic.
        if is_planar and app.camera.projection != 'orthographic':
            app.camera.projection = 'orthographic'
            changed = True

        is_ortho = (app.camera.projection == 'orthographic')
        if preset in ('right', 'left'):
            profile = 'side'
        elif preset == 'back':
            profile = 'back'
        elif preset == 'top':
            profile = 'top'
        elif preset == 'bottom':
            profile = 'bottom'
        else:
            profile = 'front'

        vc = getattr(app, 'viewport_count', 1)
        alien_clicked = False

        # --- Projection toggle alien (main) ---
        # Always sits below the cube. Orange = ortho, green = perspective.
        # Dimmed when the current preset is planar (click is a no-op —
        # planar views are locked to orthographic).
        alien_pixel_main = max(s(2.25), th * 0.255)
        alien_h = alien_pixel_main * 8
        alien_w = alien_pixel_main * 11

        # Reserve a bottom row for viewport-count invaders + grid toggle
        bottom_row_h = th * 2.4
        bottom_margin = th * 0.3
        bottom_row_y = wy + panel_h - bottom_margin - bottom_row_h

        # Cube area fits between the header strip and the bottom row
        cube_area_top = wy + th * 0.5
        cube_area_h = max(th * 6.0,
                          bottom_row_y - cube_area_top - th * 0.4)
        cube_area_cy = cube_area_top + cube_area_h * 0.45

        # Base color for main alien (dimmed if planar-locked)
        if is_planar:
            alien_color = OP1_GRAY
        elif is_ortho:
            alien_color = OP1_ORANGE
        else:
            alien_color = OP1_GREEN

        # Pre-compute an estimated alien position for the hit test; the
        # actual draw happens after the cube so we can pin to
        # cube_bottom_center. This estimate lives just below the cube.
        alien_cx_est = wx + w * 0.5
        alien_cy_est = cube_area_cy + cube_area_h * 0.38

        ax0 = alien_cx_est - alien_w * 0.5 - s(3)
        ax1 = alien_cx_est + alien_w * 0.5 + s(3)
        ay0 = alien_cy_est - alien_h * 0.5 - s(3)
        ay1 = alien_cy_est + alien_h * 0.5 + s(3)
        alien_main_hovered = (ax0 <= mxp <= ax1 and ay0 <= myp <= ay1)

        if (not alien_clicked and alien_main_hovered
                and imgui.is_mouse_clicked(0) and not is_planar):
            app.camera.projection = (
                'orthographic' if app.camera.projection == 'perspective'
                else 'perspective')
            is_ortho = not is_ortho
            alien_color = OP1_ORANGE if is_ortho else OP1_GREEN
            alien_clicked = True
            changed = True

        # --- Draw camera cube(s) ---
        # Multi-viewport: scale by 0.5 per extra view, arrange in 2x2
        if vc == 1:
            # Single cube — fills available space
            size_units = max(6.0, (cube_area_h / th) / 1.2)
            imgui.set_cursor_screen_pos((wx, cube_area_top))
            clicked_preset, cube_bottom_center = op1_camera_cube(
                app.camera, active_preset=app.camera.active_preset,
                size_units=size_units)
            if not alien_clicked and clicked_preset is not None:
                app.camera.set_preset(clicked_preset)
                if clicked_preset in PLANAR_PRESETS:
                    app.camera.projection = 'orthographic'
                changed = True
        else:
            # Multi-viewport: one cube per viewport. Layout matches
            # _viewport_rects() in main.py:
            #   vc=2: primary right, top left
            #   vc=3: primary right double-height, top top-left,
            #         front bottom-left
            #   vc=4: primary top-right, top top-left, front bottom-left,
            #         right bottom-right
            base_units = max(6.0, (cube_area_h / th) / 1.2)
            half_w = w * 0.5
            half_h = cube_area_h * 0.5

            secondaries = getattr(app, '_secondary_cameras', [])

            if vc == 2:
                slots = [
                    (app.camera, app.camera.active_preset,
                     wx + half_w, cube_area_top, half_w, cube_area_h),
                    (secondaries[0] if len(secondaries) > 0 else app.camera,
                     'top', wx, cube_area_top, half_w, cube_area_h),
                ]
            elif vc == 3:
                slots = [
                    (app.camera, app.camera.active_preset,
                     wx + half_w, cube_area_top, half_w, cube_area_h),
                    (secondaries[0] if len(secondaries) > 0 else app.camera,
                     'top', wx, cube_area_top, half_w, half_h),
                    (secondaries[1] if len(secondaries) > 1 else app.camera,
                     'front', wx, cube_area_top + half_h, half_w, half_h),
                ]
            else:  # vc == 4
                slots = [
                    (app.camera, app.camera.active_preset,
                     wx + half_w, cube_area_top, half_w, half_h),
                    (secondaries[0] if len(secondaries) > 0 else app.camera,
                     'top', wx, cube_area_top, half_w, half_h),
                    (secondaries[1] if len(secondaries) > 1 else app.camera,
                     'front', wx, cube_area_top + half_h, half_w, half_h),
                    (secondaries[2] if len(secondaries) > 2 else app.camera,
                     'right', wx + half_w, cube_area_top + half_h, half_w, half_h),
                ]

            cube_bottom_center = None
            for vi, (cam_i, preset_i, sx, sy, sw, sh) in enumerate(slots):
                imgui.set_cursor_screen_pos((sx, sy))
                # Scale cube to slot, fit both width and height
                slot_units = max(4.0, min(sw, sh) / th / 1.2)
                if vi == 0:
                    cp, cbc = op1_camera_cube(
                        cam_i, active_preset=preset_i,
                        size_units=slot_units, width=sw)
                    cube_bottom_center = cbc
                    if not alien_clicked and cp is not None:
                        app.camera.set_preset(cp)
                        if cp in PLANAR_PRESETS:
                            app.camera.projection = 'orthographic'
                        changed = True
                else:
                    op1_camera_cube(
                        cam_i, active_preset=preset_i,
                        size_units=slot_units, width=sw)

        # --- Draw main projection alien (always below the cube) ---
        if cube_bottom_center is not None:
            alien_cx = cube_bottom_center[0]
            alien_cy = cube_bottom_center[1] - alien_h * 0.5
        else:
            alien_cx = wx + w * 0.5
            alien_cy = cube_area_cy + cube_area_h * 0.38
        op1_alien(dl, alien_cx, alien_cy, alien_pixel_main,
                  col32(alien_color), angle=0.0, profile=profile)

        # --- Bottom row: 4 viewport-count invaders + grid toggle ---
        # Mini invaders on the left, grid-toggle square on the right.
        mini_pixel = max(s(1.0), th * 0.13)
        mini_gap = th * 0.35
        mini_alien_w = mini_pixel * 11
        mini_alien_h = mini_pixel * 8
        row_invaders_w = mini_alien_w * 4 + mini_gap * 3

        row_pad_x = th * 0.5
        row_cy = bottom_row_y + bottom_row_h * 0.5
        row_left = wx + row_pad_x

        # Viewport count invaders
        for mi in range(4):
            mcx = row_left + mi * (mini_alien_w + mini_gap) + mini_alien_w * 0.5
            mcy = row_cy
            is_active = (mi < vc)
            mini_col = OP1_ORANGE if is_active else OP1_GRAY
            op1_alien(dl, mcx, mcy, mini_pixel, col32(mini_col),
                      angle=0.0, profile='front')
            mhx0 = mcx - mini_alien_w * 0.5 - s(2)
            mhx1 = mcx + mini_alien_w * 0.5 + s(2)
            mhy0 = mcy - mini_alien_h * 0.5 - s(2)
            mhy1 = mcy + mini_alien_h * 0.5 + s(2)
            if (mhx0 <= mxp <= mhx1 and mhy0 <= myp <= mhy1
                    and imgui.is_mouse_clicked(0)):
                app.viewport_count = mi + 1
                vc = mi + 1
                alien_clicked = True
                changed = True

        # Grid toggle button to the right of the invaders
        btn_size = bottom_row_h * 0.85
        btn_x = row_left + row_invaders_w + th * 0.6
        btn_y = row_cy - btn_size * 0.5
        if _toggle_cell(btn_x, btn_y, btn_size, btn_size, _icon_grid,
                        app.show_grid, OP1_ORANGE):
            app.show_grid = not app.show_grid
            changed = True

        imgui.set_cursor_screen_pos((wx, wy + panel_h))
        imgui.dummy(w, s(4))
        return changed

    # --- Contact Sheets layout: BBOX + GRID + HOME in a row ---
    # Background is locked to dark mode so there is no BG knob.
    cells = ['bbox', 'grid', 'home']
    n = len(cells)
    gap = th * 0.4
    cell_w = (w - gap * (n - 1)) / max(n, 1)
    row_h = th * 2.6

    for i, kind in enumerate(cells):
        cx = wx + i * (cell_w + gap)
        if kind == 'bbox':
            if _toggle_cell(cx, wy, cell_w, row_h, _icon_bbox,
                            app.show_bbox, OP1_ORANGE):
                app.show_bbox = not app.show_bbox
                changed = True
        elif kind == 'grid':
            if _toggle_cell(cx, wy, cell_w, row_h, _icon_grid,
                            app.show_grid, OP1_ORANGE):
                app.show_grid = not app.show_grid
                changed = True
        elif kind == 'home':
            if _action_cell(cx, wy, cell_w, row_h, _icon_home, OP1_ORANGE):
                app._fit_camera(animate=True)
                changed = True

    imgui.dummy(w, row_h + s(4))

    # --- Scene unit selector ---
    _UNITS = ['mm', 'm', 'um', 'vx', 'raw']
    ux, uy = imgui.get_cursor_screen_pos()
    unit_h = th * 1.4
    unit_w_each = (w - gap * (len(_UNITS) - 1)) / len(_UNITS)
    for i, u in enumerate(_UNITS):
        bx = ux + i * (unit_w_each + gap)
        is_sel = getattr(app, 'scene_unit', 'mm') == u
        if is_sel:
            dl.add_rect_filled(bx, uy, bx + unit_w_each, uy + unit_h,
                               col32((OP1_ORANGE[0] * 0.15, OP1_ORANGE[1] * 0.15,
                                      OP1_ORANGE[2] * 0.15, 1.0)),
                               s(BUTTON_ROUNDING))
        lw = imgui.calc_text_size(u)[0]
        dl.add_text(bx + (unit_w_each - lw) * 0.5, uy + (unit_h - th) * 0.5,
                    col32(OP1_ORANGE if is_sel else DIM), u)
    imgui.dummy(w, unit_h)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx_u, _ = imgui.get_mouse_pos()
        for i, u in enumerate(_UNITS):
            bx = ux + i * (unit_w_each + gap)
            if bx <= mx_u <= bx + unit_w_each:
                app.scene_unit = u
                changed = True
                break
    imgui.spacing()

    return changed


def _handle_sidebar_resize(app):
    """Draw the sidebar drag handle and drive the resize interaction.

    A narrow vertical strip is overlaid on the sidebar's right edge.
    Hover/drag lightens the strip and pins the OS horizontal-resize
    cursor. The drag state lives on ``app`` so ``_update_tool_cursor``
    can yield the cursor while a resize is in progress.
    """
    from src.gui.scale import (
        sidebar_width, set_sidebar_width_left,
        get_scale, sidebar_min_logical,
    )

    sw = sidebar_width()
    if sw <= 0:
        return

    menu_h = int(getattr(app, '_menu_bar_height', 0))
    top = float(menu_h)
    bottom = float(app.height)

    # Hit strip is a touch wider than the visual line so the grab area
    # stays forgiving on high-DPI displays.
    hit_half = s(5)
    dl = imgui.get_foreground_draw_list()
    mx, my = imgui.get_mouse_pos()
    hovered = (sw - hit_half <= mx <= sw + hit_half and top <= my <= bottom)

    dragging = (app._sidebar_drag_active == 'left')

    # No visual divider or shadow — sidebar blends seamlessly into viewport.
    # The invisible hit strip still allows drag-to-resize.

    # Pin the OS resize cursor while hovering or dragging the handle.
    if dragging or hovered:
        try:
            import glfw as _glfw
            if app._cursor_resize_ew is not None:
                _glfw.set_cursor(app.window, app._cursor_resize_ew)
        except Exception:
            pass

    # Start drag.
    if not dragging and hovered and imgui.is_mouse_clicked(0):
        app._sidebar_drag_active = 'left'
        app._sidebar_drag_start_x = float(mx)
        scale = get_scale()
        app._sidebar_drag_start_width = int(sw / max(scale, 1e-6))

    # Drive drag.
    if dragging:
        if imgui.is_mouse_down(0):
            scale = get_scale()
            dx_fb = float(mx) - app._sidebar_drag_start_x
            dx_logical = dx_fb / max(scale, 1e-6)
            new_logical = int(app._sidebar_drag_start_width + dx_logical)
            # Clamp to [min, 60% of window width] so the viewport never
            # disappears.
            max_logical = int((app.width / max(scale, 1e-6)) * 0.60)
            new_logical = max(sidebar_min_logical(),
                              min(max_logical, new_logical))
            set_sidebar_width_left(new_logical)
        else:
            app._sidebar_drag_active = None


def _draw_tab_bar(app):
    """Browser-style tabs: rounded top, flat bottom, active tab connects
    to a baseline strip. One per workflow mode, color-coded by stage.

    In Clinician mode the research-only tabs (Contact Sheets / Light /
    Train) are hidden so the surgeon sees only the workflow they need:
    Imaging → Hologram → Overwatch."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    tab_h = th * 2.2
    gap = s(1.0)

    # Filter modes/names/colors by user_mode. Clinician sees 3 tabs;
    # researcher sees all 6.
    all_modes = [
        MODE_CONTACT_SHEETS, MODE_LIGHT_TABLE, MODE_AUTOMATION,
    ]
    visible_idx = list(range(len(all_modes)))
    visible_names = [_TAB_NAMES[i] for i in visible_idx]
    visible_colors = [_TAB_COLORS[i] for i in visible_idx]
    modes = [all_modes[i] for i in visible_idx]

    n_tabs = len(modes)
    if n_tabs == 0:
        return
    tab_w = (w - gap * (n_tabs - 1)) / n_tabs
    tab_r = s(BUTTON_ROUNDING)
    baseline_h = s(2.0)

    active_accent = None
    active_x0 = active_x1 = None

    for i, (name, color, mode) in enumerate(zip(visible_names, visible_colors, modes)):
        tx = wx + i * (tab_w + gap)
        is_active = app.mode == mode

        if is_active:
            # Dark fill, slight accent tint — no stroke borders
            fill = (color[0] * 0.10 + 0.05,
                    color[1] * 0.10 + 0.05,
                    color[2] * 0.10 + 0.05, 1.0)
            dl.add_rect_filled(tx, wy, tx + tab_w, wy + tab_h + baseline_h,
                               col32(fill), tab_r,
                               imgui.DRAW_ROUND_CORNERS_TOP)
            active_accent = color
            active_x0, active_x1 = tx, tx + tab_w
        else:
            # Inactive: subtle dark fill, no outline, dim text
            dl.add_rect_filled(tx, wy + s(2), tx + tab_w, wy + tab_h,
                               col32((0.05, 0.05, 0.05, 1.0)), tab_r,
                               imgui.DRAW_ROUND_CORNERS_TOP)

        text_col = col32(color) if is_active else col32(DIM)
        text_w = imgui.calc_text_size(name)[0]
        dl.add_text(tx + (tab_w - text_w) * 0.5,
                    wy + tab_h * 0.5 - th * 0.5, text_col, name)

    imgui.dummy(w, tab_h + baseline_h)

    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        for i, mode in enumerate(modes):
            tx = wx + i * (tab_w + gap)
            if tx <= mx <= tx + tab_w:
                app.mode = mode
                break


_ontology_popup_open: bool = False


def _draw_ontology_picker(app) -> bool:
    """Compact ontology preset selector for the Contact Sheets LABELS section.

    Shows current preset name (or 'None'). Clicking toggles a dropdown
    of available presets. Only active for user projects — hidden for
    smart views and folder views.
    """
    global _ontology_popup_open
    from src.data.labels import ONTOLOGY_PRESETS

    view = getattr(app, 'active_view', None)
    col = None
    if view is not None and view[0] == "project":
        catalog = getattr(app, 'catalog', None)
        if catalog is not None:
            col = catalog.projects.get(view[1])

    # Only show for user projects (not smart views / folder views)
    is_user_col = col is not None and not view[1].startswith("__")
    if not is_user_col:
        return False

    changed = False
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()
    row_h = th * 1.6
    _purple = (0.7, 0.5, 0.95, 1.0)

    current = col.ontology_preset_name or "None"

    # Row background
    draw_styled_rect(dl, wx, wy, w, row_h, _purple,
                     rounding=BUTTON_ROUNDING, thickness=1.0)

    # Label + value
    dl.add_text(wx + th * 0.4, wy + (row_h - th) * 0.5,
                col32(DIM), "ONTOLOGY")
    val_w = imgui.calc_text_size(current)[0]
    dl.add_text(wx + w - val_w - th * 0.5, wy + (row_h - th) * 0.5,
                col32(_purple), current)

    # Dropdown arrow
    arrow_x = wx + w - val_w - th * 1.2
    arrow_y = wy + row_h * 0.5
    arrow_sz = th * 0.2
    if _ontology_popup_open:
        dl.add_triangle_filled(
            arrow_x - arrow_sz, arrow_y + arrow_sz * 0.5,
            arrow_x + arrow_sz, arrow_y + arrow_sz * 0.5,
            arrow_x, arrow_y - arrow_sz * 0.5,
            col32(_purple))
    else:
        dl.add_triangle_filled(
            arrow_x - arrow_sz, arrow_y - arrow_sz * 0.5,
            arrow_x + arrow_sz, arrow_y - arrow_sz * 0.5,
            arrow_x, arrow_y + arrow_sz * 0.5,
            col32(_purple))

    imgui.dummy(w, row_h)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        _ontology_popup_open = not _ontology_popup_open

    # Dropdown list
    if _ontology_popup_open:
        options = ["None"] + list(ONTOLOGY_PRESETS.keys())
        for opt in options:
            ox, oy = imgui.get_cursor_screen_pos()
            opt_h = th * 1.4
            is_selected = (opt == current)

            if is_selected:
                dl.add_rect_filled(ox, oy, ox + w, oy + opt_h,
                                   col32((0.7 * 0.15, 0.5 * 0.15, 0.95 * 0.15, 1.0)),
                                   s(BUTTON_ROUNDING))
            elif imgui.is_mouse_hovering_rect(ox, oy, ox + w, oy + opt_h):
                dl.add_rect_filled(ox, oy, ox + w, oy + opt_h,
                                   col32((0.12, 0.12, 0.12, 1.0)),
                                   s(BUTTON_ROUNDING))

            dl.add_text(ox + th * 0.5, oy + (opt_h - th) * 0.5,
                        col32(WHITE if is_selected else OP1_GRAY), opt)

            imgui.dummy(w, opt_h)
            if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
                _ontology_popup_open = False
                if opt == "None":
                    if col.ontology_preset_name is not None:
                        app.catalog.clear_project_ontology(col.id)
                        app._restore_global_schema()
                        if app.label_texture is not None:
                            from src.rendering.label_texture import update_label_color_texture
                            update_label_color_texture(app.label_texture, app.label_registry)
                        app.label_count_cache.clear()
                        changed = True
                elif opt != current:
                    app.catalog.set_project_ontology(col.id, opt)
                    app._sync_project_state(app.active_view)
                    changed = True

        imgui.spacing()

    imgui.spacing()
    return changed


def _styled_button(label: str, color: tuple, active: bool = False,
                   height_units: float = 2.0, fill_alpha: float = 0.0) -> bool:
    """Full-width styled button using the unified fill+border system.

    ``fill_alpha`` (0..1) optionally drops a translucent ``color`` wash
    behind the border so primary-action buttons (LAUNCH TRAINING etc.)
    can read more prominently than secondary chrome. Default 0 keeps the
    existing pure-outline look for every other call site.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()
    h = th * height_units

    if fill_alpha > 0.0:
        r, g, b = color[:3]
        dl.add_rect_filled(
            wx, wy, wx + w, wy + h,
            imgui.get_color_u32_rgba(r, g, b, fill_alpha),
            s(BUTTON_ROUNDING),
        )

    if active:
        draw_styled_rect_active(dl, wx, wy, w, h, color,
                                rounding=BUTTON_ROUNDING, thickness=1.8)
    else:
        draw_styled_rect(dl, wx, wy, w, h, color,
                         rounding=BUTTON_ROUNDING, thickness=1.2)

    lbl_size = imgui.calc_text_size(label)
    dl.add_text(wx + (w - lbl_size[0]) * 0.5,
                wy + (h - th) * 0.5,
                imgui.get_color_u32_rgba(*color), label)

    imgui.dummy(w, h + s(4))
    return imgui.is_item_hovered() and imgui.is_mouse_clicked(0)


def _draw_library_panel(app):
    """Library browser: PROJECTS + smart collections.

    Lives inside the Contact Sheets tab. Clicking a row changes the
    app's active_view, which re-fills the grid with that view's cached
    previews without re-importing anything from disk.
    """
    global _new_project_buf, _show_new_project

    catalog = getattr(app, "catalog", None)
    if catalog is None:
        return

    th = imgui.get_text_line_height()
    row_h = th * 1.45

    # ---------------- PROJECTS ----------------
    op1_section("PROJECTS", collapsible=False)

    from src.data.library_catalog import SMART_ALL, SMART_RECENT, SMART_MISSING

    smart_items = [
        ("All Photos", ("project", SMART_ALL), len(catalog.entries), OP1_WHITE),
        ("Recent Imports", ("project", SMART_RECENT),
         len(catalog.entries_in_project(SMART_RECENT)), OP1_BLUE),
    ]
    # Cached; stats the filesystem at most once per entry per TTL
    # (~2s) instead of once per entry per frame.
    missing_count = catalog.missing_count()
    if missing_count > 0:
        smart_items.append(
            ("Missing Files", ("project", SMART_MISSING),
             missing_count, OP1_RED)
        )

    for label, view, count, color in smart_items:
        _library_row(app, label, count, color, view,
                     indent=th * 0.6, is_smart=True)

    # Divider before user projects
    if catalog.projects:
        imgui.dummy(1, s(4))
    for cid, col in sorted(catalog.projects.items(), key=lambda kv: kv[1].name.lower()):
        _library_row(app, col.name, len(col.file_keys), OP1_GREEN,
                     ("project", cid), indent=th * 0.6,
                     is_smart=False, project_id=cid)

    # --- New project inline input ---
    if _show_new_project:
        imgui.push_item_width(imgui.get_content_region_available_width() - s(60))
        changed, _new_project_buf = imgui.input_text(
            "##new_col", _new_project_buf, 64,
            flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        imgui.pop_item_width()
        committed = changed and _new_project_buf.strip()
        imgui.same_line(spacing=s(4))
        if imgui.button("OK##col", width=s(28)):
            committed = True
        imgui.same_line(spacing=s(2))
        if imgui.button("X##col", width=s(22)):
            _show_new_project = False
            _new_project_buf = ""
        if committed:
            name = _new_project_buf.strip()
            if name:
                catalog.create_project(name)
            _new_project_buf = ""
            _show_new_project = False
    else:
        if _styled_button("+ NEW PROJECT", OP1_DIM, height_units=1.6):
            _show_new_project = True

    imgui.spacing()
    # FOLDERS panel removed — filesystem tree browser lived here but
    # users navigate by projects + smart collections exclusively now.
    # _draw_folder_node is kept as dead code in case we bring the
    # folder browser back as an opt-in panel later.


def _draw_folder_node(app, node, depth: int, row_h: float):
    """Recursively draw a folder tree node with disclosure triangle."""
    th = imgui.get_text_line_height()
    has_children = bool(node["children"])
    path = node["path"]
    expanded = path in _library_expanded

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    indent = s(10) + depth * s(12)

    # Highlight when this folder is the active view
    active_view = getattr(app, "active_view", None)
    is_active = (
        active_view is not None
        and active_view[0] == "folder"
        and active_view[1] == path
    )

    if is_active:
        dl.add_rect_filled(wx + indent - s(2), wy,
                           wx + w, wy + row_h,
                           imgui.get_color_u32_rgba(0.99, 0.57, 0.14, 0.22))
        dl.add_rect_filled(wx + indent - s(2), wy,
                           wx + indent, wy + row_h,
                           imgui.get_color_u32_rgba(*OP1_ORANGE))

    # Disclosure triangle for expandable folders
    tri_x = wx + indent + th * 0.15
    if has_children:
        tri_y = wy + row_h * 0.5
        tri_sz = th * 0.28
        tri_col = imgui.get_color_u32_rgba(*(OP1_ORANGE if is_active else OP1_GRAY))
        if expanded:
            # Downward triangle
            dl.add_triangle_filled(
                tri_x, tri_y - tri_sz * 0.5,
                tri_x + tri_sz * 1.15, tri_y - tri_sz * 0.5,
                tri_x + tri_sz * 0.58, tri_y + tri_sz * 0.5,
                tri_col)
        else:
            # Rightward triangle
            dl.add_triangle_filled(
                tri_x, tri_y - tri_sz * 0.65,
                tri_x, tri_y + tri_sz * 0.65,
                tri_x + tri_sz * 0.9, tri_y,
                tri_col)

    # Folder icon
    from src.gui.op1_widgets import op1_folder_icon
    icon_cx = wx + indent + th * 0.55
    icon_cy = wy + row_h * 0.5
    icon_col = imgui.get_color_u32_rgba(*(OP1_ORANGE if is_active else OP1_GRAY))
    op1_folder_icon(dl, icon_cx, icon_cy, th * 0.95, icon_col,
                    thickness=s(1.0))

    # Label
    name_x = wx + indent + th * 1.3
    name = node["name"] or "/"
    # Elide long names
    max_name_px = w - (name_x - wx) - s(40)
    display = name
    if imgui.calc_text_size(name)[0] > max_name_px and len(name) > 6:
        keep = max(3, int(len(name) * max_name_px / max(imgui.calc_text_size(name)[0], 1)) - 1)
        display = name[:keep] + "…"

    name_col = imgui.get_color_u32_rgba(*OP1_WHITE if is_active else OP1_GRAY)
    dl.add_text(name_x, wy + (row_h - th) * 0.5, name_col, display)

    # Count on the right
    count_str = str(node["count"])
    cw = imgui.calc_text_size(count_str)[0]
    dl.add_text(wx + w - cw - s(6), wy + (row_h - th) * 0.5,
                imgui.get_color_u32_rgba(*OP1_DIM), count_str)

    imgui.dummy(w, row_h)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, _ = imgui.get_mouse_pos()
        # Click on the triangle region toggles expansion; click on the
        # row body selects the folder as the active view.
        if has_children and mx < tri_x + th * 0.85:
            if expanded:
                _library_expanded.discard(path)
            else:
                _library_expanded.add(path)
        else:
            app.set_active_view(("folder", path))

    if has_children and expanded:
        for child in node["children"]:
            _draw_folder_node(app, child, depth + 1, row_h)


def _library_row(app, label: str, count: int, color: tuple,
                 view: tuple[str, str], indent: float,
                 is_smart: bool = True, project_id: str | None = None):
    """Single selectable library row (project or smart view)."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()
    row_h = th * 1.45

    active_view = getattr(app, "active_view", None)
    is_active = (active_view == view)

    if is_active:
        r, g, b, _a = color
        dl.add_rect_filled(wx + indent - s(2), wy,
                           wx + w, wy + row_h,
                           imgui.get_color_u32_rgba(r, g, b, 0.22))
        dl.add_rect_filled(wx + indent - s(2), wy,
                           wx + indent, wy + row_h,
                           imgui.get_color_u32_rgba(*color))

    # Bullet
    dot_x = wx + indent + s(4)
    dot_y = wy + row_h * 0.5
    dl.add_circle_filled(dot_x, dot_y, th * 0.22,
                         imgui.get_color_u32_rgba(*color), 10)

    label_col = imgui.get_color_u32_rgba(*(OP1_WHITE if is_active else OP1_GRAY))
    dl.add_text(dot_x + s(10), wy + (row_h - th) * 0.5, label_col, label)

    count_str = str(count)
    cw = imgui.calc_text_size(count_str)[0]
    dl.add_text(wx + w - cw - s(6), wy + (row_h - th) * 0.5,
                imgui.get_color_u32_rgba(*OP1_DIM), count_str)

    imgui.dummy(w, row_h)

    if imgui.is_item_hovered():
        if imgui.is_mouse_clicked(0):
            app.set_active_view(view)
        elif not is_smart and project_id and imgui.is_mouse_clicked(1):
            # Right-click = open context menu. Used to delete instantly,
            # which made a stray RMB destroy the project's ontology with
            # no undo. The menu is the confirmation — user has to click
            # Delete explicitly. Cloud files and per-cloud labels are
            # untouched by delete; only the project grouping + ontology
            # mapping go away.
            app._project_ctx_target = project_id
            imgui.open_popup("##proj_ctx_menu")

    # Render the context menu when this row owns it. Only one menu is
    # ever open at a time (ImGui enforces this for popups sharing an id),
    # so the target check just routes the menu to the row that opened it.
    if (not is_smart and project_id
            and getattr(app, '_project_ctx_target', '') == project_id):
        if imgui.begin_popup("##proj_ctx_menu"):
            imgui.text_disabled(label)
            imgui.separator()
            # Duplicate — fresh project with same labels + model
            # registry, empty cloud set. Models stay shared on disk
            # (zero-copy reference); ontology is deep-copied so the
            # duplicate diverges cleanly from the source on edit.
            dup_clicked, _ = imgui.selectable("Duplicate (labels + models)")
            if dup_clicked:
                _duplicate_project_with_models(app, project_id)
                app._project_ctx_target = ""
                imgui.close_current_popup()
            imgui.spacing()
            imgui.separator()
            clicked, _ = imgui.selectable("Delete project")
            if clicked:
                app.catalog.delete_project(project_id)
                if active_view == ("project", project_id):
                    app.active_view = None
                app._project_ctx_target = ""
                imgui.close_current_popup()
            imgui.spacing()
            imgui.text_colored("ontology removed", 0.95, 0.7, 0.25)
            imgui.text_colored("clouds + labels kept", 0.45, 0.7, 0.45)
            imgui.end_popup()
        else:
            # Popup closed by clicking outside — clear the target so the
            # next RMB on a different project opens a fresh menu cleanly.
            app._project_ctx_target = ""


_SORT_KEYS = ["NAME", "POINTS", "CREATED", "IMPORTED"]


def _duplicate_project_with_models(app, source_project_id: str,
                                   include_clouds: bool = True) -> None:
    """Right-click action: clone a project into an independent copy.

    What we copy (1.1 full duplicate):
      - Project metadata (name, ontology_data, settings) — deep copy.
      - Cloud membership (``file_keys``) plus a verbatim copy of every
        label file into the duplicate's own namespace — repainting or
        re-classing the copy never perturbs the source. This is the
        "retrain with a different class set" entry point.
      - Model registry entries — new model_ids, **same** checkpoint
        paths on disk. Inference in the duplicate uses the source's
        trained .pth files for free.

    With ``include_clouds=False`` (the "setup only" variant) the
    duplicate starts with zero clouds — apply this ontology to a new
    batch.

    Checkpoint files themselves stay where they are. If the source
    project is later deleted and its training_runs folder cleaned up,
    the duplicate's references go dangling; the model registry's
    load-time self-heal covers reachable cases.

    Logs the result to the CLI for confirmation.
    """
    catalog = getattr(app, "catalog", None)
    if catalog is None:
        return
    new_proj = catalog.duplicate_project(
        source_project_id, include_clouds=include_clouds)
    if new_proj is None:
        if app.cli:
            app.cli.log(f"Duplicate failed: source project not found.", "error")
        return
    # Lazy-init the model registry the same way the TRAIN tab does.
    if getattr(app, "_train_model_registry", None) is None:
        from src.data.model_registry import ProjectModelRegistry
        from src.data.library_catalog import library_dir
        app._train_model_registry = ProjectModelRegistry(library_dir())
    n_models = app._train_model_registry.duplicate_from(
        source_project_id, new_proj.id,
    )
    src_name = catalog.projects[source_project_id].name
    msg = (f"Duplicated '{src_name}' → '{new_proj.name}'  "
           f"({len(new_proj.file_keys)} cloud(s) + labels copied, "
           f"{n_models} model(s) shared)")
    print(f"[duplicate] {msg}")
    if app.cli:
        app.cli.log(msg, "success")
    # Switch to the new project through set_active_view — the 1.0 code
    # assigned app.active_view directly, which left the entry list and
    # (now) the label namespace pointing at the source project.
    app.set_active_view(("project", new_proj.id))


def _apply_gallery_sort(app):
    """Sort app.entries in place according to the current sort setting."""
    key = getattr(app, '_gallery_sort_key', 'NAME')
    desc = getattr(app, '_gallery_sort_desc', False)
    catalog = getattr(app, 'catalog', None)

    def _sort_val(e):
        if key == 'POINTS':
            return e.point_count
        elif key == 'CREATED':
            try:
                return os.path.getmtime(e.file_path)
            except OSError:
                return 0.0
        elif key == 'IMPORTED':
            if catalog and e.file_key and e.file_key in catalog.entries:
                return catalog.entries[e.file_key].import_date
            return 0.0
        else:  # NAME
            return e.name.lower()

    try:
        app.entries.sort(key=_sort_val, reverse=desc)
    except Exception:
        pass


def _draw_sort_bar(app) -> bool:
    """Compact clickable sort labels: NAME | POINTS | CREATED | IMPORTED."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()
    row_h = th * 1.6

    cur_key = getattr(app, '_gallery_sort_key', 'NAME')
    cur_desc = getattr(app, '_gallery_sort_desc', False)
    mx, my = imgui.get_mouse_pos()
    changed = False

    n = len(_SORT_KEYS)
    cell_w = w / n

    for i, label in enumerate(_SORT_KEYS):
        cx = wx + i * cell_w
        is_active = (label == cur_key)
        # Arrow indicator
        arrow = ""
        if is_active:
            arrow = " ▼" if cur_desc else " ▲"
        display = label + arrow
        tw = imgui.calc_text_size(display)[0]
        color = OP1_BLUE if is_active else DIM
        dl.add_text(cx + (cell_w - tw) * 0.5,
                    wy + (row_h - th) * 0.5,
                    col32(color), display)
        # Click detection
        if (cx <= mx <= cx + cell_w and wy <= my <= wy + row_h
                and imgui.is_mouse_clicked(0)):
            if is_active:
                app._gallery_sort_desc = not cur_desc
            else:
                app._gallery_sort_key = label
                app._gallery_sort_desc = False
            _apply_gallery_sort(app)
            changed = True

    imgui.dummy(w, row_h)
    return changed


def _draw_clouds_section(app, child_id: str = "##clouds") -> bool:
    """Shared CLOUDS section — scrollable cloud list with adjustable height.

    Each row shows the distinct label colors present in the cloud (cached
    via app.label_count_cache.unique_label_ids), so the user can scan
    the list and see which clouds have been labelled with which classes
    at a glance.
    """
    changed = False
    th = imgui.get_text_line_height()
    from src.gui.op1_widgets import op1_cloud_row
    _green_tint = (OP1_GREEN[0] * 0.10, OP1_GREEN[1] * 0.10,
                   OP1_GREEN[2] * 0.10, 1.0)
    cache = getattr(app, 'label_count_cache', None)
    registry = getattr(app, 'label_registry', None)
    if op1_section("CLOUDS", OP1_GREEN, fill_color=_green_tint, collapsible=False):
        clouds_h = getattr(app, '_clouds_list_h', th * 5.5)
        imgui.begin_child(child_id, 0, clouds_h, border=False)
        # Multi-select state — only meaningful in Contact Sheets, but the
        # set/anchor live on app and are read here regardless. ``selected_index``
        # is still the Light Table primary; the set is a transient superset
        # used by Contact Sheets batch actions.
        multi_set = getattr(app, 'contact_sheets_selected', None)
        if not isinstance(multi_set, set):
            multi_set = None
        for i, entry in enumerate(app.entries):
            is_sel = (i == app.selected_index)
            in_multi = bool(multi_set is not None and i in multi_set)
            has_preview = entry.preview_gpu is not None or entry.full_gpu is not None
            pts = entry.point_count if entry.point_count else 0
            name = entry.name if hasattr(entry, 'name') else f"Cloud {i}"

            label_colors = None
            if cache is not None and registry is not None:
                gpu = entry.full_gpu or entry.preview_gpu
                cloud = gpu.cloud_data if gpu is not None else None
                if cloud is not None and cloud.labels is not None:
                    # Reuse the entry-cached color list when neither the
                    # cloud identity nor its labels_version has changed.
                    # Without this, every frame allocates one list +
                    # tuple-per-label across every visible row — small per
                    # frame but adds GC pressure with 50+ clouds open.
                    cache_key = (id(cloud),
                                 int(getattr(cloud, 'labels_version', 0)))
                    cached_colors = getattr(entry, '_row_colors_cache', None)
                    if cached_colors is not None and cached_colors[0] == cache_key:
                        label_colors = cached_colors[1]
                    else:
                        ids = cache.unique_label_ids(cloud)
                        if ids:
                            label_colors = [
                                tuple(info.color) for info in
                                (registry.get(lid) for lid in ids)
                                if info is not None
                            ]
                        entry._row_colors_cache = (cache_key, label_colors)

            if op1_cloud_row(i, name, pts, is_sel, has_preview,
                             accent_color=OP1_GREEN,
                             label_colors=label_colors,
                             is_in_multi_set=in_multi):
                io = imgui.get_io()
                ctrl = bool(getattr(io, 'key_ctrl', False))
                shift = bool(getattr(io, 'key_shift', False))
                if multi_set is not None and ctrl:
                    # Toggle membership; selected_index follows the click so
                    # Light Table mode still has a sensible single target.
                    if i in multi_set:
                        multi_set.discard(i)
                    else:
                        multi_set.add(i)
                    app._select_entry_by_index(i)
                    app.contact_sheets_selected_anchor = i
                elif multi_set is not None and shift and \
                        getattr(app, 'contact_sheets_selected_anchor', None) is not None:
                    a = int(app.contact_sheets_selected_anchor)
                    lo, hi = (a, i) if a <= i else (i, a)
                    multi_set.update(range(lo, hi + 1))
                    app._select_entry_by_index(i)
                else:
                    # Plain click — single select, replaces the multi set.
                    # Route through _select_entry_by_index so the
                    # camera, full-res load, clip box, and stale-label
                    # persistence all follow the click. Arrow-key
                    # navigation does the same so click + arrow stay
                    # in lockstep.
                    app._select_entry_by_index(i)
                    if multi_set is not None:
                        multi_set.clear()
                        multi_set.add(i)
                    app.contact_sheets_selected_anchor = i
                changed = True
        imgui.end_child()
    return changed


_labels_resize_dragging: bool = False


def _draw_labels_section(app, child_id: str = "##labels") -> bool:
    """Shared LABELS section — scrollable label rows with +ADD in header.

    The list height is user-adjustable via a drag handle below the
    list. Height persists across sessions in prefs.json as
    ``labels_list_h``.
    """
    global _labels_resize_dragging
    changed = False
    th = imgui.get_text_line_height()
    _label_purple = (0.7, 0.5, 0.95, 1.0)
    _label_tint = (0.7 * 0.10, 0.5 * 0.10, 0.95 * 0.10, 1.0)
    if op1_section("LABELS", _label_purple, fill_color=_label_tint, collapsible=False):
        from src.gui.label_panel import draw_label_add_button
        changed |= draw_label_add_button(app)

        stored = getattr(app, '_labels_list_h', 0.0)
        label_max_h = stored if stored > 0 else th * 10
        min_h = th * 3.0
        max_h = th * 40.0

        imgui.begin_child(child_id, 0, label_max_h, border=False)
        changed |= draw_label_panel(app)
        imgui.end_child()

        # --- Resize handle ---
        dl = imgui.get_window_draw_list()
        hx, hy = imgui.get_cursor_screen_pos()
        hw = imgui.get_content_region_available_width()
        handle_h = s(6)

        mx, my = imgui.get_mouse_pos()
        in_handle = (hx <= mx <= hx + hw and hy <= my <= hy + handle_h)

        active = _labels_resize_dragging or in_handle
        grip_col = imgui.get_color_u32_rgba(
            0.99, 0.57, 0.14, 0.75 if active else 0.0)
        base_col = imgui.get_color_u32_rgba(
            0.3, 0.3, 0.3, 0.55 if not active else 0.0)

        # Subtle grip: two short bars centered in the handle strip
        bar_y = hy + handle_h * 0.5
        bar_thick = max(1.0, s(1.0))
        bar_len = th * 1.1
        cx = hx + hw * 0.5
        gap = th * 0.25
        col = grip_col if active else base_col
        dl.add_line(cx - gap - bar_len, bar_y, cx - gap, bar_y, col, bar_thick)
        dl.add_line(cx + gap, bar_y, cx + gap + bar_len, bar_y, col, bar_thick)

        imgui.dummy(hw, handle_h)

        # Drag logic
        if _labels_resize_dragging:
            if imgui.is_mouse_down(0):
                delta_y = imgui.get_io().mouse_delta[1]
                if abs(delta_y) > 0.01:
                    new_h = label_max_h + delta_y
                    new_h = max(min_h, min(max_h, new_h))
                    app._labels_list_h = float(new_h)
            else:
                _labels_resize_dragging = False
                # Persist the final height
                try:
                    from src.utils.prefs import update_prefs
                    update_prefs({'labels_list_h': float(app._labels_list_h)})
                except Exception:
                    pass
        elif in_handle and imgui.is_mouse_clicked(0):
            _labels_resize_dragging = True
    return changed




















































# ---------------------------------------------------------------------
# Measurement cache + CSV export
# ---------------------------------------------------------------------





# ---------------------------------------------------------------------
# Selection-driven on-demand measurements
# ---------------------------------------------------------------------

























def _draw_contact_sheets_tab(app) -> bool:
    """Contact Sheets tab — project intake and shaping.

    Workflow order, top to bottom: IMPORT (bring clouds in) →
    PROJECTS (pick/create the working set) → ONTOLOGY (define the
    label classes for this project) → CLOUDS (membership + selection)
    → LABELS (this project's palette) → POINT SIZE.

    Inference moved out in 1.1: batch runs live on the TRAIN tab next
    to the models they use; single-cloud runs live on the Light Table
    INFER button.
    """
    changed = False
    th = imgui.get_text_line_height()

    # --- IMPORT ---
    if op1_section("IMPORT", collapsible=False):
        file_clicked, folder_clicked = op1_import_buttons()
        if file_clicked:
            app.import_file_dialog()
        if folder_clicked:
            app.import_folder_dialog()
        imgui.spacing()

    # --- PROJECTS + FOLDERS (library browser) ---
    _draw_library_panel(app)
    imgui.spacing()

    # --- ONTOLOGY (label-set preset for the active project) ---
    # Sits directly under PROJECTS: pick a project, then define its
    # classes, then look at its clouds. (This picker existed in 1.0
    # but was orphaned — defined and never drawn.)
    changed |= _draw_ontology_picker(app)

    # --- CLOUDS (scrollable list) ---
    changed |= _draw_clouds_section(app)

    # --- LABELS (scrollable list) ---
    changed |= _draw_labels_section(app)

    # --- POINT SIZE (sphere + knob gauge) ---
    _orange_tint = (OP1_ORANGE[0] * 0.12, OP1_ORANGE[1] * 0.12,
                    OP1_ORANGE[2] * 0.12, 1.0)
    if op1_section("POINT SIZE", OP1_ORANGE, fill_color=_orange_tint, collapsible=False):
        c, app.point_size = op1_sphere_size_gauge(app, app.point_size,
                                                   min_size=0.002, max_size=0.5)
        changed |= c
    imgui.spacing()

    return changed


def _draw_inference_section(app) -> None:
    """The body of the TRAIN tab's BATCH INFERENCE section.

    Shows a status line and the run/cancel button. Operates over the
    SELECT CLOUDS checkbox set at the top of the tab (``_train_selected``
    — the same set the export uses); falls back to the single
    ``selected_index`` when nothing is checked.
    """
    th = imgui.get_text_line_height()
    runner = getattr(app, 'contact_sheets_infer_runner', None)
    is_running = runner is not None and runner.running
    multi_set: set[int] = set(_train_selected)

    # ---- status line + progress bar ----
    if is_running and runner.current is not None:
        slot, total, name = runner.current
        status_txt = f"running {slot}/{total}: {name[:24]}"
        status_col = OP1_BLUE
        # 1-based slot count — slot=1 means "starting cloud 1 of N",
        # so fraction-done is (slot-1)/total at the moment of the
        # callback. Clamp into [0, 1] defensively.
        frac = max(0.0, min(1.0, (max(slot - 1, 0)) / max(total, 1)))
    elif runner is not None and not is_running:
        if runner.cancelled:
            status_txt = (f"cancelled — {runner.completed} done, "
                          f"{runner.failed} failed")
        else:
            status_txt = (f"done — {runner.completed} updated, "
                          f"{runner.failed} failed")
        status_col = OP1_GREEN if runner.failed == 0 else OP1_ORANGE
        # Show full bar on completion so the visual matches "done".
        total = max(getattr(runner, "completed", 0)
                    + getattr(runner, "failed", 0), 1)
        frac = 1.0
    else:
        status_txt = "idle"
        status_col = DIM
        frac = 0.0
    imgui.text_colored(status_txt, *status_col[:3])
    if runner is not None:
        bar_w = imgui.get_content_region_available_width()
        if is_running and runner.current is not None:
            slot, total, _name = runner.current
            overlay = f"{slot - 1}/{total}"
        else:
            overlay = ""
        try:
            imgui.progress_bar(frac, (bar_w, th * 0.45), overlay)
        except TypeError:
            # Older pyimgui: size as positional float height-only.
            imgui.progress_bar(frac, (-1, th * 0.45))

    # ---- launch / cancel button ----
    indices = sorted(multi_set) if multi_set else (
        [app.selected_index] if 0 <= app.selected_index < len(app.entries) else [])

    has_checkpoint = _resolve_inference_checkpoint(app) is not None
    if not has_checkpoint:
        imgui.text_colored("no trained model — run TRAINING first",
                           *DIM[:3])
        return

    _draw_inference_model_picker(app)
    imgui.spacing()

    if is_running:
        if _styled_button("CANCEL INFERENCE", DIM, height_units=2.2):
            runner.cancel()
            if app.cli:
                app.cli.log("Cancel requested.", "info")
        return

    if not indices:
        imgui.text_colored("select one or more clouds above for RUN INFERENCE",
                           *DIM[:3])
    else:
        label = (f"RUN INFERENCE on {len(indices)} clouds"
                 if len(indices) > 1 else "RUN INFERENCE")
        if _styled_button(label, OP1_BLUE, height_units=2.5):
            _start_batch_inference(app, indices)

    # ---- Batch on every unlabelled bone in the project ------------
    # Cheap unlabelled detection: no labels/<file_key>.npy on disk.
    # Mirrors check_catalog_integrity's "labeled" definition so the
    # count here matches what the catalog summary shows.
    unlabelled = _unlabelled_entry_indices(app)
    imgui.spacing()
    if not unlabelled:
        imgui.text_colored("(every cloud has labels — nothing to batch)",
                           *DIM[:3])
    else:
        batch_label = f"RUN BATCH ON {len(unlabelled)} UNLABELLED"
        if _styled_button(batch_label, OP1_BLUE, height_units=2.2,
                          fill_alpha=0.12):
            _start_batch_inference(app, unlabelled)


def _unlabelled_entry_indices(app) -> list[int]:
    """Indices of catalog entries that have never been segmented.

    Defined as "no labels file on disk" — same definition the catalog
    integrity check uses. Cheap (just a filesystem stat per entry) so
    it's safe to recompute every frame while the inference panel is
    visible.
    """
    from src.data.cloud_store import cloud_labels_path
    return [
        i for i, e in enumerate(app.entries)
        if getattr(e, "file_key", "")
        and not cloud_labels_path(e.file_key).exists()
    ]


def _draw_selected_details(app, entry):
    """Compact info strip about the currently-highlighted cloud."""
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()
    pad = 6

    rows = []
    name = entry.name[:26] if len(entry.name) > 26 else entry.name
    rows.append(("NAME", name))
    if entry.point_count > 0:
        if entry.point_count >= 1_000_000:
            pts_str = f"{entry.point_count / 1_000_000:.2f}M"
        elif entry.point_count >= 1_000:
            pts_str = f"{entry.point_count / 1_000:.1f}K"
        else:
            pts_str = str(entry.point_count)
        rows.append(("POINTS", pts_str))
    else:
        rows.append(("POINTS", "— loading —"))

    if entry.bounds_min is not None and entry.bounds_max is not None:
        span = entry.bounds_max - entry.bounds_min
        rows.append(("SIZE X", f"{span[0]:.2f}"))
        rows.append(("SIZE Y", f"{span[1]:.2f}"))
        rows.append(("SIZE Z", f"{span[2]:.2f}"))

    full_loaded = entry.full_gpu is not None
    rows.append(("FULL RES", "LOADED" if full_loaded else "PREVIEW"))

    row_h = th + 4
    total_h = row_h * len(rows) + pad * 2

    # Background panel — uses unified styled rect (mono grey)
    draw_styled_rect(dl, wx, wy, w, total_h, OP1_GRAY,
                     rounding=PANEL_ROUNDING, thickness=1.2)

    label_col = imgui.get_color_u32_rgba(*OP1_DIM)
    value_col = imgui.get_color_u32_rgba(*OP1_GRAY)
    for i, (label, value) in enumerate(rows):
        y = wy + pad + i * row_h
        dl.add_text(wx + pad, y, label_col, label)
        vw = imgui.calc_text_size(value)[0]
        dl.add_text(wx + w - pad - vw, y, value_col, value)

    imgui.dummy(w, total_h + 4)




def _draw_viewport_view_controls(app, *, for_hologram: bool = False) -> bool:
    """Shared viewport-render controls: POINT SIZE, DISPLAY tone grid,
    DISPLAY MODE, CLIP, CAMERA. Used by both LIGHT TABLE (labelling
    workflow) and HOLOGRAM (measurement workflow) so adjustments to the
    visual presentation transfer between tabs.

    ``for_hologram=True`` suppresses the point-rendering-specific
    sections (POINT SIZE, the DISPLAY tone grid that contains
    point-only knobs like FOCUS / DEPTH falloff, and DISPLAY MODE).
    HOLOGRAM is locked to the per-face flat mesh renderer; those
    controls are irrelevant there and consume real estate that the
    data panels (Cobb table, spinopelvic params, sagittal alignment
    curve, etc.) will eventually want. CLIP + CAMERA remain since
    they're useful in any view.

    All state lives on ``app`` (``point_size``, ``display_mode``,
    ``dof_strength``, ``brightness``, etc.) so this function just owns
    the widget layout — no per-tab state.
    """
    # Hoisted up here so the DISPLAY MODE block below can use these even
    # when the DISPLAY tone-grid section is collapsed. The collapsed-
    # DISPLAY path used to crash with UnboundLocalError on this exact
    # symbol the moment we extracted the block out of _draw_light_table_tab.
    from src.gui.op1_widgets import op1_tone_cell, draw_styled_rect

    changed = False
    th = imgui.get_text_line_height()

    # --- POINT SIZE ---
    if not for_hologram:
        _orange_tint = (OP1_ORANGE[0] * 0.12, OP1_ORANGE[1] * 0.12,
                        OP1_ORANGE[2] * 0.12, 1.0)
        if op1_section("POINT SIZE", OP1_ORANGE, fill_color=_orange_tint):
            c, app.point_size = op1_sphere_size_gauge(app, app.point_size)
            changed |= c

    # --- DISPLAY (4-cell tone grid: FOCUS/DEPTH/SOFT/COLOR) ---
    # Skipped for HOLOGRAM — these are point-rendering knobs that have
    # no effect on the mesh-locked HOLOGRAM view. SOFT and COLOR DO
    # influence both modes (saturation + brightness), but bundling
    # them with FOCUS / DEPTH (point-only) in one tone grid means
    # surfacing the panel surfaces all four. Cleaner to defer the
    # whole grid and add a separate "appearance" panel later if any
    # of the mesh-affecting knobs become important in HOLOGRAM.
    _blue_tint = (OP1_BLUE[0] * 0.12, OP1_BLUE[1] * 0.12,
                  OP1_BLUE[2] * 0.12, 1.0)
    if not for_hologram and op1_section("DISPLAY", OP1_BLUE, fill_color=_blue_tint):
        dl_tone = imgui.get_window_draw_list()
        tone_wx, tone_wy = imgui.get_cursor_screen_pos()
        tone_w_total = imgui.get_content_region_available_width()
        tone_th = imgui.get_text_line_height()

        tone_cols = 4
        tone_margin = tone_th * 0.4
        tone_cell_gap = tone_th * 0.15
        tone_cell_w = (tone_w_total - tone_margin * 2
                       - tone_cell_gap * (tone_cols - 1)) / tone_cols
        tone_cell_h = tone_th * 9.0
        tone_row_gap = tone_th * 0.15
        tone_panel_h = tone_th * 0.6 + tone_cell_h

        draw_styled_rect(dl_tone, tone_wx, tone_wy, tone_w_total, tone_panel_h,
                         OP1_BLUE, rounding=PANEL_ROUNDING, thickness=1.2,
                         colored_fill=True)

        tone_specs = [
            (0, 0, 'FOCUS', 'tone_focus', app.dof_strength, 0.0, 20.0, 0.0,
             'lens_tube', OP1_RED,
             lambda v: (setattr(app, 'dof_strength', v),
                        setattr(app, 'dof_enabled', v > 0.0))),
            (0, 1, 'DEPTH', 'tone_depth', app.depth_falloff, 0.0, 10.0, 0.0,
             'box', OP1_BLUE,
             lambda v: setattr(app, 'depth_falloff', v)),
            (0, 2, 'SOFT', 'tone_soft', app.saturation, 0.0, 2.0, 1.0,
             'pyramid', OP1_GREEN,
             lambda v: setattr(app, 'saturation', v)),
            (0, 3, 'COLOR', 'tone_color', app.brightness, -1.0, 1.0, 0.0,
             'pyramid', OP1_WHITE,
             lambda v: setattr(app, 'brightness', v)),
        ]
        for (row, col, lbl, uid, val, vmin, vmax, dflt,
             shape, cell_color, on_change) in tone_specs:
            cell_x = (tone_wx + tone_margin
                      + col * (tone_cell_w + tone_cell_gap))
            cell_y = (tone_wy + tone_th * 0.3
                      + row * (tone_cell_h + tone_row_gap))
            c, new_val = op1_tone_cell(
                app, cell_x, cell_y, tone_cell_w, tone_cell_h,
                uid=uid, label=lbl, value=val,
                min_val=vmin, max_val=vmax, color=cell_color,
                shape=shape, default_val=dflt,
            )
            if c:
                on_change(new_val)
                changed = True

        imgui.dummy(tone_w_total, tone_panel_h + s(4))

    # --- DISPLAY MODE (Points / Mesh / Both) ---
    # Suppressed in HOLOGRAM — locked to mesh rendering.
    if not for_hologram:
        op1_section("DISPLAY MODE", OP1_GREEN)
        dm_th = imgui.get_text_line_height()
        dm_dl = imgui.get_window_draw_list()
        dm_x, dm_y = imgui.get_cursor_screen_pos()
        dm_w_total = imgui.get_content_region_available_width()
        dm_h = dm_th * 1.7
        dm_choices = (("POINTS", "points"), ("MESH", "mesh"), ("BOTH", "both"))
        dm_cur = getattr(app, "display_mode", "points")
        dm_btn_w = (dm_w_total - s(4) * (len(dm_choices) - 1)) / len(dm_choices)
        dm_clicked = imgui.is_mouse_clicked(0)
        mx, my = imgui.get_mouse_pos()
        for i, (label, value) in enumerate(dm_choices):
            bx = dm_x + i * (dm_btn_w + s(4))
            by = dm_y
            is_active = (dm_cur == value)
            col = OP1_GREEN if is_active else OP1_DIM
            if is_active:
                draw_styled_rect_active(dm_dl, bx, by, dm_btn_w, dm_h, col,
                                        rounding=BUTTON_ROUNDING, thickness=1.8)
            else:
                draw_styled_rect(dm_dl, bx, by, dm_btn_w, dm_h, col,
                                 rounding=BUTTON_ROUNDING, thickness=1.2)
            lbl_size = imgui.calc_text_size(label)
            dm_dl.add_text(bx + (dm_btn_w - lbl_size[0]) * 0.5,
                           by + (dm_h - dm_th) * 0.5,
                           imgui.get_color_u32_rgba(*col), label)
            if (dm_clicked and bx <= mx <= bx + dm_btn_w
                    and by <= my <= by + dm_h):
                app.display_mode = value
                changed = True
        imgui.dummy(dm_w_total, dm_h + s(4))
        imgui.spacing()

    # --- CLIP ---
    if op1_section("CLIP", OP1_ORANGE):
        from src.gui.op1_wireframe import op1_clip_fader_panel
        changed |= op1_clip_fader_panel(app)

    # --- CAMERA ---
    changed |= _draw_view_panel(app, show_bg=False, show_camera=True)

    return changed


def _draw_light_table_tab(app) -> bool:
    """Light Table tab — single scrollable column.

    CLOUDS   — collapsible, internal scroll for cloud list
    LABELS   — collapsible, internal scroll for label rows
    BRUSH FALLOFF (conditional)
    DISPLAY  — 3x2 wireframe tone grid
    CAMERA   — wireframe cube + toggles
    CLIP     — axis faders
    """
    changed = False
    th = imgui.get_text_line_height()

    # Labels are always on in Light Table — no toggle button.
    app.label_blend = 1.0

    # --- CLOUDS ---
    changed |= _draw_clouds_section(app, "##lt_clouds")

    # --- VOLUME metadata (shown when selected cloud is volume-derived) ---
    if 0 <= app.selected_index < len(app.entries):
        _ve = app.entries[app.selected_index]
        _vgpu = _ve.full_gpu or _ve.preview_gpu
        if _vgpu and _vgpu.cloud_data and _vgpu.cloud_data.metadata.get('source_kind') == 'volume':
            _vmeta = _vgpu.cloud_data.metadata
            _orange_tint = (OP1_ORANGE[0] * 0.10, OP1_ORANGE[1] * 0.10,
                            OP1_ORANGE[2] * 0.10, 1.0)
            if op1_section("VOLUME", OP1_ORANGE, fill_color=_orange_tint):
                _vunit = _vmeta.get('source_unit', 'vx').upper()
                _vsp = _vmeta.get('voxel_spacing')
                dl_v = imgui.get_window_draw_list()
                vx, vy = imgui.get_cursor_screen_pos()
                vw = imgui.get_content_region_available_width()
                dim_col = imgui.get_color_u32_rgba(*OP1_GRAY)
                val_col = imgui.get_color_u32_rgba(*OP1_ORANGE)
                dl_v.add_text(vx + th * 0.3, vy, dim_col, "UNIT")
                uv = imgui.calc_text_size(_vunit)[0]
                dl_v.add_text(vx + vw - uv - th * 0.3, vy, val_col, _vunit)
                if _vsp:
                    sp_str = f"{_vsp[0]:.3f} x {_vsp[1]:.3f} x {_vsp[2]:.3f}"
                    dl_v.add_text(vx + th * 0.3, vy + th * 1.1, dim_col, "SPACING")
                    sv = imgui.calc_text_size(sp_str)[0]
                    dl_v.add_text(vx + vw - sv - th * 0.3, vy + th * 1.1, val_col, sp_str)
                imgui.dummy(vw, th * 2.4)

    # --- LABELS ---
    changed |= _draw_labels_section(app, "##lt_labels")

    # Brush distance falloff — only shown while brush is active. A
    # single 0..1 slider that shrinks the effective paint radius
    # relative to the visible brush rim. 0 = paint everywhere inside
    # the rim; 1 = paint nothing (effective radius → 0). The outer
    # rim drawn in the viewport always shows the full brush_radius
    # so you can see where the brush *could* reach, while the inner
    # ring (only drawn when falloff > 0) shows what actually paints.
    if app.active_tool == 'brush':
        if op1_section("BRUSH DISTANCE FALLOFF"):
            imgui.push_item_width(-1)
            changed_b, new_val = imgui.slider_float(
                "##brush_distance_falloff",
                float(app.brush_distance_falloff),
                0.0, 1.0, "%.2f",
            )
            imgui.pop_item_width()
            if changed_b:
                app.brush_distance_falloff = float(
                    max(0.0, min(1.0, new_val))
                )
                changed = True

    # POINT SIZE / DISPLAY / DISPLAY MODE / CLIP / CAMERA — shared with
    # HOLOGRAM so visual tweaks transfer between workflows.
    changed |= _draw_viewport_view_controls(app)

    # --- INFER (cloud-constellation loader button, 1.1) ---
    # Runs the active project's model on the current cloud through the
    # same runner/namespace/undo path as batch inference.
    from src.gui.infer_widget import draw_light_table_infer
    _ckpt = _resolve_inference_checkpoint(app)
    _project_id = _resolve_active_project_id(app)
    _chosen = _resolve_inference_model(app, _project_id) if _ckpt else None
    _mdl_label = (_chosen.name if _chosen is not None
                  else (os.path.basename(_ckpt) if _ckpt else None))
    _runner_busy = (getattr(app, 'contact_sheets_infer_runner', None)
                    is not None
                    and app.contact_sheets_infer_runner.running)
    draw_light_table_infer(
        app,
        start_infer=_start_batch_inference,
        model_label=_mdl_label,
        can_run=bool(_ckpt) and not _runner_busy,
        registry=app.label_registry,
    )

    return changed


def _draw_train_nn_header(app):
    """Animated neural-net diagram header.

    Three layers of nodes connected by edges, with dots flowing along
    the edges. During training the flowing particle + ring pulse
    animate every frame; when idle we draw a cheap static version (no
    `sin` calls, no phase maths, no per-frame flowing dot) so the
    Train tab doesn't burn CPU drawing decoration during long idle
    periods.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    h = th * 4.2
    # Flat panel — no outline, just a 95%-black fill against the 90%-black bg
    dl.add_rect_filled(wx, wy, wx + w, wy + h,
                       col32((0.05, 0.05, 0.05, 1.0)), s(PANEL_ROUNDING))

    # Node layout: 4 columns × variable rows (input → hidden → hidden → output)
    layers = [3, 4, 4, 2]
    runner = getattr(app, 'training_runner', None)
    training = runner is not None and runner.is_running

    # Detect training-completed edge so we can fire a one-shot
    # "finale" spin after the run wraps up. Track last-frame state on
    # the app so the transition fires exactly once.
    was_running = bool(getattr(app, '_train_was_running', False))
    if was_running and not training:
        # Falling edge: training just finished. Stamp the completion-
        # animation deadline ~2.5 s into the future. Drawn distinct
        # from the training spin so the user actually notices the run
        # ended (helpful for long-running training tabs left open).
        app._train_completion_anim_until = time.perf_counter() + 2.5
    app._train_was_running = training
    completion_until = float(getattr(app, '_train_completion_anim_until', 0.0))
    celebrating = (not training) and (time.perf_counter() < completion_until)

    pad_x = th * 1.0
    pad_y = th * 0.7
    cell_w = (w - pad_x * 2) / (len(layers) - 1)
    col_x = [wx + pad_x + i * cell_w for i in range(len(layers))]
    node_r = th * 0.28

    # Precompute node positions per layer
    positions = []
    for li, count in enumerate(layers):
        ys = []
        if count == 1:
            ys = [wy + h * 0.5]
        else:
            top = wy + pad_y
            bot = wy + h - pad_y
            for k in range(count):
                ys.append(top + (bot - top) * k / (count - 1))
        positions.append([(col_x[li], y) for y in ys])

    # Edges — draw first so nodes overlap them
    edge_col = col32((0.25, 0.25, 0.25, 1.0))
    for li in range(len(layers) - 1):
        for (x0, y0) in positions[li]:
            for (x1, y1) in positions[li + 1]:
                dl.add_line(x0, y0, x1, y1, edge_col, s(0.8))

    if training or celebrating:
        t = time.perf_counter()
        # While training: steady white spin at normal speed.
        # On the post-completion celebratory window: ~2x faster, still
        # white, and gain a soft intensity ramp-down as the timer
        # bleeds out — so the finale visibly fades rather than cutting.
        if celebrating:
            pulse_speed = 4.4
            decay = max(0.0, (completion_until - t) / 2.5)  # 1 → 0
            intensity = 0.6 + 0.4 * decay  # 1.0 → 0.6
        else:
            pulse_speed = 2.2
            intensity = 1.0

        # Flowing particle along the connections (one per layer gap)
        for li in range(len(layers) - 1):
            phase = (t * pulse_speed + li * 0.25) % 1.0
            beam_a = int((t * 0.7 + li) * 1.3) % len(positions[li])
            beam_b = int((t * 0.9 + li * 3) * 1.1) % len(positions[li + 1])
            ax, ay = positions[li][beam_a]
            bx, by = positions[li + 1][beam_b]
            px = ax + (bx - ax) * phase
            py = ay + (by - ay) * phase
            dl.add_circle_filled(px, py, s(2.0),
                                 col32((1.0, 1.0, 1.0, 0.95 * intensity)), 10)
            dl.add_line(ax, ay, bx, by,
                        col32((1.0, 1.0, 1.0, 0.35 * intensity)), s(1.2))

        # Nodes — pulsing
        for li, nodes in enumerate(positions):
            for (x, y) in nodes:
                fill_col = col32((0.09, 0.09, 0.09, 1.0))
                dl.add_circle_filled(x, y, node_r, fill_col, 16)
                pulse = 0.5 + 0.5 * math.sin(t * pulse_speed + li * 0.8 + x * 0.01)
                ring_alpha = (0.35 + 0.55 * pulse) * intensity
                dl.add_circle(x, y, node_r,
                              col32((1.0, 1.0, 1.0, ring_alpha)), 16, s(1.2))
    else:
        # Idle — cheap static version. No sin/cos, no flowing particle.
        ring_col = OP1_GRAY
        ring_rgba = col32((ring_col[0], ring_col[1], ring_col[2], 0.45))
        fill_col = col32((0.09, 0.09, 0.09, 1.0))
        for nodes in positions:
            for (x, y) in nodes:
                dl.add_circle_filled(x, y, node_r, fill_col, 16)
                dl.add_circle(x, y, node_r, ring_rgba, 16, s(1.2))

    # Label strip on the right edge. White while training to match
    # the spinning graphic, a brief "DONE" while the celebratory spin
    # plays, then back to neutral idle.
    if training:
        status, sc = "TRAINING", (1.0, 1.0, 1.0, 1.0)
    elif celebrating:
        status, sc = "DONE", (1.0, 1.0, 1.0, 1.0)
    else:
        status, sc = "IDLE", OP1_GRAY
    lbl_w = imgui.calc_text_size(status)[0]
    dl.add_text(wx + w - lbl_w - th * 0.4, wy + h - th * 1.1,
                col32(sc), status)

    imgui.dummy(w, h + s(4))


def _draw_train_label_histogram(app):
    """Stacked horizontal bar of label color distribution across selected clouds."""
    reg = app.label_registry
    non_unlabeled = [info for info in reg.all_labels() if info.id != 0]
    if not non_unlabeled:
        return

    # Aggregate counts across the currently-selected clouds. Go through
    # the shared LabelCountCache so each (cloud, label_id) pair is only
    # summed once after a label mutation — this panel was previously
    # re-scanning M·N·L bool masks every frame.
    counts = {info.id: 0 for info in non_unlabeled}
    total = 0
    cache = getattr(app, 'label_count_cache', None)
    for i in _train_selected:
        if i >= len(app.entries):
            continue
        entry = app.entries[i]
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None:
            continue
        cloud = gpu.cloud_data
        if cache is not None:
            for lid in counts:
                n = cache.get(cloud, lid)
                counts[lid] += n
                total += n
        else:
            labels = cloud.labels
            for lid in counts:
                n = int((labels == lid).sum())
                counts[lid] += n
                total += n

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    th = imgui.get_text_line_height()

    bar_h = th * 1.0
    dl.add_rect_filled(wx, wy, wx + w, wy + bar_h,
                       col32((0.09, 0.09, 0.09, 1.0)), s(3))
    if total > 0:
        x = wx
        for info in non_unlabeled:
            frac = counts[info.id] / total
            seg_w = w * frac
            if seg_w > 0.5:
                r, g, b, _a = info.color
                dl.add_rect_filled(x, wy, x + seg_w, wy + bar_h,
                                   col32((r, g, b, 0.85)), s(3))
            x += seg_w
    else:
        msg = "no labeled points in selection"
        mw = imgui.calc_text_size(msg)[0]
        dl.add_text(wx + (w - mw) * 0.5, wy + (bar_h - th) * 0.5,
                    col32(DIM), msg)

    imgui.dummy(w, bar_h + s(2))


def _draw_log_pulse_banner(app):
    """Decorative pulse-grid activity banner for the CLI LOG section.

    Kitbashed from Figma 154:2141 (Pulse). Encodes the recent CLI output
    levels as a square-wave: info/success = low, command/error = high.
    Falls back to an idle waveform when there's no history.
    """
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    h = s(36)

    # Flat panel — 95%-black fill, no outline
    dl.add_rect_filled(wx, wy, wx + w, wy + h,
                       col32((0.05, 0.05, 0.05, 1.0)),
                       s(PANEL_ROUNDING))

    # Build the square-wave signal from the last ~32 CLI messages.
    signal: list[int] = []
    if app.cli is not None and app.cli.output:
        for text, level in list(app.cli.output)[-32:]:
            signal.append(1 if level in ("cmd", "error") else 0)

    if not signal:
        # Idle pattern — a gentle square wave so the banner isn't dead.
        signal = [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0]

    pad = s(6)
    op1_pulse_grid(
        dl,
        wx + pad, wy + pad,
        w - pad * 2, h - pad * 2,
        signal=signal,
        color=OP1_BLUE,
        grid_color=(0.0, 0.28, 0.36, 0.8),
        divisions=12,
    )
    imgui.dummy(w, h + s(2))


def _draw_train_tab(app):
    """Train tab — export labeled point clouds for PTv3 training and launch runs.

    Layout:
      [SELECT] mini-gallery of loaded clouds with checkboxes + select-all/none
      [EXPORT] format picker, split sliders, output dir, Export button
      [TRAIN]  launch/stop controls, live status line
      [LOG]    shared CLI output pane (commands still work below)
    """
    th = imgui.get_text_line_height()


    # --- SELECT: mini-gallery ---
    op1_section("SELECT CLOUDS", TEAL)

    if not app.entries:
        imgui.text_colored("No clouds loaded.", *DIM[:3])
    else:
        # Select-all / none row
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.05, 0.05, 0.05, 1.0)
        if imgui.button("All", width=s(50)):
            _train_selected.clear()
            _train_selected.update(range(len(app.entries)))
        imgui.same_line()
        if imgui.button("None", width=s(50)):
            _train_selected.clear()
        imgui.same_line()
        if imgui.button("Labeled", width=s(70)):
            _train_selected.clear()
            _lbl_c = getattr(app, 'label_count_cache', None)
            for i, e in enumerate(app.entries):
                gpu = e.full_gpu or e.preview_gpu
                if gpu is None:
                    continue
                cloud = gpu.cloud_data
                if _lbl_c is not None:
                    if _lbl_c.any_labeled(cloud):
                        _train_selected.add(i)
                elif (cloud.labels != 0).any():
                    _train_selected.add(i)
        imgui.pop_style_color(1)
        imgui.same_line()
        imgui.text_colored(f"{len(_train_selected)}/{len(app.entries)}", *DIM[:3])

        # Scrollable mini-gallery — checkbox + name + point count + label status
        gallery_h = min(th * 12.0, th * max(3.0, len(app.entries) * 1.6))
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.05, 1.0)
        imgui.begin_child("##train_gallery", 0, gallery_h, border=False)

        _lbl_cache = getattr(app, 'label_count_cache', None)
        for i, entry in enumerate(app.entries):
            gpu = entry.full_gpu or entry.preview_gpu
            has_labels = False
            label_count = 0
            if gpu is not None:
                cloud = gpu.cloud_data
                if _lbl_cache is not None:
                    label_count = _lbl_cache.labeled_count(cloud)
                else:
                    label_count = int((cloud.labels != 0).sum())
                has_labels = label_count > 0

            was_checked = i in _train_selected
            changed, checked = imgui.checkbox(f"##chk_{i}", was_checked)
            if changed:
                if checked:
                    _train_selected.add(i)
                else:
                    _train_selected.discard(i)

            imgui.same_line()
            # Status marker — "L" if labeled, "." if loaded, "~" if loading
            dot = GREEN if has_labels else (AMBER if entry.preview_gpu else DIM)
            marker = "L" if has_labels else ("." if entry.preview_gpu else "~")
            imgui.text_colored(marker, *dot[:3])
            imgui.same_line()

            pts = f"{entry.point_count // 1000}K" if entry.point_count >= 1000 else f"{entry.point_count}"
            name = entry.name[:20]
            label_tag = f" [{label_count:,} lbl]" if has_labels else ""
            imgui.text(f"[{i:02d}] {name}")
            imgui.same_line()
            imgui.text_colored(f"{pts}{label_tag}", *DIM[:3])

        imgui.end_child()
        imgui.pop_style_color(1)

    imgui.spacing()

    # --- LABEL FILTER ---
    global _train_label_sel
    op1_section("LABELS TO EXPORT", ORANGE)

    reg = app.label_registry
    # Show every label in the registry, Unlabeled (id 0) first. Including
    # Unlabeled is required to train a model that can predict "this is
    # just bone" — without it, every foreground class gets oversaturated
    # at inference time (the no-background-class trap from earlier).
    # The list used to hide id 0 outright, which silently produced
    # exports with Unlabeled = IGNORE_INDEX. Showing it makes the choice
    # explicit and lets the user opt out only when they really mean it.
    all_labels_in_order = sorted(reg.all_labels(), key=lambda info: info.id)
    if not all_labels_in_order:
        imgui.text_colored("No labels defined. Create labels in Light Table.", *DIM[:3])
    else:
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.05, 0.05, 0.05, 1.0)
        if imgui.button("All##lbl", width=s(50)):
            _train_label_sel = None
        imgui.same_line()
        if imgui.button("None##lbl", width=s(50)):
            _train_label_sel = set()
        imgui.pop_style_color(1)
        imgui.same_line()
        n_sel = (len(all_labels_in_order) if _train_label_sel is None
                 else len(_train_label_sel))
        imgui.text_colored(f"{n_sel}/{len(all_labels_in_order)}", *DIM[:3])

        lbl_h = min(th * 8.0,
                    th * max(2.0, len(all_labels_in_order) * 1.4))
        imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.05, 1.0)
        imgui.begin_child("##train_labels", 0, lbl_h, border=False)
        for info in all_labels_in_order:
            all_on = _train_label_sel is None
            checked_now = all_on or info.id in _train_label_sel
            changed, new_val = imgui.checkbox(f"##lbl_{info.id}", checked_now)
            if changed:
                if _train_label_sel is None:
                    # Materialize from "all" so we can deselect individual items
                    _train_label_sel = {lbl.id for lbl in all_labels_in_order}
                if new_val:
                    _train_label_sel.add(info.id)
                else:
                    _train_label_sel.discard(info.id)
            imgui.same_line()
            r, g, b, _ = info.color
            imgui.text_colored("[#]", r, g, b)
            imgui.same_line()
            # Mark Unlabeled (id 0) so it reads as the background-bone
            # class and not as just-another-foreground-label.
            tag = " (bg)" if info.id == 0 else ""
            imgui.text(f"[{info.id:3d}] {info.name}{tag}")
        imgui.end_child()
        imgui.pop_style_color(1)

        # Label distribution histogram across the selected clouds
        _draw_train_label_histogram(app)

    imgui.spacing()

    # --- STAGED DATASET ---
    staging = getattr(app, 'dataset_staging', None)
    if staging is None:
        staging = []
        app.dataset_staging = staging

    # Advanced path for mixed multi-filter exports; the common
    # select→export flow doesn't need it, so it starts collapsed.
    # Stable title (no count) so the collapse-state key doesn't reset
    # every add/remove; 1.0 also ignored the open/closed return, so
    # collapsing the header hid nothing.
    if op1_section("STAGED DATASET", GREEN, default_collapsed=True):
        # Memoise the sum across staged clouds; it only changes on
        # add/remove and those paths bump `_dataset_staging_version`.
        _staging_version = getattr(app, '_dataset_staging_version', 0)
        _cached_total = getattr(app, '_dataset_staging_total_pts', None)
        _cached_version = getattr(app, '_dataset_staging_total_version', -1)
        if _cached_total is None or _cached_version != _staging_version:
            total_pts = sum(int(item['cloud'].point_count) for item in staging)
            app._dataset_staging_total_pts = total_pts
            app._dataset_staging_total_version = _staging_version
        else:
            total_pts = _cached_total
        imgui.text_colored(
            f"{len(staging)} clouds   {total_pts:,} pts total", *DIM[:3])

        can_add = bool(_train_selected) and bool(app.entries)
        btn_color = GREEN if can_add else DIM
        if _styled_button(
                f"ADD {len(_train_selected)} SELECTED → STAGED",
                btn_color):
            if can_add:
                _add_to_staged(app)

        if staging:
            stg_h = min(th * 8.0, th * max(2.0, len(staging) * 1.6))
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.05, 1.0)
            imgui.begin_child("##staged_list", 0, stg_h, border=False)
            remove_idx = -1
            for si, item in enumerate(staging):
                name = item['name'][:22]
                pts = item['cloud'].point_count
                pts_str = (f"{pts // 1000}K" if pts >= 1000 else f"{pts}")
                f = item['label_filter']
                filt_str = "all" if f is None else f"{len(f)} lbl"
                if imgui.button(f"x##rm_{si}", width=s(24)):
                    remove_idx = si
                imgui.same_line()
                imgui.text(f"[{si:02d}] {name}")
                imgui.same_line()
                imgui.text_colored(f"{pts_str}  {filt_str}", *DIM[:3])
            if remove_idx >= 0:
                staging.pop(remove_idx)
                app._dataset_staging_version = getattr(app, '_dataset_staging_version', 0) + 1
            imgui.end_child()
            imgui.pop_style_color(1)

            imgui.push_style_color(imgui.COLOR_BUTTON, 0.05, 0.05, 0.05, 1.0)
            if imgui.button("Clear staged", width=s(120)):
                staging.clear()
                app._dataset_staging_version = getattr(app, '_dataset_staging_version', 0) + 1
            imgui.pop_style_color(1)

    imgui.spacing()

    # --- EXPORT controls ---
    op1_section("EXPORT DATASET")

    # Format radio
    fmt = getattr(app, '_train_export_fmt', 'ptv3')
    imgui.text_colored("Format", *DIM[:3])
    imgui.same_line(s(80))
    for label in ('ptv3', 'npz', 'h5'):
        if imgui.radio_button(label, fmt == label):
            fmt = label
        imgui.same_line()
    app._train_export_fmt = fmt
    imgui.new_line()

    # Splits
    splits = getattr(app, '_train_splits', [0.7, 0.15, 0.15])
    imgui.push_item_width(s(90))
    changed, splits[0] = imgui.slider_float("train", splits[0], 0.0, 1.0, "%.2f")
    imgui.same_line()
    changed, splits[1] = imgui.slider_float("val", splits[1], 0.0, 1.0, "%.2f")
    imgui.same_line()
    splits[2] = max(0.0, 1.0 - splits[0] - splits[1])
    imgui.text_colored(f"test {splits[2]:.2f}", *DIM[:3])
    imgui.pop_item_width()
    app._train_splits = splits

    # Output dir
    out_dir = getattr(app, '_train_output_dir', os.path.join(os.getcwd(), 'dataset'))
    imgui.push_item_width(-s(80))
    _, out_dir = imgui.input_text("##out_dir", out_dir, 512)
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button("...", width=s(30)):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            picked = filedialog.askdirectory(title="Dataset output directory")
            root.destroy()
            if picked:
                out_dir = picked
        except Exception:
            pass
    app._train_output_dir = out_dir

    # Export source: selected gallery clouds or the staged dataset
    src = getattr(app, '_train_export_src', 'selected')
    imgui.text_colored("Source", *DIM[:3])
    imgui.same_line(s(80))
    if imgui.radio_button("selected", src == 'selected'):
        src = 'selected'
    imgui.same_line()
    if imgui.radio_button("staged", src == 'staged'):
        src = 'staged'
    imgui.new_line()
    app._train_export_src = src

    if src == 'selected':
        can_export = bool(_train_selected) and bool(out_dir)
        n_export = len(_train_selected)
    else:
        can_export = bool(staging) and bool(out_dir)
        n_export = len(staging)

    if _styled_button(
            f"EXPORT {n_export} CLOUD{'S' if n_export != 1 else ''} → {fmt.upper()}",
            TEAL if can_export else DIM):
        if can_export:
            if src == 'selected':
                _run_dataset_export(app, out_dir, fmt, splits)
            else:
                _run_staged_export(app, out_dir, fmt, splits)

    imgui.spacing()

    # --- TRAIN controls ---
    op1_section("TRAIN", CORAL)

    runner = getattr(app, 'training_runner', None)
    cfg_epochs = getattr(app, '_train_epochs', 200)
    cfg_batch = getattr(app, '_train_batch', 2)
    cfg_device = getattr(app, '_train_device', 'cuda')

    # --- Resolve active project for model registry ---
    active_project_id = None
    active_project_name = ""
    av = getattr(app, 'active_view', None)
    if av and av[0] == "project" and not str(av[1]).startswith("smart:"):
        active_project_id = av[1]
        cat = getattr(app, 'catalog', None)
        if cat and active_project_id in getattr(cat, 'projects', {}):
            active_project_name = cat.projects[active_project_id].name

    # Lazy-init the model registry
    if app._train_model_registry is None:
        from src.data.model_registry import ProjectModelRegistry
        from src.data.library_catalog import library_dir
        app._train_model_registry = ProjectModelRegistry(library_dir())
    registry = app._train_model_registry

    # Load models for active project
    project_models = registry.list_models(active_project_id) if active_project_id else []

    # --- MODEL selector dropdown ---
    sel_idx = getattr(app, '_train_selected_model_idx', 0)
    # Clamp if models were deleted
    if sel_idx > len(project_models):
        sel_idx = 0

    if active_project_id:
        # Build combo items: [+ New Model, Model A, Model B, ...]
        combo_items = ["+ New Model"]
        for m in project_models:
            miou_str = f"  ({m.best_miou:.3f})" if m.best_miou > 0 else ""
            combo_items.append(f"{m.name}{miou_str}")

        preview_label = combo_items[sel_idx] if sel_idx < len(combo_items) else combo_items[0]
        imgui.push_item_width(-1)
        if imgui.begin_combo("##model_select", preview_label):
            for i, label in enumerate(combo_items):
                is_selected = (i == sel_idx)
                clicked, _ = imgui.selectable(label, is_selected)
                if clicked:
                    sel_idx = i
                    app._train_model_finalized = False
                if is_selected:
                    imgui.set_item_default_focus()
            imgui.end_combo()
        imgui.pop_item_width()
        app._train_selected_model_idx = sel_idx

        # New model name input
        if sel_idx == 0:
            new_name = getattr(app, '_train_new_model_name', '')
            imgui.push_item_width(-1)
            _, new_name = imgui.input_text("##new_model_name", new_name, 128,
                                            imgui.INPUT_TEXT_AUTO_SELECT_ALL)
            imgui.pop_item_width()
            imgui.text_colored("model name", *DIM[:3])
            app._train_new_model_name = new_name
        else:
            # Show selected model stats
            sel_model = project_models[sel_idx - 1]
            status_colors = {
                "completed": OP1_GREEN, "training": ORANGE,
                "failed": RED, "stopped": DIM, "pending": DIM,
            }
            sc = status_colors.get(sel_model.status, DIM)
            miou_str = f"mIoU {sel_model.best_miou:.3f}  |  " if sel_model.best_miou > 0 else ""
            parent_str = ""
            if sel_model.parent_model_id:
                parent = registry.get_model(active_project_id, sel_model.parent_model_id)
                if parent:
                    parent_str = f"  |  from {parent.name}"
            imgui.text_colored(
                f"{miou_str}{sel_model.epochs} ep  |  {sel_model.architecture}  |  "
                f"{sel_model.status}{parent_str}",
                *sc[:3])
            if sel_model.has_checkpoint:
                imgui.text_colored("FINE-TUNE from this checkpoint", *CORAL[:3])
            elif sel_model.status == "completed":
                imgui.text_colored("no checkpoint found on disk", *DIM[:3])

        imgui.spacing()

        # --- Model list (compact scrollable) ---
        if project_models:
            list_h = min(th * (len(project_models) + 0.5), th * 5)
            imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.04, 0.04, 0.04, 1.0)
            imgui.begin_child("##model_list", 0, list_h, border=False)
            for i, m in enumerate(project_models):
                status_colors = {
                    "completed": OP1_GREEN, "training": ORANGE,
                    "failed": RED, "stopped": DIM, "pending": DIM,
                }
                dot_col = status_colors.get(m.status, DIM)

                # Status dot
                dl = imgui.get_window_draw_list()
                cx, cy = imgui.get_cursor_screen_pos()
                dot_r = th * 0.2
                dl.add_circle_filled(cx + dot_r + 2, cy + th * 0.5, dot_r,
                                     imgui.get_color_u32_rgba(*dot_col))

                # Selectable row
                imgui.indent(th * 0.7)
                is_sel = (sel_idx == i + 1)
                miou_str = f"{m.best_miou:.3f}" if m.best_miou > 0 else "---"
                row_label = f"{m.name:<20s} {miou_str}  {m.epochs}ep"
                clicked, _ = imgui.selectable(f"{row_label}##model_{i}", is_sel)
                if clicked:
                    sel_idx = i + 1
                    app._train_selected_model_idx = sel_idx
                imgui.unindent(th * 0.7)
            imgui.end_child()
            imgui.pop_style_color(1)

        imgui.spacing()
    else:
        imgui.text_colored("open a project to track models", *DIM[:3])
        imgui.spacing()

    # --- Training hyperparameters ---
    imgui.push_item_width(s(90))
    _, cfg_epochs = imgui.input_int("epochs", cfg_epochs, 10, 50)
    imgui.same_line()
    _, cfg_batch = imgui.input_int("batch", cfg_batch, 1, 4)
    imgui.pop_item_width()
    imgui.text_colored("device", *DIM[:3])
    imgui.same_line()
    for d in ('cuda', 'cpu'):
        if imgui.radio_button(d, cfg_device == d):
            cfg_device = d
        imgui.same_line()
    imgui.new_line()
    app._train_epochs = max(1, cfg_epochs)
    app._train_batch = max(1, cfg_batch)
    app._train_device = cfg_device

    # --- ADVANCED hyperparameters (collapsible) ---
    # These map 1:1 to PTv3TrainParams fields. Defaults match the
    # dataclass; the small-dataset suggestions in the docstring on each
    # input below come from runs where Pointcept's stock defaults
    # (tuned for ScanNet-scale data) over-regularized the model on
    # tens-of-samples spinal data.
    from src.gui.op1_widgets import _section_collapsed
    if "ADVANCED HYPERPARAMS" not in _section_collapsed:
        _section_collapsed["ADVANCED HYPERPARAMS"] = True
    if op1_section("ADVANCED HYPERPARAMS", DIM, collapsible=True):
        cfg_lr = float(getattr(app, '_train_base_lr', 0.006))
        cfg_wd = float(getattr(app, '_train_weight_decay', 0.05))
        cfg_warm = int(getattr(app, '_train_warmup_epochs', 10))
        cfg_dp = float(getattr(app, '_train_drop_path', 0.3))
        cfg_mb = float(getattr(app, '_train_minority_boost', 1.0))
        cfg_so3 = bool(getattr(app, '_train_random_rotate_so3', True))

        imgui.push_item_width(s(90))
        _, cfg_lr = imgui.input_float("base_lr", cfg_lr, 0.001, 0.005, "%.4f")
        imgui.same_line()
        _, cfg_wd = imgui.input_float("weight_decay", cfg_wd, 0.005, 0.02, "%.4f")
        _, cfg_warm = imgui.input_int("warmup_epochs", cfg_warm, 1, 5)
        imgui.same_line()
        _, cfg_dp = imgui.input_float("drop_path", cfg_dp, 0.05, 0.1, "%.2f")
        _, cfg_mb = imgui.input_float("minority_boost", cfg_mb, 0.1, 0.5, "%.1f")
        imgui.pop_item_width()
        imgui.text_colored(
            "minority_boost multiplies the LAST 2 class weights "
            "(pedicles, in the 5-class spinal ontology).",
            *DIM[:3])
        imgui.spacing()

        # Rotation augmentation toggle. SO(3) rotation makes the model
        # rotation-invariant but kills absolute up/down cues, which
        # turns Sup_End vs Inf_End into a much harder distinction. For
        # supine CT data (always consistent orientation), turning it
        # off lets the model use Z=up as a free anchor and resolves
        # the Sup/Inf collapse seen on cervical inference. Turn back
        # on once the dataset is large enough that the model can
        # distinguish endplates from shape alone.
        _, cfg_so3 = imgui.checkbox("random_rotate_so3", cfg_so3)
        imgui.text_colored(
            "off = trust absolute orientation (Z=up). Best for CT data "
            "in a consistent frame, esp. when Sup/Inf is hard to learn.",
            *DIM[:3])

        app._train_base_lr = max(1e-6, cfg_lr)
        app._train_weight_decay = max(0.0, cfg_wd)
        app._train_warmup_epochs = max(0, cfg_warm)
        app._train_drop_path = max(0.0, min(0.9, cfg_dp))
        app._train_minority_boost = max(0.1, cfg_mb)
        app._train_random_rotate_so3 = bool(cfg_so3)

    # --- Pointcept ENV (collapsible, collapsed by default) ---
    from src.gui.op1_widgets import _section_collapsed
    if "POINTCEPT ENV" not in _section_collapsed:
        _section_collapsed["POINTCEPT ENV"] = True
    cfg_python = getattr(app, '_train_python_exe', '')
    cfg_pcept_dir = getattr(app, '_train_pointcept_dir', '')

    if op1_section("POINTCEPT ENV", DIM, collapsible=True):
        imgui.push_item_width(-1)
        changed_py, cfg_python = imgui.input_text("##python_exe", cfg_python, 512,
                                                   imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        imgui.pop_item_width()
        imgui.text_colored("python exe", *DIM[:3])

        imgui.push_item_width(-1)
        changed_pd, cfg_pcept_dir = imgui.input_text("##pcept_dir", cfg_pcept_dir, 512,
                                                      imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        imgui.pop_item_width()
        imgui.text_colored("pointcept dir", *DIM[:3])

        app._train_python_exe = cfg_python
        app._train_pointcept_dir = cfg_pcept_dir
        if changed_py or changed_pd:
            from src.utils.prefs import update_prefs
            update_prefs({
                'train_python_exe': cfg_python,
                'train_pointcept_dir': cfg_pcept_dir,
            })

    imgui.spacing()

    # --- Training progress net ---
    global _net_smooth_progress, _net_last_running, _net_run_seed
    running = runner is not None and runner.is_running
    finished = runner is not None and getattr(
        runner.status, 'finished', False)

    target_progress = _training_progress_fraction(runner)

    # Re-seed anchor grid dots on each fresh launch (False → True edge)
    if running and not _net_last_running:
        _net_run_seed = int(time.time() * 1000) & 0xFFFFFFFF
        _net_smooth_progress = 0.0
    _net_last_running = running

    # Ease smoothed progress toward target each frame (~0.2s half-life)
    _net_smooth_progress += (target_progress - _net_smooth_progress) * 0.05

    # Draw the net panel
    net_dl = imgui.get_window_draw_list()
    net_wx, net_wy = imgui.get_cursor_screen_pos()
    net_w = imgui.get_content_region_available_width()
    net_h = min(net_w * 0.9, s(260))
    op1_training_progress_net(
        net_dl, app, net_wx, net_wy, net_w, net_h,
        progress=_net_smooth_progress,
        running=running,
        finished=finished,
        run_seed=_net_run_seed,
        shape_kind="axes_cross",
    )
    imgui.dummy(net_w, net_h + s(4))

    # Live status line — epoch / loss / mIoU while running, IDLE otherwise
    if running:
        status = runner.status
        step_str = ""
        if hasattr(status, 'current_step') and status.total_steps > 0:
            step_str = f"  step {status.current_step}/{status.total_steps}"
        best_str = ""
        if hasattr(status, 'best_miou') and status.best_miou > 0:
            best_str = f"  best {status.best_miou:.3f}"
        imgui.text_colored(
            f"epoch {status.current_epoch}/{status.total_epochs}{step_str}  "
            f"loss {status.last_loss:.4f}  mIoU {status.last_miou:.3f}{best_str}",
            *CORAL[:3])
    elif finished:
        status = runner.status if runner else None
        best = getattr(status, 'best_miou', 0) if status else 0
        if best > 0:
            imgui.text_colored(f"COMPLETE  —  best mIoU {best:.3f}", *OP1_GREEN[:3])
        else:
            imgui.text_colored("TRAINING COMPLETE", *OP1_GREEN[:3])
        err = getattr(status, 'error', '') if status else ''
        if err:
            imgui.text_colored(err, *RED[:3])
    else:
        imgui.text_colored("IDLE", *DIM[:3])

    # --- Finalize model status when training completes ---
    if finished and not getattr(app, '_train_model_finalized', False):
        _finalize_model_status(app, active_project_id)
        app._train_model_finalized = True

    # --- Launch / Fine-tune / Stop button ---
    if running:
        if _styled_button("STOP TRAINING", RED, height_units=3.0):
            runner.stop()
    else:
        # Determine button label and parent model for fine-tuning
        parent_model = None
        model_name = ""
        if active_project_id and sel_idx > 0 and sel_idx <= len(project_models):
            parent_model = project_models[sel_idx - 1]
            # Unique fine-tune name: strip any existing " ft"/" ftN"
            # suffix so repeated fine-tunes yield "base ft2", "base
            # ft3" instead of "base ft ft ft".
            import re as _re
            _base = _re.sub(r"( ft\d*)+$", "", parent_model.name)
            _taken = {m.name for m in project_models}
            model_name = f"{_base} ft"
            _n = 2
            while model_name in _taken:
                model_name = f"{_base} ft{_n}"
                _n += 1
            btn_label = f"FINE-TUNE FROM {parent_model.name.upper()}"
            btn_color = CORAL
        else:
            if active_project_id:
                model_name = getattr(app, '_train_new_model_name', '').strip()
            btn_label = "LAUNCH TRAINING"
            btn_color = RED

        train_ok = _path_ok('dir', out_dir)
        if not train_ok:
            btn_color = DIM

        # Pre-launch summary: exactly what this run will train on, read
        # from the exported dataset's classes.json + scene dirs so it
        # reflects the data on disk (what training actually consumes),
        # not the transient UI selection.
        summary = _dataset_summary_cached(app, out_dir)
        if summary is not None:
            n_scenes, n_pts, cls_names = summary
            imgui.text_colored(
                f"{n_scenes} scenes   {n_pts:,} pts   "
                f"{len(cls_names)} classes", *OP1_GRAY[:3])
            if cls_names:
                cls_line = ", ".join(cls_names[:6])
                if len(cls_names) > 6:
                    cls_line += f", +{len(cls_names) - 6} more"
                imgui.text_colored(f"  {cls_line}", *DIM[:3])

        if _styled_button(btn_label, btn_color, height_units=3.0,
                          fill_alpha=0.15):
            if train_ok:
                _launch_training(app, out_dir,
                                 model_name=model_name,
                                 parent_model=parent_model,
                                 project_id=active_project_id)

    imgui.spacing()

    # --- RUN INFERENCE button ---
    # Visible when a model with a checkpoint is selected and not training.
    if not running:
        _infer_runner = getattr(app, '_infer_runner', None)
        _infer_running = _infer_runner is not None and _infer_runner.poll() is None
        has_checkpoint = (
            parent_model is not None and getattr(parent_model, 'has_checkpoint', False)
        )
        # Also allow inference if there's a best checkpoint from the latest run
        if not has_checkpoint and app.training_runner is not None:
            _last_best = getattr(app.training_runner.status, 'best_checkpoint', '')
            if _last_best and _path_ok('file', _last_best):
                has_checkpoint = True

        if _infer_running:
            if _styled_button("STOP INFERENCE", DIM, height_units=2.5):
                _infer_runner.kill()
                if app.cli:
                    app.cli.log("Inference stopped.", "info")
        elif has_checkpoint and _path_ok('dir', out_dir):
            if _styled_button("RUN INFERENCE", BLUE, height_units=2.5):
                _launch_inference(app, out_dir,
                                  parent_model=parent_model)
        imgui.spacing()

    # --- BATCH INFERENCE (moved here from Contact Sheets in 1.1) ---
    # Lives next to the models it uses: model picker, run-on-selected,
    # run-on-unlabelled. Single-cloud inference is the Light Table's
    # INFER button.
    _blue_tint = (OP1_BLUE[0] * 0.10, OP1_BLUE[1] * 0.10,
                  OP1_BLUE[2] * 0.10, 1.0)
    if op1_section("BATCH INFERENCE", OP1_BLUE, fill_color=_blue_tint):
        _draw_inference_section(app)
    imgui.spacing()

    # --- LOG pane (shared CLI) ---
    # The pulse-grid activity banner that used to sit above the LOG
    # header was decorative-only — square-wave signal of the recent
    # CLI log level history — and was removed at user request to
    # quiet the train tab. The helper ``_draw_log_pulse_banner`` is
    # kept in this file as dead code for now in case it's wanted back
    # somewhere; delete it once another iteration confirms it isn't.
    op1_section("LOG", DIM)
    if app.cli is None:
        imgui.text_colored("CLI not initialized.", *DIM[:3])
        return

    cli = app.cli
    output_h = max(th * 4.0, imgui.get_content_region_available()[1] - th * 2.5)
    imgui.push_style_color(imgui.COLOR_CHILD_BACKGROUND, 0.05, 0.05, 0.05, 1.0)
    imgui.begin_child("##cli_output", 0, output_h, border=False)

    level_colors = {"info": DIM, "error": RED, "cmd": ORANGE, "success": GREEN}
    for text, level in cli.output:
        color = level_colors.get(level, DIM)
        imgui.text_colored(text, *color[:3])

    if cli._pending_scroll:
        imgui.set_scroll_here_y(1.0)
        cli._pending_scroll = False
    imgui.end_child()
    imgui.pop_style_color(1)

    # Command input
    imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.08, 0.08, 0.08, 1.0)
    imgui.push_style_color(imgui.COLOR_TEXT, *ORANGE[:3])
    imgui.push_item_width(-1)
    entered, cli.input_buf = imgui.input_text(
        "##cli_input", cli.input_buf, 512,
        imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
    )
    imgui.pop_item_width()
    imgui.pop_style_color(2)
    if entered and cli.input_buf.strip():
        cli.execute(cli.input_buf)
        cli.input_buf = ""
        imgui.set_keyboard_focus_here(-1)


def _dataset_summary_cached(app, out_dir: str):
    """(n_scenes, total_points, class_names) for an exported dataset dir.

    Reads classes.json + the train/val/test scene dirs. The scan is
    cached for 5 seconds per directory so the TRAIN tab never stats the
    dataset tree on every frame; an export bumps mtime and shows up on
    the next refresh. Returns None when the dir has no dataset.
    """
    cache = getattr(app, '_dataset_summary_cache', None)
    if cache is None:
        cache = {}
        app._dataset_summary_cache = cache
    now = time.time()
    hit = cache.get(out_dir)
    if hit is not None and (now - hit[0]) < 5.0:
        return hit[1]

    summary = None
    classes_path = os.path.join(out_dir, "classes.json")
    if os.path.isfile(classes_path):
        try:
            with open(classes_path) as f:
                meta = json.load(f)
            cls_names = [n for n in meta.get("class_names", [])
                         if n.lower() != "unlabeled"]
            n_scenes = 0
            n_pts = 0
            for split in ("train", "val", "test"):
                split_dir = os.path.join(out_dir, split)
                if not os.path.isdir(split_dir):
                    continue
                for scene in os.listdir(split_dir):
                    coord = os.path.join(split_dir, scene, "coord.npy")
                    if os.path.isfile(coord):
                        n_scenes += 1
                        # 12 bytes per float32 xyz point — header ~128B.
                        n_pts += max(0, (os.path.getsize(coord) - 128) // 12)
            if n_scenes:
                summary = (n_scenes, int(n_pts), cls_names)
        except (OSError, json.JSONDecodeError, ValueError):
            summary = None
    cache[out_dir] = (now, summary)
    return summary


def _collect_selected_clouds(app) -> list:
    """Load full-res data for every checked gallery entry, return cloud list."""
    clouds = []
    for i in sorted(_train_selected):
        if i >= len(app.entries):
            continue
        entry = app.entries[i]
        app._ensure_full_resolution(i)
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None:
            if app.cli:
                app.cli.log(f"  [{i}] {entry.name}: no data, skipping", "error")
            continue
        clouds.append(gpu.cloud_data)
    return clouds


def _run_dataset_export(app, output_dir: str, fmt: str, splits: list):
    """Export the currently checked Train-tab clouds as a dataset."""
    if not _train_selected:
        if app.cli:
            app.cli.log("No clouds selected for export.", "error")
        return

    clouds = _collect_selected_clouds(app)
    if not clouds:
        if app.cli:
            app.cli.log("No cloud data available.", "error")
        return

    allowed = None if _train_label_sel is None else set(_train_label_sel)
    try:
        from src.export.dataset_export import export_dataset
        summary = export_dataset(
            clouds, app.label_registry, output_dir, format=fmt,
            train_ratio=splits[0], val_ratio=splits[1], test_ratio=splits[2],
            allowed_ids=allowed,
        )
        if app.cli:
            filt_str = "all labels" if allowed is None else f"{len(allowed)} labels"
            app.cli.log(
                f"Exported {summary['train']}t/{summary['val']}v/{summary['test']}ts "
                f"({filt_str}) → {output_dir} ({fmt})", "success")
    except Exception as e:
        if app.cli:
            app.cli.log(f"Export failed: {e}", "error")


def _add_to_staged(app):
    """Append the selected gallery clouds to the dataset staging list.

    Each staged item captures the cloud reference and the currently
    selected label filter, so the user can change labels between adds
    and have each entry remember its own filter at export time.
    """
    staging = app.dataset_staging
    clouds = _collect_selected_clouds(app)
    if not clouds:
        if app.cli:
            app.cli.log("Nothing selected to stage.", "error")
        return

    label_filter = None if _train_label_sel is None else frozenset(_train_label_sel)

    # Avoid adding the exact same cloud object twice
    existing = {id(item['cloud']) for item in staging}
    added = 0
    for cloud in clouds:
        if id(cloud) in existing:
            continue
        name = os.path.basename(cloud.file_path) if cloud.file_path else f"cloud_{id(cloud)}"
        staging.append({
            'cloud': cloud,
            'name': name,
            'label_filter': label_filter,
        })
        added += 1
    if added:
        app._dataset_staging_version = getattr(app, '_dataset_staging_version', 0) + 1
    if app.cli:
        app.cli.log(
            f"Staged {added} new cloud(s). Total: {len(staging)}", "success")


def _run_staged_export(app, output_dir: str, fmt: str, splits: list):
    """Export the staged dataset, honoring per-cloud label filters."""
    staging = getattr(app, 'dataset_staging', None) or []
    if not staging:
        if app.cli:
            app.cli.log("Staged dataset is empty.", "error")
        return

    clouds = [item['cloud'] for item in staging]
    per_cloud_filters = [
        (None if item['label_filter'] is None else set(item['label_filter']))
        for item in staging
    ]
    try:
        from src.export.dataset_export import export_dataset
        summary = export_dataset(
            clouds, app.label_registry, output_dir, format=fmt,
            train_ratio=splits[0], val_ratio=splits[1], test_ratio=splits[2],
            per_cloud_filters=per_cloud_filters,
        )
        if app.cli:
            app.cli.log(
                f"Staged exported: {summary['train']}t/{summary['val']}v/{summary['test']}ts "
                f"→ {output_dir} ({fmt})", "success")
    except Exception as e:
        if app.cli:
            app.cli.log(f"Staged export failed: {e}", "error")


def _launch_training(app, data_dir: str, model_name: str = "",
                     parent_model=None, project_id: str | None = None):
    """Launch a Pointcept PTv3 training run against the given dataset directory.

    When ``parent_model`` is set (a TrainedModel), the run fine-tunes from
    that model's best checkpoint. A new TrainedModel entry is created in
    the project's model registry if ``project_id`` is provided.
    """
    python_exe = getattr(app, '_train_python_exe', '').strip()
    pointcept_dir = getattr(app, '_train_pointcept_dir', '').strip()

    # Validate Pointcept paths
    if not python_exe or not pointcept_dir:
        if app.cli:
            app.cli.log(
                "Set python exe and pointcept dir in POINTCEPT ENV first.",
                "error")
        return
    if not os.path.isfile(python_exe) and not python_exe.startswith("wsl"):
        if app.cli:
            app.cli.log(f"Python exe not found: {python_exe}", "error")
        return
    if not os.path.isdir(pointcept_dir):
        if app.cli:
            app.cli.log(f"Pointcept dir not found: {pointcept_dir}", "error")
        return

    from src.training.ptv3_runner import PointceptRunner, PointceptLaunchConfig
    from src.training.config_gen import PTv3TrainParams, generate_ptv3_config

    if app.training_runner is None:
        app.training_runner = PointceptRunner()
    if app.training_runner.is_running:
        if app.cli:
            app.cli.log("Training already running.", "error")
        return

    # Build work directory — include model name for readability
    import time as _time
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in model_name).strip()
    safe_name = safe_name.replace(" ", "_")
    run_name = f"{safe_name}_{int(_time.time())}" if safe_name else f"run_{int(_time.time())}"
    work_dir = os.path.join(data_dir, "training_runs", run_name)
    os.makedirs(work_dir, exist_ok=True)

    # Resolve num_classes from the dataset's classes.json (the export
    # writes "Unlabeled" as class 0 + one entry per non-zero registry
    # label). Falls back to the registry size if classes.json isn't
    # there yet — happens if the user clicks LAUNCH before staging.
    num_classes = max(2, len(app.label_registry))
    cj = os.path.join(data_dir, "classes.json")
    if os.path.isfile(cj):
        try:
            with open(cj) as f:
                num_classes = int(json.load(f).get("num_classes", num_classes))
        except Exception:
            pass

    # Class-weighted CE so the dominant Unlabeled class doesn't drown
    # out the foreground anatomy classes. ~80% of every cloud is
    # Unlabeled (background bone), only ~10% each is sup/inf endplate;
    # plain CE collapses to predicting Unlabeled everywhere. Down-
    # weight class 0 by 10x so foreground gets a real gradient.
    class_weights = (0.1,) + (1.0,) * (num_classes - 1) if num_classes >= 2 else ()
    # Minority-class boost: multiply the LAST K class weights so small
    # structures (pedicles in the 5-class spinal ontology) get a
    # heavier loss signal than ~10x-larger endplate classes. Default
    # K = 2 since that matches Pedicle_L + Pedicle_R; falls back to 0
    # safely when there aren't two classes to boost.
    minority_boost = float(getattr(app, '_train_minority_boost', 1.0))
    if (class_weights and minority_boost != 1.0
            and len(class_weights) >= 3):
        k = min(2, len(class_weights) - 1)
        boosted = list(class_weights)
        for i in range(len(boosted) - k, len(boosted)):
            boosted[i] = boosted[i] * minority_boost
        class_weights = tuple(boosted)

    # Resolve fine-tune weight path
    weight_path = ""
    if parent_model is not None and parent_model.has_checkpoint:
        weight_path = parent_model.best_checkpoint

    epochs = int(getattr(app, '_train_epochs', 200))
    batch_size = int(getattr(app, '_train_batch', 2))
    device = getattr(app, '_train_device', 'cuda')

    # Generate the Pointcept config. Advanced knobs (base_lr,
    # weight_decay, warmup_epochs, drop_path) come from the TRAIN tab's
    # ADVANCED section via app._train_*; fall back to the dataclass
    # defaults when the user hasn't touched them.
    base_lr_kw = float(getattr(app, '_train_base_lr', 0.006))
    weight_decay_kw = float(getattr(app, '_train_weight_decay', 0.05))
    warmup_kw = int(getattr(app, '_train_warmup_epochs', 10))
    drop_path_kw = float(getattr(app, '_train_drop_path', 0.3))
    so3_kw = bool(getattr(app, '_train_random_rotate_so3', True))
    params = PTv3TrainParams(
        data_root=data_dir,
        num_classes=num_classes,
        epochs=epochs,
        batch_size=batch_size,
        weight=weight_path,
        class_weights=class_weights,
        base_lr=base_lr_kw,
        weight_decay=weight_decay_kw,
        warmup_epochs=warmup_kw,
        drop_path=drop_path_kw,
        random_rotate_so3=so3_kw,
    )
    config_path = os.path.join(work_dir, "config.py")
    pointcept_ext_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training", "pointcept_ext",
    )
    try:
        generate_ptv3_config(params, pointcept_ext_dir, config_path)
    except Exception as e:
        if app.cli:
            app.cli.log(f"Config generation failed: {e}", "error")
        return

    # Build launch config and spawn
    launch_cfg = PointceptLaunchConfig(
        python_exe=python_exe,
        pointcept_dir=pointcept_dir,
        config_file=config_path,
        work_dir=work_dir,
        pointcept_ext_dir=pointcept_ext_dir,
    )

    # Freeze the class map at launch: the dataset's classes.json is the
    # ground truth for what each output channel means. Recording it on
    # the registry entry (and mirroring it into the run's work_dir)
    # makes inference class resolution independent of cwd and of later
    # ontology edits.
    frozen_class_map: dict[str, str] = {}
    try:
        with open(os.path.join(data_dir, "classes.json")) as _cf:
            _cj = json.load(_cf)
        for _i, _name in enumerate(_cj.get("class_names", [])):
            frozen_class_map[str(_i)] = _name
        if frozen_class_map:
            with open(os.path.join(work_dir, "classes.json"), "w") as _wf:
                json.dump(_cj, _wf, indent=2)
    except (OSError, json.JSONDecodeError, TypeError) as e:
        if app.cli:
            app.cli.log(f"classes.json snapshot failed: {e}", "error")

    # Create model registry entry before launch
    model_entry = None
    if project_id and app._train_model_registry:
        from src.data.model_registry import TrainedModel
        display_name = model_name or run_name
        model_entry = TrainedModel(
            name=display_name,
            architecture="PT-v3m1",
            status="training",
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            num_classes=num_classes,
            work_dir=work_dir,
            config_snapshot=params.to_dict(),
            class_map=frozen_class_map,
            parent_model_id=parent_model.model_id if parent_model else "",
        )
        app._train_model_registry.add_model(project_id, model_entry)
        app._train_active_model_id = model_entry.model_id
        app._train_model_finalized = False

    def _on_event(parsed, raw_line):
        """Forward parsed training events through the Pattern A queue.

        Runs on the PT-v3 runner's reader thread — must not touch
        ``app._train_model_registry`` (writes JSON + mutates dict) or
        ``app.cli`` (which the GUI iterates each frame) directly. We
        only touch the per-run model_entry locally (GIL-atomic
        attribute writes; safe), then post a payload for the main
        thread to flush into the registry + CLI log.
        """
        if not hasattr(parsed, 'kind') or parsed.kind == "raw":
            # Raw lines stay in the runner's own log_lines ring; we
            # don't need to fan them out to the CLI panel.
            return
        # TS-002 (post-audit): only the main-thread handler writes to
        # ``model_entry`` — the registry's ``update_model`` re-loads from
        # disk, applies ``updates``, and persists. We previously wrote
        # the three fields directly on the reader thread too, redundant
        # with the handler AND a data race with the main-thread reads
        # in list_models()/get_model() the TRAIN panel does each frame.
        # Reader thread now only computes the updates dict (pure read of
        # ``parsed``) and ships it via post_event.
        updates = {}
        if (model_entry and project_id and app._train_model_registry):
            if (parsed.kind == "val_result"
                    and hasattr(parsed, 'miou') and parsed.miou > 0
                    and parsed.miou > model_entry.best_miou):
                updates['best_miou'] = parsed.miou
            if parsed.kind == "checkpoint" and hasattr(parsed, 'checkpoint_path'):
                updates['last_checkpoint'] = parsed.checkpoint_path
            if parsed.kind == "best_miou" and hasattr(parsed, 'best_miou'):
                ckpt = model_entry.last_checkpoint
                if ckpt:
                    updates['best_checkpoint'] = ckpt
        # ST-5: ship the (raw_line, updates) tuple to the main thread
        # via Pattern A. The handler appends to app.cli.output and
        # calls _train_model_registry.update_model — both are now
        # safely main-thread-only.
        app.post_event("training_event", {
            "raw_line": raw_line,
            "project_id": project_id,
            "model_id": model_entry.model_id if model_entry else None,
            "updates": updates,
        })

    ok = app.training_runner.launch(launch_cfg, on_event=_on_event)
    if app.cli:
        if ok:
            ft_str = f" (fine-tune from {parent_model.name})" if parent_model else ""
            app.cli.log(
                f"Pointcept PTv3 launched: {epochs} epochs, "
                f"batch {batch_size}, {num_classes} classes{ft_str}",
                "success")
            app.cli.log(f"  work_dir: {work_dir}", "info")
        else:
            err = app.training_runner.status.error
            app.cli.log(f"Launch failed: {err}", "error")
            if model_entry and project_id and app._train_model_registry:
                app._train_model_registry.update_model(
                    project_id, model_entry.model_id,
                    status="failed", error=err)


def _launch_inference(app, data_dir: str, parent_model=None):
    """Launch inference using a trained PT-v3 checkpoint.

    Runs single-pass prediction on test split scenes, saves per-scene
    .npy predictions, then imports them as labels on the loaded clouds.
    Uses a subprocess so it doesn't block the UI.
    """
    import subprocess as _sp

    python_exe = getattr(app, '_train_python_exe', '').strip()
    pointcept_dir = getattr(app, '_train_pointcept_dir', '').strip()

    if not python_exe or not pointcept_dir:
        if app.cli:
            app.cli.log("Set python exe and pointcept dir first.", "error")
        return

    # Find the best checkpoint. Precedence:
    #   1. parent_model    (caller passed a specific model — Train tab path)
    #   2. inference selector pick (or auto-best) within the active project
    #   3. live training runner's running best (mid-session fallback)
    checkpoint = None
    if parent_model is not None and getattr(parent_model, 'has_checkpoint', False):
        checkpoint = parent_model.best_checkpoint
    if not checkpoint:
        project_id = _resolve_active_project_id(app)
        chosen = _resolve_inference_model(app, project_id)
        if chosen is not None and chosen.has_checkpoint:
            checkpoint = chosen.best_checkpoint
    if not checkpoint and app.training_runner is not None:
        checkpoint = getattr(app.training_runner.status, 'best_checkpoint', '')
    if not checkpoint or not os.path.isfile(checkpoint):
        if app.cli:
            app.cli.log("No checkpoint found for inference.", "error")
        return

    # Output directory for predictions
    pred_dir = os.path.join(os.path.dirname(checkpoint), "..", "predictions")
    pred_dir = os.path.normpath(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)

    # Find the data root — look for train/val/test subdirs
    test_dir = None
    for split in ("test", "val"):
        candidate = os.path.join(data_dir, split)
        if os.path.isdir(candidate):
            test_dir = candidate
            break
    if test_dir is None:
        test_dir = data_dir

    # Build the inference command using our infer script. The S3DIS-
    # flavoured script is hardcoded for 13 furniture classes + ScanNet
    # pdnorm conditions; calling it on a 3Photon checkpoint blows up in
    # load_state_dict (strict=True) the moment the seg_head shapes
    # disagree. infer_3photon.py reads num_classes from the dataset's
    # classes.json and uses pdnorm_conditions=("Vertebrae",), matching
    # what config_gen.py emits for training.
    infer_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tools", "infer_3photon.py",
    )

    # If the dedicated script doesn't exist, use a generic approach
    if not os.path.isfile(infer_script):
        if app.cli:
            app.cli.log("Inference script not found.", "error")
        return

    cmd = [
        python_exe, "-u", infer_script,
        "--checkpoint", checkpoint,
        "--data-root", os.path.dirname(test_dir),
        "--split", os.path.basename(test_dir),
        "--output", pred_dir,
    ]

    if app.cli:
        app.cli.log(f"Running inference: {os.path.basename(checkpoint)}", "cmd")
        app.cli.log(f"  data: {test_dir}", "info")
        app.cli.log(f"  output: {pred_dir}", "info")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = pointcept_dir + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"
        # UTF-8 child encoding so PT-v3 inference's tqdm bars don't kill
        # the reader thread on Windows.
        env["PYTHONIOENCODING"] = "utf-8"
        proc = _sp.Popen(
            cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT,
            text=True,
            encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        app._infer_runner = proc

        # Stream output to CLI in a background thread.
        # TS-001 (post-audit): reader thread no longer touches
        # ``app.cli`` directly. ``CLIEngine.log`` appends to and
        # rebinds ``self.output`` (a plain list) which the main
        # thread reads each frame — the same Pattern A violation
        # the ST-1..5 fixes eliminated. Reader posts events instead.
        import threading

        def _read_infer_output():
            try:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        app.post_event("infer_log",
                                       {"line": line, "level": "info"})
            except Exception as e:
                app.post_event("infer_log", {
                    "line": f"[infer reader] {type(e).__name__}: {e}",
                    "level": "error",
                })
            proc.wait()
            app.post_event("infer_finished", {
                "returncode": int(proc.returncode or 0),
                "pred_dir": pred_dir,
            })

        t = threading.Thread(
            target=_read_infer_output,
            name="infer-reader",
            daemon=True,
        )
        t.start()

    except Exception as e:
        if app.cli:
            app.cli.log(f"Failed to launch inference: {e}", "error")


def _finalize_model_status(app, project_id: str | None):
    """Update the model registry when a training run finishes."""
    if not project_id or not app._train_model_registry:
        return
    model_id = getattr(app, '_train_active_model_id', '')
    if not model_id:
        return
    runner = getattr(app, 'training_runner', None)
    if not runner:
        return
    status = runner.status
    updates = {}
    if getattr(status, 'error', ''):
        updates['status'] = 'failed'
        updates['error'] = status.error
    elif getattr(status, 'returncode', None) == 0:
        updates['status'] = 'completed'
    else:
        updates['status'] = 'stopped'
    updates['finished'] = time.time()
    if getattr(status, 'best_miou', 0) > 0:
        updates['best_miou'] = status.best_miou
    if getattr(status, 'last_checkpoint', ''):
        updates['last_checkpoint'] = status.last_checkpoint
    # Prefer the runner's tracked best_checkpoint (probed at val-improve
    # time, points at model_best.pth on disk). Fall back to last only if
    # best was never set — e.g. training crashed before the first val
    # pass. Without this guard the registry stored garbage paths because
    # the old code copied last_checkpoint into best, and the log parser
    # was capturing "to:" as the path string.
    best_cp = getattr(status, 'best_checkpoint', '')
    if best_cp and os.path.isfile(best_cp):
        updates['best_checkpoint'] = best_cp
    elif getattr(status, 'last_checkpoint', '') and os.path.isfile(
            status.last_checkpoint):
        updates['best_checkpoint'] = status.last_checkpoint
    app._train_model_registry.update_model(project_id, model_id, **updates)


def _draw_gallery_ctx_popup(app):
    """Right-click context menu for a gallery cell.

    If the right-clicked cell is part of a multi-selection, LOAD/DELETE
    and the 'Add to project' submenu operate on the whole group.
    Otherwise the target is just that one cell.
    """
    if app._gallery_ctx_open_pending:
        imgui.open_popup("##gallery_ctx")
        app._gallery_ctx_open_pending = False

    if imgui.begin_popup("##gallery_ctx"):
        target = app._gallery_ctx_target
        valid = 0 <= target < len(app.entries)

        # Build the effective target set: the multi-selection if the
        # right-clicked cell is in it, else just the single target.
        multi = sorted(getattr(app, '_gallery_multi_sel', set()))
        if target in multi and len(multi) > 1:
            targets = multi
        elif valid:
            targets = [target]
        else:
            targets = []

        if valid:
            entry = app.entries[target]
            header = (f"{len(targets)} clouds selected"
                      if len(targets) > 1 else entry.name[:32])
            imgui.text_colored(header, *OP1_WHITE[:3])
            imgui.separator()

            if imgui.menu_item("Load →  Light Table")[0]:
                app._on_cloud_selected(target, enter_light_table=True)
                imgui.close_current_popup()

            # Add to project — submenu over user-owned projects.
            catalog = getattr(app, 'catalog', None)
            user_cols = []
            if catalog is not None:
                user_cols = [
                    c for c in catalog.projects.values()
                ]
            add_label = (f"Add {len(targets)} to project"
                         if len(targets) > 1 else "Add to project")
            if imgui.begin_menu(add_label, bool(user_cols)):
                for col in user_cols:
                    if imgui.menu_item(col.name)[0]:
                        file_keys = [
                            app.entries[i].file_key for i in targets
                            if app.entries[i].file_key is not None
                        ]
                        if file_keys:
                            # Carry painted labels into the destination
                            # project's ontology — merge into existing
                            # labels by name, add new ones otherwise.
                            # Only migrate keys that aren't already in
                            # the project (an idempotent re-add should
                            # not rewrite the labels file).
                            from src.data.cloud_store import (
                                migrate_cloud_labels_to_project,
                            )
                            already = set(col.file_keys)
                            new_keys = [k for k in file_keys if k not in already]
                            if new_keys:
                                migrate_cloud_labels_to_project(
                                    new_keys, app.label_registry, col,
                                )
                            catalog.add_to_project(col.id, file_keys)
                        imgui.close_current_popup()
                if not user_cols:
                    imgui.menu_item("(no projects — create one first)",
                                    "", False, False)
                imgui.end_menu()
            elif not user_cols:
                # Grey-disabled row when there are no user projects yet
                imgui.menu_item(add_label, "", False, False)

            if imgui.menu_item("Rename...", "",
                               False, len(targets) == 1)[0]:
                app._gallery_rename_target = target
                app._gallery_rename_buf = entry.name
                imgui.close_current_popup()
                imgui.open_popup("##gallery_rename")

            # CP-5: re-attach a moved/renamed source file. Only offered
            # when the entry's source is actually missing — the catalog
            # copy of the data keeps working either way, but a live
            # source path re-enables stale-source detection + re-import.
            _src_missing = (len(targets) == 1 and catalog is not None
                            and entry.file_key is not None
                            and not catalog.entry_exists(entry))
            if _src_missing and imgui.menu_item("Re-attach source...")[0]:
                imgui.close_current_popup()
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                    _root = tk.Tk()
                    _root.withdraw()
                    _new = filedialog.askopenfilename(
                        title=f"Re-attach source for {entry.name}")
                    _root.destroy()
                except Exception:
                    _new = ""
                if _new:
                    if catalog.reattach_entry_source(entry.file_key, _new):
                        entry.file_path = _new
                        if app.cli:
                            app.cli.log(
                                f"Re-attached {entry.name} -> {_new}",
                                "success")
                    else:
                        app.set_status_banner(
                            f"Re-attach failed for {entry.name}",
                            level="error", source="reattach")

            imgui.separator()

            # Two persistent destructive options instead of the old
            # in-memory-only Delete. The old version just popped from
            # app.entries, so the cloud reappeared the next time you
            # opened the same view. These two write through to the
            # catalog and survive a restart. The source file on disk is
            # untouched by either action — neither path calls os.remove
            # on the cloud data file. (The catalog-managed preview .npz
            # IS deleted by remove_entry, which is correct: it'll just
            # be regenerated if you ever re-import.)
            project_id = _resolve_active_project_id(app)
            in_project = project_id is not None

            # 1) Remove from project — only meaningful when the
            #    currently-active view is a user project. Keeps the
            #    cloud in the catalog so it remains visible under
            #    other views.
            if in_project:
                rm_label = (f"Remove {len(targets)} from project"
                            if len(targets) > 1 else "Remove from project")
                if imgui.menu_item(rm_label)[0]:
                    catalog = app.catalog
                    file_keys = [
                        app.entries[i].file_key for i in targets
                        if app.entries[i].file_key is not None
                    ]
                    if catalog is not None and file_keys:
                        catalog.remove_from_project(project_id, file_keys)
                    # Also pop from the in-memory session so the cell
                    # vanishes immediately; the next view rebuild will
                    # honour the persisted state.
                    for i in sorted(targets, reverse=True):
                        app.entries[i].release()
                        app.entries.pop(i)
                    if app.selected_index >= len(app.entries):
                        app.selected_index = max(0, len(app.entries) - 1)
                    app._gallery_multi_sel.clear()
                    app._overlays_dirty = True
                    imgui.close_current_popup()

            # 2) Delete from program — always available. Strips the
            #    entry from the catalog index AND from every project
            #    that references it. Source file on disk stays put,
            #    so re-importing brings it back if needed.
            del_label = (f"Delete {len(targets)} from program"
                         if len(targets) > 1 else "Delete from program")
            if imgui.menu_item(del_label)[0]:
                catalog = app.catalog
                file_keys = [
                    app.entries[i].file_key for i in targets
                    if app.entries[i].file_key is not None
                ]
                if catalog is not None:
                    for fk in file_keys:
                        catalog.remove_entry(fk)
                for i in sorted(targets, reverse=True):
                    app.entries[i].release()
                    app.entries.pop(i)
                if app.selected_index >= len(app.entries):
                    app.selected_index = max(0, len(app.entries) - 1)
                app._gallery_multi_sel.clear()
                app._overlays_dirty = True
                imgui.close_current_popup()
            imgui.text_colored("source file on disk is kept",
                               *DIM[:3])
        else:
            imgui.text_colored("(no target)", *DIM[:3])
        imgui.end_popup()


def _draw_gallery_rename_modal(app):
    """Modal with a text input for renaming a cloud entry."""
    if imgui.begin_popup_modal(
            "##gallery_rename", flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)[0]:
        imgui.text("New name:")
        imgui.push_item_width(s(260))
        entered, app._gallery_rename_buf = imgui.input_text(
            "##rename_buf", app._gallery_rename_buf, 256,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE,
        )
        imgui.pop_item_width()

        confirm = entered or imgui.button("OK", width=s(80))
        imgui.same_line()
        cancel = imgui.button("Cancel", width=s(80))

        target = app._gallery_rename_target
        if confirm and 0 <= target < len(app.entries):
            new_name = app._gallery_rename_buf.strip()
            if new_name:
                # CP-9: persist to catalog so the rename survives restart.
                # In-memory entry's ``name`` (file basename) stays — we
                # only set the display override + push it to the catalog.
                entry = app.entries[target]
                entry.name = new_name
                fk = getattr(entry, "file_key", None)
                if fk and app.catalog is not None:
                    try:
                        app.catalog.rename_entry(fk, new_name)
                    except Exception as e:
                        print(f"[rename] catalog persist failed for {fk}: {e}")
            app._gallery_rename_target = -1
            imgui.close_current_popup()
        elif cancel:
            app._gallery_rename_target = -1
            imgui.close_current_popup()
        imgui.end_popup()


def _parse_resolution(app) -> tuple[int, int]:
    return (1920, 1080)


def _do_save_points(app):
    """Force-save labels for every loaded cloud, then write a timestamped
    snapshot to ``<catalog>/backups/<timestamp>/`` so the user has an
    explicit recovery point. Prints a summary so the action's outcome
    is visible without digging in the filesystem.

    Also writes preview-resolution labels alongside each full save so
    the Train tab's "labelled" filter and the CLOUDS row swatches see
    the cloud as labelled on next relaunch even before the user opens
    it in Light Table.
    """
    from src.data.cloud_store import save_preview_labels

    saved = 0
    skipped = 0
    preview_saved = 0
    saved_keys: list[str] = []
    for entry in app.entries:
        gpu = entry.full_gpu or entry.preview_gpu
        cloud = gpu.cloud_data if gpu is not None else None
        file_key = getattr(entry, 'file_key', None)
        if cloud is None or cloud.labels is None or not file_key:
            skipped += 1
            continue
        try:
            app._persist_cloud_labels(entry, cloud)
            saved += 1
            saved_keys.append(file_key)
        except Exception as e:
            print(f"[save points] {entry.name}: {type(e).__name__}: {e}")
            skipped += 1
            continue

        # Also persist preview labels so the cloud reads as labelled
        # without needing to re-open it in Light Table next session.
        prev_gpu = entry.preview_gpu
        if (prev_gpu is not None
                and prev_gpu.cloud_data is not None
                and prev_gpu.cloud_data.labels is not None):
            try:
                save_preview_labels(file_key, prev_gpu.cloud_data.labels)
                preview_saved += 1
            except Exception as e:
                print(f"[save points] preview {entry.name}: {type(e).__name__}: {e}")

    backup_msg = ""
    if saved_keys:
        try:
            from src.data.cloud_store import snapshot_labels
            backup_dir, copied, snap_skipped = snapshot_labels(saved_keys)
            backup_msg = f"  backup: {copied} file(s) -> {backup_dir}"
            if snap_skipped:
                backup_msg += f" ({snap_skipped} skipped)"
        except Exception as e:
            backup_msg = f"  backup failed: {type(e).__name__}: {e}"

    print(f"[save points] saved {saved} cloud(s) ({preview_saved} preview), skipped {skipped}")
    if backup_msg:
        print(backup_msg)


def _resolve_active_project_id(app) -> str | None:
    """Return the active project's id, or None for smart views / no view."""
    av = getattr(app, 'active_view', None)
    if av and av[0] == 'project' and not str(av[1]).startswith('smart:'):
        return av[1]
    return None


def _inference_eligible_models(app, project_id: str | None) -> list:
    """Return models that can drive an inference run, newest first.

    Eligibility = the model has a ``best_checkpoint`` path that still
    exists on disk. ``has_checkpoint`` on TrainedModel does both checks.
    Failed runs that never reached the first eval are filtered out
    because they never produced a checkpoint to point at.
    """
    registry = getattr(app, '_train_model_registry', None)
    if not project_id or registry is None:
        return []
    return [m for m in registry.list_models(project_id) if m.has_checkpoint]


def _auto_best_inference_model(models: list):
    """Pick the highest-mIoU eligible model, or the newest if none scored.

    ``list_models`` returns newest-first, so falling through to ``[0]``
    when no entry has a positive ``best_miou`` (e.g. only imported
    orphan runs whose mIoU was never extracted from logs) still gives
    the user the most recently completed checkpoint as a sensible
    default.
    """
    if not models:
        return None
    with_miou = [m for m in models if m.best_miou > 0]
    if with_miou:
        return max(with_miou, key=lambda m: m.best_miou)
    return models[0]


def _resolve_inference_model(app, project_id: str | None):
    """Return the TrainedModel that should drive inference, or None.

    Selection order:
    1. User-selected pick on ``app._inference_selector_picks[project_id]``
       — set by the model dropdown so the user can A/B between
       checkpoints. ``None`` in the dict means "auto" (explicit clear).
       Persisted in the project's settings (``active_model_id``) so the
       pick survives restarts and project switches (1.1).
    2. Highest-mIoU eligible model in the registry (auto default).
    """
    eligibles = _inference_eligible_models(app, project_id)
    if not eligibles:
        return None
    picks = getattr(app, "_inference_selector_picks", None)
    pick_id = None
    if isinstance(picks, dict) and project_id in picks:
        pick_id = picks[project_id]
    else:
        # Restore a persisted pick from project settings on first touch.
        catalog = getattr(app, 'catalog', None)
        proj = catalog.projects.get(project_id) if catalog else None
        if proj is not None and isinstance(proj.settings, dict):
            pick_id = proj.settings.get("active_model_id") or None
            if isinstance(picks, dict):
                picks[project_id] = pick_id
    if pick_id is not None:
        for m in eligibles:
            if m.model_id == pick_id:
                return m
        # Stale pick (model was deleted) — fall through to auto.
    return _auto_best_inference_model(eligibles)


def _draw_inference_model_picker(app) -> None:
    """Dropdown above RUN INFERENCE listing every eligible model for
    the active project. Selection drives ``_resolve_inference_model``
    via ``app._inference_selector_picks[project_id]``.

    Per-project picks so switching active project doesn't carry a
    stale selection across. "(auto)" is the default — picks the
    highest-mIoU model; equivalent to leaving the picks dict alone.
    """
    project_id = _resolve_active_project_id(app)
    if not project_id:
        return

    # Lazy-init the registry. TRAIN tab does the same on first visit;
    # without this branch, switching to SHEETS first would silently
    # show "no models" until you bounce through TRAIN.
    if getattr(app, "_train_model_registry", None) is None:
        from src.data.model_registry import ProjectModelRegistry
        from src.data.library_catalog import library_dir
        app._train_model_registry = ProjectModelRegistry(library_dir())

    eligibles = _inference_eligible_models(app, project_id)
    if not eligibles:
        return  # Nothing to pick from — picker hides itself

    picks_dict = getattr(app, "_inference_selector_picks", None)
    if not isinstance(picks_dict, dict):
        picks_dict = {}
        app._inference_selector_picks = picks_dict
    current_pick = picks_dict.get(project_id)

    # Build dropdown items. First entry is "(auto)" — explicit way to
    # clear an override without having to know which model would win.
    items = ["(auto — highest mIoU)"]
    item_model_ids: list[str | None] = [None]
    for m in eligibles:
        miou_str = f" — mIoU {m.best_miou:.2f}" if m.best_miou > 0 else ""
        cls_str = f" · {m.num_classes}cls" if m.num_classes else ""
        items.append(f"{m.name}{miou_str}{cls_str}")
        item_model_ids.append(m.model_id)

    current_idx = 0
    if current_pick is not None:
        for i, mid in enumerate(item_model_ids):
            if mid == current_pick:
                current_idx = i
                break

    imgui.text_colored("MODEL", *DIM[:3])
    imgui.push_item_width(-1)
    changed, new_idx = imgui.combo(
        "##inference_model_pick", current_idx, items,
    )
    imgui.pop_item_width()
    if changed and 0 <= new_idx < len(item_model_ids):
        picks_dict[project_id] = item_model_ids[new_idx]
        # Persist the pick on the project so it survives restarts.
        catalog = getattr(app, 'catalog', None)
        proj = catalog.projects.get(project_id) if catalog else None
        if proj is not None:
            if not isinstance(proj.settings, dict):
                proj.settings = {}
            proj.settings["active_model_id"] = item_model_ids[new_idx]
            catalog._save_projects()

    # Show the checkpoint basename that RUN INFERENCE would actually
    # use right now — disambiguates "(auto)" by naming the winner.
    chosen = _resolve_inference_model(app, project_id)
    if chosen is not None and chosen.best_checkpoint:
        ckpt_name = os.path.basename(chosen.best_checkpoint)
        imgui.text_colored(f"  → {ckpt_name}", *OP1_GRAY[:3])


def _resolve_inference_checkpoint(app) -> str | None:
    """Find a ``model_best.pth`` to run inference against.

    Selection precedence:
      1. Highest-mIoU completed model in the active project's registry
         (via ``_resolve_inference_model``).
      2. Live training runner's running best — ergonomic fallback for
         mid-session use before the run lands in the registry.
      3. Legacy mtime-scan of ``<cwd>/dataset/training_runs/*/model/
         model_best.pth`` — preserves behaviour for projects predating
         the registry, but no longer the primary path.
    """
    project_id = _resolve_active_project_id(app)
    chosen = _resolve_inference_model(app, project_id)
    if chosen is not None and chosen.has_checkpoint:
        return chosen.best_checkpoint
    if app.training_runner is not None:
        cp = getattr(app.training_runner.status, 'best_checkpoint', '') or ""
        if cp and os.path.isfile(cp):
            return cp
    runs_root = os.path.join(os.getcwd(), 'dataset', 'training_runs')
    candidates = []
    if os.path.isdir(runs_root):
        for run in os.listdir(runs_root):
            p = os.path.join(runs_root, run, 'model', 'model_best.pth')
            if os.path.isfile(p):
                candidates.append((os.path.getmtime(p), p))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def _find_classes_json_for_checkpoint(checkpoint_path: str) -> str | None:
    """Walk up from a checkpoint path looking for a ``classes.json`` sibling.

    Standard 3Photon layout puts the file 4 levels above the checkpoint
    (``<data_root>/classes.json`` vs. ``<data_root>/training_runs/<run>/
    model/model_best.pth``). We search a few extra levels so off-standard
    layouts (e.g. a manually-moved checkpoint) still resolve.

    Returns the absolute path or None.
    """
    cur = os.path.dirname(os.path.abspath(checkpoint_path))
    for _ in range(6):
        candidate = os.path.join(cur, 'classes.json')
        if os.path.isfile(candidate):
            return candidate
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent
    return None


def _resolve_inference_class_map(
    app, checkpoint_path: str | None = None,
) -> tuple[list[str], "np.ndarray"] | None:
    """Build the class-id → registry-id translation for inference.

    PT-v3 emits dense class IDs 0..N-1; the order is fixed by the
    ``classes.json`` that was written next to the dataset at export
    time. The model was trained against those exact names, so we MUST
    use that file to interpret the predictions — using the current
    project's registry instead silently mismaps classes (or, more often,
    crashes ``load_state_dict`` when the lengths disagree).

    Resolution order:
      0. The registry model whose checkpoint this is — its ``class_map``
         was frozen at launch time (1.1). Fully immune to moved files
         and later ontology edits.
      1. ``classes.json`` co-located with the checkpoint (walks up from
         the checkpoint dir). This is the training-time file and the
         only one guaranteed to line up with the checkpoint's seg_head.
      2. ``<cwd>/dataset/classes.json`` — legacy fallback for callers
         that haven't passed the checkpoint path through. Brittle (cwd
         can move) but kept for backwards compat.
      3. The active project's non-zero registry labels, ordered by id.
         True last-resort — likely to mismatch the checkpoint and is
         what caused the seg_head-shape crashes we just fixed.

    Returns ``(class_names, cls_to_rid)`` or None if no class list
    could be resolved at all.
    """
    import numpy as _np

    classes_name_order: list[str] = []

    # 0. Frozen class_map on the registry entry that owns this checkpoint.
    if checkpoint_path and getattr(app, '_train_model_registry', None):
        project_id = _resolve_active_project_id(app)
        if project_id:
            for m in app._train_model_registry.list_models(project_id):
                if (m.class_map
                        and checkpoint_path in (m.best_checkpoint,
                                                m.last_checkpoint)):
                    try:
                        ordered = sorted(m.class_map.items(),
                                         key=lambda kv: int(kv[0]))
                        classes_name_order = [name for _k, name in ordered]
                    except (ValueError, TypeError):
                        classes_name_order = []
                    break

    # 1. Co-located with the checkpoint — authoritative.
    if not classes_name_order and checkpoint_path:
        cj = _find_classes_json_for_checkpoint(checkpoint_path)
        if cj:
            try:
                with open(cj) as f:
                    meta = json.load(f)
                classes_name_order = list(meta.get("class_names", []))
            except Exception:
                pass

    # 2. Legacy cwd-relative lookup.
    if not classes_name_order:
        dataset_classes_json = os.path.join(os.getcwd(), 'dataset', 'classes.json')
        if os.path.isfile(dataset_classes_json):
            try:
                with open(dataset_classes_json) as f:
                    meta = json.load(f)
                classes_name_order = list(meta.get("class_names", []))
            except Exception:
                pass

    # 3. Project-registry fallback (last resort).
    if not classes_name_order:
        ordered = sorted(
            (info for info in app.label_registry.all_labels() if info.id != 0),
            key=lambda info: info.id,
        )
        classes_name_order = [info.name for info in ordered]
    if not classes_name_order:
        return None

    name_to_rid: dict[str, int] = {}
    for info in app.label_registry.all_labels():
        if info.id != 0:
            name_to_rid[info.name.lower()] = info.id
    cls_to_rid = _np.zeros(len(classes_name_order), dtype=_np.int32)
    for i, name in enumerate(classes_name_order):
        cls_to_rid[i] = name_to_rid.get(name.lower(), 0)
    return classes_name_order, cls_to_rid


def _predict_cloud(
    app, entry, *, checkpoint: str, python_exe: str,
    class_names: list[str], cls_to_rid,
    label_namespace: str | None = None,
) -> "np.ndarray | None":
    """Run inference on a single entry and return mapped registry-id labels.

    Pure compute — does NOT touch the GPU or call ``_after_label_mutation``,
    so it's safe to call from a worker thread. Caller is responsible for
    applying the returned array on the main thread:

        new_labels = _predict_cloud(app, entry, ...)
        if new_labels is not None:
            cloud.labels[:] = new_labels
            app._after_label_mutation(gpu, cloud)

    Returns None on any failure (subprocess error, length mismatch, missing
    GPU buffer, etc.) and logs to ``app.cli`` / stdout for diagnosis.

    ``label_namespace`` pins the disk writes to the project namespace
    captured at submit time — this function runs on worker threads, so
    relying on the process-global active namespace would misroute the
    save if the user switches projects mid-batch.
    """
    import subprocess as _sp
    import tempfile
    import shutil as _shutil
    import numpy as _np
    from src.data.cloud_store import save_cloud_labels

    # Inference must run on the FULL-resolution cloud. CONTACT SHEETS
    # only loads preview-sized GPU buffers (~5k pts), so predicting
    # against entry.preview_gpu would (a) train-distribution-mismatched
    # because the model never saw downsampled inputs, and (b) the
    # resulting preview-length labels can never persist to disk —
    # _persist_cloud_labels rejects writes whose length doesn't match
    # the entry's full-res point_count. The original workflow required
    # the user to open each cloud in LIGHT TABLE first to force a
    # full-res load before inference could touch it. That defeats the
    # whole point of batch inference.
    #
    # Resolution: when full_gpu isn't loaded, fetch the full cloud
    # data from the catalog on this worker thread (no GL ops, safe to
    # call off the main thread) and run inference against it. We never
    # upload to GPU here — the disk-saved labels surface the next time
    # the cloud is opened, and clouds that already have a full_gpu
    # also get an in-memory apply via the drain queue.
    full_gpu = entry.full_gpu
    if full_gpu is not None and full_gpu.cloud_data is not None:
        cloud = full_gpu.cloud_data
    else:
        file_key = getattr(entry, 'file_key', None)
        try:
            cloud, _ = app._load_cloud_via_catalog(
                file_key, entry.file_path)
        except Exception as e:
            print(f"[infer] {entry.name}: full-cloud load failed: {e}")
            return None
        if cloud is None or cloud.point_count == 0:
            return None

    # Use a TemporaryDirectory context so the input.npz + pred.npy get
    # cleaned up even if the subprocess raises, the user kills the
    # batch, or this function returns early. Without this, every
    # inference click leaks a directory under %TEMP%.
    tmp_dir = tempfile.mkdtemp(prefix="3photon_infer_")
    try:
        in_path = os.path.join(tmp_dir, "input.npz")
        out_path = os.path.join(tmp_dir, "pred.npy")
        coord = cloud.positions.astype("float32")
        color = (cloud.colors.astype("float32")
                 if cloud.colors is not None else _np.full_like(coord, 0.5))
        # Match the training-time feature vector exactly. The export
        # pipeline (dataset_export.compute_normals) computes per-point
        # normals via KNN-PCA and writes them to normal.npy alongside
        # every training scene; the trained config asks for
        # feat_keys=['coord','color','normal']. None of the cloud loaders
        # populate cloud.scalars['normal'] though — so without this on-
        # the-fly compute, inference silently feeds zero-normals to the
        # model, the feature vector no longer matches what was trained,
        # and predictions collapse to garbage on every cloud (training-
        # set clouds included). Always compute normals here unless the
        # caller already attached real ones.
        normal = (cloud.scalars.get("normal")
                  if hasattr(cloud, "scalars") and cloud.scalars else None)
        if normal is None:
            from src.export.dataset_export import compute_normals
            normal = compute_normals(coord)
            # Cache on the cloud so repeated inference calls (the
            # predict-refine-retrain flywheel hits the same cloud often)
            # don't re-run KNN PCA every time. ~100-200ms saved per click.
            if hasattr(cloud, "scalars") and cloud.scalars is not None:
                cloud.scalars["normal"] = normal
        _np.savez(in_path, coord=coord, color=color,
                  normal=normal.astype("float32"))

        infer_script = os.path.normpath(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "tools", "infer_single.py"))

        cmd = [python_exe, "-u", infer_script,
               "--checkpoint", checkpoint,
               "--input", in_path,
               "--output", out_path,
               "--num-classes", str(len(class_names)),
               "--grid-size", "0.5"]

        try:
            # encoding="utf-8" + errors="replace" so a Unicode glyph in
            # the inference subprocess's stdout (tqdm / logging) doesn't
            # raise UnicodeDecodeError on the captured output — same fix
            # the long-running runners use.
            infer_env = os.environ.copy()
            infer_env["PYTHONIOENCODING"] = "utf-8"
            proc = _sp.run(
                cmd, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                env=infer_env, timeout=300,
            )
        except _sp.TimeoutExpired:
            print(f"[infer] {entry.name}: timed out (>5 min)")
            return None
        except Exception as e:
            print(f"[infer] {entry.name}: launch failed: {e}")
            return None

        if proc.returncode != 0:
            print(f"[infer] {entry.name}: subprocess failed")
            print(proc.stdout)
            print(proc.stderr)
            return None

        if not os.path.isfile(out_path):
            print(f"[infer] {entry.name}: no output file produced")
            return None

        pred = _np.load(out_path).reshape(-1)
        if pred.shape[0] != cloud.point_count:
            print(f"[infer] {entry.name}: pred length {pred.shape[0]} "
                  f"!= cloud {cloud.point_count}")
            return None

        valid = (pred >= 0) & (pred < len(class_names))
        new_labels = _np.zeros(cloud.point_count, dtype=_np.int32)
        new_labels[valid] = cls_to_rid[pred[valid]]

        # LS-4: snapshot the existing labels to a pre-inference
        # backup BEFORE overwriting with the prediction. The catalog
        # labels file is otherwise overwritten in-place; without a
        # backup the user's prior paint is lost with no recovery
        # path. Drain handler in main.py is expected to push a
        # LabelCommand built from (prior, new_labels) onto the undo
        # stack when it sees the new labels, so Ctrl+Z restores.
        file_key = getattr(entry, 'file_key', None)
        if file_key and len(new_labels) == cloud.point_count:
            try:
                from src.data import cloud_store as _cs
                prior = _cs.load_cloud_labels(
                    file_key, namespace=label_namespace)
                if prior is not None and prior.size == new_labels.size:
                    import time as _time
                    from pathlib import Path as _Path
                    from src.data import library_paths as _lp
                    ts = _time.strftime("%Y-%m-%d_%H-%M-%S")
                    backup_root = (_Path(_lp.library_dir())
                                   / "backups" / ts / "inference_pre")
                    backup_root.mkdir(parents=True, exist_ok=True)
                    _np.save(str(backup_root / f"{file_key}.npy"), prior)
            except Exception as e:
                print(f"[infer] {entry.name}: snapshot failed: {e}")
            try:
                # CP-4 + LS-7 (post-audit): pass catalog so the per-
                # file_key lock serialises this background-runner write
                # against any concurrent main-thread paint save. Surface
                # any error through the sticky banner — print() is
                # invisible in the packaged .exe and inference is one
                # of the high-impact destructive writes.
                err = save_cloud_labels(
                    file_key, new_labels,
                    catalog=getattr(app, 'catalog', None),
                    namespace=label_namespace)
                if err:
                    print(f"[infer] {entry.name}: {err}")
                    try:
                        app.set_status_banner(
                            f"Inference label save failed for "
                            f"{entry.name}: {err}",
                            level="error",
                            source=f"cloud_store.save_cloud_labels:{file_key}",
                        )
                    except AttributeError:
                        pass
            except Exception as e:
                print(f"[infer] {entry.name}: disk save failed: {e}")
                try:
                    app.set_status_banner(
                        f"Inference label save raised for "
                        f"{entry.name}: {e}",
                        level="error",
                        source=f"cloud_store.save_cloud_labels:{file_key}",
                    )
                except AttributeError:
                    pass
        return new_labels
    finally:
        try:
            _shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def _do_predict_current_cloud(app):
    """Run PT-v3 inference on the currently-selected cloud and apply the
    predictions directly as labels — no dataset export, no PLY round-trip.

    Thin wrapper around :func:`_predict_cloud`: resolves checkpoint + class
    map once, predicts, applies on the main thread via the same
    ``_after_label_mutation`` path the brush uses (GPU upload, label
    count cache, preview propagation, catalog persistence — all free).
    """
    import numpy as _np

    if not (0 <= app.selected_index < len(app.entries)):
        if app.cli:
            app.cli.log("No cloud selected.", "error")
        return
    entry = app.entries[app.selected_index]
    gpu = entry.full_gpu or entry.preview_gpu
    if gpu is None or gpu.cloud_data is None:
        if app.cli:
            app.cli.log("Cloud not loaded yet.", "error")
        return

    python_exe = getattr(app, '_train_python_exe', '').strip()
    if not python_exe or not os.path.isfile(python_exe):
        if app.cli:
            app.cli.log("Set python exe in POINTCEPT ENV first.", "error")
        return

    checkpoint = _resolve_inference_checkpoint(app)
    if not checkpoint:
        if app.cli:
            app.cli.log("No model_best.pth found in any training run.", "error")
        return

    cmap = _resolve_inference_class_map(app)
    if cmap is None:
        if app.cli:
            app.cli.log("No classes found for inference mapping.", "error")
        return
    class_names, cls_to_rid = cmap

    if app.cli:
        app.cli.log(f"Predicting {entry.name} via {os.path.basename(checkpoint)}",
                    "cmd")
    new_labels = _predict_cloud(
        app, entry,
        checkpoint=checkpoint, python_exe=python_exe,
        class_names=class_names, cls_to_rid=cls_to_rid,
    )
    if new_labels is None:
        if app.cli:
            app.cli.log(f"Inference failed for {entry.name}", "error")
        return

    cloud = gpu.cloud_data
    cloud.labels[:] = new_labels
    app._after_label_mutation(gpu, cloud)

    unique, counts = _np.unique(new_labels, return_counts=True)
    summary = ", ".join(f"{int(u)}={int(c)}" for u, c in zip(unique, counts))
    msg = f"Predicted {entry.name}: {summary}"
    if app.cli:
        app.cli.log(msg, "ok")
    print(msg)


class _BatchInferenceRunner:
    """Background worker that runs ``_predict_cloud`` over a list of entries
    sequentially and queues results for the main thread to apply.

    The worker thread never touches OpenGL or ImGui. It writes one
    ``(entry, new_labels)`` tuple per success into ``app._pending_label_applies``;
    the main loop drains that queue each frame in ``_drain_pending_label_applies``
    and runs ``_after_label_mutation`` on the GL-owning thread.

    Status fields read by the UI:
      ``running``   — True while the thread is alive
      ``cancelled`` — True if cancel was requested
      ``current``   — (idx_in_list, total, entry_name) or None
      ``completed`` — count of successfully predicted clouds
      ``failed``    — count of clouds that returned None
    """

    def __init__(self, app, indices: list[int],
                 checkpoint: str, python_exe: str,
                 class_names: list[str], cls_to_rid):
        import threading
        self.app = app
        # ST-3: snapshot the (index, entry) pairs at submit. Once the
        # worker thread starts, app.entries can mutate (user imports
        # / removes / reorders) — looking up by stale index would
        # operate on the wrong cloud.
        self.indices = list(indices)
        self._submit_entries: list[tuple[int, object]] = []
        for idx in self.indices:
            if 0 <= idx < len(app.entries):
                self._submit_entries.append((idx, app.entries[idx]))
        self.checkpoint = checkpoint
        self.python_exe = python_exe
        self.class_names = class_names
        self.cls_to_rid = cls_to_rid
        # v2: freeze the label namespace at submit so a mid-batch
        # project switch can't misroute the worker's disk writes.
        from src.data import cloud_store as _cs
        self.label_namespace = _cs.active_label_namespace()
        self.running = False
        self.cancelled = False
        self.current: tuple[int, int, str] | None = None
        self.completed = 0
        self.failed = 0
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="3photon-batch-infer")

    def start(self):
        self.running = True
        self._thread.start()

    def cancel(self):
        self.cancelled = True

    def _run(self):
        # ST-3: iterate the snapshot, not live self.app.entries.
        total = len(self._submit_entries)
        try:
            for slot, (idx, entry) in enumerate(self._submit_entries):
                if self.cancelled:
                    break
                self.current = (slot + 1, total, entry.name)
                print(f"[batch infer] ({slot+1}/{total}) {entry.name}")
                new_labels = _predict_cloud(
                    self.app, entry,
                    checkpoint=self.checkpoint,
                    python_exe=self.python_exe,
                    class_names=self.class_names,
                    cls_to_rid=self.cls_to_rid,
                    label_namespace=self.label_namespace,
                )
                if new_labels is None:
                    self.failed += 1
                    continue
                # Disk save already happened in _predict_cloud. Only
                # queue an in-memory apply when there's a GPU buffer
                # whose point count matches — otherwise the drain
                # function would print a confusing "length mismatch"
                # warning every time, even though the labels are
                # already correctly persisted to disk.
                gpu_now = entry.full_gpu or entry.preview_gpu
                cloud_now = gpu_now.cloud_data if gpu_now is not None else None
                if (cloud_now is not None
                        and cloud_now.point_count == new_labels.shape[0]):
                    self.app._pending_label_applies.append((entry, new_labels))
                self.completed += 1
        finally:
            self.current = None
            self.running = False


def _start_batch_inference(app, indices: list[int]) -> bool:
    """Resolve checkpoint + class map, spawn a batch runner. Returns True
    if the runner started, False if a precondition failed (logged via
    ``app.cli``).

    Re-entrancy guard: the existing runner check covers the common case
    of a click during an active run, but a rapid double-click can fire
    twice before ``runner.start()`` flips ``running`` to True. We set
    a sentinel runner on ``app`` synchronously to make the second click
    short-circuit immediately.
    """
    if app.contact_sheets_infer_runner is not None and \
            getattr(app.contact_sheets_infer_runner, 'running', False):
        if app.cli:
            app.cli.log("Batch inference already running.", "error")
        return False
    # Stale runner with running=False but not torn down: also bail to
    # avoid two runners writing to _pending_label_applies for the same
    # cloud in quick succession.
    if app.contact_sheets_infer_runner is not None and \
            getattr(app.contact_sheets_infer_runner, '_thread', None) is not None and \
            app.contact_sheets_infer_runner._thread.is_alive():
        if app.cli:
            app.cli.log("Batch inference still cleaning up.", "error")
        return False

    python_exe = getattr(app, '_train_python_exe', '').strip()
    if not python_exe or not os.path.isfile(python_exe):
        if app.cli:
            app.cli.log("Set python exe in POINTCEPT ENV first.", "error")
        return False

    checkpoint = _resolve_inference_checkpoint(app)
    if not checkpoint:
        if app.cli:
            app.cli.log("No model_best.pth found in any training run.", "error")
        return False

    # Pass the checkpoint into the class-map resolver so it reads
    # classes.json next to the training run, not from the current
    # working directory — that prevented per-project ontology drift
    # from being silently misinterpreted as the checkpoint's class list.
    cmap = _resolve_inference_class_map(app, checkpoint)
    if cmap is None:
        if app.cli:
            app.cli.log("No classes found for inference mapping.", "error")
        return False
    class_names, cls_to_rid = cmap

    # Filter out indices that no longer point to a valid entry — guards
    # against the user deleting a cloud while it was in the multi-select
    # set, which would otherwise IndexError mid-batch.
    valid_indices = [i for i in indices if 0 <= i < len(app.entries)]
    dropped = len(indices) - len(valid_indices)
    if dropped:
        if app.cli:
            app.cli.log(f"Dropping {dropped} stale selection(s).", "info")
    indices = valid_indices
    if not indices:
        if app.cli:
            app.cli.log("No clouds selected.", "error")
        return False

    runner = _BatchInferenceRunner(
        app, indices,
        checkpoint=checkpoint, python_exe=python_exe,
        class_names=class_names, cls_to_rid=cls_to_rid,
    )
    app.contact_sheets_infer_runner = runner
    runner.start()
    if app.cli:
        app.cli.log(
            f"Batch inference: {len(indices)} cloud(s) via "
            f"{os.path.basename(checkpoint)}", "cmd")
    return True


def drain_pending_label_applies(app, max_per_frame: int = 1) -> int:
    """Apply queued (entry, new_labels) pairs on the main thread.

    Called once per frame from the main loop. Caps work per frame so a
    big batch doesn't stall rendering — at most one cloud's labels get
    re-uploaded per frame, which still feels instant for the user but
    keeps frame time bounded.

    Pulls items from a ``collections.deque`` via ``popleft``, which is
    atomic under the GIL — safe even if the worker thread is calling
    ``append`` concurrently. The ``while`` check sees a snapshot of
    ``len(queue)``; we may exit one item early if the worker appends
    between iterations, but that item gets picked up next frame.
    """
    import numpy as _np
    queue = getattr(app, '_pending_label_applies', None)
    if not queue:
        return 0
    applied = 0
    while applied < max_per_frame:
        try:
            entry, new_labels = queue.popleft()
        except IndexError:
            break
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None or gpu.cloud_data is None:
            continue
        cloud = gpu.cloud_data
        if new_labels.shape[0] != cloud.point_count:
            print(f"[batch infer] apply: length mismatch on {entry.name} "
                  f"({new_labels.shape[0]} vs {cloud.point_count}), skip")
            continue
        # LS-4 (post-audit): the drain previously did
        # ``cloud.labels[:] = new_labels`` and never pushed a
        # LabelCommand — so Ctrl+Z couldn't restore the user's prior
        # paint after batch inference. _predict_cloud writes a disk
        # backup under ``<catalog>/backups/<ts>/inference_pre/`` but
        # there's no UI to roll back from it.
        #
        # Same approach as LS-8's propagation port: one LabelCommand
        # per distinct new label id, mirroring ``apply_label`` from
        # ``core/undo.py``. The user gets one undo step per class —
        # consistent with how propagate-all and apply_label_array work
        # today. The mutation lands as expected; cloud.labels matches
        # new_labels after the loop because each call writes only its
        # own ``new_labels == lid`` slice.
        from src.core.undo import apply_label as _apply_label
        unique = _np.unique(new_labels)
        for lid in unique:
            idx = _np.where(new_labels == lid)[0].astype(_np.int32)
            if idx.size == 0:
                continue
            _apply_label(cloud, idx, int(lid), app.undo_stack,
                         description=f"infer {entry.name} -> label {lid}")
        app._after_label_mutation(gpu, cloud)
        counts = _np.bincount(new_labels.astype(_np.int64).clip(min=0))
        summary = ", ".join(
            f"{int(u)}={int(counts[int(u)])}" for u in unique if int(u) >= 0)
        msg = f"Predicted {entry.name}: {summary}"
        if app.cli:
            app.cli.log(msg, "ok")
        print(msg)
        applied += 1
    return applied


def _do_screenshot(app):
    from src.export.image_export import export_still
    w, h = _parse_resolution(app)
    output = os.path.join(os.getcwd(), 'export', 'screenshot.png')
    export_still(app, output, w, h)


def _do_spin(app):
    from src.export.spin_export import export_spin
    w, h = _parse_resolution(app)
    output_dir = os.path.join(os.getcwd(), 'export', 'spin')
    export_spin(app, output_dir, frames=360, width=w, height=h)
