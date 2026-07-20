"""Hierarchical label layer panel — Rhino-style layer tree + color picker modal."""

import math
import imgui
import numpy as np

from src.gui.theme import *
from src.gui.scale import s
from src.gui.op1_widgets import BUTTON_ROUNDING
from src.data.labels import LabelRegistry, LabelInfo, DEFAULT_PALETTE


def draw_label_panel(app) -> bool:
    """Draw the hierarchical label layer panel. Returns True if anything changed."""
    changed = False
    registry: LabelRegistry = app.label_registry

    # --- Label rows (no header — +ADD is in the section strip) ---
    hierarchy = registry.get_hierarchy()
    for info, depth in hierarchy:
        row_changed = _draw_label_row(app, info, depth, registry)
        changed |= row_changed

    return changed


def _project_active(app) -> bool:
    """True if the current view is a user project (labels are editable)."""
    av = getattr(app, 'active_view', None)
    return bool(av and av[0] == 'project' and not str(av[1]).startswith('smart:'))


def draw_label_add_button(app) -> bool:
    """Draw the +ADD and −DEL buttons inline in the section header.
    Called from panels.py.

    Labels are project-scoped: both buttons are dimmed and inert when no
    project is active. The −DEL button is also inert when no removable
    label is active (active_label_id == 0 means unlabeled/root, which
    can't be removed).
    """
    th = imgui.get_text_line_height()
    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()

    add_w = th * 3.2
    add_h = th * 1.2
    add_x = wx + w - add_w - th * 0.3
    add_y = wy - add_h - th * 0.15  # overlap into section header

    rem_w = th * 2.8
    rem_gap = th * 0.25
    rem_x = add_x - rem_w - rem_gap
    rem_y = add_y

    project_active = _project_active(app)
    active_lid = int(getattr(app, 'active_label_id', 0) or 0)
    can_remove = (project_active and active_lid > 0
                  and app.label_registry.get(active_lid) is not None)

    btn_r = s(BUTTON_ROUNDING)
    mx, my = imgui.get_mouse_pos()
    changed = False

    # --- −DEL button (left) ---
    rc = OP1_RED if can_remove else OP1_GRAY
    dl.add_rect_filled(rem_x, rem_y, rem_x + rem_w, rem_y + add_h,
                       imgui.get_color_u32_rgba(rc[0] * 0.12, rc[1] * 0.12,
                                                rc[2] * 0.12, 1.0), btn_r)
    dl.add_rect_filled(rem_x, rem_y, rem_x + rem_w, rem_y + add_h,
                       imgui.get_color_u32_rgba(rc[0] * 0.08, rc[1] * 0.08,
                                                rc[2] * 0.08, 0.6), btn_r)
    rlbl = "− DEL"
    rlbl_sz = imgui.calc_text_size(rlbl)
    dl.add_text(rem_x + (rem_w - rlbl_sz[0]) * 0.5,
                rem_y + (add_h - th) * 0.5,
                col32(rc), rlbl)

    if (can_remove
            and rem_x <= mx <= rem_x + rem_w
            and rem_y <= my <= rem_y + add_h
            and imgui.is_mouse_clicked(0)):
        # Confirm before deleting if the label has any points
        # assigned. Otherwise the user can silently destroy hours of
        # paint work by clicking the wrong row + DEL. With zero
        # painted points the delete is harmless, so skip the prompt.
        n_points = _count_points_with_label(app, active_lid)
        if n_points > 0:
            # Stash what's pending so the modal (drawn later this frame)
            # can read which label + count without re-scanning.
            app._pending_label_delete = {
                "label_id": int(active_lid),
                "label_name": app.label_registry.get(active_lid).name,
                "point_count": int(n_points),
            }
            imgui.open_popup("##confirm_label_delete")
        else:
            if app.label_registry.remove_label(active_lid):
                app.active_label_id = 0
                cache = getattr(app, 'label_count_cache', None)
                if cache is not None:
                    cache.clear()
                _update_label_texture(app)
                changed = True

    # Modal must be drawn at the same nesting level as open_popup. The
    # popup() guards on its own visible state, so this is a no-op when
    # the user hasn't clicked DEL on a label-with-points.
    if _draw_label_delete_confirm_modal(app):
        # Returned True means the modal committed a deletion this
        # frame; let the caller refresh.
        changed = True

    # --- +ADD button (right) ---
    ac = OP1_ORANGE if project_active else OP1_GRAY
    dl.add_rect_filled(add_x, add_y, add_x + add_w, add_y + add_h,
                       imgui.get_color_u32_rgba(ac[0] * 0.12, ac[1] * 0.12,
                                                ac[2] * 0.12, 1.0), btn_r)
    dl.add_rect_filled(add_x, add_y, add_x + add_w, add_y + add_h,
                       imgui.get_color_u32_rgba(ac[0] * 0.08, ac[1] * 0.08,
                                                ac[2] * 0.08, 0.6), btn_r)
    lbl = "+ ADD"
    lbl_sz = imgui.calc_text_size(lbl)
    dl.add_text(add_x + (add_w - lbl_sz[0]) * 0.5,
                add_y + (add_h - th) * 0.5,
                col32(ac), lbl)

    if (project_active
            and add_x <= mx <= add_x + add_w
            and add_y <= my <= add_y + add_h
            and imgui.is_mouse_clicked(0)):
        registry = app.label_registry
        name = f"Label {len(registry)}"
        new_id = registry.add_label(name)
        app.active_label_id = new_id
        _update_label_texture(app)
        changed = True

    return changed


def _draw_label_row(app, info: LabelInfo, depth: int,
                    registry: LabelRegistry) -> bool:
    """Draw a single label row with icons and controls."""
    changed = False
    th = imgui.get_text_line_height()
    dl = imgui.get_window_draw_list()

    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    row_h = th * 1.6
    indent = depth * th * 1.2

    is_active = app.active_label_id == info.id

    # Background — active row fills with purple tint matching LABELS section,
    # inactive rows use a very light grey.
    if is_active:
        bg = (OP1_ORANGE[0] * 0.15, OP1_ORANGE[1] * 0.15,
              OP1_ORANGE[2] * 0.15, 1.0)  # orange tint
    else:
        bg = (0.09, 0.09, 0.09, 1.0)  # subtle dark
    row_r = s(BUTTON_ROUNDING)
    dl.add_rect_filled(wx, wy, wx + w, wy + row_h, col32(bg), row_r)

    # Layout: [indent] [swatch] [name] ... [count] [eye]
    x = wx + indent + th * 0.3

    # Color swatch
    swatch_x = x
    swatch_color = col32(info.color)
    dl.add_rect_filled(swatch_x, wy + row_h * 0.3,
                       swatch_x + th * 0.9, wy + row_h * 0.7, swatch_color)

    # Name — either inline rename input or static text
    name_x = swatch_x + th * 1.3
    renaming_this = (getattr(app, '_renaming_label_id', None) == info.id)

    if renaming_this:
        # Inline rename: ImGui text input in a tiny invisible window
        # positioned right over the name area.
        rename_w = w - (name_x - wx) - s(40)
        imgui.set_next_window_position(name_x, wy + (row_h - th) * 0.5 - 2)
        imgui.set_next_window_size(rename_w, th * 1.6)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.0, 0.0, 0.0, 0.0)
        imgui.push_style_color(imgui.COLOR_FRAME_BACKGROUND, 0.12, 0.12, 0.12, 1.0)
        imgui.begin(f"##rename_{info.id}", flags=(
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR |
            imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BACKGROUND))
        imgui.push_item_width(rename_w - 4)
        # Auto-focus on first frame
        if getattr(app, '_renaming_focus_pending', False):
            imgui.set_keyboard_focus_here()
            app._renaming_focus_pending = False
        enter, new_name = imgui.input_text(
            f"##rn_{info.id}", app._renaming_buf, 128,
            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE | imgui.INPUT_TEXT_AUTO_SELECT_ALL)
        imgui.pop_item_width()
        imgui.end()
        imgui.pop_style_color(2)

        if enter and new_name.strip():
            registry.set_name(info.id, new_name.strip())
            app._renaming_label_id = None
            changed = True
        elif imgui.is_key_pressed(imgui.KEY_ESCAPE):
            app._renaming_label_id = None
        else:
            app._renaming_buf = new_name
    else:
        name_color = col32(WHITE)
        display_name = info.name[:24]
        dl.add_text(name_x, wy + (row_h - th) * 0.5, name_color, display_name)

    # Right side: [count] [eye] — eye is far right, scaled down
    eye_r = th * 0.22  # smaller than before
    eye_cx = wx + w - th * 0.5
    eye_cy = wy + row_h * 0.5
    if info.visible:
        eye_col = col32(AMBER)
        dl.add_circle(eye_cx, eye_cy, eye_r, eye_col, 12, 1.2)
        ray_inner = eye_r + th * 0.06
        ray_outer = eye_r + th * 0.18
        for k in range(6):
            a = math.pi * 2.0 * k / 6.0 + math.pi / 6.0
            ca, sa = math.cos(a), math.sin(a)
            dl.add_line(eye_cx + ca * ray_inner, eye_cy + sa * ray_inner,
                        eye_cx + ca * ray_outer, eye_cy + sa * ray_outer,
                        eye_col, 1.2)
    else:
        dl.add_circle(eye_cx, eye_cy, eye_r, col32(FAINT), 12, 1.0)

    # Point count to the left of eye
    count = _count_label_points(app, info.id)
    if count > 0:
        count_str = _format_count(count)
        count_w = imgui.calc_text_size(count_str)[0]
        dl.add_text(wx + w - count_w - th * 1.3, wy + (row_h - th) * 0.5,
                    col32(DIM), count_str)

    # Hit detection
    imgui.dummy(w, row_h + 2)
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(0):
        mx, my = imgui.get_mouse_pos()

        # Eye click region (far right)
        if mx >= wx + w - th * 1.0:
            registry.set_visible(info.id, not info.visible)
            _update_label_texture(app)
            changed = True
        # Swatch click region
        elif swatch_x <= mx < swatch_x + th * 1.0:
            if info.id != 0:
                app._editing_label_id = info.id
                app._editing_color = tuple(info.color)
                app._editing_name_buf = info.name
                app._editing_opened_frame = app.frame_count
                # Store swatch screen position for popup placement
                app._editing_swatch_pos = (swatch_x, wy + row_h)
                changed = True
        else:
            # Click anywhere else sets active
            app.active_label_id = info.id
            if getattr(app, '_renaming_label_id', None) is not None:
                app._renaming_label_id = None
            changed = True

    # Double-click the name area → inline rename
    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
        mx, my = imgui.get_mouse_pos()
        if mx > name_x and info.id != 0:
            app._renaming_label_id = info.id
            app._renaming_buf = info.name
            app._renaming_focus_pending = True

    return changed


def _count_label_points(app, label_id: int) -> int:
    """Count points with a given label in the currently selected cloud.

    Uses app.label_count_cache so this is O(1) after the first frame
    for each (cloud, label_id) pair. Invalidated by label edits.
    """
    if not (0 <= app.selected_index < len(app.entries)):
        return 0
    entry = app.entries[app.selected_index]
    gpu = entry.full_gpu or entry.preview_gpu
    if gpu is None:
        return 0
    cache = getattr(app, 'label_count_cache', None)
    if cache is None:
        return int((gpu.cloud_data.labels == label_id).sum())
    return cache.get(gpu.cloud_data, label_id)


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def _update_label_texture(app):
    """Re-upload the label color texture after registry change."""
    if app.label_texture is not None:
        from src.rendering.label_texture import update_label_color_texture
        update_label_color_texture(app.label_texture, app.label_registry)


def draw_color_picker_popup(app):
    """Mini color picker popup positioned at the clicked swatch.

    Compact 4x4 palette grid that appears right over the swatch.
    Click a color to apply immediately. Click outside or Escape to dismiss.
    """
    label_id = getattr(app, '_editing_label_id', None)
    if label_id is None:
        return
    info = app.label_registry.get(label_id)
    if info is None:
        app._editing_label_id = None
        return

    dl = imgui.get_foreground_draw_list()
    th = imgui.get_text_line_height()

    # Position at swatch location (stored when swatch was clicked)
    swatch_pos = getattr(app, '_editing_swatch_pos', None)
    if swatch_pos is None:
        swatch_pos = (100, 100)

    # Compact popup: 4x4 grid of color swatches
    cols = 4
    rows = 4
    gap = 3.0
    cell_sz = th * 1.4
    pad = th * 0.4
    panel_w = cols * cell_sz + (cols - 1) * gap + pad * 2
    panel_h = rows * cell_sz + (rows - 1) * gap + pad * 2

    px = swatch_pos[0]
    py = swatch_pos[1] + 2

    # Keep on screen
    if px + panel_w > app.width:
        px = app.width - panel_w - 4
    if py + panel_h > app.height:
        py = swatch_pos[1] - panel_h - 2

    # Panel background
    purple = (0.7, 0.5, 0.95, 1.0)
    dl.add_rect_filled(px, py, px + panel_w, py + panel_h,
                       imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.96), 6.0)
    dl.add_rect(px, py, px + panel_w, py + panel_h,
                imgui.get_color_u32_rgba(*purple), 6.0, 0, 1.5)

    new_color = app._editing_color or tuple(info.color)
    mx, my = imgui.get_mouse_pos()

    for i, color in enumerate(DEFAULT_PALETTE[:16]):
        r = i // cols
        c = i % cols
        cx = px + pad + c * (cell_sz + gap)
        cy = py + pad + r * (cell_sz + gap)
        dl.add_rect_filled(cx, cy, cx + cell_sz, cy + cell_sz,
                           imgui.get_color_u32_rgba(*color), 3.0)
        is_current = (abs(color[0] - new_color[0]) < 0.01 and
                      abs(color[1] - new_color[1]) < 0.01 and
                      abs(color[2] - new_color[2]) < 0.01)
        if is_current:
            dl.add_rect(cx, cy, cx + cell_sz, cy + cell_sz,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.9), 3.0, 0, 2.0)

        # Click = apply immediately and close
        if (cx <= mx <= cx + cell_sz and cy <= my <= cy + cell_sz
                and imgui.is_mouse_clicked(0)):
            app.label_registry.set_color(label_id, color)
            _update_label_texture(app)
            app._editing_label_id = None
            app._editing_color = None
            return

    # Click outside popup = dismiss
    opened_this_frame = (getattr(app, '_editing_opened_frame', -1) == app.frame_count)
    if not opened_this_frame and imgui.is_mouse_clicked(0):
        if not (px <= mx <= px + panel_w and py <= my <= py + panel_h):
            app._editing_label_id = None
            app._editing_color = None

    if imgui.is_key_pressed(imgui.KEY_ESCAPE):
        app._editing_label_id = None
        app._editing_color = None


# ---------------------------------------------------------------------------
# Label deletion safety
# ---------------------------------------------------------------------------

def _count_points_with_label(app, label_id: int) -> int:
    """Count points across all loaded clouds whose label == ``label_id``.

    Uses the project-wide label_count_cache so this is O(loaded clouds)
    of hash lookups, not O(total points). Clouds that aren't loaded
    into ``app.entries`` (catalog-only, no preview_gpu yet) don't get
    counted — the cache is an in-memory structure keyed by the loaded
    cloud objects. That's acceptable for the "are you sure?" use case
    because if the user can see the count badge on the LABELS panel
    showing N points painted, those clouds are loaded and the count
    is correct; clouds outside the working set would only inflate the
    number, not lead to silent data loss.
    """
    cache = getattr(app, 'label_count_cache', None)
    if cache is None:
        return 0
    total = 0
    for entry in getattr(app, 'entries', []):
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None or gpu.cloud_data is None:
            continue
        total += cache.get(gpu.cloud_data, int(label_id))
    return total


def _draw_label_delete_confirm_modal(app) -> bool:
    """Modal that gates label deletion behind an explicit confirmation.

    Triggered when the user clicks DEL on a label that has painted
    points. Shows the label name + count + a brief explanation of
    what soft-deletion means in this codebase (points keep their
    numeric label_id; the registry just drops the entry, so the
    points become "ghost-labeled" until you add a label with the same
    id again).

    Returns True if a deletion was actually committed this frame, so
    the caller can refresh its display state.
    """
    committed = False
    pending = getattr(app, '_pending_label_delete', None)
    is_open, _ = imgui.begin_popup_modal(
        "##confirm_label_delete",
        flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE,
    )
    if is_open:
        if pending is None:
            # Stale open without a payload — just close it.
            imgui.close_current_popup()
            imgui.end_popup()
            return False

        name = pending.get("label_name", "?")
        n = pending.get("point_count", 0)
        imgui.text_colored(f"Delete label '{name}'?", *OP1_WHITE[:3])
        imgui.spacing()
        imgui.text_colored(
            f"{n:,} points are currently labelled as '{name}'.",
            *OP1_GRAY[:3])
        imgui.text_colored(
            f"They will be reassigned to Unlabeled. Permanent —", *OP1_RED[:3])
        imgui.text_colored(
            f"there is no automatic recovery if you change your mind.",
            *OP1_RED[:3])
        imgui.spacing()
        imgui.text_colored(
            "Tip: rename the label instead if you just want a different",
            *DIM[:3])
        imgui.text_colored(
            "name + colour but keep the painted points.", *DIM[:3])
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Right-aligned destructive button + cancel.
        # Destructive on the right so muscle memory for "click OK to
        # continue" doesn't auto-delete.
        cancel = imgui.button("Cancel", width=s(100))
        imgui.same_line()
        # Style the destructive button red.
        imgui.push_style_color(imgui.COLOR_BUTTON,
                               OP1_RED[0] * 0.35, OP1_RED[1] * 0.10,
                               OP1_RED[2] * 0.10, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED,
                               OP1_RED[0] * 0.55, OP1_RED[1] * 0.18,
                               OP1_RED[2] * 0.18, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE,
                               OP1_RED[0] * 0.7, OP1_RED[1] * 0.25,
                               OP1_RED[2] * 0.25, 1.0)
        confirm = imgui.button(f"Delete '{name}'", width=s(160))
        imgui.pop_style_color(3)

        if confirm:
            lid = pending.get("label_id")
            if lid is not None:
                # Zero the numeric label_id on every cloud BEFORE
                # removing the registry entry. Order matters only for
                # caches — zeroing first means the count cache sees
                # the new state when the next render pass refreshes.
                # The registry removal itself is essentially free.
                try:
                    loaded, on_disk = app.zero_label_id_across_catalog(int(lid))
                    print(f"[delete-label] {pending.get('label_name')!r} "
                          f"(id={lid}): zeroed {loaded} loaded + "
                          f"{on_disk} on-disk clouds")
                except Exception as e:
                    print(f"[delete-label] zeroing failed: {e}")
                if app.label_registry.remove_label(int(lid)):
                    if app.active_label_id == lid:
                        app.active_label_id = 0
                    cache = getattr(app, 'label_count_cache', None)
                    if cache is not None:
                        cache.clear()
                    _update_label_texture(app)
                    committed = True
            app._pending_label_delete = None
            imgui.close_current_popup()
        elif cancel:
            app._pending_label_delete = None
            imgui.close_current_popup()
        imgui.end_popup()
    return committed
