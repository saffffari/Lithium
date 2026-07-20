"""Right-hand measurement registry panel.

Auto-opens on first committed measurement. Shows all committed
measurements (lines, angles, landmarks) in a collapsible folder-tree.

Features
--------
- Click to select / Shift+click range-select
- Small × delete button per row
- RMB context menu: "Send to Group" > group list
- Toolbar: "New Group" button
- Folder icon (📁) for group headers
- Keyboard Delete handled in main.py via app.measure_selection
"""

import imgui
from src.gui.scale import s, sidebar_width_right
from src.gui.theme import OP1_RED, OP1_ORANGE


# ── colour helpers ────────────────────────────────────────────────────────────

def _rgba(r, g, b, a=1.0):
    return imgui.get_color_u32_rgba(r, g, b, a)


def _fmt_value(item, resolve, unit: str = "u") -> str:
    """Format item value with unit suffix.

    ``resolve`` is the cloud-local->world resolver that the measurement
    layer requires now that anchors live in cloud-local space. Pass
    ``app._build_measure_resolver()`` from the panel call site so the
    formatting reflects the current scene state.
    """
    v = item.get_value(resolve)
    if v is None:
        # Landmark: show XYZ position
        pos = item.get_position(resolve)
        if pos is not None:
            return f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
        return "—"
    if item.mode == 'measure_line':
        if v >= 10_000:
            return f"{v:.1f} {unit}"
        if v >= 100:
            return f"{v:.2f} {unit}"
        if v >= 1:
            return f"{v:.3f} {unit}"
        return f"{v:.4f} {unit}"
    else:
        return f"{v:.1f}\u00b0"


# ── module-level state (ImGui immediate-mode) ─────────────────────────────────

# Rename-in-place state: which item/group id is being renamed and its buffer.
_rename_id: str | None = None
_rename_buf: str = ""
_rename_kind: str = ""   # 'item' | 'group'

# Context-menu state — deferred one frame so ImGui popup opens cleanly.
_ctx_pending_ids: set | None = None   # selection snapshot for the popup


# ── public entry point ────────────────────────────────────────────────────────

def draw_measure_panel(app) -> None:
    """Draw the right-side measurement panel if open.

    Called from the main render loop after draw_settings_panel().
    Reads / writes:
      app.measure_panel_open     bool
      app.measure_selection      set[str]  — selected item ids
      app.measure_registry       MeasureRegistry
      app._measure_last_sel_id   str | None — last singly-clicked id for shift-range
    """
    if not getattr(app, 'measure_panel_open', False):
        return

    global _rename_id, _rename_buf, _rename_kind, _ctx_pending_ids

    registry = getattr(app, 'measure_registry', None)
    if registry is None:
        return

    # ── window geometry ────────────────────────────────────────────────────────
    rw = sidebar_width_right()
    menu_h = int(getattr(app, '_menu_bar_height', 0.0))
    panel_h = max(1, app.height - menu_h)
    panel_x = app.width - rw

    imgui.set_next_window_position(panel_x, menu_h, imgui.ALWAYS)
    imgui.set_next_window_size(rw, panel_h, imgui.ALWAYS)

    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
             imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)

    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.07, 0.07, 0.07, 0.97)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (s(8), s(6)))
    opened, _ = imgui.begin("##measure_panel", True, flags=flags)
    imgui.pop_style_var(1)
    imgui.pop_style_color(1)

    if not opened:
        imgui.end()
        return

    # ── unit for the active cloud ──────────────────────────────────────────────
    unit = _get_unit(app)

    # ── anchor resolver for this frame ─────────────────────────────────────────
    # Anchors are cloud-local now; the resolver converts each one to a
    # world position using the current mode + reference-vertebra state.
    # Built once per panel draw so every row sees the same scene snapshot.
    resolve = app._build_measure_resolver()

    # ── panel header ──────────────────────────────────────────────────────────
    _draw_header(app, registry)

    imgui.separator()
    imgui.spacing()

    # ── tree ──────────────────────────────────────────────────────────────────
    sel: set = getattr(app, 'measure_selection', set())
    flat = registry.flat_tree()

    # Build a list of item ids in tree-display order for Shift-range selection.
    ordered_ids = [obj.id for kind, obj in flat if kind in ('item', 'ungrouped')]

    for kind, obj in flat:
        if kind == 'group':
            _draw_group_row(app, registry, obj, sel)
        else:
            in_group = (kind == 'item')
            _draw_item_row(app, registry, obj, sel, ordered_ids,
                           unit, in_group, resolve)

    # ── context menu (deferred popup open) ─────────────────────────────────────
    # The popup is opened ONE frame after the RMB click sets _ctx_pending_ids.
    # This avoids the rapid open/close cycle that causes flashing: ImGui needs
    # the popup to be opened outside the widget loop that detected the RMB.
    if _ctx_pending_ids is not None:
        imgui.open_popup("##measure_ctx")

    if imgui.begin_popup("##measure_ctx"):
        # Use the snapshot taken at click time — NOT the live sel set
        _draw_context_menu(app, registry, _ctx_pending_ids or set())
        imgui.end_popup()
    else:
        # Popup closed (user clicked away or action completed) — clear snapshot
        _ctx_pending_ids = None

    imgui.end()


# ── internal helpers ──────────────────────────────────────────────────────────

def _get_unit(app) -> str:
    """Best-effort unit label from the active cloud."""
    try:
        from src.main import _cloud_unit_label
        if 0 <= app.selected_index < len(app.entries):
            return _cloud_unit_label(app.entries[app.selected_index])
    except Exception:
        pass
    return "u"


def _draw_folder_icon(dl, x, y, sz, expanded, col):
    """Small folder icon drawn via the draw list."""
    # Tab on the top-left
    tab_w = sz * 0.45
    tab_h = sz * 0.22
    dl.add_rect_filled(x, y, x + tab_w, y + tab_h, col, 1.0)
    # Body
    body_y = y + tab_h - 1
    body_h = sz * 0.62
    dl.add_rect_filled(x, body_y, x + sz, body_y + body_h, col, 1.5)
    if not expanded:
        # Small inner line to indicate closed
        dl.add_line(x + sz * 0.2, body_y + body_h * 0.5,
                    x + sz * 0.8, body_y + body_h * 0.5,
                    imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.3), 1.0)


def _draw_header(app, registry) -> None:
    """Panel title + toolbar buttons."""
    imgui.push_style_color(imgui.COLOR_TEXT, 0.65, 0.65, 0.65, 1.0)
    imgui.text("MEASUREMENTS")
    imgui.pop_style_color(1)

    imgui.same_line()

    # Close button (×) far right
    rw = sidebar_width_right()
    pad = s(8)
    close_x = rw - s(24) - pad
    imgui.set_cursor_pos_x(close_x)
    imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.25, 0.25, 0.25, 1.0)
    if imgui.button("\u00d7##mp_close", s(22), s(18)):
        app.measure_panel_open = False
    imgui.pop_style_color(2)

    imgui.spacing()

    # Toolbar: [+ Group]  [Delete sel]
    imgui.push_style_color(imgui.COLOR_BUTTON, 0.18, 0.18, 0.18, 1.0)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.28, 0.28, 0.28, 1.0)
    if imgui.button("+ Group##mp_grp", s(80), s(22)):
        registry.add_group()
    imgui.pop_style_color(2)

    sel: set = getattr(app, 'measure_selection', set())
    if sel:
        imgui.same_line()
        imgui.push_style_color(imgui.COLOR_BUTTON, 0.38, 0.10, 0.10, 1.0)
        imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.55, 0.15, 0.15, 1.0)
        if imgui.button(f"Delete ({len(sel)})##mp_del", s(100), s(22)):
            registry.remove_items(sel)
            app.measure_selection = set()
            app._measure_last_sel_id = None
        imgui.pop_style_color(2)


def _draw_group_row(app, registry, grp, sel) -> None:
    """Collapsible group folder row with folder icon, rename, and delete."""
    global _rename_id, _rename_buf, _rename_kind

    dl = imgui.get_window_draw_list()
    cx, cy = imgui.get_cursor_screen_pos()

    # Folder icon (neutral grey)
    icon_sz = s(13)
    folder_col = imgui.get_color_u32_rgba(0.50, 0.50, 0.50, 0.90)
    _draw_folder_icon(dl, cx + s(2), cy + s(3), icon_sz, grp.expanded, folder_col)

    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + icon_sz + s(6))

    # Name only — no unicode arrow (the folder icon already indicates state)
    label = f"{grp.name}##grp_{grp.id}"

    imgui.push_style_color(imgui.COLOR_TEXT, 0.80, 0.80, 0.80, 1.0)
    if imgui.selectable(label, False, imgui.SELECTABLE_SPAN_ALL_COLUMNS, 0, s(20))[0]:
        grp.expanded = not grp.expanded
    imgui.pop_style_color(1)

    # RMB on group header → context popup
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
        imgui.open_popup(f"##grp_ctx_{grp.id}")

    if imgui.begin_popup(f"##grp_ctx_{grp.id}"):
        imgui.text_disabled("Group options")
        imgui.separator()
        if imgui.selectable("Rename", False)[0]:
            _rename_id = grp.id
            _rename_buf = grp.name
            _rename_kind = 'group'
            imgui.close_current_popup()
        if imgui.selectable("Delete group", False)[0]:
            registry.remove_group(grp.id)
            imgui.close_current_popup()
        imgui.end_popup()

    # Inline rename (double-click)
    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
        _rename_id = grp.id
        _rename_buf = grp.name
        _rename_kind = 'group'

    if _rename_id == grp.id and _rename_kind == 'group':
        imgui.set_next_item_width(sidebar_width_right() - s(40))
        changed, new_val = imgui.input_text(f"##ren_{grp.id}", _rename_buf, 128,
                                            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        _rename_buf = new_val
        if changed or (not imgui.is_item_active() and _rename_id == grp.id):
            if changed and new_val.strip():
                grp.name = new_val.strip()
            _rename_id = None

    # × delete button on far right
    _draw_delete_btn(f"##del_grp_{grp.id}", lambda: registry.remove_group(grp.id))


def _draw_item_row(app, registry, item, sel: set, ordered_ids: list,
                   unit: str, indented: bool, resolve=None) -> None:
    """Single measurement item row."""
    global _ctx_pending_ids, _rename_id, _rename_buf, _rename_kind

    is_sel = item.id in sel

    if indented:
        imgui.indent(s(16))

    rw = sidebar_width_right()
    row_w = rw - s(16) - (s(16) if indented else 0) - s(26)  # leave room for × btn

    # Colour dot before name
    r, g, b, a = item.color
    draw_list = imgui.get_window_draw_list()
    cx, cy = imgui.get_cursor_screen_pos()
    dot_r = s(4.5)
    draw_list.add_circle_filled(cx + dot_r + s(2), cy + s(10),
                                dot_r, imgui.get_color_u32_rgba(r, g, b, 1.0), 12)
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + dot_r * 2 + s(6))

    # Build label: name + value
    # Resolver supplied by draw_measure_panel; the fallback handles
    # legacy callers / tests that haven't been updated yet.
    if resolve is None:
        resolve = app._build_measure_resolver()
    val_str = _fmt_value(item, resolve, unit)
    disp = f"{item.name}   {val_str}##sel_{item.id}"

    imgui.push_style_color(imgui.COLOR_HEADER,
                           0.22, 0.22, 0.22, 1.0 if is_sel else 0.0)
    imgui.push_style_color(imgui.COLOR_HEADER_HOVERED, 0.28, 0.28, 0.28, 1.0)
    imgui.push_style_color(imgui.COLOR_HEADER_ACTIVE, 0.30, 0.30, 0.30, 1.0)

    clicked, _ = imgui.selectable(disp, is_sel,
                                  imgui.SELECTABLE_SPAN_ALL_COLUMNS,
                                  row_w, s(20))
    imgui.pop_style_color(3)

    if clicked:
        shift = imgui.get_io().key_shift
        if shift and getattr(app, '_measure_last_sel_id', None) in ordered_ids:
            # Range select
            try:
                a_idx = ordered_ids.index(app._measure_last_sel_id)
                b_idx = ordered_ids.index(item.id)
                lo, hi = sorted((a_idx, b_idx))
                sel.clear()
                sel.update(ordered_ids[lo:hi + 1])
            except ValueError:
                sel.clear()
                sel.add(item.id)
        else:
            sel.clear()
            sel.add(item.id)
        app._measure_last_sel_id = item.id
        app.measure_selection = sel

    # RMB → snapshot selection and defer popup open to next frame
    if imgui.is_item_hovered() and imgui.is_mouse_clicked(1):
        if item.id not in sel:
            sel.clear()
            sel.add(item.id)
            app.measure_selection = sel
            app._measure_last_sel_id = item.id
        # Take a frozen snapshot of the selection for the context menu.
        # This prevents the flashing caused by tree restructuring mid-popup.
        _ctx_pending_ids = set(sel)

    # Double-click → start inline rename
    if imgui.is_item_hovered() and imgui.is_mouse_double_clicked(0):
        _rename_id = item.id
        _rename_buf = item.name
        _rename_kind = 'item'

    # Inline rename input
    if _rename_id == item.id and _rename_kind == 'item':
        w = rw - s(60) - (s(16) if indented else 0)
        imgui.set_next_item_width(w)
        changed, new_val = imgui.input_text(f"##ren_{item.id}", _rename_buf, 128,
                                            imgui.INPUT_TEXT_ENTER_RETURNS_TRUE)
        _rename_buf = new_val
        if changed or (not imgui.is_item_active() and _rename_id == item.id):
            if changed and new_val.strip():
                item.name = new_val.strip()
            _rename_id = None

    if indented:
        imgui.unindent(s(16))

    # × button on the same line (right-aligned)
    _draw_delete_btn(f"##del_{item.id}",
                     lambda iid=item.id: _delete_items(app, registry, {iid}))


def _draw_delete_btn(label: str, callback) -> None:
    """Small × button right-aligned on the current row."""
    rw = sidebar_width_right()
    btn_w = s(20)
    btn_x = rw - btn_w - s(10)
    imgui.same_line()
    imgui.set_cursor_pos_x(btn_x)
    imgui.push_style_color(imgui.COLOR_BUTTON, 0.0, 0.0, 0.0, 0.0)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.40, 0.12, 0.12, 1.0)
    imgui.push_style_color(imgui.COLOR_TEXT, 0.55, 0.55, 0.55, 1.0)
    if imgui.button(f"\u00d7{label}", btn_w, s(18)):
        callback()
    imgui.pop_style_color(3)


def _delete_items(app, registry, ids: set) -> None:
    """Remove items and update selection."""
    registry.remove_items(ids)
    sel = getattr(app, 'measure_selection', set())
    sel -= ids
    app.measure_selection = sel
    if getattr(app, '_measure_last_sel_id', None) in ids:
        app._measure_last_sel_id = None


def _draw_context_menu(app, registry, frozen_sel: set) -> None:
    """RMB context menu for selected items.

    `frozen_sel` is a snapshot taken at click time — NOT the live sel
    reference, so tree restructuring during the popup can't cause
    the selection set to thrash and produce flashing.
    """
    global _ctx_pending_ids

    n = len(frozen_sel)
    imgui.text_disabled(f"{n} selected" if n > 1 else "Measurement")
    imgui.separator()

    # Send to Group submenu
    if imgui.begin_menu("Send to Group"):
        # Ungrouped option
        if imgui.selectable("(Ungrouped)", False)[0]:
            registry.send_to_group(frozen_sel, None)
            _ctx_pending_ids = None
            imgui.close_current_popup()
        imgui.separator()
        for grp in registry.groups:
            if imgui.selectable(grp.name, False)[0]:
                registry.send_to_group(frozen_sel, grp.id)
                _ctx_pending_ids = None
                imgui.close_current_popup()
        imgui.end_menu()

    imgui.separator()

    if imgui.selectable("Delete", False)[0]:
        _delete_items(app, registry, set(frozen_sel))
        _ctx_pending_ids = None
        imgui.close_current_popup()
