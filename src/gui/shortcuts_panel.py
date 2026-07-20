"""Keyboard shortcut reference overlay (press ? to toggle)."""

import imgui

from src.gui.theme import *


SHORTCUTS = [
    ("MOUSE", []),
    (None, [
        ("RMB drag", "Orbit camera (flick to spin)"),
        ("RMB hold", "Radial preset menu (right/top/left/bottom)"),
        ("MMB drag", "Pan camera"),
        ("Scroll", "Zoom (dolly-under-cursor)"),
        ("MMB + scroll", "Adjust point size"),
        ("LMB double-click", "Place 3D cursor + focus"),
        ("LMB drag", "Reserved for selection tools"),
    ]),
    ("NAVIGATION", []),
    (None, [
        ("1 / 2 / 3", "Switch tab (Sheets / Light Table / Train)"),
        ("ESC", "Back to Contact Sheets / Exit"),
        ("H", "Toggle GUI visibility"),
        ("G", "Toggle Contact Sheets <-> Light Table"),
        ("T / F / R / I", "Camera presets (top/front/right/iso)"),
        ("SPACE", "Fit view"),
        ("F", "Focus on 3D cursor"),
        ("LEFT / RIGHT", "Previous / next frame (4D)"),
    ]),
    ("SELECTION", []),
    (None, [
        ("P", "Pick tool"),
        ("B", "Box tool"),
        ("O", "Lasso tool"),
        ("K", "Brush tool"),
        ("U", "Curve tool"),
        ("Shift + click", "Add to selection"),
        ("Ctrl + click", "Remove from selection"),
        ("Ctrl + scroll", "Adjust brush radius / depth limit"),
        ("Enter", "Apply active label"),
        ("Ctrl+Z / Ctrl+Shift+Z", "Undo / Redo"),
        ("L", "Toggle label color view"),
    ]),
    ("OVERLAYS", []),
    (None, [
        ("Shift+B", "Toggle bounding box"),
        ("Shift+G", "Toggle grid floor"),
    ]),
    ("EXPORT", []),
    (None, [
        ("Ctrl+E", "Screenshot"),
        ("Ctrl+Shift+E", "Spin render"),
    ]),
    ("HELP", []),
    (None, [
        ("?", "Toggle this overlay"),
    ]),
]


def draw_shortcuts_overlay(app):
    """Draw a centered overlay showing all keyboard shortcuts."""
    if not getattr(app, 'show_shortcuts', False):
        return

    w = min(app.width - 200, 700)
    h = min(app.height - 100, 640)
    x = (app.width - w) * 0.5
    y = (app.height - h) * 0.5

    dl = imgui.get_foreground_draw_list()
    th = imgui.get_text_line_height()

    # Background
    dl.add_rect_filled(x, y, x + w, y + h, col32((0.04, 0.04, 0.04, 0.95)))
    dl.add_rect(x, y, x + w, y + h, col32(ORANGE), 0.0, 0, 2.0)

    # Title
    title = "KEYBOARD SHORTCUTS"
    title_w = imgui.calc_text_size(title)[0]
    dl.add_text(x + (w - title_w) * 0.5, y + th * 0.5, col32(ORANGE), title)
    dl.add_line(x + 20, y + th * 2, x + w - 20, y + th * 2, col32(FAINT), 1.0)

    cy = y + th * 2.8
    for header, rows in SHORTCUTS:
        if header is not None:
            dl.add_text(x + 20, cy, col32(CORAL), header)
            cy += th * 1.3
        for key, desc in rows:
            dl.add_text(x + 40, cy, col32(WHITE), key)
            dl.add_text(x + 200, cy, col32(DIM), desc)
            cy += th * 1.1
        cy += th * 0.3

    # Close hint
    close_hint = "Press ? to close"
    ch_w = imgui.calc_text_size(close_hint)[0]
    dl.add_text(x + (w - ch_w) * 0.5, y + h - th * 1.5, col32(FAINT), close_hint)
