"""Keyboard shortcut reference overlay (press ? to toggle).

The overlay sizes itself to its content: the sections are measured
first, laid out in one or two columns depending on the window width,
and the frame is drawn around the result. The old fixed 700x640 box
overflowed once the list grew past ~30 rows, so the SELECTION block
and the "Press ? to close" footer drew on top of each other.
"""

import imgui

from src.gui.theme import *


# A new user's first ten minutes, in order. Kept short on purpose: the
# full workflow lives in docs/user_guide.md; this is the card on the
# wall next to the machine.
GETTING_STARTED = [
    ("1.", "CONTACT SHEETS: double-click a bone to open it"),
    ("2.", "LIGHT TABLE: Choose a label and a tool (P / O / K)"),
    ("3.", "Shift+drag paints, Ctrl+drag erases; G cycles labels"),
    ("4.", "Thumb wheel or LEFT / RIGHT: next bone (autosaves)"),
    ("5.", "Mistake? Ctrl+Z. Lost? F focuses the view"),
]

SHORTCUTS = [
    ("MOUSE", [
        ("RMB drag", "Orbit camera (flick to spin)"),
        ("RMB hold", "Radial preset menu (right/top/left/bottom)"),
        ("MMB drag", "Pan camera"),
        ("Scroll", "Zoom (dolly-under-cursor)"),
        ("Thumb wheel", "Previous / next cloud"),
        ("Alt + LMB", "Place 3D cursor (no zoom)"),
        ("MMB + scroll", "Adjust point size"),
        ("LMB double-click", "Place 3D cursor + focus"),
        ("LMB drag", "Pan camera"),
        ("Shift + LMB drag", "Paint with the active tool"),
        ("Ctrl + LMB drag", "Erase the active label"),
    ]),
    ("NAVIGATION", [
        ("1 / 2 / 3", "Switch tab (Sheets / Light Table / Train)"),
        ("ESC", "Back to Contact Sheets / Exit"),
        ("H / C", "Hide active label / show all labels (Light Table)"),
        ("G", "Next label (Light Table)"),
        ("T / R / I", "Top / right / isometric view"),
        ("SPACE", "Place 3D cursor + focus"),
        ("F", "Focus on 3D cursor"),
        ("LEFT / RIGHT", "Previous / next cloud or 4D frame"),
    ]),
    ("SELECTION", [
        ("P", "Pick tool"),
        ("B / V", "Previous / next cloud (Light Table)"),
        ("O", "Lasso tool"),
        ("K", "Brush tool"),
        ("U", "Curve tool"),
        ("Shift + click", "Paint (Pick tool)"),
        ("Ctrl + click", "Erase (Pick tool)"),
        ("Ctrl + scroll", "Adjust brush radius / depth limit"),
        ("Shift+H", "Toggle GUI visibility"),
        ("Ctrl+Z / Ctrl+Shift+Z", "Undo / Redo"),
        ("Ctrl+Shift+X", "Clear labels (cloud, or selected sheets)"),
        ("Ctrl/Cmd + / - / 0", "UI scale up / down / reset"),
        ("L", "Toggle label color view"),
    ]),
    ("OVERLAYS", [
        ("Shift+B", "Toggle bounding box"),
        ("Shift+G", "Toggle grid floor"),
    ]),
    ("EXPORT", [
        ("Ctrl+E", "Screenshot"),
        ("Ctrl+Shift+E", "Spin render"),
    ]),
    ("HELP", [
        ("?", "Toggle this overlay"),
    ]),
]

_PAD = 20          # frame inset
_INDENT = 20       # key column inset inside a section
_GAP = 24          # gap between the key and description columns
_COL_GAP = 32      # gap between the two layout columns
_MARGIN = 10       # minimum gap between the frame and the window edge


def _measure_section(header, rows, th):
    """Height of one section in pixels, matching ``_draw_section``."""
    return th * 1.3 + len(rows) * th * 1.1 + th * 0.3


def _split_columns(sections, th):
    """Greedy two-column split that keeps sections whole and the
    columns as even as possible."""
    heights = [_measure_section(h, r, th) for h, r in sections]
    total = sum(heights)
    best, best_diff = len(sections), float("inf")
    acc = 0.0
    for i, hgt in enumerate(heights):
        acc += hgt
        diff = abs(acc - (total - acc))
        if diff < best_diff:
            best, best_diff = i + 1, diff
    left, right = sections[:best], sections[best:]
    return [left, right] if right else [left]


def _draw_section(dl, x, cy, header, rows, key_w, th):
    dl.add_text(x, cy, col32(CORAL), header)
    cy += th * 1.3
    for key, desc in rows:
        dl.add_text(x + _INDENT, cy, col32(WHITE), key)
        dl.add_text(x + _INDENT + key_w + _GAP, cy, col32(DIM), desc)
        cy += th * 1.1
    return cy + th * 0.3


def draw_shortcuts_overlay(app):
    """Draw a centered overlay showing all keyboard shortcuts."""
    if not getattr(app, 'show_shortcuts', False):
        return

    dl = imgui.get_foreground_draw_list()
    th = imgui.get_text_line_height()
    calc = imgui.calc_text_size

    sections = [("GETTING STARTED", GETTING_STARTED)] + SHORTCUTS
    all_rows = [row for _, rows in sections for row in rows]
    key_w = max(calc(k)[0] for k, _ in all_rows)
    desc_w = max(calc(d)[0] for _, d in all_rows)
    col_w = _INDENT + key_w + _GAP + desc_w

    # One column if it fits the window height, otherwise two; the frame
    # follows the content either way.
    head_h = th * 2.8                 # title + rule
    foot_h = th * 2.0                 # close hint
    avail_h = app.height - 2 * _MARGIN
    avail_w = app.width - 2 * _MARGIN
    one_col_h = sum(_measure_section(h, r, th) for h, r in sections)
    columns = [sections]
    if head_h + one_col_h + foot_h > avail_h and avail_w >= 2 * _PAD + 2 * col_w + _COL_GAP:
        columns = _split_columns(sections, th)
    body_h = max(sum(_measure_section(h, r, th) for h, r in col) for col in columns)
    w = min(avail_w, 2 * _PAD + len(columns) * col_w + (len(columns) - 1) * _COL_GAP)
    h = min(avail_h, head_h + body_h + foot_h)
    x = (app.width - w) * 0.5
    y = (app.height - h) * 0.5

    # Background
    dl.add_rect_filled(x, y, x + w, y + h, col32((0.04, 0.04, 0.04, 0.95)))
    dl.add_rect(x, y, x + w, y + h, col32(ORANGE), 0.0, 0, 2.0)

    # Title
    title = "KEYBOARD SHORTCUTS"
    title_w = calc(title)[0]
    dl.add_text(x + (w - title_w) * 0.5, y + th * 0.5, col32(ORANGE), title)
    dl.add_line(x + _PAD, y + th * 2, x + w - _PAD, y + th * 2, col32(FAINT), 1.0)

    # Body: clip to the frame so a very short window degrades to a
    # cropped list instead of text outside the box.
    dl.push_clip_rect(x, y + head_h, x + w, y + h - foot_h, True)
    cx = x + _PAD
    for col in columns:
        cy = y + head_h
        for header, rows in col:
            cy = _draw_section(dl, cx, cy, header, rows, key_w, th)
        cx += col_w + _COL_GAP
    dl.pop_clip_rect()

    # Close hint
    close_hint = "Press ? to close"
    ch_w = calc(close_hint)[0]
    dl.add_text(x + (w - ch_w) * 0.5, y + h - th * 1.5, col32(FAINT), close_hint)
