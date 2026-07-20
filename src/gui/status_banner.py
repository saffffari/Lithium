"""Sticky status banner for save/load errors and recovery hints.

The packaged 3Photon .exe has no stdout — every ``print()`` on failure
is invisible to the user. This module gives App a single place to
surface critical messages (label save failed, project lock denied,
catalog corruption recovered, etc.) as a dismissible sticky bar at the
top of the viewport.

Construction is intentionally minimal so the eventual Phase 9/10
fixes can replace a ``print(...)`` site with one helper call:

    app.set_status_banner("Catalog save failed: disk full",
                          level="error", source="cloud_store.save")

Auto-coalescing: when ``set_status_banner`` runs with the same
``source`` as the current banner, the count bumps instead of stacking
a second bar. This stops a runaway error loop (e.g. autosave failing
every 15 s) from drowning the UI.
"""

from __future__ import annotations

from dataclasses import dataclass

import imgui


@dataclass
class ErrorBanner:
    """One banner's worth of state."""
    message: str
    level: str = "error"           # 'error' | 'warn' | 'info'
    suppressed_count: int = 0      # additional same-source occurrences after the first
    source: str = ""               # opaque identifier for coalescing (e.g. "cloud_store.save")


# Colors picked to read clearly on the ultra-dark viewport while keeping
# the warm OP-1 palette discipline (see CLAUDE.md). Tuples are RGBA in
# [0, 1]; ImGui consumes them as col32 via imgui.color_convert_float4_to_u32.
_LEVEL_BG = {
    "error": (0.82, 0.18, 0.18, 0.92),
    "warn":  (0.92, 0.62, 0.18, 0.92),
    "info":  (0.18, 0.55, 0.92, 0.92),
}
_LEVEL_FG = {
    "error": (1.0, 1.0, 1.0, 1.0),
    "warn":  (0.05, 0.05, 0.05, 1.0),
    "info":  (1.0, 1.0, 1.0, 1.0),
}


def draw_status_banner(app) -> float:
    """Render the active banner if any. Returns the drawn height in px.

    Must be called from ``App._render_frame`` BEFORE any other
    ``imgui.begin`` so the banner sits flush at the top edge. Callers
    can offset other layout by the returned height when needed; the
    primary panels are already drawn at ``app._menu_bar_height``, which
    they query separately.
    """
    banner = getattr(app, "_status_banner", None)
    if banner is None:
        return 0.0

    width = float(app.width)
    th = imgui.get_text_line_height()
    h = th * 1.8

    flags = (imgui.WINDOW_NO_DECORATION
             | imgui.WINDOW_NO_MOVE
             | imgui.WINDOW_NO_SAVED_SETTINGS
             | imgui.WINDOW_ALWAYS_AUTO_RESIZE
             | imgui.WINDOW_NO_FOCUS_ON_APPEARING
             | imgui.WINDOW_NO_NAV
             | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS)

    # Sit it just under the main menu bar — height queried each frame
    # because the menu bar can be hidden via H.
    top_y = float(getattr(app, "_menu_bar_height", 0.0))
    imgui.set_next_window_position(0, top_y)
    imgui.set_next_window_size(width, h)

    bg = _LEVEL_BG.get(banner.level, _LEVEL_BG["error"])
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *bg)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (th * 0.5, th * 0.35))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)

    opened = imgui.begin("##status_banner", flags=flags)
    if opened:
        fg = _LEVEL_FG.get(banner.level, (1.0, 1.0, 1.0, 1.0))
        text = banner.message
        if banner.suppressed_count > 0:
            text = f"{text}  (+{banner.suppressed_count} more)"
        imgui.push_style_color(imgui.COLOR_TEXT, *fg)
        imgui.text(text)
        imgui.pop_style_color()

        # Dismiss button on the right edge.
        imgui.same_line()
        label_w = imgui.calc_text_size(" ✕ ").x
        imgui.set_cursor_pos_x(width - label_w - th)
        if imgui.small_button("✕"):
            app._status_banner = None
    imgui.end()

    imgui.pop_style_var(2)
    imgui.pop_style_color()

    return h
