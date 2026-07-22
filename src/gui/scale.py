"""GUI scaling utilities — DPI-aware, driven by GLFW content scale."""

import glfw

# Cached scale factor (set once at init, updated on window move between monitors)
_dpi_scale = 1.0
_base_font_size = 15.0

# Mutable sidebar width (logical, pre-DPI pixels). The drag handle in
# panels.py mutates this at runtime. Single sidebar shared across tabs.
_LEFT_DEFAULT = 380
_SIDEBAR_MIN = 280

_left_w: int = _LEFT_DEFAULT

# Right sidebar (measurement panel). Wider default than the left
# so the measurement table's long row names + value column + units
# have room to breathe; 280px clipped the labels and made values
# look cramped. 480px is comfortable.
_RIGHT_DEFAULT = 480
_RIGHT_MIN = 320
_right_w: int = _RIGHT_DEFAULT

# Second left sidebar (popout column to the right of the main left
# sidebar). Used by HOLOGRAM to keep scene-composition controls within
# eye-line of the main controls rather than across the screen.
_LEFT2_DEFAULT = 280
_LEFT2_MIN = 220
_left2_w: int = _LEFT2_DEFAULT


def init_scale(window) -> float:
    """Read the DPI content scale from GLFW and cache it. Returns the scale factor."""
    global _dpi_scale
    sx, sy = glfw.get_window_content_scale(window)
    _dpi_scale = max(sx, sy, 1.0)
    return _dpi_scale


def set_scale(value: float) -> float:
    """Explicitly set the cached scale (monitor-change rescale path)."""
    global _dpi_scale
    _dpi_scale = max(float(value), 0.5)
    return _dpi_scale


def get_scale() -> float:
    """Get the current DPI scale factor."""
    return _dpi_scale


def s(pixels: float) -> float:
    """Scale a pixel value by the current DPI factor."""
    return pixels * _dpi_scale


def font_size() -> float:
    """Get the font size to load at (in pixels, already scaled for DPI)."""
    return _base_font_size * _dpi_scale


def sidebar_width() -> int:
    """Current left sidebar width in framebuffer pixels.

    Back-compat alias for sidebar_width_left() — every legacy caller
    gets the outer column width.
    """
    return int(_left_w * _dpi_scale)


def sidebar_width_left() -> int:
    """Sidebar width in framebuffer pixels (single sidebar)."""
    return sidebar_width()


def set_sidebar_width_left(logical_px: int):
    """Update the sidebar logical width. Clamped to the shared min."""
    global _left_w
    _left_w = int(max(_SIDEBAR_MIN, logical_px))


def sidebar_min_logical() -> int:
    """Shared minimum sidebar width in logical pixels."""
    return _SIDEBAR_MIN


def sidebar_width_right() -> int:
    """Right sidebar (measurement panel) width in framebuffer pixels."""
    return int(_right_w * _dpi_scale)


def set_sidebar_width_right(logical_px: int) -> None:
    """Update right sidebar logical width. Clamped to minimum."""
    global _right_w
    _right_w = int(max(_RIGHT_MIN, logical_px))


def sidebar_width_left2() -> int:
    """Second left sidebar width in framebuffer pixels."""
    return int(_left2_w * _dpi_scale)


def set_sidebar_width_left2(logical_px: int) -> None:
    global _left2_w
    _left2_w = int(max(_LEFT2_MIN, logical_px))
