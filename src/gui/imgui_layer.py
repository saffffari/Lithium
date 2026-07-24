"""ImGui initialization, frame management, and GLFW integration."""

import sys

import glfw
import imgui
from imgui.integrations.glfw import GlfwRenderer
from src.gui.scale import init_scale, get_scale, s


def _apply_dark_flat_style():
    """Apply ultra-dark flat neutral grey theme."""
    style = imgui.get_style()

    sc = get_scale()

    style.window_rounding = 0.0
    style.frame_rounding = 2.0 * sc
    style.grab_rounding = 2.0 * sc
    style.scrollbar_rounding = 2.0 * sc
    style.child_rounding = 0.0
    style.popup_rounding = 0.0
    style.tab_rounding = 0.0

    style.window_padding = imgui.Vec2(12 * sc, 8 * sc)
    style.frame_padding = imgui.Vec2(6 * sc, 3 * sc)
    style.item_spacing = imgui.Vec2(8 * sc, 4 * sc)
    style.item_inner_spacing = imgui.Vec2(4 * sc, 4 * sc)
    style.indent_spacing = 16 * sc
    style.scrollbar_size = 10 * sc
    style.grab_min_size = 8 * sc

    style.window_border_size = 0.0
    style.frame_border_size = 0.0
    style.child_border_size = 0.0
    style.alpha = 1.0

    colors = style.colors

    # Background = 90% K (0.10), nested panels = 95% K (0.05).
    # The nested wells are visibly DARKER than the background so they
    # read as recessed surfaces, not raised cards. Bright accents
    # (orange) inside a 0.05 well stay high-contrast.
    colors[imgui.COLOR_WINDOW_BACKGROUND]        = (0.10, 0.10, 0.10, 1.0)
    colors[imgui.COLOR_CHILD_BACKGROUND]          = (0.05, 0.05, 0.05, 1.0)
    colors[imgui.COLOR_POPUP_BACKGROUND]          = (0.10, 0.10, 0.10, 0.95)
    colors[imgui.COLOR_MENUBAR_BACKGROUND]        = (0.10, 0.10, 0.10, 1.0)

    colors[imgui.COLOR_BORDER]                    = (0.18, 0.18, 0.18, 0.5)
    colors[imgui.COLOR_BORDER_SHADOW]             = (0.0, 0.0, 0.0, 0.0)

    colors[imgui.COLOR_TEXT]                      = (0.75, 0.75, 0.75, 1.0)
    colors[imgui.COLOR_TEXT_DISABLED]              = (0.40, 0.40, 0.40, 1.0)

    # Headers / buttons sit on the 0.10 background so they're slightly
    # lighter than the panels but still dark.
    colors[imgui.COLOR_HEADER]                    = (0.13, 0.13, 0.13, 1.0)
    colors[imgui.COLOR_HEADER_HOVERED]            = (0.20, 0.20, 0.20, 1.0)
    colors[imgui.COLOR_HEADER_ACTIVE]             = (0.16, 0.16, 0.16, 1.0)

    colors[imgui.COLOR_BUTTON]                    = (0.13, 0.13, 0.13, 1.0)
    colors[imgui.COLOR_BUTTON_HOVERED]            = (0.22, 0.22, 0.22, 1.0)
    colors[imgui.COLOR_BUTTON_ACTIVE]             = (0.18, 0.18, 0.18, 1.0)

    # Inputs are wells: a notch DARKER than the panel surrounding them.
    colors[imgui.COLOR_FRAME_BACKGROUND]          = (0.04, 0.04, 0.04, 1.0)
    colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED]  = (0.06, 0.06, 0.06, 1.0)
    colors[imgui.COLOR_FRAME_BACKGROUND_ACTIVE]   = (0.05, 0.05, 0.05, 1.0)

    colors[imgui.COLOR_TITLE_BACKGROUND]          = (0.05, 0.05, 0.05, 1.0)
    colors[imgui.COLOR_TITLE_BACKGROUND_ACTIVE]   = (0.07, 0.07, 0.07, 1.0)
    colors[imgui.COLOR_TITLE_BACKGROUND_COLLAPSED] = (0.05, 0.05, 0.05, 0.75)

    colors[imgui.COLOR_SCROLLBAR_BACKGROUND]      = (0.06, 0.06, 0.06, 0.5)
    colors[imgui.COLOR_SCROLLBAR_GRAB]            = (0.25, 0.25, 0.25, 1.0)
    colors[imgui.COLOR_SCROLLBAR_GRAB_HOVERED]    = (0.35, 0.35, 0.35, 1.0)
    colors[imgui.COLOR_SCROLLBAR_GRAB_ACTIVE]     = (0.30, 0.30, 0.30, 1.0)

    colors[imgui.COLOR_SLIDER_GRAB]               = (0.85, 0.45, 0.15, 1.0)
    colors[imgui.COLOR_SLIDER_GRAB_ACTIVE]        = (0.95, 0.55, 0.15, 1.0)

    colors[imgui.COLOR_CHECK_MARK]                = (0.95, 0.4, 0.2, 1.0)

    colors[imgui.COLOR_SEPARATOR]                 = (0.20, 0.20, 0.20, 0.5)
    colors[imgui.COLOR_SEPARATOR_HOVERED]         = (0.30, 0.30, 0.30, 0.8)
    colors[imgui.COLOR_SEPARATOR_ACTIVE]          = (0.40, 0.40, 0.40, 1.0)

    colors[imgui.COLOR_RESIZE_GRIP]               = (0.20, 0.20, 0.20, 0.25)
    colors[imgui.COLOR_RESIZE_GRIP_HOVERED]       = (0.30, 0.30, 0.30, 0.67)
    colors[imgui.COLOR_RESIZE_GRIP_ACTIVE]        = (0.40, 0.40, 0.40, 0.95)

    colors[imgui.COLOR_TAB]                       = (0.08, 0.08, 0.08, 1.0)
    colors[imgui.COLOR_TAB_HOVERED]               = (0.18, 0.18, 0.18, 1.0)
    colors[imgui.COLOR_TAB_ACTIVE]                = (0.16, 0.16, 0.16, 1.0)

    colors[imgui.COLOR_PLOT_LINES]                = (0.50, 0.50, 0.50, 1.0)
    colors[imgui.COLOR_PLOT_HISTOGRAM]            = (0.70, 0.35, 0.15, 1.0)


class ImGuiLayer:
    """Manages ImGui lifecycle within a GLFW/OpenGL context."""

    def __init__(self, window):
        imgui.create_context()

        # Get DPI scale before creating renderer
        dpi = init_scale(window)

        self._build_fonts(dpi)

        self.impl = GlfwRenderer(window, attach_callbacks=False)
        self.window = window
        self.visible = True

        _apply_dark_flat_style()

    def _build_fonts(self, dpi: float) -> None:
        """(Re)build the font atlas at the given DPI scale."""
        io = imgui.get_io()
        io.fonts.clear()
        io.font_global_scale = 1.0

        # 1. UI font — compact, for general text
        font_path = self._find_system_font()
        if font_path:
            self.font_ui = io.fonts.add_font_from_file_ttf(
                font_path, int(15 * dpi))
        else:
            self.font_ui = io.fonts.add_font_default()
            io.font_global_scale = dpi

        # 2. Display font — large, OP-1 style geometric for big numerals
        display_path = self._find_display_font()
        if display_path:
            self.font_display = io.fonts.add_font_from_file_ttf(
                display_path, int(56 * dpi))
        else:
            self.font_display = None

    def rebuild_for_scale(self, new_dpi: float) -> None:
        """Rescale the whole GUI for a monitor change (live, per-frame safe
        when called OUTSIDE an imgui frame).

        The scale used to be frozen at window creation, so dragging the
        window from the studio display to the mini (different content
        scales) left every font and padding sized for the wrong
        monitor. Rebuilds the font atlas at the new size, re-uploads
        the texture, and re-applies the style block (its paddings are
        baked in scaled pixels).
        """
        from src.gui import scale as _scale
        _scale.set_scale(new_dpi)
        self._build_fonts(new_dpi)
        self.impl.refresh_font_texture()
        # Style paddings/rounding are absolute values computed from the
        # scale at apply time — reset to defaults isn't needed since
        # _apply_dark_flat_style writes every field it touches.
        _apply_dark_flat_style()

    def _find_system_font(self) -> str | None:
        """Find a good clean sans font for general UI text."""
        import os
        candidates = [
            os.path.expandvars(r"%WINDIR%\Fonts\segoeui.ttf"),
            os.path.expandvars(r"%WINDIR%\Fonts\consola.ttf"),
            os.path.expandvars(r"%WINDIR%\Fonts\arial.ttf"),
            # Linux (X11/Wayland) — common sans fonts across distros.
            "/usr/share/fonts/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _find_display_font(self) -> str | None:
        """Find a light geometric display font that approximates OP-1 glyphs.

        Preference order: Segoe UI Light (thin geometric, best at large sizes),
        Segoe UI Semilight, Bahnschrift (heavier), Arial.
        """
        import os
        candidates = [
            os.path.expandvars(r"%WINDIR%\Fonts\segoeuil.ttf"),     # Segoe UI Light
            os.path.expandvars(r"%WINDIR%\Fonts\segoeuisl.ttf"),    # Segoe UI Semilight
            os.path.expandvars(r"%WINDIR%\Fonts\bahnschrift.ttf"),
            os.path.expandvars(r"%WINDIR%\Fonts\arial.ttf"),
            # Linux — light/geometric display fonts, falling back to plain sans.
            "/usr/share/fonts/TTF/FiraSansCondensed-UltraLight.ttf",
            "/usr/share/fonts/noto/NotoSans-Light.ttf",
            "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def new_frame(self):
        self.impl.process_inputs()
        # Unify ImGui into FRAMEBUFFER space. The stock GlfwRenderer
        # runs ImGui in window-logical coords with display_fb_scale
        # doing the stretch at render time — which (a) upscales the
        # font atlas on fractional-scale monitors (blurry text) and
        # (b) disagrees by the scale factor with every piece of app
        # math done in framebuffer pixels (self.width/height, the GL
        # gallery, selection rects). One space for everything instead:
        # display_size = framebuffer, fb_scale = 1, mouse scaled up.
        io = imgui.get_io()
        try:
            fbw, fbh = glfw.get_framebuffer_size(self.window)
            ww, wh = glfw.get_window_size(self.window)
            if fbw > 0 and fbh > 0 and ww > 0 and wh > 0:
                io.display_size = (fbw, fbh)
                io.display_fb_scale = (1.0, 1.0)
                mx, my = io.mouse_pos
                if mx > -sys.float_info.max / 2:  # imgui's "no mouse" sentinel
                    io.mouse_pos = (mx * fbw / ww, my * fbh / wh)
        except Exception:
            pass
        imgui.new_frame()

    def render(self):
        imgui.render()
        self.impl.render(imgui.get_draw_data())

    def shutdown(self):
        self.impl.shutdown()

    def wants_mouse(self) -> bool:
        return imgui.get_io().want_capture_mouse

    def wants_keyboard(self) -> bool:
        return imgui.get_io().want_capture_keyboard

    def feed_mouse_button(self, button: int, pressed: bool):
        io = imgui.get_io()
        if button < 5:
            io.mouse_down[button] = pressed

    def feed_scroll(self, y_offset: float):
        io = imgui.get_io()
        io.mouse_wheel = y_offset

    def feed_key(self, key: int, action: int, mods: int):
        io = imgui.get_io()
        if action == glfw.PRESS:
            io.keys_down[key] = True
        elif action == glfw.RELEASE:
            io.keys_down[key] = False
        io.key_ctrl = bool(mods & glfw.MOD_CONTROL)
        io.key_shift = bool(mods & glfw.MOD_SHIFT)
        io.key_alt = bool(mods & glfw.MOD_ALT)

    def feed_char(self, char: int):
        io = imgui.get_io()
        if 0 < char < 0x10000:
            io.add_input_character(char)
