"""Window controls rendered in the same row as the application menus."""
import glfw
import imgui

from src.gui.scale import s


def draw_brand(app, bar_height):
    """Logo and fixed product name precede the menus at the left edge."""
    if getattr(app, '_window_chrome', None) is None:
        return
    import sys
    from pathlib import Path
    from PIL import Image
    texture = getattr(app, '_header_logo', None)
    if texture is None:
        root = Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parents[2]))
        with Image.open(root / 'assets' / 'lithium.png') as im:
            im = im.convert('RGBA')
            texture = app.ctx.texture(im.size, 4, im.tobytes())
        app._header_logo = texture
    x, y = imgui.get_window_position()
    size = max(s(12), bar_height - s(4))
    dl = imgui.get_window_draw_list()
    dl.add_image(texture.glo, (x + s(2), y + s(2)),
                 (x + s(2) + size, y + s(2) + size))
    text_x = x + size + s(8)
    dl.add_text(text_x, y + (bar_height - imgui.get_text_line_height()) * .5,
                imgui.get_color_u32_rgba(.8, .8, .8, 1), 'LITHIUM')
    imgui.set_cursor_screen_pos((text_x + imgui.calc_text_size('LITHIUM')[0] + s(20),
                                 imgui.get_cursor_screen_pos()[1]))


def draw_header(app, bar_height):
    chrome = getattr(app, '_window_chrome', None)
    if chrome is None:
        return
    width = imgui.get_window_width()
    origin_x, origin_y = imgui.get_window_position()
    menu_right = imgui.get_cursor_screen_pos()[0] - origin_x + s(12)
    button_width = s(44)
    controls_left = width - button_width * 3
    dl = imgui.get_window_draw_list()
    for i, name in enumerate(('minimize', 'maximize', 'close')):
        x = origin_x + controls_left + i * button_width
        imgui.set_cursor_screen_pos((x, origin_y))
        clicked = imgui.invisible_button('##window_' + name, button_width, bar_height)
        if imgui.is_item_hovered():
            color = (0.65, 0.15, 0.13, 1) if name == 'close' else (0.17, 0.17, 0.17, 1)
            dl.add_rect_filled(x, origin_y, x + button_width, origin_y + bar_height,
                               imgui.get_color_u32_rgba(*color))
            imgui.set_tooltip({'minimize': 'Minimize', 'maximize': 'Restore / Maximize', 'close': 'Close'}[name])
        cx, cy = x + button_width * 0.5, origin_y + bar_height * 0.5
        r = s(4)
        ink = imgui.get_color_u32_rgba(0.8, 0.8, 0.8, 1)
        if name == 'minimize':
            dl.add_line(cx-r, cy, cx+r, cy, ink, s(1))
        elif name == 'maximize':
            if glfw.get_window_attrib(app.window, glfw.MAXIMIZED):
                dl.add_rect(cx-r+s(2), cy-r-s(2), cx+r+s(2), cy+r-s(2), ink)
            dl.add_rect(cx-r, cy-r, cx+r, cy+r, ink)
        else:
            dl.add_line(cx-r, cy-r, cx+r, cy+r, ink, s(1))
            dl.add_line(cx+r, cy-r, cx-r, cy+r, ink, s(1))
        if clicked:
            if name == 'minimize': glfw.iconify_window(app.window)
            elif name == 'close': glfw.set_window_should_close(app.window, True)
            elif glfw.get_window_attrib(app.window, glfw.MAXIMIZED): glfw.restore_window(app.window)
            else: glfw.maximize_window(app.window)
    chrome.update(bar_height, menu_right, controls_left, width)
