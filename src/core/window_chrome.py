"""A Windows client title bar with native dragging and resize borders."""
from __future__ import annotations

import sys


class WindowsChrome:
    def __init__(self, window):
        import ctypes as ct
        from ctypes import wintypes as wt
        import glfw

        self.ct = ct
        self.user = ct.WinDLL('user32', use_last_error=True)
        self.hwnd = glfw.get_win32_window(window)
        self.header_height = 30
        self.menu_right = 210
        self.controls_left = 100000
        self.frame_width = 1
        self.old_proc = None
        self.user.CallWindowProcW.argtypes = [ct.c_void_p, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
        self.user.CallWindowProcW.restype = ct.c_ssize_t
        self.user.SetWindowLongPtrW.argtypes = [wt.HWND, ct.c_int, ct.c_void_p]
        self.user.SetWindowLongPtrW.restype = ct.c_void_p
        self.user.IsZoomed.argtypes = [wt.HWND]
        self.user.GetWindowRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
        self.user.GetClientRect.argtypes = [wt.HWND, ct.POINTER(wt.RECT)]
        self.user.MonitorFromWindow.argtypes = [wt.HWND, wt.DWORD]
        self.user.MonitorFromWindow.restype = wt.HMONITOR

        class MonitorInfo(ct.Structure):
            _fields_ = [('size', wt.DWORD), ('monitor', wt.RECT), ('work', wt.RECT), ('flags', wt.DWORD)]

        self.user.GetMonitorInfoW.argtypes = [wt.HMONITOR, ct.POINTER(MonitorInfo)]
        callback = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

        def proc(hwnd, message, wparam, lparam):
            try:
                if message == 0x0083 and wparam:  # WM_NCCALCSIZE
                    if self.user.IsZoomed(hwnd):
                        info = MonitorInfo()
                        info.size = ct.sizeof(info)
                        if self.user.GetMonitorInfoW(self.user.MonitorFromWindow(hwnd, 2), ct.byref(info)):
                            # The first member of NCCALCSIZE_PARAMS is its client RECT.
                            rect = ct.cast(lparam, ct.POINTER(wt.RECT)).contents
                            rect.left, rect.top = info.work.left, info.work.top
                            rect.right, rect.bottom = info.work.right, info.work.bottom
                    return 0
                if message == 0x0084:  # WM_NCHITTEST, signed screen coordinates
                    x, y = ct.c_short(lparam & 0xffff).value, ct.c_short((lparam >> 16) & 0xffff).value
                    rect = wt.RECT()
                    self.user.GetWindowRect(hwnd, ct.byref(rect))
                    x, y = x - rect.left, y - rect.top
                    width, height = rect.right - rect.left, rect.bottom - rect.top
                    scale = width / max(1, self.frame_width)
                    border = max(4, round(6 * scale))
                    if not self.user.IsZoomed(hwnd):
                        left, right = x < border, x >= width - border
                        top, bottom = y < border, y >= height - border
                        if top: return 13 if left else 14 if right else 12
                        if bottom: return 16 if left else 17 if right else 15
                        if left: return 10
                        if right: return 11
                    if (0 <= y < self.header_height * scale
                            and self.menu_right * scale <= x < self.controls_left * scale):
                        return 2  # HTCAPTION: native move, double-click maximize, system menu
                    return 1  # HTCLIENT: menus and custom window controls
            except Exception:
                pass
            return self.user.CallWindowProcW(self.old_proc, hwnd, message, wparam, lparam)

        self.proc = callback(proc)  # Retain the callback until the window is destroyed.
        self.old_proc = self.user.SetWindowLongPtrW(self.hwnd, -4, ct.cast(self.proc, ct.c_void_p))
        if not self.old_proc:
            raise ct.WinError(ct.get_last_error())
        self.user.SetWindowPos.argtypes = [wt.HWND, wt.HWND, ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.UINT]
        self.user.SetWindowPos(self.hwnd, None, 0, 0, 0, 0, 0x0037)  # FRAMECHANGED, NOMOVE, NOSIZE, NOZORDER, NOACTIVATE

    def update(self, height, menu_right, controls_left, frame_width):
        self.header_height, self.menu_right = height, menu_right
        self.controls_left, self.frame_width = controls_left, frame_width

    def close(self):
        if self.old_proc:
            self.user.SetWindowLongPtrW(self.hwnd, -4, self.old_proc)
            self.old_proc = None


def install(window):
    if sys.platform != 'win32':
        return None
    return WindowsChrome(window)
