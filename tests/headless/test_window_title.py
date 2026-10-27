"""App._update_fps builds the window title once a second.

It runs after _render_frame and before swap_buffers, so an exception here
drops a frame every second and is invisible in the packaged .exe.
"""

from __future__ import annotations

import time

import src.main as main_module
from src.core.modes import MODE_LIGHT_TABLE


def test_update_fps_sets_title(monkeypatch):
    titles = []
    monkeypatch.setattr(main_module.glfw, "set_window_title",
                        lambda window, title: titles.append(title))

    class _Stub:
        _update_fps = main_module.App._update_fps

    s = _Stub()
    s.window = object()
    s.entries = []
    s.catalog = None
    s.mode = MODE_LIGHT_TABLE
    s.fps = 0.0
    s._fps_frame_count = 60
    s.last_fps_time = time.time() - 1.0

    s._update_fps()

    assert titles and "Light Table" in titles[0]
    assert s._fps_frame_count == 0
