from types import SimpleNamespace
from unittest.mock import Mock

from src.core.cloud_wheel import CloudWheel
from src.core.modes import MODE_CONTACT_SHEETS, MODE_LIGHT_TABLE


def test_fractional_wheel_and_large_bursts_never_skip_clouds():
    wheel = CloudWheel()
    assert [wheel.step(0.25, t) for t in (0, .01, .02, .03)] == [0, 0, 0, 1]
    assert wheel.step(120, .05) == 0
    assert wheel.step(120, .18) == 1
    assert wheel.step(-120, .4) == -1


def test_partial_gestures_expire_and_reverse_without_stored_motion():
    wheel = CloudWheel()
    assert wheel.step(.75, 0) == 0
    assert wheel.step(-.25, .1) == 0
    assert wheel.step(-.75, .2) == -1
    assert wheel.step(.75, 1) == 0
    assert wheel.step(.25, 2) == 0
    assert wheel.step(float('nan'), 3) == 0
    wheel.reset()
    assert wheel.step(1, 3) == 1


def _app(mode=MODE_LIGHT_TABLE):
    from src.main import App
    app = App.__new__(App)
    app.gui = SimpleNamespace(feed_scroll=Mock(), wants_mouse=lambda: False)
    app._cloud_wheel = CloudWheel()
    app.thumb_wheel_navigation = True
    app.thumb_wheel_reverse = False
    app.mode = mode
    app._lmb_held = app._rmb_held = app._mmb_held = False
    app._radial_menu_active = app._gallery_scrollbar_dragging = False
    app.sequence = None
    app._navigate_cloud = Mock()
    app._seek_sequence = Mock()
    return app


def test_horizontal_event_uses_cloud_navigation_without_zoom_or_paint():
    app = _app()
    app._on_scroll(None, 1, 0)
    app._navigate_cloud.assert_called_once_with(1)
    app._cloud_wheel.reset()
    app.thumb_wheel_reverse = True
    app._on_scroll(None, 1, 0)
    assert app._navigate_cloud.call_args.args == (-1,)


def test_horizontal_event_never_switches_during_drag_or_gui_capture():
    app = _app()
    app._lmb_held = True
    app._on_scroll(None, 1, 0)
    app._lmb_held = False
    app.gui.wants_mouse = lambda: True
    app._on_scroll(None, 1, 0)
    app._navigate_cloud.assert_not_called()


def test_gallery_and_sequences_follow_existing_navigation():
    app = _app(MODE_CONTACT_SHEETS)
    app._on_scroll(None, -1, 0)
    app._navigate_cloud.assert_called_once_with(-1)
    app._cloud_wheel.reset()
    app.sequence = SimpleNamespace(current_index=5)
    app._on_scroll(None, 1, 0)
    app._seek_sequence.assert_called_once_with(6)


def test_vertical_wheel_keeps_gallery_scroll(monkeypatch):
    import glfw
    app = _app(MODE_CONTACT_SHEETS)
    app._gallery_scroll_by = Mock()
    monkeypatch.setattr(glfw, 'get_key', lambda *args: glfw.RELEASE)
    app._on_scroll(None, 0, -2)
    app._gallery_scroll_by.assert_called_once_with(-2)
    app._navigate_cloud.assert_not_called()
