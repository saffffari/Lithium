"""Mixed Linux display environments must use the context GLFW created."""
import pytest
from src.core import gl_platform as platform

@pytest.mark.parametrize('api,window_platform,expected', [
    (platform.glfw.NATIVE_CONTEXT_API,platform.glfw.PLATFORM_X11,'glx'),
    (platform.glfw.NATIVE_CONTEXT_API,platform.glfw.PLATFORM_WAYLAND,'egl'),
    (platform.glfw.EGL_CONTEXT_API,platform.glfw.PLATFORM_X11,'egl'),
])
def test_match_actual_context_despite_both_display_variables(monkeypatch,api,window_platform,expected):
    monkeypatch.setattr(platform.sys,'platform','linux')
    monkeypatch.delenv('PYOPENGL_PLATFORM',raising=False)
    monkeypatch.setenv('DISPLAY',':0');monkeypatch.setenv('WAYLAND_DISPLAY','wayland-0')
    monkeypatch.setattr(platform.glfw,'get_window_attrib',lambda *a:api)
    monkeypatch.setattr(platform.glfw,'get_platform',lambda:window_platform)
    platform.configure_opengl_platform(None)
    assert platform.os.environ['PYOPENGL_PLATFORM']==expected

def test_explicit_backend_override_is_preserved(monkeypatch):
    monkeypatch.setattr(platform.sys,'platform','linux')
    monkeypatch.setenv('PYOPENGL_PLATFORM','egl')
    platform.configure_opengl_platform(None)
    assert platform.os.environ['PYOPENGL_PLATFORM']=='egl'
