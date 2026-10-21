"""Match PyOpenGL to the context GLFW actually created on Linux."""
import os
import sys
import glfw


def configure_opengl_platform(window):
    # PyOpenGL may infer EGL from WAYLAND_DISPLAY even when GLFW selected
    # X11/GLX. Choose after window creation but before importing PyOpenGL.
    # Preserve an explicit user override and leave WGL/Cocoa unchanged.
    if not sys.platform.startswith('linux') or os.environ.get('PYOPENGL_PLATFORM'):
        return
    api = glfw.get_window_attrib(window, glfw.CONTEXT_CREATION_API)
    platform = glfw.get_platform()
    egl = api == glfw.EGL_CONTEXT_API or platform == glfw.PLATFORM_WAYLAND
    os.environ['PYOPENGL_PLATFORM'] = 'egl' if egl else 'glx'
