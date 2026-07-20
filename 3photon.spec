# -*- mode: python ; coding: utf-8 -*-
# Cross-platform PyInstaller spec (Windows + Linux).
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs

IS_WIN = sys.platform == "win32"

# Let PyInstaller's collector locate the glfw native lib for the current
# platform (glfw3.dll on Windows, libglfw.so on Linux) instead of a
# hardcoded path.
glfw_binaries = collect_dynamic_libs("glfw")

# OpenGL's platform backend differs by OS.
gl_platform = "OpenGL.platform.win32" if IS_WIN else "OpenGL.platform.glx"

# .ico is Windows-only; Linux builds ship without an embedded icon.
app_icon = "assets/3photon.ico" if IS_WIN else None

a = Analysis(
    ['src/main.py'],
    pathex=['.'],
    binaries=glfw_binaries,
    datas=[
        ('shaders', 'shaders'),
        ('assets', 'assets'),
    ],
    hiddenimports=[
        'glfw', 'moderngl', 'glcontext',
        'imgui', 'imgui.integrations', 'imgui.integrations.glfw',
        'OpenGL', 'OpenGL.GL', 'OpenGL.platform', gl_platform,
        'laspy', 'lazrs', 'plyfile',
        'PIL', 'PIL.Image',
        'numpy', 'scipy', 'scipy.spatial',
        'tkinter', 'tkinter.filedialog',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='3Photon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=app_icon,
)
