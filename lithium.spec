# -*- mode: python ; coding: utf-8 -*-
# Cross-platform PyInstaller spec (Windows + Linux + macOS).
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# Let PyInstaller's collector locate the glfw native lib for the current
# platform (glfw3.dll on Windows, libglfw.so on Linux) instead of a
# hardcoded path.
glfw_binaries = collect_dynamic_libs("glfw")

# OpenGL's platform backend is chosen dynamically at runtime — bundle
# every candidate for the OS (Linux picks glx under X11, egl under
# Wayland), plus the array-format helpers PyOpenGL lazy-imports.
if IS_WIN:
    gl_platforms = ["OpenGL.platform.win32"]
elif IS_MAC:
    gl_platforms = ["OpenGL.platform.darwin"]
else:
    gl_platforms = ["OpenGL.platform.glx", "OpenGL.platform.egl",
                    "OpenGL.platform.osmesa"]
gl_arrays = [
    "OpenGL.arrays.ctypesarrays", "OpenGL.arrays.ctypesparameters",
    "OpenGL.arrays.ctypespointers", "OpenGL.arrays.lists",
    "OpenGL.arrays.numbers", "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.strings", "OpenGL.arrays.nones",
]

# .ico on Windows, .icns on macOS; Linux builds ship without an embedded icon.
app_icon = "assets/lithium.ico" if IS_WIN else ("assets/lithium.icns" if IS_MAC else None)

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
        'OpenGL', 'OpenGL.GL', 'OpenGL.platform', *gl_platforms, *gl_arrays,
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
    name='Lithium',
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

if IS_MAC:
    # A double-clickable bundle. Unsigned until there is an Apple Developer ID:
    # first launch needs right-click -> Open (or xattr -dr com.apple.quarantine).
    app = BUNDLE(
        exe,
        name='Lithium.app',
        icon=app_icon,
        bundle_identifier='io.2photon.lithium',
        info_plist={
            'CFBundleDisplayName': 'Lithium',
            'CFBundleShortVersionString': '1.1',
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '12.0',
        },
    )
