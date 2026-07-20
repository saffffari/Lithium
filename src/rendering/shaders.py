"""Shader loading, compilation, and a tiny live-reload registry.

The registry lets the app rebuild every registered program from disk on a
single keypress during demo crunch. Anything that wants to participate in
hot-reload must be created through ``register_program`` (or a helper that
calls it) and referenced via the returned handle/attribute — never by
caching the ``moderngl.Program`` object directly in a place that won't
be re-assigned after reload.
"""

import os
import sys
from typing import Callable

import moderngl


def _shader_dir() -> str:
    """Get the shaders directory, handling PyInstaller frozen apps."""
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'shaders')
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), 'shaders')


def load_shader(filename: str) -> str:
    """Load a shader source file from the shaders directory."""
    path = os.path.join(_shader_dir(), filename)
    with open(path, 'r') as f:
        return f.read()


def load_post_program(ctx: moderngl.Context, vert_filename: str, frag_filename: str) -> moderngl.Program:
    """Compile a simple vert+frag program for a fullscreen postprocess pass."""
    return ctx.program(
        vertex_shader=load_shader(vert_filename),
        fragment_shader=load_shader(frag_filename),
    )


def create_point_cloud_program(ctx: moderngl.Context) -> moderngl.Program:
    """Compile and link the point cloud shader program."""
    try:
        prog = ctx.program(
            vertex_shader=load_shader('point_cloud.vert'),
            fragment_shader=load_shader('point_cloud.frag'),
        )
    except Exception as e:
        raise RuntimeError(f"Shader compilation failed: {e}") from e

    # Set safe defaults for color adjustment uniforms
    prog['u_brightness'].value = 0.0
    prog['u_contrast'].value = 1.0
    prog['u_saturation'].value = 1.0
    prog['u_rgb_gain'].value = (1.0, 1.0, 1.0)
    prog['u_label_blend'].value = 0.0
    # Default clip AABB: huge extents = no clipping
    prog['u_clip_min'].value = (-1e9, -1e9, -1e9)
    prog['u_clip_max'].value = (1e9, 1e9, 1e9)
    # DOF defaults: disabled (strength = 0), focus at a safe
    # distance. draw_cloud overrides both per frame when enabled.
    prog['u_dof_strength'].value = 0.0
    prog['u_dof_focus_dist'].value = 1.0

    return prog


def get_uniform_safe(prog: moderngl.Program, name: str):
    """Return a uniform by name or None if it doesn't exist on this program."""
    try:
        return prog[name]
    except KeyError:
        return None


# ---------------------------------------------------------------------------
# Hot-reload registry
# ---------------------------------------------------------------------------

# name -> (current object, factory that rebuilds it from disk)
# The "object" is either a moderngl.Program or a moderngl.ComputeShader.
_PROGRAM_REGISTRY: dict[str, tuple[object, Callable[[moderngl.Context], object]]] = {}


def register_program(name: str, factory: Callable[[moderngl.Context], object], ctx: moderngl.Context):
    """Build the program via ``factory(ctx)``, store it, and return it.

    Use the same ``name`` on subsequent reloads. The factory must be a pure
    function of the context — it gets re-called on every reload.
    """
    obj = factory(ctx)
    _PROGRAM_REGISTRY[name] = (obj, factory)
    return obj


def get_registered(name: str):
    """Fetch the current object for a previously registered program."""
    entry = _PROGRAM_REGISTRY.get(name)
    return entry[0] if entry else None


def reload_all(ctx: moderngl.Context) -> tuple[dict[str, object], list[str]]:
    """Rebuild every registered program from disk.

    Returns ``(new_objects, failed_names)``. ``new_objects`` maps name to the
    new program/compute shader so the caller can rewire any attribute
    references. On failure we keep the previous object in place for that
    name so rendering stays alive.
    """
    new_objs: dict[str, object] = {}
    failed: list[str] = []
    for name, (old, factory) in list(_PROGRAM_REGISTRY.items()):
        try:
            obj = factory(ctx)
        except Exception as e:
            print(f"[shader reload] {name}: {e}")
            failed.append(name)
            continue
        _PROGRAM_REGISTRY[name] = (obj, factory)
        new_objs[name] = obj
        # Best-effort release of the old object after the swap
        try:
            if hasattr(old, 'release'):
                old.release()
        except Exception:
            pass
    return new_objs, failed
