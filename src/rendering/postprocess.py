"""Postprocess helpers: fullscreen-triangle passes and the main chain.

``FullscreenPass`` owns one shader program and a single-triangle VAO
covering clip space. You bind whichever ``RenderTarget`` (or the screen)
you want to write to, then call ``render()`` with uniforms and textures.

The pass plays nicely with hot-reload: programs are created via
``register_program`` so ``shaders.reload_all(ctx)`` rebuilds them in
place. After reload, the app rebinds ``pass.prog`` to the new program
via ``_refresh_from_registry``.
"""

from typing import Callable, Dict, Iterable

import moderngl
import numpy as np

from src.rendering.shaders import (
    load_post_program,
    register_program,
    get_registered,
)


# A 3-vertex triangle that covers the clip-space square [-1,1]^2.
#   v0 = (-1, -1)    v1 = ( 3, -1)    v2 = (-1,  3)
# When rasterized and clipped, this yields the full screen with one
# triangle (cheaper than two and avoids the diagonal seam).
_FULLSCREEN_TRI = np.array([
    -1.0, -1.0,
     3.0, -1.0,
    -1.0,  3.0,
], dtype='f4')


class FullscreenPass:
    def __init__(
        self,
        ctx: moderngl.Context,
        name: str,
        frag_filename: str,
        vert_filename: str = 'post_fullscreen.vert',
    ):
        self.ctx = ctx
        self.name = name
        self._vert = vert_filename
        self._frag = frag_filename
        self.prog = register_program(
            name,
            lambda c: load_post_program(c, vert_filename, frag_filename),
            ctx,
        )
        self.vbo = ctx.buffer(_FULLSCREEN_TRI.tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_pos')])

    def refresh_from_registry(self) -> None:
        """Rebind VAO to the latest registered program (after hot reload)."""
        new_prog = get_registered(self.name)
        if new_prog is None or new_prog is self.prog:
            return
        try:
            self.vao.release()
        except Exception:
            pass
        self.prog = new_prog
        self.vao = self.ctx.vertex_array(self.prog, [(self.vbo, '2f', 'in_pos')])

    def render(
        self,
        uniforms: Dict[str, object] | None = None,
        textures: Dict[int, moderngl.Texture] | None = None,
    ) -> None:
        if textures:
            for loc, tex in textures.items():
                if tex is not None:
                    tex.use(location=loc)
        if uniforms:
            for name, value in uniforms.items():
                try:
                    u = self.prog[name]
                except KeyError:
                    continue
                if isinstance(value, np.ndarray):
                    if value.ndim == 0:
                        u.value = float(value)
                    else:
                        u.write(value.astype('f4', copy=False).tobytes())
                elif isinstance(value, (tuple, list)):
                    u.value = tuple(value)
                else:
                    u.value = value
        self.vao.render(moderngl.TRIANGLES, vertices=3)

    def release(self) -> None:
        try:
            self.vao.release()
        except Exception:
            pass
        try:
            self.vbo.release()
        except Exception:
            pass
