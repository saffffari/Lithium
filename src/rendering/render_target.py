"""HDR offscreen render target.

Wraps a ``moderngl.Framebuffer`` with a float16 color attachment (so
values > 1.0 survive for bloom and tonemapping) and an optional depth
texture. Resizable — the GLFW framebuffer-size callback should call
``resize()`` on every target it owns.
"""

import moderngl


class RenderTarget:
    def __init__(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
        components: int = 4,
        color_dtype: str = 'f2',     # RGBA16F
        with_depth: bool = True,
    ):
        self.ctx = ctx
        self.components = components
        self.color_dtype = color_dtype
        self.with_depth = with_depth
        self.color: moderngl.Texture | None = None
        self.depth: moderngl.Texture | None = None
        self.fbo: moderngl.Framebuffer | None = None
        self.width = 0
        self.height = 0
        self._alloc(max(1, int(width)), max(1, int(height)))

    # --- lifecycle ----------------------------------------------------------

    def _alloc(self, w: int, h: int) -> None:
        self.width = w
        self.height = h
        self.color = self.ctx.texture((w, h), self.components, dtype=self.color_dtype)
        self.color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.color.repeat_x = False
        self.color.repeat_y = False

        if self.with_depth:
            self.depth = self.ctx.depth_texture((w, h))
            self.depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.depth.repeat_x = False
            self.depth.repeat_y = False
            # Sample as raw depth (no PCF comparison)
            try:
                self.depth.compare_func = ''
            except Exception:
                pass
            self.fbo = self.ctx.framebuffer(
                color_attachments=[self.color],
                depth_attachment=self.depth,
            )
        else:
            self.fbo = self.ctx.framebuffer(color_attachments=[self.color])

    def resize(self, w: int, h: int) -> None:
        w = max(1, int(w))
        h = max(1, int(h))
        if (w, h) == (self.width, self.height):
            return
        self.release()
        self._alloc(w, h)

    def release(self) -> None:
        try:
            if self.fbo is not None:
                self.fbo.release()
        except Exception:
            pass
        try:
            if self.color is not None:
                self.color.release()
        except Exception:
            pass
        try:
            if self.depth is not None:
                self.depth.release()
        except Exception:
            pass
        self.fbo = None
        self.color = None
        self.depth = None

    # --- binding ------------------------------------------------------------

    def use(self) -> None:
        if self.fbo is not None:
            self.fbo.use()

    def clear(self, r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0, depth: float = 1.0) -> None:
        if self.fbo is not None:
            self.fbo.clear(r, g, b, a, depth=depth)
