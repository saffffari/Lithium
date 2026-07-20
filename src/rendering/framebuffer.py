"""Offscreen framebuffer for image export."""

import moderngl
import numpy as np
from PIL import Image


class ExportFramebuffer:
    """Offscreen FBO for rendering to an image."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.color = ctx.texture((width, height), 4, dtype='f1')
        self.depth = ctx.depth_renderbuffer((width, height))
        self.fbo = ctx.framebuffer(
            color_attachments=[self.color],
            depth_attachment=self.depth,
        )

    def use(self):
        self.fbo.use()

    def clear(self, r=0.0, g=0.0, b=0.0, a=1.0):
        self.fbo.clear(r, g, b, a)

    def read_image(self) -> Image.Image:
        """Read the framebuffer content and return a PIL Image."""
        raw = self.fbo.read(components=4)
        img = Image.frombytes('RGBA', (self.width, self.height), raw)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

    def release(self):
        self.color.release()
        self.depth.release()
        self.fbo.release()
