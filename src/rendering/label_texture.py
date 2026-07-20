"""Label color lookup texture for GPU point rendering.

Packs the LabelRegistry's color palette into a 256x1 RGBA float texture
that the fragment shader samples via texelFetch(label_id).
"""

import numpy as np
import moderngl

from src.data.labels import LabelRegistry


def create_label_color_texture(ctx: moderngl.Context,
                                registry: LabelRegistry) -> moderngl.Texture:
    """Create a 256x1 RGBA float texture from a LabelRegistry."""
    lut = registry.get_color_lut()  # (256, 4) float32
    tex = ctx.texture((256, 1), 4, lut.tobytes(), dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def update_label_color_texture(tex: moderngl.Texture,
                                registry: LabelRegistry):
    """Re-upload the LUT data after label colors or visibility change."""
    lut = registry.get_color_lut()
    tex.write(lut.tobytes())
