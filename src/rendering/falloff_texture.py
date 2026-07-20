"""Falloff LUT texture for point cloud rendering.

Bakes an Envelope into a 256x1 single-channel float texture that the
fragment shader samples by distance from the point center.
"""

import numpy as np
import moderngl

from src.core.envelope import Envelope


def create_falloff_texture(ctx: moderngl.Context, envelope: Envelope,
                           size: int = 256) -> moderngl.Texture:
    """Create a 256x1 single-channel float texture from an Envelope."""
    lut = envelope.to_lut(size)
    tex = ctx.texture((size, 1), 1, lut.tobytes(), dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


def update_falloff_texture(tex: moderngl.Texture, envelope: Envelope):
    """Re-upload the LUT after the envelope changes."""
    lut = envelope.to_lut(tex.size[0])
    tex.write(lut.tobytes())
