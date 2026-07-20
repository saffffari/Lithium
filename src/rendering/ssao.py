"""SSAO kernel + noise generation utilities.

Called once at app init to build the hemisphere sample kernel and the
4×4 random-rotation noise texture used by ``shaders/ssao.frag``. The
kernel is uploaded as a uniform array; the noise is uploaded as a
GL_REPEAT 2D texture so it tiles across the screen.

Kept separate from main.py so the math is unit-testable and the init
path stays readable.
"""

from __future__ import annotations

import numpy as np


SSAO_KERNEL_SIZE = 32


def generate_kernel(n: int = SSAO_KERNEL_SIZE,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Hemisphere sample kernel for SSAO.

    Returns ``(n, 3)`` float32 array of vectors in the +Z hemisphere.
    Magnitudes are weighted toward zero so most samples cluster near
    the fragment — this matches the falloff of real ambient occlusion,
    where nearby occluders dominate.

    Each sample is unit-normalised then scaled by a quadratic ramp
    from 0.1 to 1.0; with 32 samples the smallest is at 10% radius
    and the largest at full radius, biased toward small radii.
    """
    if rng is None:
        rng = np.random.default_rng(seed=20260514)
    out = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        # Random direction in the +Z half-space.
        v = np.array([
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(0.0, 1.0),
        ], dtype=np.float32)
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-6:
            v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            v = v / v_norm
        # Quadratic ramp: t = i/n in [0, 1), scale = lerp(0.1, 1.0, t²).
        # Heavier weight on near-origin samples reproduces the AO
        # falloff curve you'd expect from real light integration.
        t = i / max(n - 1, 1)
        scale = 0.1 + 0.9 * t * t
        v = v * scale
        out[i] = v
    return out


def generate_noise(size: int = 4,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """4×4 (or NxN) noise texture of random tangent-plane rotation vectors.

    Returns ``(size, size, 3)`` float32. Each pixel is a unit vector in
    the XY plane (z=0) used by the SSAO shader to rotate the hemisphere
    kernel per-fragment. Tiling with GL_REPEAT across the screen breaks
    the regular kernel pattern; the AO blur pass denoises the result.
    """
    if rng is None:
        rng = np.random.default_rng(seed=20260515)
    out = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            v = np.array([
                rng.uniform(-1.0, 1.0),
                rng.uniform(-1.0, 1.0),
                0.0,
            ], dtype=np.float32)
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                v = v / n
            out[y, x] = v
    return out
