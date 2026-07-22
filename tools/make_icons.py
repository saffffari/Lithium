#!/usr/bin/env python
"""Generate the Lithium isotope icon set.

Mark: three hairline elliptical orbits at 0/60/120 degrees around a
warm-white nucleus, one photon per orbit in the app's accent colors
(OP-1 orange / teal / coral), each with a short comet tail — the same
visual grammar as the Light Table INFER constellation loader.

Two variants:
  lithium_isotope      mark only (window icon, dock, .ico)
  lithium_isotope_Li   mark + "Li" wordmark (splash / about / site)
"""

import math
import subprocess
from pathlib import Path

OUT = Path("/home/alex/icons/out/Lithium")
OUT.mkdir(parents=True, exist_ok=True)

BG = "#0B0B0C"
ORBIT = "#3A3A3E"
NUCLEUS = "#EBE0D1"          # WARM_WHITE
ACCENTS = ["#F99224",        # OP1_ORANGE
           "#26BFA6",        # TEAL
           "#F26650"]        # CORAL

CX = CY = 256.0
RX, RY = 172.0, 66.0
ORBIT_W = 7.0
PHOTON_R = 17.0
TAIL_DEG = 34.0

# Photon parametric angles per orbit — spread so no two crowd a
# crossing, asymmetric enough to feel in-motion.
PHOTON_T = [35.0, 170.0, 320.0]
ROTS = [0.0, 60.0, 120.0]


def orbit_point(t_deg, rot_deg):
    t = math.radians(t_deg)
    r = math.radians(rot_deg)
    x, y = RX * math.cos(t), RY * math.sin(t)
    return (CX + x * math.cos(r) - y * math.sin(r),
            CY + x * math.sin(r) + y * math.cos(r))


def tail_path(t_deg, rot_deg, span_deg, steps=14):
    pts = [orbit_point(t_deg - k * span_deg / steps, rot_deg)
           for k in range(steps + 1)]
    d = f"M {pts[0][0]:.2f} {pts[0][1]:.2f} " + " ".join(
        f"L {p[0]:.2f} {p[1]:.2f}" for p in pts[1:])
    return d


def build_svg(with_wordmark: bool, small: bool = False) -> str:
    """``small=True`` = bold variant for <=64 px rasters: thick orbits,
    fat photons, no tails — hairlines vanish below ~1 device px."""
    orbit_w = 16.0 if small else ORBIT_W
    photon_r = 30.0 if small else PHOTON_R
    nucleus_r = 36.0 if small else 24.0
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">',
        f'<rect x="0" y="0" width="512" height="512" rx="96" fill="{BG}"/>',
    ]
    scale = 0.78 if with_wordmark else 1.0
    ty = -46.0 if with_wordmark else 0.0
    parts.append(f'<g transform="translate({CX * (1 - scale)},'
                 f'{CY * (1 - scale) + ty}) scale({scale})">')
    orbit_col = "#4A4A50" if small else ORBIT
    for rot in ROTS:
        parts.append(
            f'<ellipse cx="{CX}" cy="{CY}" rx="{RX}" ry="{RY}" '
            f'fill="none" stroke="{orbit_col}" stroke-width="{orbit_w}" '
            f'transform="rotate({rot} {CX} {CY})"/>')
    parts.append(
        f'<circle cx="{CX}" cy="{CY}" r="{nucleus_r}" fill="{NUCLEUS}"/>')
    for (t, rot, col) in zip(PHOTON_T, ROTS, ACCENTS):
        if not small:
            parts.append(
                f'<path d="{tail_path(t, rot, TAIL_DEG)}" fill="none" '
                f'stroke="{col}" stroke-width="{ORBIT_W + 1.5}" '
                f'stroke-linecap="round" opacity="0.42"/>')
        px, py = orbit_point(t, rot)
        parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" '
                     f'r="{photon_r}" fill="{col}"/>')
    parts.append('</g>')
    if with_wordmark:
        parts.append(
            f'<text x="{CX}" y="436" font-family="Inter, DejaVu Sans, '
            f'sans-serif" font-weight="800" font-size="132" '
            f'text-anchor="middle" fill="{NUCLEUS}" '
            f'letter-spacing="4">Li</text>')
    parts.append('</svg>')
    return "\n".join(parts)


for name, wordmark in (("lithium_isotope", False),
                       ("lithium_isotope_Li", True)):
    svg_path = OUT / f"{name}.svg"
    svg_path.write_text(build_svg(wordmark))
    small_svg = OUT / f"{name}_small.svg"
    small_svg.write_text(build_svg(wordmark, small=True))
    sizes = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
    for s in sizes:
        src = small_svg if s <= 64 else svg_path
        subprocess.run(["rsvg-convert", "-w", str(s), "-h", str(s),
                        str(src), "-o", str(OUT / f"{name}_{s}.png")],
                       check=True)
    # Windows .ico: bundle the small-to-medium set.
    subprocess.run(["magick"] +
                   [str(OUT / f"{name}_{s}.png")
                    for s in (16, 24, 32, 48, 64, 128, 256)] +
                   [str(OUT / f"{name}.ico")], check=True)
    print(f"{name}: svg + {len(sizes)} png + ico")
print(f"wrote {len(list(OUT.iterdir()))} files -> {OUT}")
