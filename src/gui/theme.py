"""Lithium color palette and theme constants — OP-1 inspired, warm bias."""

# Accent colors (saturated on dark background)
CYAN = (0.0, 0.85, 0.85, 1.0)
TEAL = (0.15, 0.75, 0.65, 1.0)
GREEN = (0.2, 0.9, 0.4, 1.0)
BLUE = (0.35, 0.55, 0.95, 1.0)
LAVENDER = (0.6, 0.5, 0.9, 1.0)
CORAL = (0.95, 0.4, 0.3, 1.0)
PINK = (0.9, 0.35, 0.55, 1.0)
ORANGE = (0.95, 0.6, 0.15, 1.0)
RED = (0.95, 0.25, 0.25, 1.0)
# Saturated, eye-searing red — for measurements/alerts that need to
# read as RED, not coral/pink. Use sparingly; the default RED above
# is the right call when red is *one* of many accent colours.
VIVID_RED = (1.0, 0.08, 0.06, 1.0)
AMBER = (1.0, 0.75, 0.2, 1.0)
WARM_WHITE = (0.92, 0.88, 0.82, 1.0)
WHITE = (0.85, 0.85, 0.85, 1.0)
DIM = (0.4, 0.4, 0.4, 1.0)
FAINT = (0.25, 0.25, 0.25, 1.0)
# Background / panel scheme (recessed nested wells, per design call):
#   background  90% K  →  0.10  (window + main canvas backdrop)
#   panel       95% K  →  0.05  (sidebar groups, knob rows, label list,
#                                 CLI input — visibly DARKER than the
#                                 background so they read as wells the
#                                 content sits inside, not raised cards)
BG_PANEL = (0.05, 0.05, 0.05, 1.0)
BG_CELL = (0.05, 0.05, 0.05, 1.0)
BG_DARK = (0.05, 0.05, 0.05, 1.0)  # alias retained for legacy call sites

# OP-1 exact colors from Figma (OP-1 Screen Graphics kit)
OP1_BG = (0.10, 0.10, 0.10, 1.0)         # 90% K window backdrop
OP1_BLUE = (0.388, 0.545, 1.0, 1.0)      # #638BFF
OP1_GREEN = (0.0, 0.929, 0.584, 1.0)     # #00ED95
OP1_WHITE = (0.996, 1.0, 1.0, 1.0)       # #FEFFFF
OP1_RED = (1.0, 0.31, 0.22, 1.0)         # #FF4F38 — orange side of red, never magenta
OP1_ORANGE = (0.976, 0.573, 0.141, 1.0)  # #F99224
OP1_DIM = (0.28, 0.28, 0.28, 1.0)        # flat mid grey (was purple-tinted)
OP1_GRAY = (0.46, 0.46, 0.46, 1.0)       # flat light grey (was purple-tinted)

# Functional color mapping — warm bias, less purple/blue
COLOR_HEADER = CORAL
COLOR_SECTION = DIM
COLOR_VALUE = WHITE
COLOR_ACTIVE = ORANGE
COLOR_ACCENT = ORANGE
COLOR_X_AXIS = CORAL
COLOR_Y_AXIS = GREEN
COLOR_Z_AXIS = TEAL
COLOR_GALLERY = AMBER
COLOR_EXPORT = CORAL
COLOR_CAMERA = ORANGE

def im_col(c):
    """Convert (r,g,b,a) tuple to imgui-compatible color args."""
    return c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0

def im_u32(r, g, b, a=1.0):
    """Convert float RGBA to packed u32 for draw list."""
    return (int(a * 255) << 24) | (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255)

def col32(color):
    """Convert tuple to draw list u32."""
    return im_u32(*color[:4] if len(color) >= 4 else (*color[:3], 1.0))
