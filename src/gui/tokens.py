"""Design tokens — the single source of truth for visual decisions.

Three layers, top to bottom:

  PRIMITIVE  raw values. greys, base type sizes. no meaning. don't import.
  SEMANTIC   what a value MEANS in the product. UI code imports from here.
  COMPONENT  small per-component compositions, when a widget has enough
             states that inline references would get confusing.

References: Dieter Rams (as little design as possible — every token
must earn its place); Claude Code (typography-led, single warm accent,
dark canvas, no chrome); Teenage Engineering OP-1 (flat surfaces,
hard edges, generous space, one bright thing at a time).

Rules of use:
  - UI code references SEMANTIC or COMPONENT tokens, never PRIMITIVE.
  - No raw tuples in panel/widget code. If you reach for one, the
    token is missing — add it here first.
  - Spacing values are LOGICAL pixels; pass through `scale.s()` at the
    call site so DPI scaling stays centralised.
"""

from .theme import (
    # accent family (semantic uses these by name, not by hue)
    ORANGE, CORAL, AMBER, GREEN, TEAL, BLUE, VIVID_RED, WARM_WHITE,
    DIM, FAINT,
)


# ─────────────────────────────────────────────────────────────────────
# PRIMITIVE — raw values. do not import outside this file.
# ─────────────────────────────────────────────────────────────────────

# Neutral ramp (warm-neutral, not pure grey — matches Claude Code feel).
# Values are float 0..1 with alpha. Each step is a meaningful surface.
_GREY_00 = (0.00, 0.00, 0.00, 1.0)   # absolute black — reserved
_GREY_03 = (0.03, 0.03, 0.03, 1.0)   # void / beneath wells
_GREY_05 = (0.05, 0.05, 0.05, 1.0)   # recessed wells (panels, lists)
_GREY_07 = (0.07, 0.07, 0.07, 1.0)   # cells, code blocks
_GREY_10 = (0.10, 0.10, 0.10, 1.0)   # main canvas / window backdrop
_GREY_13 = (0.13, 0.13, 0.13, 1.0)   # hover surface
_GREY_17 = (0.17, 0.17, 0.17, 1.0)   # active / pressed surface
_GREY_22 = (0.22, 0.22, 0.22, 1.0)   # hairline dividers
_GREY_35 = (0.35, 0.35, 0.35, 1.0)   # disabled stroke
_GREY_55 = (0.55, 0.55, 0.55, 1.0)   # muted text

# Type base sizes (logical px, pre-DPI). Sizes form a small ramp; we
# never invent a new size at the call site.
_TYPE_10 = 10
_TYPE_12 = 12
_TYPE_13 = 13
_TYPE_15 = 15
_TYPE_18 = 18

# Spacing base unit. Everything multiplies this. Picked 2 because the
# UI has fine detail (chips, axis labels) that need sub-4px control.
_UNIT = 2


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — colour
# ─────────────────────────────────────────────────────────────────────

# Surfaces — read top to bottom as the visual depth stack.
BG_VOID     = _GREY_03   # deepest, behind everything
BG_CANVAS   = _GREY_10   # main window, viewport backdrop
BG_PANEL    = _GREY_05   # sidebar wells, recessed groups
BG_CELL     = _GREY_07   # list rows, code-block-like surfaces
BG_HOVER    = _GREY_13   # interactive hover
BG_ACTIVE   = _GREY_17   # pressed / selected / focused
BG_TOOLTIP  = _GREY_17   # tooltip body
BG_OVERLAY  = (0.00, 0.00, 0.00, 0.55)  # modal scrim

# Borders — used sparingly. Rams rule: only when space alone can't
# carry the separation.
BORDER_HAIRLINE = _GREY_22
BORDER_STRONG   = _GREY_35
BORDER_FOCUS    = ORANGE

# Text — hierarchy via weight of presence, not hue.
TEXT_PRIMARY   = WARM_WHITE
TEXT_SECONDARY = (0.68, 0.66, 0.62, 1.0)  # softer warm grey
TEXT_MUTED     = DIM
TEXT_DISABLED  = FAINT
TEXT_ON_ACCENT = _GREY_05                 # text sitting on a bright fill
TEXT_LINK      = ORANGE

# Accent — exactly one. Every "highlight" in the product is this colour.
# Resist adding a second. If a second is needed, it's probably a status.
ACCENT          = ORANGE
ACCENT_HOVER    = (1.00, 0.66, 0.20, 1.0)
ACCENT_PRESSED  = (0.82, 0.50, 0.10, 1.0)
ACCENT_MUTED    = (0.55, 0.36, 0.12, 1.0)  # accent on a quiet ground

# Status — used to communicate STATE, never decoration.
STATUS_OK    = GREEN
STATUS_WARN  = AMBER
STATUS_ERROR = VIVID_RED
STATUS_INFO  = BLUE

# Axis / mode tags — kept distinct from status because they label
# THINGS, not state. Existing convention; preserved.
AXIS_X = CORAL
AXIS_Y = GREEN
AXIS_Z = TEAL


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — Phase 13 token-drift consolidation (added Phase 4 Batch 10)
# ─────────────────────────────────────────────────────────────────────
# These tokens consolidate the raw RGBA tuples that crept into panels
# during the SpineLab build-out. Port sites land in Phase 13 — adding
# the names here first so the migration commits can stay focused on
# "replace this tuple with this token" diffs.

# Semi-transparent panel background for overlay floats (open dialogs,
# popouts) — sits on top of the active scene at ~85 % opacity.
BG_PANEL_SEMI       = (0.05, 0.05, 0.05, 0.85)

# Deeper than BG_PANEL — used by inset wells that should read as a
# step *below* a regular sidebar surface (HOLOGRAM measurement well,
# the bottom log area in TRAIN, etc).
BG_DEEP             = _GREY_03

# Active-state fill for input fields (text inputs, sliders, knobs).
# Slightly brighter than BG_PANEL so the focus ring around an active
# input doesn't read as a hairline pressed-against-panel border.
BG_INPUT_ACTIVE     = _GREY_10

# Dark-red ground for destructive button fills. Sits one tone darker
# than STATUS_ERROR so the button reads as "red" without yelling — the
# label/icon foreground carries the alert.
STATUS_DESTRUCTIVE_BG = (0.30, 0.06, 0.06, 1.0)

# Accent at low alpha — used for selection-row backgrounds in lists
# and gallery grids. Reads as a tinted glow over the row's BG_PANEL.
ACCENT_SELECTION_TINT = (ACCENT[0], ACCENT[1], ACCENT[2], 0.18)

# Even lower-alpha accent — used by section dividers / "active section"
# bands in the OP-1 layout where a full ACCENT fill would dominate.
ACCENT_SECTION_TINT   = (ACCENT[0], ACCENT[1], ACCENT[2], 0.10)

# Purple — reserved for label-kit / measurement glyphs that need to
# read as a distinct family from the warm accent palette. Matches the
# existing local ``_purple`` constants in panels.py + measure_panel.py
# (Phase 13 consolidates those usages onto this token).
LABEL_PURPLE = (0.70, 0.50, 0.95, 1.0)


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — spacing  (logical px; pass through scale.s() at call site)
# ─────────────────────────────────────────────────────────────────────

# Generic scale. Use these when no semantic name fits.
SPACE_0  = 0
SPACE_1  = _UNIT * 1     # 2  — hairline gap
SPACE_2  = _UNIT * 2     # 4  — tight inline
SPACE_3  = _UNIT * 3     # 6  — row vertical
SPACE_4  = _UNIT * 4     # 8  — default inline
SPACE_5  = _UNIT * 5     # 10 — panel padding
SPACE_6  = _UNIT * 7     # 14 — section gap
SPACE_7  = _UNIT * 10    # 20 — group gap
SPACE_8  = _UNIT * 14    # 28 — module gap

# Semantic aliases — prefer these in UI code.
PAD_PANEL_X  = SPACE_5
PAD_PANEL_Y  = SPACE_3
PAD_ROW_X    = SPACE_5
PAD_ROW_Y    = SPACE_2
PAD_CHIP_X   = SPACE_3
PAD_CHIP_Y   = SPACE_1
PAD_TOOLTIP  = SPACE_4

GAP_INLINE   = SPACE_2   # between sibling controls on a row
GAP_ROW      = SPACE_3   # between rows in a list
GAP_SECTION  = SPACE_6   # between titled sections
GAP_GROUP    = SPACE_7   # between top-level groups in a sidebar
GAP_MODULE   = SPACE_8   # between major regions (e.g. mode panels)


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — typography
# ─────────────────────────────────────────────────────────────────────

# Sizes (logical px). All type comes from this ramp.
FONT_TINY     = _TYPE_10   # keyboard hints, version strings, axis ticks
FONT_LABEL    = _TYPE_12   # condensed labels, small caps captions
FONT_NUMERIC  = _TYPE_13   # values, coordinates, mm/deg readouts
FONT_BODY     = _TYPE_15   # default UI text
FONT_HEADER   = _TYPE_18   # mode titles, panel headers

# Roles. Pair size + colour + casing intent in one name so call sites
# express WHAT, not HOW.
ROLE_HEADER    = (FONT_HEADER,  TEXT_PRIMARY)
ROLE_SECTION   = (FONT_LABEL,   TEXT_SECONDARY)   # uppercase at call site
ROLE_BODY      = (FONT_BODY,    TEXT_PRIMARY)
ROLE_SECONDARY = (FONT_BODY,    TEXT_SECONDARY)
ROLE_MUTED     = (FONT_BODY,    TEXT_MUTED)
ROLE_NUMERIC   = (FONT_NUMERIC, TEXT_PRIMARY)
ROLE_HINT      = (FONT_TINY,    TEXT_MUTED)


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — geometry
# ─────────────────────────────────────────────────────────────────────

# Corner radii. Rams/OP-1: hard corners by default. RADIUS_CHIP exists
# for the few cases (pills, tags) where a tiny round makes sense.
RADIUS_NONE  = 0
RADIUS_CHIP  = 2
RADIUS_PILL  = 999   # used only for circular knobs / dots

# Stroke widths.
BORDER_W_HAIRLINE = 1
BORDER_W_FOCUS    = 1   # focus is colour, not weight — keep 1px


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — opacity
# ─────────────────────────────────────────────────────────────────────

OPACITY_DISABLED = 0.38
OPACITY_MUTED    = 0.65
OPACITY_OVERLAY  = 0.85
OPACITY_GHOST    = 0.18   # ghosted preview geometry, hover-only labels


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — motion  (seconds; pair durations with eases)
# ─────────────────────────────────────────────────────────────────────

DUR_INSTANT    = 0.04   # button-down feedback
DUR_QUICK      = 0.08   # hover settle
DUR_BASE       = 0.16   # panel slide, expand/collapse
DUR_SLOW       = 0.28   # mode transition
DUR_DELIBERATE = 0.45   # one-shot reveal, splash, first-paint

# Cubic-bezier control points (x1, y1, x2, y2).
EASE_OUT        = (0.00, 0.00, 0.20, 1.00)   # default — settle
EASE_IN_OUT     = (0.40, 0.00, 0.20, 1.00)   # symmetric transition
EASE_LINEAR     = (0.00, 0.00, 1.00, 1.00)   # progress meters
EASE_STANDARD   = EASE_OUT                   # alias


# ─────────────────────────────────────────────────────────────────────
# SEMANTIC — elevation  (documentation only; ImGui handles draw order)
# ─────────────────────────────────────────────────────────────────────

Z_CANVAS  = 0
Z_PANEL   = 10
Z_POPOUT  = 30
Z_OVERLAY = 50
Z_MODAL   = 100
Z_TOAST   = 200


# ─────────────────────────────────────────────────────────────────────
# COMPONENT — only when a widget has enough states to earn it.
# ─────────────────────────────────────────────────────────────────────

# Button — neutral default, accent variant lives at the call site.
BUTTON_BG          = BG_PANEL
BUTTON_BG_HOVER    = BG_HOVER
BUTTON_BG_ACTIVE   = BG_ACTIVE
BUTTON_TEXT        = TEXT_PRIMARY
BUTTON_TEXT_MUTED  = TEXT_SECONDARY
BUTTON_PAD         = (PAD_ROW_X, PAD_ROW_Y)

# Primary button (accent fill).
BUTTON_PRIMARY_BG        = ACCENT
BUTTON_PRIMARY_BG_HOVER  = ACCENT_HOVER
BUTTON_PRIMARY_BG_ACTIVE = ACCENT_PRESSED
BUTTON_PRIMARY_TEXT      = TEXT_ON_ACCENT

# Input field.
INPUT_BG           = BG_PANEL
INPUT_BG_FOCUS     = BG_CELL
INPUT_BORDER       = BORDER_HAIRLINE
INPUT_BORDER_FOCUS = BORDER_FOCUS
INPUT_TEXT         = TEXT_PRIMARY
INPUT_PLACEHOLDER  = TEXT_MUTED
INPUT_PAD          = (PAD_ROW_X, PAD_ROW_Y)

# Slider / knob.
SLIDER_TRACK       = BG_PANEL
SLIDER_FILL        = ACCENT
SLIDER_THUMB       = TEXT_PRIMARY
SLIDER_THUMB_HOVER = ACCENT

# Chip / pill (status tags, mode labels).
CHIP_BG       = BG_CELL
CHIP_BG_ON    = ACCENT
CHIP_TEXT     = TEXT_SECONDARY
CHIP_TEXT_ON  = TEXT_ON_ACCENT
CHIP_PAD      = (PAD_CHIP_X, PAD_CHIP_Y)
CHIP_RADIUS   = RADIUS_CHIP

# Divider.
DIVIDER_COLOR  = BORDER_HAIRLINE
DIVIDER_WEIGHT = BORDER_W_HAIRLINE

# Tooltip.
TOOLTIP_BG     = BG_TOOLTIP
TOOLTIP_TEXT   = TEXT_PRIMARY
TOOLTIP_PAD    = PAD_TOOLTIP
TOOLTIP_RADIUS = RADIUS_CHIP

# List row.
ROW_BG          = BG_PANEL
ROW_BG_HOVER    = BG_HOVER
ROW_BG_SELECTED = BG_ACTIVE
ROW_TEXT        = TEXT_PRIMARY
ROW_TEXT_MUTED  = TEXT_SECONDARY
ROW_PAD         = (PAD_ROW_X, PAD_ROW_Y)


# ─────────────────────────────────────────────────────────────────────
# Tiny helpers — keep this file the only place tuples are manipulated.
# ─────────────────────────────────────────────────────────────────────

def with_alpha(color, alpha: float):
    """Return a copy of an RGBA token with alpha replaced."""
    r, g, b = color[0], color[1], color[2]
    return (r, g, b, float(alpha))


def mix(a, b, t: float):
    """Linearly interpolate two RGBA tokens. t=0 → a, t=1 → b."""
    t = max(0.0, min(1.0, float(t)))
    return tuple(a[i] * (1.0 - t) + b[i] * t for i in range(4))


def disabled(color):
    """Standard disabled treatment for any foreground token."""
    return with_alpha(color, OPACITY_DISABLED)


def muted(color):
    """Standard muted treatment for any foreground token."""
    return with_alpha(color, OPACITY_MUTED)
