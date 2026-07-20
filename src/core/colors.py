"""Cross-cutting color constants — core only, no GUI dependencies.

Lives in ``src.core`` so rendering modules (``gizmo_renderer``,
overlays) can import canonical axis / highlight colors without
reaching back into ``src.gui.theme``. The GUI theme imports the same
constants from here so a single source of truth survives — see AR-5
in REPORT.md.
"""

from __future__ import annotations


# Canonical XYZ axis colors. RGB tuples in [0, 1]; alpha appended at the
# call site to match the consumer's per-channel needs.
AXIS_X_COLOR = (0.95, 0.4, 0.3)        # CORAL
AXIS_Y_COLOR = (0.0, 0.929, 0.584)     # OP1_GREEN
AXIS_Z_COLOR = (0.388, 0.545, 1.0)     # OP1_BLUE

# Gizmo highlight (active drag axis).
GIZMO_HIGHLIGHT_COLOR = (1.0, 0.85, 0.2, 1.0)
