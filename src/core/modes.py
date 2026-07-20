"""View-mode constants for the 3Photon top-level UI.

Three modes; ordering follows the research workflow
browse → label → train.  Lives in ``src.core`` so panels,
widgets, and automation can import these without pulling in
``src.main`` (the App god module) — that lazy-import chain was the
root cause of the gui→main import cycle flagged as AR-1.

The integer values (1 / 2 / 5) are preserved from the historical
six-mode layout so persisted settings + saved mode state from before
the SpineLab clinical surface was split out stay valid. The retired
indices 0 (IMAGING), 3 (HOLOGRAM) and 4 (OVERWATCH) left with SpineLab
and are intentionally not reused.
"""

MODE_CONTACT_SHEETS = 1   # browse the per-vertebra / per-cloud gallery
MODE_LIGHT_TABLE = 2      # paint / refine labels
MODE_AUTOMATION = 5       # build models (PT-v3 training)

# Legacy aliases — pre-existing callers expect these names. Don't add
# new ones; the canonical mode constants above are authoritative.
MODE_GALLERY = MODE_CONTACT_SHEETS
MODE_INDIVIDUAL = MODE_LIGHT_TABLE
