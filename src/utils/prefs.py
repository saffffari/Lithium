"""Cross-session UI preferences for Lithium.

Persists small key/value UI state to ``~/.lithium/prefs.json``. This is
distinct from ``schema.json`` (label taxonomy) and ``library/`` (cloud
catalog) — it's purely for UI tweaks that should remember themselves
across launches but don't belong in any of the bigger persistence
files.

Currently stores:
- ``gallery_cell_size``: user-controlled Contact Sheets icon size in
  pixels (Ctrl+wheel sets it).

Both load/save are best-effort: failures are logged and swallowed.
The UI must work fine when the file is missing, malformed, or
unwritable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


PREFS_DIR = Path.home() / ".lithium"
PREFS_PATH = PREFS_DIR / "prefs.json"


def load_prefs() -> dict:
    """Return the saved UI prefs dict, or an empty dict on any failure."""
    if not PREFS_PATH.exists():
        return {}
    try:
        with open(PREFS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except (OSError, json.JSONDecodeError) as e:
        print(f"UI prefs load failed ({PREFS_PATH}): {e}")
        return {}


def save_prefs(prefs: dict) -> None:
    """Atomically persist ``prefs`` to disk. Best-effort; failures logged."""
    try:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_PATH.with_suffix(PREFS_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2)
        os.replace(tmp, PREFS_PATH)
    except OSError as e:
        print(f"UI prefs save failed: {e}")


def update_prefs(updates: dict) -> None:
    """Merge ``updates`` into the on-disk prefs and save.

    Loads the current prefs, applies the updates, and rewrites. Used by
    the UI when a single key changes (e.g. user resized the gallery
    cells) so we don't have to know about every preference here.
    """
    prefs = load_prefs()
    prefs.update(updates)
    save_prefs(prefs)
