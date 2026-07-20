"""Cross-session persistence for the label taxonomy (LabelRegistry).

A single default schema file at ``~/.3photon/schema.json`` is auto-saved
every time the registry is mutated and auto-loaded at app start when no
project is being opened. This lets the user define their anatomical
label set once and have it available for every new vertebra mesh they
import, even across app restarts, without having to manage explicit
project files.

The serialization format is exactly ``LabelRegistry.to_json()`` — so
``from_json`` handles validation (max-label cap, malformed ids, etc.)
the same way project load does.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.data.labels import LabelRegistry


SCHEMA_DIR = Path.home() / ".3photon"
SCHEMA_PATH = SCHEMA_DIR / "schema.json"


def save_schema(registry: LabelRegistry) -> None:
    """Atomically persist ``registry`` to the default schema path.

    Write failures are logged and swallowed — persistence is best-effort
    and should never break the annotation UI.
    """
    try:
        SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SCHEMA_PATH.with_suffix(SCHEMA_PATH.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(registry.to_json(), f, indent=2)
        os.replace(tmp, SCHEMA_PATH)
    except OSError as e:
        print(f"Label schema save failed: {e}")


def load_schema() -> LabelRegistry | None:
    """Return the saved registry, or ``None`` if missing / unreadable.

    ``None`` lets the caller fall back to a fresh ``LabelRegistry()``.
    Corrupt files are reported and treated as missing — we never raise
    out of here because failing to load a schema should never block app
    startup.
    """
    if not SCHEMA_PATH.exists():
        return None
    try:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LabelRegistry.from_json(data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        print(f"Label schema load failed ({SCHEMA_PATH}): {e}")
        return None
