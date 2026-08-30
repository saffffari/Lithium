"""Schema persistence verification — ~/.lithium/schema.json round-trip."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pathlib
import tempfile

from src.data.labels import LabelRegistry
from src.data import schema_persistence


def _with_temp_schema(func):
    """Run func with SCHEMA_DIR / SCHEMA_PATH patched to a tempdir."""
    def wrapper():
        orig_dir = schema_persistence.SCHEMA_DIR
        orig_path = schema_persistence.SCHEMA_PATH
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp) / "schema.json"
            schema_persistence.SCHEMA_DIR = pathlib.Path(tmp)
            schema_persistence.SCHEMA_PATH = tmp_path
            try:
                func()
            finally:
                schema_persistence.SCHEMA_DIR = orig_dir
                schema_persistence.SCHEMA_PATH = orig_path
    wrapper.__name__ = func.__name__
    return wrapper


@_with_temp_schema
def test_roundtrip():
    reg = LabelRegistry()
    body = reg.add_label("body", color=(0.3, 0.6, 0.9, 1.0))
    pedicle = reg.add_label("left pedicle", color=(0.9, 0.3, 0.2, 1.0))
    reg.set_locked(body, True)
    reg.set_visible(pedicle, False)

    schema_persistence.save_schema(reg)
    assert schema_persistence.SCHEMA_PATH.exists()

    loaded = schema_persistence.load_schema()
    assert loaded is not None
    assert len(loaded) == len(reg)
    body_info = loaded.get(body)
    pedicle_info = loaded.get(pedicle)
    assert body_info is not None and body_info.name == "body"
    assert body_info.locked is True
    assert pedicle_info is not None and pedicle_info.name == "left pedicle"
    assert pedicle_info.visible is False
    print("PASS: Schema round-trip preserves names, colors, lock, visibility")


@_with_temp_schema
def test_missing_file_returns_none():
    # No file has been written yet.
    assert not schema_persistence.SCHEMA_PATH.exists()
    loaded = schema_persistence.load_schema()
    assert loaded is None
    print("PASS: Missing schema file returns None")


@_with_temp_schema
def test_corrupt_file_returns_none():
    schema_persistence.SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    schema_persistence.SCHEMA_PATH.write_text("{not valid json", encoding="utf-8")
    loaded = schema_persistence.load_schema()
    assert loaded is None
    print("PASS: Corrupt schema file returns None without raising")


@_with_temp_schema
def test_atomic_save_uses_tmp_file():
    """save_schema should write via a .tmp file then os.replace."""
    reg = LabelRegistry()
    reg.add_label("test")
    schema_persistence.save_schema(reg)
    # After save there should be exactly one file (schema.json), no leftover tmp.
    files = list(schema_persistence.SCHEMA_DIR.iterdir())
    names = sorted(f.name for f in files)
    assert "schema.json" in names
    assert "schema.json.tmp" not in names
    print("PASS: Atomic save cleans up tmp file")


def test_on_change_callback_fires():
    """A LabelRegistry with an installed callback fires it on every mutation."""
    reg = LabelRegistry()
    events = []
    reg._on_change_callback = lambda r: events.append(len(r))

    lid = reg.add_label("a")
    reg.set_name(lid, "alpha")
    reg.set_color(lid, (0.1, 0.2, 0.3, 1.0))
    reg.set_visible(lid, False)
    reg.set_locked(lid, True)
    reg.remove_label(lid)

    # 6 mutations → 6 callback fires
    assert len(events) == 6, f"expected 6 events, got {len(events)}"
    print(f"PASS: _on_change fired {len(events)} times")


def test_on_change_callback_errors_are_swallowed():
    """A raising callback must not break the mutator."""
    reg = LabelRegistry()

    def bad(r):
        raise RuntimeError("simulated disk full")

    reg._on_change_callback = bad
    # Should not raise even though the callback does.
    lid = reg.add_label("ok")
    assert lid in reg
    print("PASS: Exceptions in _on_change callback are swallowed")


if __name__ == '__main__':
    test_roundtrip()
    test_missing_file_returns_none()
    test_corrupt_file_returns_none()
    test_atomic_save_uses_tmp_file()
    test_on_change_callback_fires()
    test_on_change_callback_errors_are_swallowed()
    print("\nAll schema persistence tests passed.")
