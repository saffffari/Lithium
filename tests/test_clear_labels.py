"""Contact Sheets "Clear labels" (Ctrl+Shift+X) on clouds that aren't loaded.

Drives the real App methods on a stub app: labels live only on disk, a
locked label must survive, and Ctrl+Z (``_undo_bulk_label_clear``)
must put everything back.
"""
import types

import numpy as np

from src.data import library_paths
from src.data.cloud_store import (
    load_cloud_labels, load_preview_labels, save_cloud_labels,
    save_preview_labels,
)
from src.data.labels import LabelRegistry
from src.main import App, MODE_CONTACT_SHEETS


def _stub_app(entries, registry):
    app = types.SimpleNamespace(
        entries=entries, label_registry=registry, catalog=None, cli=None,
        mode=MODE_CONTACT_SHEETS, selected_index=0, _gallery_multi_sel=set(),
        _bulk_label_undo=None, _snapshot_dirty=False,
        undo_stack=types.SimpleNamespace(clear=lambda: None),
    )
    for name in ("_entry_label_array", "_store_entry_labels",
                 "clear_labels_for_entries", "_undo_bulk_label_clear",
                 "_clear_labels_targets"):
        setattr(app, name, types.MethodType(getattr(App, name), app))
    return app


def _entry(fk):
    return types.SimpleNamespace(file_key=fk, name=fk, full_gpu=None,
                                 preview_gpu=None, mesh_dirty=False)


def test_clear_unloaded_clouds_keeps_locked_and_undoes(tmp_path, monkeypatch):
    monkeypatch.setattr(library_paths, "library_dir", lambda: str(tmp_path))
    reg = LabelRegistry()
    body = reg.add_label("body")
    rim = reg.add_label("rim")
    reg.get(rim).locked = True

    full = np.array([0, body, body, rim, body, 0], dtype=np.int32)
    prev = np.array([body, rim, 0], dtype=np.int32)
    for fk in ("a", "b"):
        save_cloud_labels(fk, full)
        save_preview_labels(fk, prev)
    untouched = np.array([body] * 4, dtype=np.int32)
    save_cloud_labels("c", untouched)

    app = _stub_app([_entry("a"), _entry("b"), _entry("c")], reg)
    app._gallery_multi_sel = {0, 1}
    assert app._clear_labels_targets() == [0, 1]
    assert app.clear_labels_for_entries([0, 1]) == 2

    for fk in ("a", "b"):
        np.testing.assert_array_equal(load_cloud_labels(fk),
                                      [0, 0, 0, rim, 0, 0])
        np.testing.assert_array_equal(load_preview_labels(fk), [0, rim, 0])
    np.testing.assert_array_equal(load_cloud_labels("c"), untouched)
    assert list((tmp_path / "backups").glob("*/*/a.npy"))

    assert app._undo_bulk_label_clear() is True
    for fk in ("a", "b"):
        np.testing.assert_array_equal(load_cloud_labels(fk), full)
        np.testing.assert_array_equal(load_preview_labels(fk), prev)
    assert app._undo_bulk_label_clear() is False
