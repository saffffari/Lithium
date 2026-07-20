"""Tier-1 smoke test: minimum end-to-end harness viability.

Goal: a single test that proves the harness can stand up an empty
catalog, load a tiny PLY, paint a label, persist it, and reload it on
a fresh HeadlessApp — observing the same labels we just wrote. If this
breaks, every other label-safety test in this folder is suspect.
"""

from __future__ import annotations

import numpy as np

from tests.headless.app_fixture import HeadlessApp
from tests.headless.helpers import (
    count_label_files_on_disk,
    labels_npy_path,
)


def test_smoke_create_load_paint_save_reload(tmp_library, tiny_ply):
    """Full lifecycle: load → paint → save → reload → assert persistence."""
    # ----- Session 1 -----
    app = HeadlessApp()
    entry = app.load_file(tiny_ply)
    assert entry is not None, "load_file should return an entry for a valid PLY"
    assert entry.point_count == 100
    assert entry.file_key, "every loaded entry must have a stable file_key"

    # Fresh load: no cached labels on disk yet, so the file_key has no
    # labels file under <library>/labels/.
    assert count_label_files_on_disk(tmp_library) == 0
    assert (entry.cloud.labels == 0).all()

    # Paint the first 30 points to label id 7. In-memory only — disk
    # should still be empty until save_labels runs.
    label_id = 7
    app.paint_labels(np.arange(30), label_id)
    assert (entry.cloud.labels[:30] == label_id).all()
    assert (entry.cloud.labels[30:] == 0).all()
    assert count_label_files_on_disk(tmp_library) == 0

    # Persist and confirm the file appears with the expected name.
    app.save_labels()
    expected_path = labels_npy_path(tmp_library, entry.file_key)
    assert expected_path.exists(), f"expected labels file at {expected_path}"
    assert count_label_files_on_disk(tmp_library) == 1
    app.close()

    # ----- Session 2: fresh HeadlessApp, same library dir -----
    # The tmp_library monkeypatch is still active for the rest of the
    # test, so a new HeadlessApp picks up the same on-disk catalog.
    app2 = HeadlessApp()
    entry2 = app2.load_file(tiny_ply)
    assert entry2 is not None
    assert entry2.file_key == entry.file_key, (
        "file_key must be stable across sessions for the same source file"
    )
    labels2 = app2.get_labels()
    assert labels2 is not None
    assert (labels2[:30] == label_id).all(), (
        "labels written in session 1 must survive into session 2"
    )
    assert (labels2[30:] == 0).all()
    app2.close()
