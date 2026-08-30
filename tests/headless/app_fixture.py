"""Tier-1 headless test fixture for Lithium.

A minimal stand-in for ``src.main.App`` exercising the catalog + label
persistence code paths without the GLFW window, ImGui layer, or the
ModernGL upload pipeline. Tests that need a GL context get one via the
``gl_context`` fixture in conftest.py.

Mirrors just the parts of App that matter for label-safety regression
tests: register-with-catalog, apply-cached-labels-on-load, persist on
mutation, and reload-from-disk on a fresh instance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from src.data import cloud_store
from src.data.labels import LabelRegistry
from src.data.labels_io import (
    apply_label_array_to_cloud,
    find_companion_label_array,
)
from src.data.library_catalog import LibraryCatalog
from src.data.loader import load_point_cloud
from src.data.point_cloud import PointCloudData


@dataclass
class HeadlessEntry:
    """Per-loaded-cloud state — file path, file_key, and the live cloud."""
    file_path: str
    file_key: str
    cloud: PointCloudData
    point_count: int


class HeadlessApp:
    """Lightweight App stand-in for label-safety regression tests.

    The library directory used by the catalog is whatever
    ``library_catalog.library_dir()`` resolves to at call time — tests
    monkey-patch it to a tempdir via the ``tmp_library`` fixture, so a
    fresh HeadlessApp inside a test always sees an empty catalog.
    """

    def __init__(self):
        self.catalog: LibraryCatalog | None = None
        self.label_registry = LabelRegistry()
        self.entries: list[HeadlessEntry] = []
        self.selected_index: int = -1

    def _ensure_catalog(self) -> LibraryCatalog:
        if self.catalog is None:
            self.catalog = LibraryCatalog()
        return self.catalog

    # ---- loading / selection ----------------------------------------

    def load_file(self, path: str) -> HeadlessEntry | None:
        """Register a file with the catalog, load it, apply cached labels.

        Returns the newly-loaded entry, or None if the file can't be read.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            return None
        catalog = self._ensure_catalog()
        lib_entry = catalog.register_file(path)
        if lib_entry is None:
            return None

        try:
            cloud = load_point_cloud(path)
        except (OSError, ValueError, IOError) as e:
            print(f"HeadlessApp.load_file: {path} failed to load: {e}")
            return None

        catalog.update_metrics(
            lib_entry.file_key, cloud.point_count,
            cloud.bounds_min, cloud.bounds_max,
        )

        # Apply any cached labels from the catalog so a reload picks
        # back up where the last session left off.
        cached = cloud_store.load_cloud_labels(lib_entry.file_key)
        if cached is not None and cached.shape[0] == cloud.point_count:
            cloud.labels = cached.astype(np.int32, copy=True)
        elif cached is not None:
            print(
                f"HeadlessApp.load_file: cached labels for {lib_entry.file_key} "
                f"length {cached.shape[0]} != cloud {cloud.point_count}; "
                f"ignoring stale array."
            )

        entry = HeadlessEntry(
            file_path=path,
            file_key=lib_entry.file_key,
            cloud=cloud,
            point_count=cloud.point_count,
        )
        self.entries.append(entry)
        self.selected_index = len(self.entries) - 1
        return entry

    def apply_companion_label_array(self, entry: HeadlessEntry) -> int:
        """If a ``{stem}_labels.npy`` sits next to the source file, apply it.

        Returns the number of points that ended up labeled, or 0 if no
        companion file is present. Mirrors what ``_maybe_auto_apply_label_array``
        in src/main.py does on import.
        """
        path = find_companion_label_array(entry.file_path)
        if not path:
            return 0
        labeled, _ = apply_label_array_to_cloud(
            entry.cloud, path, self.label_registry)
        # Auto-persist so the catalog reflects the applied labels.
        self._persist_entry(entry)
        return labeled

    # ---- selection helpers ------------------------------------------

    @property
    def selected_entry(self) -> HeadlessEntry | None:
        if 0 <= self.selected_index < len(self.entries):
            return self.entries[self.selected_index]
        return None

    def select(self, index: int) -> None:
        if 0 <= index < len(self.entries):
            self.selected_index = index

    # ---- label mutation + persistence -------------------------------

    def paint_labels(self, indices, label_id: int,
                     entry: HeadlessEntry | None = None) -> None:
        """Set ``cloud.labels[indices] = label_id`` and bump labels_version.

        In-memory only — persistence is explicit via ``save_labels`` so
        tests can distinguish "painted but not saved" from "persisted".
        """
        target = entry if entry is not None else self.selected_entry
        if target is None:
            raise RuntimeError("paint_labels: no entry selected")
        idx = np.asarray(indices, dtype=np.int64)
        target.cloud.labels[idx] = int(label_id)
        target.cloud.labels_version += 1

    def save_labels(self, entry: HeadlessEntry | None = None) -> None:
        """Persist the entry's labels via cloud_store. Mirrors _persist_cloud_labels."""
        target = entry if entry is not None else self.selected_entry
        if target is None:
            return
        self._persist_entry(target)

    def _persist_entry(self, entry: HeadlessEntry) -> None:
        labels = entry.cloud.labels
        if labels is None:
            return
        # Same length guard as src/main.py:_persist_cloud_labels — refuse
        # to write a wrong-length array to the full-resolution slot.
        if entry.point_count and labels.shape[0] != entry.point_count:
            print(
                f"[persist] skip {os.path.basename(entry.file_path)}: "
                f"labels length {labels.shape[0]} != entry full-res "
                f"{entry.point_count}"
            )
            return
        cloud_store.save_cloud_labels(entry.file_key, labels)

    def get_labels(self, entry: HeadlessEntry | None = None) -> np.ndarray | None:
        target = entry if entry is not None else self.selected_entry
        if target is None or target.cloud.labels is None:
            return None
        return target.cloud.labels

    # ---- session control --------------------------------------------

    def close(self) -> None:
        """Release any background workers the catalog holds."""
        if self.catalog is not None:
            try:
                self.catalog._executor.shutdown(wait=False, cancel_futures=True)
            except (AttributeError, RuntimeError):
                pass
        self.entries.clear()
        self.selected_index = -1
        self.catalog = None
