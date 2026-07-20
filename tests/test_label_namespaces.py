"""v2 per-project label namespaces (1.1).

The project is the label namespace: labels/<project_id>/<file_key>.npy.
These tests cover the properties the 1.0 flat layout could not provide:

- the same cloud holds independent labels in two projects
- copying labels into a project translates ids by name and never
  touches the source namespace
- project duplication copies label files into the new namespace
- deleting a project removes only its own namespace
- the v1 flat layout migrates losslessly into _library + per-project
  copies
"""

import numpy as np
import pytest

from src.data import cloud_store, library_paths
from src.data.labels import LabelRegistry
from src.data.library_catalog import LibraryCatalog


@pytest.fixture
def tmp_library(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(library_paths, "library_dir", lambda: str(lib))
    cloud_store.set_active_label_namespace(None)
    yield str(lib)
    cloud_store.set_active_label_namespace(None)


def test_namespaces_are_independent(tmp_library):
    a = np.array([0, 1, 2, 1], dtype=np.int32)
    b = np.array([0, 3, 0, 3], dtype=np.int32)
    cloud_store.save_cloud_labels("k1", a, namespace="proj:aaa")
    cloud_store.save_cloud_labels("k1", b, namespace="proj:bbb")

    got_a = cloud_store.load_cloud_labels("k1", namespace="proj:aaa")
    got_b = cloud_store.load_cloud_labels("k1", namespace="proj:bbb")
    np.testing.assert_array_equal(got_a, a)
    np.testing.assert_array_equal(got_b, b)
    # And the library baseline is untouched.
    assert cloud_store.load_cloud_labels("k1") is None


def test_active_namespace_routes_default_calls(tmp_library):
    arr = np.array([1, 1, 0], dtype=np.int32)
    cloud_store.set_active_label_namespace("proj:xyz")
    cloud_store.save_cloud_labels("k2", arr)
    assert cloud_store.load_cloud_labels("k2", namespace="proj:xyz") is not None
    cloud_store.set_active_label_namespace(None)
    assert cloud_store.load_cloud_labels("k2") is None  # _library is empty


def test_migrate_to_project_copies_and_translates(tmp_library):
    # Source registry: id 1 = "bone" (red-ish); paint in _library.
    src_reg = LabelRegistry()
    bone_id = src_reg.add_label(name="bone", color=(1.0, 0.0, 0.0, 1.0))
    labels = np.array([0, bone_id, bone_id], dtype=np.int32)
    cloud_store.save_cloud_labels("k3", labels)  # active = _library

    # Destination project already names "bone" under a different id.
    catalog = LibraryCatalog()
    proj = catalog.create_project("dest")
    dest_reg = LabelRegistry()
    dest_reg.add_label(name="filler", color=(0.0, 1.0, 0.0, 1.0))
    dest_bone = dest_reg.add_label(name="bone", color=(1.0, 0.0, 0.0, 1.0))
    proj.ontology_data = dest_reg.to_json()

    summary = cloud_store.migrate_cloud_labels_to_project(
        ["k3"], src_reg, proj)
    assert summary["clouds_copied"] == 1

    # Destination namespace got the translated ids...
    got = cloud_store.load_cloud_labels("k3", namespace=proj.id)
    np.testing.assert_array_equal(
        got, np.array([0, dest_bone, dest_bone], dtype=np.int32))
    # ...and the source namespace is byte-identical to before.
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("k3"), labels)


def test_migrate_never_clobbers_dest_paint(tmp_library):
    src_reg = LabelRegistry()
    src_reg.add_label(name="bone", color=(1.0, 0.0, 0.0, 1.0))
    catalog = LibraryCatalog()
    proj = catalog.create_project("dest")
    proj.ontology_data = src_reg.to_json()

    theirs = np.array([9, 9], dtype=np.int32)
    cloud_store.save_cloud_labels("k4", theirs, namespace=proj.id)
    cloud_store.save_cloud_labels(
        "k4", np.array([1, 1], dtype=np.int32))  # _library source

    cloud_store.migrate_cloud_labels_to_project(["k4"], src_reg, proj)
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("k4", namespace=proj.id), theirs)


def test_duplicate_project_copies_clouds_and_labels(tmp_library):
    catalog = LibraryCatalog()
    src = catalog.create_project("source")
    reg = LabelRegistry()
    reg.add_label(name="bone", color=(1.0, 0.0, 0.0, 1.0))
    src.ontology_data = reg.to_json()
    src.file_keys.append("k5")
    labels = np.array([0, 1, 1, 0], dtype=np.int32)
    cloud_store.save_cloud_labels("k5", labels, namespace=src.id)

    dup = catalog.duplicate_project(src.id, "copy")
    assert dup is not None
    assert dup.file_keys == ["k5"]
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("k5", namespace=dup.id), labels)

    # Repainting the duplicate leaves the source untouched.
    cloud_store.save_cloud_labels(
        "k5", np.array([2, 2, 2, 2], dtype=np.int32), namespace=dup.id)
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("k5", namespace=src.id), labels)


def test_duplicate_project_setup_only(tmp_library):
    catalog = LibraryCatalog()
    src = catalog.create_project("source")
    src.file_keys.append("k6")
    cloud_store.save_cloud_labels(
        "k6", np.array([1], dtype=np.int32), namespace=src.id)
    dup = catalog.duplicate_project(src.id, "setup", include_clouds=False)
    assert dup.file_keys == []
    assert cloud_store.load_cloud_labels("k6", namespace=dup.id) is None


def test_delete_project_drops_only_its_namespace(tmp_library):
    catalog = LibraryCatalog()
    p1 = catalog.create_project("one")
    p2 = catalog.create_project("two")
    arr = np.array([1], dtype=np.int32)
    cloud_store.save_cloud_labels("k7", arr, namespace=p1.id)
    cloud_store.save_cloud_labels("k7", arr, namespace=p2.id)
    cloud_store.save_cloud_labels("k7", arr)  # _library

    catalog.delete_project(p1.id)
    assert cloud_store.load_cloud_labels("k7", namespace=p1.id) is None
    assert cloud_store.load_cloud_labels("k7", namespace=p2.id) is not None
    assert cloud_store.load_cloud_labels("k7") is not None


def test_drop_cloud_sweeps_all_namespaces(tmp_library):
    arr = np.array([1], dtype=np.int32)
    cloud_store.save_cloud_labels("k8", arr, namespace="proj:a")
    cloud_store.save_cloud_labels("k8", arr)
    cloud_store.drop_cloud("k8")
    assert not cloud_store.has_cloud_labels_any_namespace("k8")


def test_v1_flat_layout_migrates(tmp_library, tmp_path):
    # Fabricate a v1 library: flat labels/*.npy + a project containing
    # one of the clouds.
    import json
    from pathlib import Path
    lib = Path(tmp_library)
    (lib / "labels").mkdir(parents=True, exist_ok=True)
    (lib / "preview_labels").mkdir(parents=True, exist_ok=True)
    a = np.array([1, 2], dtype=np.int32)
    b = np.array([3], dtype=np.int32)
    np.save(lib / "labels" / "ka.npy", a)
    np.save(lib / "labels" / "kb.npy", b)
    np.save(lib / "preview_labels" / "ka.npy", a)
    with open(lib / "projects.json", "w") as f:
        json.dump({"proj:11112222": {
            "id": "proj:11112222", "name": "legacy",
            "file_keys": ["ka"], "created": 0.0,
        }}, f)

    catalog = LibraryCatalog()  # runs the migration in __init__

    # Flat files moved to _library.
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("ka"), a)
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("kb"), b)
    assert not (lib / "labels" / "ka.npy").exists()
    # Project got its own copy of its member cloud only.
    np.testing.assert_array_equal(
        cloud_store.load_cloud_labels("ka", namespace="proj:11112222"), a)
    assert cloud_store.load_cloud_labels(
        "kb", namespace="proj:11112222") is None
    # Preview labels came along.
    np.testing.assert_array_equal(
        cloud_store.load_preview_labels("ka", namespace="proj:11112222"), a)
    # Marker written; a second catalog init is a no-op.
    assert cloud_store.labels_layout_is_v2()
    LibraryCatalog()
    np.testing.assert_array_equal(cloud_store.load_cloud_labels("ka"), a)
