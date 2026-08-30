"""SANDBOX pseudo-project + cloud-level inference layers (1.2)."""
import numpy as np
import pytest

from src.data import cloud_store, library_paths, sandbox
from src.data.labels import LabelRegistry
from src.data.library_catalog import SANDBOX_PROJECT_ID, LibraryCatalog
from src.data.model_registry import ProjectModelRegistry, TrainedModel


@pytest.fixture
def tmp_library(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(library_paths, "library_dir", lambda: str(lib))
    cloud_store.set_active_label_namespace(None)
    yield str(lib)
    cloud_store.set_active_label_namespace(None)


def test_ensure_sandbox_is_idempotent_persistent_and_undeletable(tmp_library):
    cat = LibraryCatalog()
    n_before = len(cat.projects)
    sb = sandbox.ensure_sandbox(cat)
    assert sb.id == SANDBOX_PROJECT_ID and sb.settings == {"sandbox": True}
    assert sandbox.ensure_sandbox(cat) is sb
    assert len(cat.projects) == n_before + 1
    cat2 = LibraryCatalog()  # reload from disk
    assert SANDBOX_PROJECT_ID in cat2.projects
    cat2.delete_project(SANDBOX_PROJECT_ID)
    assert SANDBOX_PROJECT_ID in cat2.projects
    assert all(p.id != SANDBOX_PROJECT_ID for p in cat2.user_projects())


def test_layers_are_cloud_level_and_independent_of_project_labels(tmp_library):
    proj_labels = np.array([0, 1, 2, 1], dtype=np.int32)
    cloud_store.save_cloud_labels("k1", proj_labels, namespace="proj:aaa")
    meta = sandbox.write_layer_meta("m123", ["Unlabeled", "A", "B"], model_name="model A",
                                    source_project_id="proj:aaa", source_project_name="aaa")
    assert meta["created"] > 0
    layer_labels = np.array([0, 2, 2, 0], dtype=np.int32)
    cloud_store.save_cloud_labels("k1", layer_labels, namespace=sandbox.layer_namespace("m123"))
    # project labels untouched, layer readable through its namespace
    assert np.array_equal(cloud_store.load_cloud_labels("k1", namespace="proj:aaa"), proj_labels)
    assert np.array_equal(
        cloud_store.load_cloud_labels("k1", namespace=sandbox.layer_namespace("m123")), layer_labels)
    layers = sandbox.list_layers()
    assert [m["model_id"] for m in layers] == ["m123"]
    assert sandbox.read_layer_meta("m123")["class_names"] == ["Unlabeled", "A", "B"]
    assert [m["model_id"] for m in sandbox.layers_for_cloud("k1")] == ["m123"]
    assert sandbox.layers_for_cloud("other") == []
    # re-writing meta keeps the first-seen time
    again = sandbox.write_layer_meta("m123", ["Unlabeled", "A", "B"])
    assert again["created"] == meta["created"]
    sandbox.drop_layer("m123")
    assert sandbox.list_layers() == []


def test_registry_for_layer_uses_model_indices_and_source_palette():
    src = LabelRegistry()
    src.add_label_at(3, "Pedicle", (0.1, 0.2, 0.3, 1.0))
    reg = sandbox.registry_for_layer(["Unlabeled", "Endplate", "Pedicle"], palette_source=src)
    ids = {info.id: info for info in reg.all_labels()}
    assert ids[1].name == "Endplate" and ids[2].name == "Pedicle"
    assert tuple(ids[2].color)[:3] == pytest.approx((0.1, 0.2, 0.3))
    assert 3 not in ids  # ids follow the model, not the source project


def test_models_compatible_is_name_subset_excluding_background():
    proj = ["Superior_Endplate", "Inferior_Endplate", "Pedicle", "Body_Wall", "Spinous_Process"]
    assert sandbox.models_compatible(proj, ["Unlabeled", "Pedicle", "body_wall"])
    assert not sandbox.models_compatible(proj, ["Unlabeled", "Pedicle", "Transverse_Process"])
    assert not sandbox.models_compatible(proj, ["Unlabeled"])
    assert not sandbox.models_compatible(proj, [])


def test_all_models_dedupes_across_projects(tmp_library):
    reg = ProjectModelRegistry(tmp_library)
    m = TrainedModel(name="shared", status="completed")
    reg.add_model("proj:a", m)
    reg.add_model("proj:b", m)
    reg.add_model("proj:b", TrainedModel(name="own", status="completed"))
    got = reg.all_models(["proj:a", "proj:b"])
    assert [(pid, x.name) for pid, x in got] == [("proj:a", "shared"), ("proj:b", "own")]
    assert sandbox.class_names_from_model(TrainedModel(name="x", class_map={"1": "B", "0": "U"})) == ["U", "B"]
