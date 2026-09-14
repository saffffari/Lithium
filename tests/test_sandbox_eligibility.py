"""Cross-project model eligibility + SANDBOX model listing (1.2), exercised
against the real catalog/registry with a stub app (no GL needed)."""
import types

import numpy as np
import pytest

from src.data import cloud_store, library_paths
from src.data.labels import LabelRegistry
from src.data.library_catalog import SANDBOX_PROJECT_ID, LibraryCatalog
from src.data.model_registry import ProjectModelRegistry, TrainedModel
from src.gui import panels


@pytest.fixture
def world(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(library_paths, "library_dir", lambda: str(lib))
    cloud_store.set_active_label_namespace(None)
    cat = LibraryCatalog()
    spine = cat.create_project("spine")
    reg = LabelRegistry()
    reg.add_label_at(1, "Superior_Endplate", (1, 0, 0, 1))
    reg.add_label_at(2, "Inferior_Endplate", (0, 1, 0, 1))
    reg.add_label_at(3, "Pedicle", (0, 0, 1, 1))
    cat.update_project_ontology(spine.id, reg.to_json())
    other = cat.create_project("other")

    def ckpt(name):
        p = tmp_path / f"{name}.pth"
        p.write_bytes(b"x")
        return str(p)

    mreg = ProjectModelRegistry(str(lib))
    own = TrainedModel(name="own", status="completed", best_miou=0.5, best_checkpoint=ckpt("own"),
                       class_map={"0": "Unlabeled", "1": "Superior_Endplate", "2": "Inferior_Endplate"})
    compat = TrainedModel(name="compat", status="completed", best_miou=0.9, best_checkpoint=ckpt("compat"),
                          class_map={"0": "Unlabeled", "1": "Pedicle"})
    alien = TrainedModel(name="alien", status="completed", best_miou=0.95, best_checkpoint=ckpt("alien"),
                         class_map={"0": "Unlabeled", "1": "Transverse_Process"})
    gone = TrainedModel(name="gone", status="completed", best_miou=0.99, best_checkpoint=str(tmp_path / "missing.pth"),
                        class_map={"0": "Unlabeled", "1": "Pedicle"})
    mreg.add_model(spine.id, own)
    for m in (compat, alien, gone):
        mreg.add_model(other.id, m)

    app = types.SimpleNamespace(catalog=cat, _train_model_registry=mreg,
                                label_registry=LabelRegistry.from_json(spine.ontology_data),
                                active_view=("project", spine.id), cli=None)
    yield app, spine, other, {"own": own, "compat": compat, "alien": alien, "gone": gone}
    cloud_store.set_active_label_namespace(None)


def test_project_lists_own_then_name_compatible_foreign_models(world):
    app, spine, other, m = world
    got = panels._inference_eligible_models(app, spine.id)
    assert [x.name for x in got] == ["own", "compat"]
    assert got[0]._source_project == "" and got[1]._source_project == "other"
    # auto pick = highest mIoU among eligibles, foreign allowed
    assert panels._resolve_inference_model(app, spine.id).name == "compat"
    # class map of a foreign checkpoint resolves from its own registry entry
    names, cls_to_rid = panels._resolve_inference_class_map(app, m["compat"].best_checkpoint)
    assert names == ["Unlabeled", "Pedicle"]
    assert cls_to_rid.tolist() == [0, 3]  # Pedicle is id 3 in this project


def test_sandbox_lists_every_model_with_a_checkpoint(world):
    app, spine, other, m = world
    app.active_view = ("project", SANDBOX_PROJECT_ID)
    app.label_registry = LabelRegistry()
    got = panels._inference_eligible_models(app, SANDBOX_PROJECT_ID)
    assert sorted(x.name for x in got) == ["alien", "compat", "own"]
    assert {x._source_project for x in got} == {"spine", "other"}
    assert panels._model_class_names(m["alien"]) == ["Unlabeled", "Transverse_Process"]


def test_cross_project_inference_uses_exact_registered_config(world, tmp_path):
    app, _, _, models = world
    model = models["compat"]
    config = tmp_path / "recovered" / "source_config.py"
    config.parent.mkdir()
    config.write_text("model = dict(type='example')\n")
    model.config_snapshot = {"source_config": str(config)}
    assert panels._resolve_inference_config(app, model.best_checkpoint) == str(config)
    config.unlink()
    with pytest.raises(FileNotFoundError, match="Registered model configuration"):
        panels._resolve_inference_config(app, model.best_checkpoint)


def test_predict_subprocess_receives_bound_config_before_any_label_write(world, tmp_path, monkeypatch):
    import subprocess
    app, project, _, models = world
    model = models["compat"]
    config = tmp_path / "exact.py"
    config.write_text("model = dict(type='example')\n")
    model.config_snapshot = {"source_config": str(config)}
    coord = np.zeros((4, 3), dtype=np.float32)
    cloud = types.SimpleNamespace(positions=coord, colors=coord, scalars={"normal": coord}, point_count=4)
    entry = types.SimpleNamespace(full_gpu=types.SimpleNamespace(cloud_data=cloud), name="fixture")
    commands = []
    def capture(cmd, **kwargs):
        commands.append(cmd)
        return types.SimpleNamespace(returncode=1, stdout="intentional stop", stderr="")
    monkeypatch.setattr(subprocess, "run", capture)
    def no_save(*args, **kwargs):
        pytest.fail("Failed subprocess must not save labels")
    monkeypatch.setattr(cloud_store, "save_cloud_labels", no_save)
    assert panels._predict_cloud(app, entry, checkpoint=model.best_checkpoint,
        python_exe="unused", class_names=["Unlabeled", "Pedicle"],
        cls_to_rid=np.array([0, 3]), label_namespace=project.id) is None
    assert commands[0][-2:] == ["--config", str(config)]
