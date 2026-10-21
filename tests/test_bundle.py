"""Round-trip tests for ``.li`` project bundles (src/data/bundle.py).

Two catalogs in two temp dirs stand in for two machines: export from
A, import into B, compare. Headless — no imgui, no GL.
"""

import json
import os
import zipfile

import numpy as np
import pytest

from src.data import bundle, cloud_store, library_paths
from src.data.library_catalog import LibraryCatalog, LibraryEntry
from src.data.point_cloud import PointCloudData


def _redirect(monkeypatch, path):
    path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(library_paths, "library_dir", lambda: str(path))


def _seed_catalog(n_clouds=3, n_pts=500, with_labels=True):
    """Build a project with clouds whose npz carry leaky scalars."""
    cat = LibraryCatalog()
    proj = cat.create_project("Seed project")
    proj.ontology_data = {"labels": [{"id": 0, "name": "Unlabeled"},
                                     {"id": 1, "name": "Superior_Endplate"}]}
    rng = np.random.default_rng(0)
    for i in range(n_clouds):
        fk = f"{i:016x}"
        xyz = rng.normal(size=(n_pts, 3)).astype(np.float32)
        rgb = rng.random((n_pts, 3)).astype(np.float32)
        scal = {"normal": rng.normal(size=(n_pts, 3)).astype(np.float32),
                "source__derived__labels_spitfire": rng.integers(0, 5, n_pts).astype(np.float32),
                "source__raw__hu0": rng.random(n_pts).astype(np.float32)}
        cloud_store.save_cloud_data(fk, PointCloudData(positions=xyz, colors=rgb, scalars=scal))
        e = LibraryEntry(file_key=fk, file_path=str(cloud_store.cloud_data_path(fk)),
                         point_count=n_pts, bounds_min=xyz.min(0), bounds_max=xyz.max(0),
                         display_name=f"bone {i}")
        cat.entries[fk] = e
        if with_labels:
            lbl = (rng.random(n_pts) > 0.5).astype(np.int32)
            assert cloud_store.save_cloud_labels(fk, lbl, catalog=None, namespace=proj.id) is None
        proj.file_keys.append(fk)
    cat._save_index()
    cat._save_projects()
    return cat, proj


def test_export_strips_label_scalars_and_import_round_trips(tmp_path, monkeypatch):
    lib_a = tmp_path / "A"
    _redirect(monkeypatch, lib_a)
    cat_a, proj = _seed_catalog()
    plan = bundle.plan_export(cat_a, proj.id)
    assert len(plan.cloud_keys) == 3 and len(plan.label_keys) == 3 and plan.missing_clouds == 0

    out = tmp_path / "seed.li"
    events = []
    man = bundle.export_bundle(cat_a, proj.id, bundle.ExportSpec(clouds=True, labels=True, models=False),
                               str(out), progress=lambda *a: events.append(a))
    assert out.exists() and events[-1][0] == "done"
    assert man["scalars_dropped"] == 6          # 2 leaky scalars x 3 clouds
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert sum(n.startswith("clouds/") for n in names) == 3
        assert sum(n.startswith("labels/") for n in names) == 3
        with zf.open("clouds/0000000000000000.npz") as f:
            z = np.load(f)
            assert set(z.files) == {"coord", "color", "scalar_normal", "_meta_json"}

    # Machine B: empty catalog
    lib_b = tmp_path / "B"
    _redirect(monkeypatch, lib_b)
    cat_b = LibraryCatalog()
    assert not cat_b.entries
    res = bundle.import_bundle(cat_b, str(out))
    assert res.clouds_new == 3 and res.labels_written == 3 and res.labels_skipped == 0
    new_proj = bundle.apply_import(cat_b, res)
    assert new_proj.name == "Seed project"
    assert new_proj.file_keys == proj.file_keys
    assert new_proj.ontology_data == proj.ontology_data
    for fk in proj.file_keys:
        assert cat_b.entries[fk].exists()           # no "missing file" badge
        assert cat_b.entries[fk].display_name.startswith("bone ")
        loaded = cloud_store.load_cloud_data(fk)
        assert loaded is not None and loaded[0].point_count == 500
        lbl_b = cloud_store.load_cloud_labels(fk, new_proj.id)
        assert lbl_b is not None and lbl_b.sum() > 0
    # persisted
    cat_b2 = LibraryCatalog()
    assert new_proj.id in cat_b2.projects and len(cat_b2.entries) == 3


def test_clouds_only_then_labels_only_merge(tmp_path, monkeypatch):
    lib_a = tmp_path / "A"
    _redirect(monkeypatch, lib_a)
    cat_a, proj = _seed_catalog()
    clouds_li = tmp_path / "clouds.li"
    labels_li = tmp_path / "labels.li"
    bundle.export_bundle(cat_a, proj.id, bundle.ExportSpec(True, False, False), str(clouds_li))
    bundle.export_bundle(cat_a, proj.id, bundle.ExportSpec(False, True, False), str(labels_li))
    with zipfile.ZipFile(clouds_li) as zf:
        assert not any(n.startswith("labels/") for n in zf.namelist())

    lib_b = tmp_path / "B"
    _redirect(monkeypatch, lib_b)
    cat_b = LibraryCatalog()
    p = bundle.apply_import(cat_b, bundle.import_bundle(cat_b, str(clouds_li)))
    assert all(not cloud_store.has_cloud_labels(fk, p.id) for fk in p.file_keys)
    # a blinded reader labels... then the labels come back into the same project
    res = bundle.import_bundle(cat_b, str(labels_li), into_project_id=p.id)
    bundle.apply_import(cat_b, res)
    assert res.labels_written == 3
    assert all(cloud_store.has_cloud_labels(fk, p.id) for fk in p.file_keys)
    # a second import does not clobber without overwrite
    res2 = bundle.import_bundle(cat_b, str(labels_li), into_project_id=p.id)
    assert res2.labels_written == 0 and res2.labels_skipped == 3


def test_reject_non_bundles_and_corruption(tmp_path, monkeypatch):
    _redirect(monkeypatch, tmp_path / "A")
    cat, proj = _seed_catalog(n_clouds=1)
    junk = tmp_path / "junk.li"
    junk.write_bytes(b"not a zip")
    with pytest.raises(bundle.BundleError):
        bundle.inspect_bundle(str(junk))
    nomani = tmp_path / "nomani.li"
    with zipfile.ZipFile(nomani, "w") as zf:
        zf.writestr("hello.txt", "x")
    with pytest.raises(bundle.BundleError):
        bundle.inspect_bundle(str(nomani))
    good = tmp_path / "good.li"
    bundle.export_bundle(cat, proj.id, bundle.ExportSpec(True, True, False), str(good))
    # flip a byte in the cloud and keep the manifest: checksum must catch it
    bad = tmp_path / "bad.li"
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(bad, "w") as dst:
        for n in src.namelist():
            data = src.read(n)
            if n.startswith("clouds/"):
                data = data[:-1] + bytes([data[-1] ^ 0xFF])
            dst.writestr(n, data)
    _redirect(monkeypatch, tmp_path / "B")
    with pytest.raises(bundle.BundleError, match="checksum"):
        bundle.import_bundle(LibraryCatalog(), str(bad))


def test_cancel_leaves_no_partial_file(tmp_path, monkeypatch):
    import threading
    _redirect(monkeypatch, tmp_path / "A")
    cat, proj = _seed_catalog(n_clouds=4)
    out = tmp_path / "c.li"
    ev = threading.Event()
    def prog(stage, done, total, nbytes):
        if done >= 2:
            ev.set()
    with pytest.raises(bundle.BundleCancelled):
        bundle.export_bundle(cat, proj.id, bundle.ExportSpec(True, True, False), str(out),
                             progress=prog, cancel=ev)
    assert not out.exists() and not list(tmp_path.glob("*.part"))


def test_model_export_and_import(tmp_path, monkeypatch):
    from src.data.model_registry import ProjectModelRegistry, TrainedModel
    _redirect(monkeypatch, tmp_path / "A")
    cat, proj = _seed_catalog(n_clouds=1)
    run = tmp_path / "run"
    (run / "model").mkdir(parents=True)
    (run / "model" / "model_best.pth").write_bytes(os.urandom(4096))
    (run / "config.py").write_bytes(b"x = 1\n")
    reg = ProjectModelRegistry(library_paths.library_dir())
    tm = TrainedModel(name="toy", status="completed", best_checkpoint=str(run / "model" / "model_best.pth"),
                      last_checkpoint=str(run / "model" / "model_best.pth"), work_dir=str(run),
                      num_classes=2, class_map={"0": "Unlabeled", "1": "Superior_Endplate"})
    reg.add_model(proj.id, tm)
    proj.settings = {"active_model_id": tm.model_id}
    cat._save_projects()
    plan = bundle.plan_export(cat, proj.id)
    assert len(plan.models) == 1 and plan.model_bytes == 4096 + 6
    out = tmp_path / "m.li"
    bundle.export_bundle(cat, proj.id, bundle.ExportSpec(True, True, True), str(out))

    _redirect(monkeypatch, tmp_path / "B")
    cat_b = LibraryCatalog()
    res = bundle.import_bundle(cat_b, str(out))
    p = bundle.apply_import(cat_b, res)
    assert res.models_registered == 1
    models = ProjectModelRegistry(library_paths.library_dir()).load(p.id)
    assert len(models) == 1 and models[0].has_checkpoint
    assert models[0].best_checkpoint.startswith(library_paths.library_dir())
    assert models[0].class_map["1"] == "Superior_Endplate"
    from pathlib import Path
    assert Path(models[0].best_checkpoint).read_bytes() == (run / "model/model_best.pth").read_bytes()
    assert (Path(models[0].work_dir) / "config.py").read_bytes() == b"x = 1\n"
    assert p.settings["active_model_id"] == tm.model_id

    # exporting without models drops the active_model_id so the other
    # machine never points INFER at a checkpoint it doesn't have
    out2 = tmp_path / "nm.li"
    _redirect(monkeypatch, tmp_path / "A")
    bundle.export_bundle(cat, proj.id, bundle.ExportSpec(True, True, False), str(out2))
    with zipfile.ZipFile(out2) as zf:
        man = json.loads(zf.read("manifest.json"))
    assert "active_model_id" not in man["project"]["settings"]
