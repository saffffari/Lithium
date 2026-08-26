#!/usr/bin/env python3
"""Create the Lithium *Validation* project and register a checkpoint for it.

The project holds the Gold247 validation bones (subjects never seen in
training) with the six-class ontology and the manual labels copied in as
the starting state (in the project's own label namespace, so the LOCKED
Gold247 project is untouched). The given checkpoint is registered to the
project with its class map frozen, so the INFER button runs it live and
its predictions overwrite the manual labels in this namespace only.

Run with Lithium CLOSED (the catalog lock must be free):

    .venv/bin/python tools/make_validation_project.py \\
        --checkpoint /home/alex/3Photon_1.1/training/runs/sonata_full6_gold247_v1/model/model_best.pth \\
        --name "Deepfield · Validation · Sonata (val 38)" --apply

Without --apply it only audits. Idempotent: existing project / labels /
model records are verified, never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DATASET = Path("/home/alex/Projects/spinelab/cloud_models/yamato_gold247_v1/dataset")
GOLD_PROJECT_PREFIX = "Deepfield · Gold247"
FULL6 = ("Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Pedicle", "Body_Wall", "Spinous_Process")
COLORS = ((128, 128, 128), (214, 94, 0), (0, 115, 178), (87, 232, 126), (38, 191, 166), (148, 87, 235))


def _ontology(names, colors):
    labels = [{"id": i, "name": n, "color": [c[0] / 255, c[1] / 255, c[2] / 255, 0.3 if i == 0 else 1.0],
               "parent_id": None, "visible": True, "locked": False} for i, (n, c) in enumerate(zip(names, colors))]
    return {"version": 1, "labels": labels}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_app_closed(library: Path):
    lock = library / ".lock"
    if lock.exists():
        raise SystemExit(f"Lithium is running (catalog lock present: {lock}). Close it and re-run.")


def _best_miou(train_log: Path) -> float:
    import re
    if not train_log.is_file():
        return 0.0
    vals = re.findall(r"Best mIoU:\s*([0-9.]+)", train_log.read_text(errors="ignore"))
    return max((float(v) for v in vals), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--name", default="Deepfield · Validation · Sonata (val 38)")
    ap.add_argument("--model-name", default="Sonata full6 Gold247 v1 (ep112)")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    from src.data import cloud_store
    from src.data.library_catalog import LibraryCatalog, Project
    from src.data.model_registry import ProjectModelRegistry, TrainedModel
    from src.data.resampler import voxel_downsample  # noqa: F401  (import check)
    from scipy.spatial import cKDTree

    library = Path(os.environ.get("THREEPHOTON_LIBRARY_DIR", str(Path.home() / ".3photon" / "library")))
    if a.apply:
        _assert_app_closed(library)          # audit is read-only and may run while Lithium is open
    catalog = LibraryCatalog()
    ckpt = Path(a.checkpoint).resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    work_dir = ckpt.parent.parent
    config = next((work_dir / n for n in ("config_src.py", "config.py") if (work_dir / n).is_file()), None)
    if config is None:
        raise SystemExit(f"no config_src.py/config.py in {work_dir} — infer_single.py needs it for this architecture")

    manifest = json.load(open(DATASET / "manifest.json"))
    members = [(k, v) for k, v in manifest.items() if k.startswith(f"{a.split}/")]
    gold = [p for p in catalog.projects.values() if p.name.startswith(GOLD_PROJECT_PREFIX)]
    if len(gold) != 1:
        raise SystemExit(f"expected exactly one Gold247 project, found {[p.name for p in gold]}")
    gold = gold[0]
    file_keys = []
    for scene, m in sorted(members):
        key = m["key"]
        if key not in catalog.entries:
            raise SystemExit(f"{m['cloud']} ({key}) is not in the catalog")
        if not cloud_store.cloud_labels_path(key, gold.id).is_file():
            raise SystemExit(f"{m['cloud']} has no Gold247 labels")
        file_keys.append(key)

    existing = [p for p in catalog.projects.values() if p.name == a.name]
    print(f"project      : {a.name}  ({'exists' if existing else 'will be created'})")
    print(f"members      : {len(file_keys)} {a.split} bones from {sorted({m['subject'] for _, m in members})}")
    print(f"checkpoint   : {ckpt}  ({ckpt.stat().st_size/1e6:.0f} MB)")
    print(f"run config   : {config}")
    print(f"best mIoU    : {_best_miou(work_dir / 'train.log'):.4f}")
    if not a.apply:
        print("audit only — re-run with --apply")
        return

    # ---- project ---------------------------------------------------------
    if existing:
        project = existing[0]
    else:
        pid = "proj:" + hashlib.sha256(("deepfield-validation-v1|" + a.name).encode()).hexdigest()[:8]
        project = Project(id=pid, name=a.name, ontology_locked=True)
        project.ontology_data = _ontology(FULL6, COLORS)
        catalog.projects[pid] = project
        print(f"created project {pid}")
    present = set(project.file_keys)
    added = 0
    for key in file_keys:
        if key not in present:
            project.file_keys.append(key); present.add(key); added += 1
    print(f"memberships added: {added}")

    # ---- labels: copy manual GT into the new namespace -------------------
    written = kept = 0
    preview_written = 0
    for key in file_keys:
        src = cloud_store.cloud_labels_path(key, gold.id)
        dst = cloud_store.cloud_labels_path(key, project.id)
        labels = np.asarray(np.load(src), dtype=np.int32).reshape(-1)
        if dst.is_file():
            kept += 1
        else:
            err = cloud_store.save_cloud_labels(key, labels, namespace=project.id)
            if err:
                raise SystemExit(err)
            written += 1
        pv = cloud_store.preview_labels_path(key, project.id)
        if not pv.is_file():
            loaded = cloud_store.load_cloud_data(key)
            if loaded is None:
                print(f"  (no cached full cloud for {key}; preview labels skipped)")
                continue
            full_pos = np.asarray(loaded[0].positions, dtype=np.float32)
            prev = np.load(library / "previews" / f"{key}.npz")
            prev_pos = np.asarray(prev["positions"], dtype=np.float32)
            _, nearest = cKDTree(np.ascontiguousarray(full_pos)).query(prev_pos, k=1, workers=4)
            cloud_store.save_preview_labels(key, labels[np.asarray(nearest, dtype=np.int64)].astype(np.int32), namespace=project.id)
            preview_written += 1
    print(f"labels: {written} copied from Gold247, {kept} already present; preview labels written: {preview_written}")

    # ---- model registration ---------------------------------------------
    registry = ProjectModelRegistry(str(library))
    model_id = hashlib.sha256((str(ckpt) + "|" + json.dumps(FULL6)).encode()).hexdigest()[:12]
    models = registry.load(project.id)
    if any(m.model_id == model_id for m in models):
        print(f"model already registered: {model_id}")
    else:
        item = TrainedModel(
            model_id=model_id, name=a.model_name, architecture="PT-v3m2 (Sonata)", status="completed",
            created=ckpt.stat().st_mtime, finished=ckpt.stat().st_mtime, epochs=200, batch_size=1, device="cuda",
            num_classes=len(FULL6), best_miou=_best_miou(work_dir / "train.log"),
            best_checkpoint=str(ckpt), last_checkpoint=str(work_dir / "model" / "model_last.pth"), work_dir=str(work_dir),
            config_snapshot={"source_config": str(config), "checkpoint_sha256": _sha256(ckpt),
                             "checkpoint_bytes": ckpt.stat().st_size, "note": "registered by tools/make_validation_project.py"},
            class_map={str(i): n for i, n in enumerate(FULL6)},
        )
        registry.add_model(project.id, item)
        print(f"registered model {model_id}: {a.model_name}")
    project.settings = dict(project.settings or {}); project.settings["active_model_id"] = model_id
    catalog._save_projects()
    print("done — open Lithium, select the project, INFER runs the registered checkpoint.")


if __name__ == "__main__":
    main()
