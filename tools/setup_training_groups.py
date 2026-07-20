#!/usr/bin/env python
"""Wipe the catalog, set up three training-group projects, and import
the PLYs staged under ``E:\\data\\verse\\spinelab_training_samples\\``.

Each project gets the same four-label ontology so the new training
round has consistent class semantics across all three groups:

    id=1  Superior_Endplate
    id=2  Inferior_Endplate
    id=3  Pedicle_Left
    id=4  Pedicle_Right

The existing 4-class ``tuned_4class`` checkpoint is also registered
into each project so RUN INFERENCE auto-paints **endplates**
immediately. Pedicle classes have no matching model output in
tuned_4class, so those points stay Unlabeled — paint pedicles by
hand in LIGHT TABLE before kicking off a fresh training run.

Usage::

    python tools/setup_training_groups.py

    # Skip the wipe — keep the current catalog, just add projects.
    python tools/setup_training_groups.py --keep-catalog

    # Different source / target groups:
    python tools/setup_training_groups.py \\
        --source E:/data/verse/spinelab_training_samples \\
        --groups group_1 group_2 group_3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_SOURCE = Path(r"E:/data/verse/spinelab_training_samples")
_DEFAULT_GROUPS = ("group_1", "group_2", "group_3")
_PROJECT_NAME_TEMPLATE = "spinelab_training_g{n}"

# Label definitions. Colours chosen for high mutual contrast in a
# busy viewport — superior endplate (light blue) vs inferior endplate
# (amber) is the convention we already use; pedicles get green vs
# magenta so left/right don't collide visually with the endplates or
# with each other on overlapping voxels.
_LABELS = [
    (1, "Superior_Endplate", (0.341, 0.706, 0.910, 1.0)),   # light blue
    (2, "Inferior_Endplate", (0.949, 0.722, 0.149, 1.0)),   # amber
    (3, "Pedicle_Left",      (0.341, 0.910, 0.494, 1.0)),   # green
    (4, "Pedicle_Right",     (0.910, 0.341, 0.706, 1.0)),   # magenta
]

# Default inference model — 4 classes including Unlabeled. The user's
# tuned_4class checkpoint. Its 4 outputs are Unlabeled / Superior /
# Inferior / Spinous; pedicles are not in this model's vocabulary, so
# any pedicle predictions you want need a new training round.
_DEFAULT_MODEL_WORK_DIR = Path(
    r"D:/3Photon/dataset/training_runs/tuned_4class_1778457554")
_DEFAULT_MODEL_NAME = "tuned_4class"
_DEFAULT_MODEL_CLASSES = [
    "Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Spinous Process",
]
_DEFAULT_MODEL_COLORS_255 = [
    [128, 128, 128],   # Unlabeled
    [242, 102, 76],    # Superior_Endplate
    [255, 191, 51],    # Inferior_Endplate
    [87, 181, 232],    # Spinous Process
]


def _wipe(no_backup: bool) -> None:
    from src.data.cloud_store import wipe_catalog
    if no_backup:
        print("Wiping catalog (no backup, user has previous one)...")
    else:
        print("Wiping catalog (backing up first)...")
    result = wipe_catalog(backup=not no_backup)
    if result is not None:
        print(f"  backup at: {result}")
    print("  catalog wiped.\n")


def _ensure_default_model_classes_json() -> bool:
    """Write the 4-class classes.json sidecar next to the canonical
    checkpoint so the inference class-map resolver finds it BEFORE the
    dataset-level 5-class file. Idempotent."""
    target = _DEFAULT_MODEL_WORK_DIR / "classes.json"
    if not _DEFAULT_MODEL_WORK_DIR.is_dir():
        print(f"WARNING: {_DEFAULT_MODEL_WORK_DIR} not found, "
              f"skipping classes.json sidecar", file=sys.stderr)
        return False
    payload = {
        "class_names": list(_DEFAULT_MODEL_CLASSES),
        "class_colors": list(_DEFAULT_MODEL_COLORS_255),
        "ignore_index": 255,
        "source_layer": "default",
        "flatten_mode": "leaves",
        "depth": None,
        "num_classes": len(_DEFAULT_MODEL_CLASSES),
        "format": "ptv3",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote 4-class sidecar: {target}")
    return True


def _register_default_model(project_id: str) -> bool:
    """Add the tuned_4class checkpoint to a project's model registry so
    RUN INFERENCE picks it deterministically."""
    best = _DEFAULT_MODEL_WORK_DIR / "model" / "model_best.pth"
    last = _DEFAULT_MODEL_WORK_DIR / "model" / "model_last.pth"
    if not best.is_file():
        print(f"  WARNING: default checkpoint not at {best} — project "
              f"will fall back to mtime-scan for inference",
              file=sys.stderr)
        return False
    from src.data.library_catalog import library_dir
    from src.data.model_registry import ProjectModelRegistry, TrainedModel
    registry = ProjectModelRegistry(library_dir())
    model = TrainedModel(
        name=_DEFAULT_MODEL_NAME,
        architecture="PT-v3m1",
        status="completed",
        num_classes=len(_DEFAULT_MODEL_CLASSES),
        best_checkpoint=str(best),
        last_checkpoint=str(last) if last.is_file() else str(best),
        work_dir=str(_DEFAULT_MODEL_WORK_DIR),
        config_snapshot={
            "num_classes": len(_DEFAULT_MODEL_CLASSES),
            "class_names": list(_DEFAULT_MODEL_CLASSES),
        },
    )
    registry.add_model(project_id, model)
    print(f"  registered model '{_DEFAULT_MODEL_NAME}' (model_id="
          f"{model.model_id})")
    return True


def _setup_one_group(catalog, group_idx: int, source: Path) -> str | None:
    from src.data.labels import LabelRegistry
    from src.data.ply_loader import load_ply
    from src.data.cloud_store import save_cloud_data

    project_name = _PROJECT_NAME_TEMPLATE.format(n=group_idx)
    project = catalog.create_project(project_name)
    print(f"=== {project_name} (id={project.id}) ===")

    # Ontology — 4 labels.
    registry = LabelRegistry()
    for lid, name, color in _LABELS:
        registry.add_label_at(lid, name, color)
    project.ontology_data = registry.to_json()
    catalog._save_projects()
    print(f"  labels: " + ", ".join(
        f"id={lid} '{name}'" for lid, name, _ in _LABELS))

    # Register the default inference model.
    _register_default_model(project.id)

    # Import PLYs from the group's source folder.
    plys = sorted(p for p in source.iterdir()
                  if p.is_file() and p.suffix.lower() == ".ply")
    if not plys:
        print(f"  (no PLYs in {source} — project created empty)")
        return project.id

    print(f"  importing {len(plys)} PLYs from {source}")
    t0 = time.time()
    file_keys: list[str] = []
    failed = 0
    for i, ply in enumerate(plys, start=1):
        try:
            cloud = load_ply(str(ply))
        except Exception as e:
            print(f"    [{i}/{len(plys)}] {ply.name}: load failed: {e}")
            failed += 1
            continue
        lib_entry = catalog.register_file(str(ply))
        if lib_entry is None:
            failed += 1
            continue
        catalog.update_metrics(
            lib_entry.file_key,
            cloud.point_count,
            cloud.bounds_min,
            cloud.bounds_max,
        )
        save_cloud_data(lib_entry.file_key, cloud, source_path=str(ply))
        lib_entry.region = "spine"
        file_keys.append(lib_entry.file_key)
        if i == 1 or i % 25 == 0 or i == len(plys):
            print(f"    [{i}/{len(plys)}] {ply.name}")

    if file_keys:
        catalog.add_to_project(project.id, file_keys)
    catalog._save_index()
    wall = time.time() - t0
    print(f"  imported {len(file_keys)}/{len(plys)} clouds in {wall:.1f}s "
          f"(failed: {failed})")
    return project.id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, default=_DEFAULT_SOURCE,
                    help=f"Base dir containing group_N subdirs "
                         f"(default {_DEFAULT_SOURCE})")
    ap.add_argument("--groups", nargs="+", default=list(_DEFAULT_GROUPS),
                    help="Group subdirectory names to import. Each "
                         "becomes its own project.")
    ap.add_argument("--keep-catalog", action="store_true",
                    help="Skip the catalog wipe step.")
    ap.add_argument("--backup", action="store_true",
                    help="Tarball the library before wiping. Default is "
                         "no-backup since the user typically has one "
                         "from a previous wipe.")
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: source dir not found: {args.source}", file=sys.stderr)
        return 2

    # 1. Wipe (or skip).
    if not args.keep_catalog:
        _wipe(no_backup=not args.backup)

    # 2. Drop the 4-class classes.json sidecar so inference resolves
    #    cleanly. Independent of any project; one file on disk.
    _ensure_default_model_classes_json()
    print()

    # 3. Per-group: create project, add labels, register model, import.
    from src.data.library_catalog import LibraryCatalog
    catalog = LibraryCatalog()
    setup_count = 0
    for i, gname in enumerate(args.groups, start=1):
        gpath = args.source / gname
        if not gpath.is_dir():
            print(f"WARNING: {gpath} not found; skipping g{i}",
                  file=sys.stderr)
            continue
        pid = _setup_one_group(catalog, i, gpath)
        if pid:
            setup_count += 1
        print()

    print(f"Done. {setup_count}/{len(args.groups)} group(s) configured.")
    print()
    print("Next: launch the app and:")
    print("  1. SHEETS → click a project (e.g. spinelab_training_g1) to "
          "make it active")
    print("  2. RUN BATCH ON N UNLABELLED → auto-paints endplates")
    print("  3. LIGHT TABLE → manually paint Pedicle_Left + Pedicle_Right")
    print("  4. Repeat for the other groups (or retrain after group 1)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
