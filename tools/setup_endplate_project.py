#!/usr/bin/env python
"""Create a fresh project with the 2 endplate labels and (optionally)
import a folder of PLYs into it.

Use case: after staging training-sample groups via
``tools/build_training_samples.py``, you want each group as its own
project with **just** the 2 endplate label classes so that running
inference with an existing 5-class checkpoint cleanly assigns:

  - predicted class "Superior_Endplate" → project label id 1
  - predicted class "Inferior_Endplate" → project label id 2
  - predicted class "Spinous Process"   → no matching label → id 0 (Unlabeled)
  - predicted class "Process_Tips"      → no matching label → id 0
  - predicted class "Unlabeled"         → id 0

The inference class-name-to-registry-id mapping in
``src/gui/panels.py::_resolve_inference_class_map`` is case-
insensitive name match, so the project's label names must be exactly
``Superior_Endplate`` and ``Inferior_Endplate`` (matching the model's
classes.json) for the predicted points to land in the right places.

Usage::

    python tools/setup_endplate_project.py \\
        --name spinelab_endplates_g1 \\
        --source E:/data/verse/spinelab_training_samples/group_1

    # Just create project + labels, no import:
    python tools/setup_endplate_project.py --name spinelab_endplates_g1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Endplate label colours match the existing convention in the
# spinelab_vertebral_subregions project so muscle-memory carries over
# (cyan-ish superior, amber-ish inferior — easy to tell apart even
# with both rendered on the same vertebra).
_LABEL_COLORS = {
    "Superior_Endplate": (0.341, 0.706, 0.910, 1.0),  # light blue
    "Inferior_Endplate": (0.949, 0.722, 0.149, 1.0),  # amber
}

# Default inference model: the 4-class PT-v3 checkpoint trained on
# Unlabeled / Superior_Endplate / Inferior_Endplate / Spinous Process.
# Class 0 = Unlabeled means the model can abstain on broken vertebrae
# (matches the calibrated-uncertainty behaviour we want), and the
# Spinous Process class lands at index 3 — without a matching project
# label, those predictions silently fall to Unlabeled in the new
# project, which is exactly what an endplates-only workflow needs.
_DEFAULT_MODEL_WORK_DIR = Path(
    r"D:/3Photon/dataset/training_runs/tuned_4class_1778457554")
_DEFAULT_MODEL_NAME = "tuned_4class"
_DEFAULT_MODEL_CLASSES = [
    "Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Spinous Process",
]
# RGB triples in 0..255 matching the dataset's existing classes.json for
# the shared names — so any visualisation that consumes classes.json
# stays consistent across the 4-class and 5-class checkpoints.
_DEFAULT_MODEL_COLORS_255 = [
    [128, 128, 128],   # Unlabeled
    [242, 102, 76],    # Superior_Endplate
    [255, 191, 51],    # Inferior_Endplate
    [87, 181, 232],    # Spinous Process
]


def _ensure_default_model_classes_json() -> bool:
    """Write the 4-class ``classes.json`` sidecar next to the tuned_4class
    checkpoint so inference's class-map resolver finds it BEFORE the
    dataset-level 5-class file.

    The resolver in ``panels._find_classes_json_for_checkpoint`` walks up
    from the checkpoint dir and stops at the first ``classes.json`` it
    finds. Without this sidecar, the walk would reach
    ``D:/3Photon/dataset/classes.json`` (5-class) and the lengths would
    disagree with the model's seg_head (shape 4). Idempotent — overwrites
    if the file already exists, since the canonical content for this
    checkpoint never changes.
    """
    target = _DEFAULT_MODEL_WORK_DIR / "classes.json"
    if not _DEFAULT_MODEL_WORK_DIR.is_dir():
        print(f"  (default model dir not found at {_DEFAULT_MODEL_WORK_DIR} "
              f"— skipping sidecar)")
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
    import json
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote 4-class sidecar: {target}")
    return True


def _register_default_model(project_id: str) -> bool:
    """Add the tuned_4class checkpoint to the project's model registry.

    Pins inference for this project to the 4-class model regardless of
    what other checkpoints exist on disk. Without this, the legacy
    fallback in ``_resolve_inference_checkpoint`` would scan
    ``training_runs/*/model_best.pth`` and pick whichever has the newest
    mtime — currently ``fight_club`` (5-class), which would crash on a
    project with only 2 endplate labels.
    """
    best = _DEFAULT_MODEL_WORK_DIR / "model" / "model_best.pth"
    last = _DEFAULT_MODEL_WORK_DIR / "model" / "model_last.pth"
    if not best.is_file():
        print(f"  (default checkpoint not at {best} — skipping registration)")
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
    print(f"  registered model '{_DEFAULT_MODEL_NAME}' under project "
          f"{project_id} (model_id={model.model_id})")
    return True


def _setup_project(name: str) -> tuple[str, object]:
    """Create the project + 2 endplate labels + default inference model.
    Returns ``(project_id, catalog)`` so the caller can chain the import
    without reloading."""
    from src.data.library_catalog import LibraryCatalog
    from src.data.labels import LabelRegistry

    catalog = LibraryCatalog()
    project = catalog.create_project(name)
    print(f"Created project: '{project.name}' (id={project.id})")

    # Build the ontology + persist it onto the project. We use
    # add_label_at to pin the ids to 1 and 2 — matches the
    # spinelab_vertebral_subregions convention so any downstream
    # tooling that looks up "endplate at id 1/2" keeps working.
    registry = LabelRegistry()
    registry.add_label_at(1, "Superior_Endplate",
                          _LABEL_COLORS["Superior_Endplate"])
    registry.add_label_at(2, "Inferior_Endplate",
                          _LABEL_COLORS["Inferior_Endplate"])
    project.ontology_data = registry.to_json()
    catalog._save_projects()
    print(f"  added labels:  id=1 'Superior_Endplate'  id=2 'Inferior_Endplate'")

    # Default inference: tuned_4class (Unlabeled + Superior + Inferior
    # + Spinous). The Spinous predictions silently fall to Unlabeled
    # because the project has no Spinous label; endplates land cleanly.
    _ensure_default_model_classes_json()
    _register_default_model(project.id)
    return project.id, catalog


def _import_folder(catalog, project_id: str, source: Path,
                   max_per_run: int | None = None) -> int:
    """Import every .ply in ``source`` into the project."""
    from src.data.ply_loader import load_ply
    from src.data.cloud_store import save_cloud_data

    plys = sorted(p for p in source.iterdir()
                  if p.is_file() and p.suffix.lower() == ".ply")
    if max_per_run is not None:
        plys = plys[:max_per_run]
    if not plys:
        print(f"  (no PLYs found in {source})")
        return 0

    print(f"Importing {len(plys)} PLYs from {source} ...")
    t0 = time.time()
    file_keys: list[str] = []
    failed = 0
    for i, ply in enumerate(plys, start=1):
        try:
            cloud = load_ply(str(ply))
        except Exception as e:
            print(f"  [{i}/{len(plys)}] {ply.name}: load failed: {e}")
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
            print(f"  [{i}/{len(plys)}] {ply.name} ({cloud.point_count} pts)")

    if file_keys:
        catalog.add_to_project(project_id, file_keys)
    catalog._save_index()
    wall = time.time() - t0
    print(f"  imported {len(file_keys)}/{len(plys)} in {wall:.1f}s "
          f"(failed: {failed})")
    print(f"  added all {len(file_keys)} clouds to project {project_id}")
    return len(file_keys)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True,
                    help="Project name (e.g. 'spinelab_endplates_g1').")
    ap.add_argument("--source", type=Path, default=None,
                    help="Folder of .ply files to import. Omit to just "
                         "create the project + labels with no clouds.")
    ap.add_argument("--max", type=int, default=None,
                    help="Cap the number of PLYs imported (for testing).")
    args = ap.parse_args()

    project_id, catalog = _setup_project(args.name)
    if args.source is not None:
        if not args.source.is_dir():
            print(f"ERROR: source dir not found: {args.source}", file=sys.stderr)
            return 2
        _import_folder(catalog, project_id, args.source, max_per_run=args.max)

    print()
    print(f"Next step: launch the app and:")
    print(f"  1. SHEETS tab → click '{args.name}' to make it active")
    print(f"  2. Multi-select the clouds (or click RUN BATCH ON N UNLABELLED)")
    print(f"  3. RUN INFERENCE — points land in Superior_Endplate /")
    print(f"     Inferior_Endplate; spinous / tips predictions silently")
    print(f"     fall to Unlabeled because no matching project label exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
