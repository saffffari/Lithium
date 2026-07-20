#!/usr/bin/env python
"""Register fight_club_heavy under the spinelab_density_test_g1 project
and drop a classes.json next to the checkpoint so the inference class-
map resolver finds it deterministically.

Mirrors what setup_endplate_project.py does for the tuned_4class model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


MODEL_WORK_DIR = Path(r"D:/3Photon/dataset/training_runs/fight_club_heavy_1778724923")
MODEL_NAME = "fight_club_heavy"
DEST_PROJECT_NAME = "spinelab_density_test_g1"

# 6-class ontology — must match the names used in spinelab_training_g1.
MODEL_CLASSES = [
    "Unlabeled", "Superior_Endplate", "Inferior_Endplate",
    "Pedicle_Left", "Pedicle_Right", "Body_Wall",
]
MODEL_COLORS_255 = [
    [128, 128, 128],   # Unlabeled
    [214, 94, 0],      # Superior_Endplate
    [0, 115, 178],     # Inferior_Endplate
    [87, 232, 126],    # Pedicle_Left
    [232, 87, 180],    # Pedicle_Right
    [38, 191, 166],    # Body_Wall
]


def _write_classes_json() -> None:
    target = MODEL_WORK_DIR / "classes.json"
    payload = {
        "class_names": list(MODEL_CLASSES),
        "class_colors": list(MODEL_COLORS_255),
        "ignore_index": 255,
        "source_layer": "default",
        "flatten_mode": "leaves",
        "depth": None,
        "num_classes": len(MODEL_CLASSES),
        "format": "ptv3",
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {target}")


def main() -> int:
    from src.data.library_catalog import LibraryCatalog, library_dir
    from src.data.model_registry import ProjectModelRegistry, TrainedModel

    catalog = LibraryCatalog()
    proj = None
    for p in catalog.projects.values():
        if p.name == DEST_PROJECT_NAME:
            proj = p
            break
    if proj is None:
        print(f"ERROR: project {DEST_PROJECT_NAME!r} not found.")
        return 2

    _write_classes_json()

    best = MODEL_WORK_DIR / "model" / "model_best.pth"
    last = MODEL_WORK_DIR / "model" / "model_last.pth"
    if not best.is_file():
        print(f"ERROR: checkpoint not at {best}")
        return 2

    registry = ProjectModelRegistry(library_dir())
    model = TrainedModel(
        name=MODEL_NAME,
        architecture="PT-v3m1",
        status="completed",
        num_classes=len(MODEL_CLASSES),
        best_checkpoint=str(best),
        last_checkpoint=str(last) if last.is_file() else str(best),
        work_dir=str(MODEL_WORK_DIR),
        config_snapshot={
            "num_classes": len(MODEL_CLASSES),
            "class_names": list(MODEL_CLASSES),
        },
    )
    registry.add_model(proj.id, model)
    print(f"registered '{MODEL_NAME}' under {proj.name} (model_id={model.model_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
