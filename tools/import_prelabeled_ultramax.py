"""Import the 214 ultramax-prelabeled VerSe bones into a 3Photon catalog project,
WITH the ultramax predictions as editable starting labels + the ultramax checkpoint attached.

Why a bespoke importer: 3Photon's ``load_ply`` deliberately DROPS the PLY 'label' field
(every imported point lands Unlabeled), and none of the 5 stock tools call
``save_cloud_labels`` from a PLY's own label field. This reuses the exact catalog
primitives (so the result auto-registers identically under ~/.3photon/library/) and adds
the one missing ``save_cloud_labels`` call so the ultramax predictions become the
*correctable starting labels* the user edits in the GUI.

Purely ADDITIVE + idempotent: each bone is content-hashed (re-run dedupes), a project is
reused by name if it already exists, and existing clouds/projects are never touched.

Run (app MUST be closed — single-instance catalog):
    /run/media/alex/board_rack/3Photon/.venv/bin/python tools/import_prelabeled_ultramax.py
    # add --dry-run to preview without writing
"""
import sys
import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData

TP = Path("/run/media/alex/board_rack/3Photon")
sys.path.insert(0, str(TP))

from src.data import library_paths
from src.data.library_catalog import LibraryCatalog
from src.data.ply_loader import load_ply
from src.data.cloud_store import save_cloud_data, save_cloud_labels
from src.data.labels import LabelRegistry
from src.data.model_registry import ProjectModelRegistry, TrainedModel

SRC = Path("/run/media/alex/citadel/data/verse/exports_32k_prelabeled")
PROJECT_NAME = "verse_supermax_prelabel"
CKPT_DIR = TP / "dataset_32k/training_runs/fight_club_ultramax_1778969431"
CKPT = CKPT_DIR / "model/model_best.pth"

# Ontology. Ids 1-5 = dataset_32k/classes.json (NAMES must match the ultramax model's
# classes.json so inference / predictions_to_project map by name; ids align 1:1).
# Id 6 Spinous_Process is NEW (7-class project) — ultramax never predicts it (the 214
# prelabels stay {0..5}); the user box-selects it by hand. It feeds the Nash-Moe frontal
# axial-rotation datum downstream and gives the pedicle class a posterior sink.
CLASSES = [  # (id, name, rgb 0-255)
    (1, "Superior_Endplate", (214, 94, 0)),
    (2, "Inferior_Endplate", (0, 115, 178)),
    (3, "Pedicle_Left", (87, 232, 126)),
    (4, "Pedicle_Right", (232, 87, 180)),
    (5, "Body_Wall", (38, 191, 166)),
    (6, "Spinous_Process", (148, 87, 235)),  # vivid violet — midline posterior; Nash-Moe axial datum
    (7, "Transverse_Process", (240, 200, 30)),  # gold — lateral pedicle neighbor; best pedicle-boundary help
]
NAMES = ["Unlabeled"] + [c[1] for c in CLASSES]


def build_ontology() -> dict:
    reg = LabelRegistry()  # id 0 = Unlabeled auto-added
    for lid, name, (r, g, b) in CLASSES:
        reg.add_label_at(lid, name, (r / 255.0, g / 255.0, b / 255.0, 1.0))
    return reg.to_json()


def find_or_create_project(cat: LibraryCatalog, dry: bool):
    for p in cat.projects.values():
        if p.name == PROJECT_NAME:
            print(f"  reusing existing project '{PROJECT_NAME}' ({p.id})")
            return p
    if dry:
        print(f"  [dry-run] would create project '{PROJECT_NAME}'")
        return None
    p = cat.create_project(PROJECT_NAME)
    cat.update_project_ontology(p.id, build_ontology())
    print(f"  created project '{PROJECT_NAME}' ({p.id}) with 6-class ultramax ontology")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plys = sorted(SRC.glob("*.ply"))
    if not plys:
        print(f"ERROR: no PLYs at {SRC}")
        return 1
    print(f"library : {library_paths.library_dir()}")
    print(f"source  : {SRC}  ({len(plys)} prelabeled PLYs)")
    print(f"mode    : {'DRY RUN' if args.dry_run else 'WRITE'}\n")

    cat = LibraryCatalog()
    proj = find_or_create_project(cat, args.dry_run)

    keys: list[str] = []
    hist = np.zeros(len(NAMES), dtype=np.int64)
    n_ok = 0
    for i, ply in enumerate(plys):
        cloud = load_ply(str(ply))
        lbl = np.asarray(PlyData.read(str(ply))["vertex"]["label"], dtype=np.int32)
        if lbl.shape[0] != cloud.point_count:
            print(f"  WARN {ply.name}: {lbl.shape[0]} labels vs {cloud.point_count} pts")
        for c in range(len(NAMES)):
            hist[c] += int((lbl == c).sum())
        if args.dry_run:
            n_ok += 1
            continue
        e = cat.register_file(str(ply))
        if e is None:
            print(f"  SKIP (register failed): {ply.name}")
            continue
        cat.update_metrics(e.file_key, cloud.point_count, cloud.bounds_min, cloud.bounds_max)
        save_cloud_data(e.file_key, cloud, source_path=str(ply))
        e.region = "spine"
        err = save_cloud_labels(e.file_key, lbl, catalog=None)  # <-- the missing call
        if err:
            print(f"  LABEL SAVE ERR {ply.name}: {err}")
            continue
        keys.append(e.file_key)
        n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(plys)}")

    if not args.dry_run and proj is not None:
        cat.add_to_project(proj.id, keys)
        cat._save_index()
        if CKPT.is_file():
            mreg = ProjectModelRegistry(library_paths.library_dir())
            model = TrainedModel(
                name="fight_club_ultramax (32k)",
                architecture="PT-v3m1",
                status="completed",
                num_classes=6,
                best_checkpoint=str(CKPT),
                last_checkpoint=str(CKPT),
                work_dir=str(CKPT_DIR),
            )
            mreg.add_model(proj.id, model)
            print(f"\n  attached inference model 'fight_club_ultramax (32k)' "
                  f"-> {proj.id} (model_id={model.model_id})")
        else:
            print(f"\n  WARN: ultramax checkpoint not found at {CKPT} — model NOT attached")

    tot = max(1, int(hist.sum()))
    print(f"\n{'[dry-run] ' if args.dry_run else ''}"
          f"{'would import' if args.dry_run else 'imported'} {n_ok}/{len(plys)} clouds"
          + (f" into '{PROJECT_NAME}' ({proj.id})" if proj is not None else ""))
    print("ultramax prelabel distribution across all bones:")
    for c in range(len(NAMES)):
        print(f"  {c}  {NAMES[c]:18} {hist[c]:>11,}  ({100 * hist[c] / tot:5.1f}%)")
    if not args.dry_run:
        print(f"\nOpen 3Photon ({TP}/run.sh), pick project '{PROJECT_NAME}', "
              f"and correct the seeds. Run-Inference uses the attached ultramax model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
