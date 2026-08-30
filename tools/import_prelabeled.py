#!/usr/bin/env python
"""Bulk-import prelabeled point clouds into a Lithium catalog project.

Built for the incoming 64k-point / ~15k-cloud Sonata-labelled cohort
coming down from the HPC, but generic: any directory of clouds that
already carry per-point class ids becomes a project whose labels are
the *editable starting labels*, with the model that produced them
attached so INFER keeps working in that project.

Accepted inputs (auto-detected per file, mixed is fine):
  *.ply            'label' (or segment/class/pred/scalar_label) vertex field
  *.npz            coord + color [+ normal] + label/segment/pred/scalar_label
  <scene>/coord.npy + color.npy + segment.npy|pred.npy   (Pointcept layout)

Ontology: ``--ontology-from`` copies an existing project's classes +
palette (default: the Sonata Validation project, so colours match), or
``--classes`` takes a classes.json (``{"class_names": [...]}``) / a
comma list. Class ids in the files must index that list (0 = Unlabeled).

Idempotent + additive: clouds are content-hashed (re-runs dedupe), the
project is reused by name, existing clouds/labels are never rewritten
unless ``--force-labels``. One index/projects save at the end — 15k
registrations do NOT rewrite index.json 15k times.

Run with the app CLOSED (single-instance catalog):
    ~/Lithium/.venv/bin/python tools/import_prelabeled.py \
        --src /board_rack/Lithium/datasets/incoming/<cohort> \
        --project "Deepfield · 64k Sonata · 15k" --region spine --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import library_paths  # noqa: E402
from src.data import cloud_store  # noqa: E402
from src.data.cloud_store import save_cloud_data, save_cloud_labels  # noqa: E402
from src.data.labels import LabelRegistry  # noqa: E402
from src.data.library_catalog import LibraryCatalog, LibraryEntry  # noqa: E402
from src.data.model_registry import ProjectModelRegistry, TrainedModel  # noqa: E402
from src.data.point_cloud import PointCloudData  # noqa: E402
from src.utils.file_hash import compute_file_key  # noqa: E402

LABEL_FIELDS = ("label", "labels", "segment", "class", "pred", "scalar_label")
DEFAULT_ONTOLOGY_PROJECT = "Deepfield · Validation · Sonata (val 38)"
DEFAULT_CHECKPOINT = REPO / "training/runs/sonata_full6_gold247_v1/model/model_best.pth"
FULL6 = ["Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Pedicle",
         "Body_Wall", "Spinous_Process"]


# --------------------------------------------------------------------------
# input readers
# --------------------------------------------------------------------------
def _pick_label_field(names) -> str | None:
    for f in LABEL_FIELDS:
        if f in names:
            return f
    return None


def read_ply(path: Path, label_field: str | None):
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    names = v.data.dtype.names
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    field = label_field or _pick_label_field(names)
    lbl = np.asarray(v[field], dtype=np.int32) if field else None
    scal = {}
    if "intensity" in names:
        scal["intensity"] = np.asarray(v["intensity"], dtype=np.float32)
    return xyz, rgb, lbl, scal


def read_npz(path: Path, label_field: str | None):
    z = np.load(str(path))
    keys = set(z.files)
    ck = "coord" if "coord" in keys else ("positions" if "positions" in keys else "xyz")
    xyz = np.asarray(z[ck], dtype=np.float32)
    if "color" in keys or "colors" in keys:
        rgb = np.asarray(z["color" if "color" in keys else "colors"], dtype=np.float32)
        if rgb.size and rgb.max() > 1.0:
            rgb = rgb / 255.0
    else:
        rgb = np.full_like(xyz, 0.5)
    field = label_field or _pick_label_field(keys)
    lbl = np.asarray(z[field], dtype=np.int32).reshape(-1) if field else None
    scal = {"normal": np.asarray(z["normal"], dtype=np.float32)} if "normal" in keys else {}
    return xyz, rgb, lbl, scal


def read_scene_dir(path: Path, label_field: str | None):
    xyz = np.load(path / "coord.npy").astype(np.float32)
    rgb = (np.load(path / "color.npy").astype(np.float32) if (path / "color.npy").exists()
           else np.full_like(xyz, 0.5))
    if rgb.size and rgb.max() > 1.0:
        rgb = rgb / 255.0
    lbl = None
    for f in ([label_field] if label_field else []) + ["segment", "pred", "label"]:
        if f and (path / f"{f}.npy").exists():
            lbl = np.load(path / f"{f}.npy").astype(np.int32).reshape(-1)
            break
    scal = {}
    if (path / "normal.npy").exists():
        scal["normal"] = np.load(path / "normal.npy").astype(np.float32)
    return xyz, rgb, lbl, scal


def discover(src: Path, pattern: str | None):
    """Yield (path_for_hash, display_stem, reader)."""
    items = []
    if pattern:
        for p in sorted(src.rglob(pattern)):
            items.append(p)
    else:
        items = sorted(list(src.rglob("*.ply")) + list(src.rglob("*.npz")))
    out = []
    for p in items:
        if p.suffix.lower() == ".ply":
            out.append((p, p.stem, read_ply))
        elif p.suffix.lower() == ".npz":
            out.append((p, p.stem, read_npz))
    for coord in sorted(src.rglob("coord.npy")):
        out.append((coord, coord.parent.name, lambda _p, lf: read_scene_dir(_p.parent, lf)))
    return out


# --------------------------------------------------------------------------
# ontology / model helpers
# --------------------------------------------------------------------------
def ontology_from_project(cat: LibraryCatalog, name_or_id: str) -> tuple[dict, list[str]]:
    for pid, p in cat.projects.items():
        if pid == name_or_id or p.name == name_or_id:
            if not p.ontology_data:
                raise SystemExit(f"project '{p.name}' has no ontology to copy")
            reg = LabelRegistry.from_json(p.ontology_data)
            names = [None] * (max(i.id for i in reg.all_labels()) + 1)
            for info in reg.all_labels():
                names[info.id] = info.name
            return p.ontology_data, [n or "Unlabeled" for n in names]
    raise SystemExit(f"no project named '{name_or_id}' to copy the ontology from")


def ontology_from_names(names: list[str]) -> tuple[dict, list[str]]:
    reg = LabelRegistry()  # id 0 = Unlabeled auto-added
    palette = [(214, 94, 0), (0, 115, 178), (87, 232, 126), (232, 87, 180), (38, 191, 166),
               (148, 87, 235), (240, 200, 30), (255, 120, 120), (120, 200, 255), (200, 255, 120)]
    for i, n in enumerate(names):
        if i == 0:
            continue
        r, g, b = palette[(i - 1) % len(palette)]
        reg.add_label_at(i, n, (r / 255.0, g / 255.0, b / 255.0, 1.0))
    return reg.to_json(), list(names)


def load_class_names(spec: str) -> list[str]:
    p = Path(spec)
    if p.is_file():
        data = json.loads(p.read_text())
        names = data.get("class_names", data) if isinstance(data, dict) else data
        return [str(n) for n in names]
    if spec.lower() in ("sonata6", "full6"):
        return list(FULL6)
    return [s.strip() for s in spec.split(",") if s.strip()]


def sha256(path: Path, limit: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model(cat: LibraryCatalog, project, names: list[str], ckpt: Path,
                   config: Path | None, model_name: str, arch: str, best_miou: float,
                   skip_sha: bool) -> str:
    registry = ProjectModelRegistry(library_paths.library_dir())
    model_id = hashlib.sha256((str(ckpt) + "|" + json.dumps(names)).encode()).hexdigest()[:12]
    if any(m.model_id == model_id for m in registry.load(project.id)):
        print(f"  model already registered: {model_id} ({model_name})")
    else:
        work_dir = ckpt.parent.parent if ckpt.parent.name == "model" else ckpt.parent
        item = TrainedModel(
            model_id=model_id, name=model_name, architecture=arch, status="completed",
            created=ckpt.stat().st_mtime, finished=ckpt.stat().st_mtime,
            num_classes=len(names), best_miou=best_miou, device="cuda",
            best_checkpoint=str(ckpt),
            last_checkpoint=str(work_dir / "model" / "model_last.pth")
            if (work_dir / "model" / "model_last.pth").exists() else str(ckpt),
            work_dir=str(work_dir),
            config_snapshot={"source_config": str(config) if config else "",
                             "checkpoint_sha256": "" if skip_sha else sha256(ckpt),
                             "checkpoint_bytes": ckpt.stat().st_size,
                             "note": "registered by tools/import_prelabeled.py"},
            class_map={str(i): n for i, n in enumerate(names)},
        )
        registry.add_model(project.id, item)
        print(f"  registered model {model_id}: {model_name}")
    project.settings = dict(project.settings or {})
    project.settings["active_model_id"] = model_id
    return model_id


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="directory to scan (recursive)")
    ap.add_argument("--project", required=True, help="project name (reused if it exists)")
    ap.add_argument("--pattern", default=None, help="glob (e.g. '*.ply'); default: ply+npz+scene dirs")
    ap.add_argument("--label-field", default=None, help="force the per-point label field name")
    ap.add_argument("--ontology-from", default=DEFAULT_ONTOLOGY_PROJECT,
                    help="copy classes + palette from this project (name or id)")
    ap.add_argument("--classes", default=None,
                    help="instead: classes.json, 'sonata6', or 'Unlabeled,A,B,...'")
    ap.add_argument("--region", default="spine")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                    help="model that produced the labels; attached for INFER (''=none)")
    ap.add_argument("--config", type=Path, default=None, help="run config next to the checkpoint")
    ap.add_argument("--model-name", default="Sonata full6 Gold247 v1 (ep112)")
    ap.add_argument("--arch", default="PT-v3m2 (Sonata)")
    ap.add_argument("--best-miou", type=float, default=0.8139)
    ap.add_argument("--skip-sha", action="store_true", help="skip hashing the 1.5 GB checkpoint")
    ap.add_argument("--limit", type=int, default=0, help="import only the first N (smoke test)")
    ap.add_argument("--force-labels", action="store_true", help="rewrite labels of known clouds")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lib = Path(library_paths.library_dir())
    lock = lib / ".lock"
    if lock.exists() and not a.dry_run:
        raise SystemExit(f"Lithium appears to be running ({lock}) — close it first")

    items = discover(a.src, a.pattern)
    if a.limit:
        items = items[: a.limit]
    if not items:
        raise SystemExit(f"nothing importable under {a.src}")
    print(f"library : {lib}\nsource  : {a.src}  ({len(items)} clouds)\n"
          f"mode    : {'DRY RUN' if a.dry_run else 'WRITE'}")

    cat = LibraryCatalog()
    if a.classes:
        ontology, names = ontology_from_names(load_class_names(a.classes))
    else:
        ontology, names = ontology_from_project(cat, a.ontology_from)
    print(f"classes : {names}")

    project = next((p for p in cat.projects.values() if p.name == a.project), None)
    if project is None:
        if a.dry_run:
            print(f"  [dry-run] would create project '{a.project}'")
        else:
            project = cat.create_project(a.project)
            cat.update_project_ontology(project.id, ontology)
            print(f"  created project '{a.project}' ({project.id})")
    else:
        print(f"  reusing project '{project.name}' ({project.id})")
    if project is not None and not a.dry_run:
        cloud_store.set_active_label_namespace(project.id)

    # -- pass 1: register everything (one index save) -----------------------
    t0 = time.time()
    keys: list[str] = []
    hist = np.zeros(len(names), dtype=np.int64)
    n_new = n_known = n_bad = 0
    pending: list[tuple[str, Path, str, object]] = []
    for path, stem, reader in items:
        try:
            fk = compute_file_key(str(path))
        except OSError as e:
            print(f"  SKIP {path.name}: {e}")
            n_bad += 1
            continue
        if fk in cat.entries:
            n_known += 1
        else:
            n_new += 1
        pending.append((fk, path, stem, reader))
    print(f"  hashed {len(pending)} files in {time.time() - t0:.1f}s: {n_new} new, {n_known} known")

    # -- pass 2: data + labels --------------------------------------------
    t0 = time.time()
    for i, (fk, path, stem, reader) in enumerate(pending):
        try:
            xyz, rgb, lbl, scal = reader(path, a.label_field)
        except Exception as e:  # noqa: BLE001
            print(f"  SKIP {path.name}: unreadable ({e})")
            n_bad += 1
            continue
        if lbl is None:
            print(f"  SKIP {path.name}: no label field (use --label-field)")
            n_bad += 1
            continue
        if lbl.shape[0] != xyz.shape[0]:
            print(f"  SKIP {path.name}: {lbl.shape[0]} labels vs {xyz.shape[0]} points")
            n_bad += 1
            continue
        if lbl.max() >= len(names) or lbl.min() < 0:
            print(f"  SKIP {path.name}: label ids {lbl.min()}..{lbl.max()} outside 0..{len(names) - 1}")
            n_bad += 1
            continue
        hist += np.bincount(lbl, minlength=len(names))[: len(names)]
        if a.dry_run:
            keys.append(fk)
            continue
        entry = cat.entries.get(fk)
        if entry is None:
            entry = LibraryEntry(file_key=fk, file_path=str(path))
            cat.entries[fk] = entry
        entry.point_count = int(xyz.shape[0])
        entry.bounds_min = xyz.min(axis=0).astype(np.float32)
        entry.bounds_max = xyz.max(axis=0).astype(np.float32)
        entry.region = a.region
        if not entry.display_name:
            entry.display_name = stem
        if not cloud_store.has_cloud_data(fk):
            save_cloud_data(fk, PointCloudData(positions=xyz, colors=rgb, scalars=scal),
                            source_path=str(path))
        if a.force_labels or not cloud_store.has_cloud_labels(fk):
            err = save_cloud_labels(fk, lbl.astype(np.int32), catalog=None)
            if err:
                print(f"  LABEL SAVE ERR {path.name}: {err}")
                n_bad += 1
                continue
        keys.append(fk)
        if (i + 1) % 250 == 0:
            rate = (i + 1) / max(1e-6, time.time() - t0)
            print(f"  ... {i + 1}/{len(pending)}  ({rate:.1f} clouds/s, "
                  f"eta {(len(pending) - i - 1) / max(rate, 1e-6) / 60:.1f} min)")

    if not a.dry_run and project is not None:
        cat._save_index()
        cat.add_to_project(project.id, keys)
        if a.checkpoint and str(a.checkpoint) and a.checkpoint.is_file():
            cfg = a.config
            if cfg is None:
                for cand in ("config_src.py", "config.py"):
                    c = a.checkpoint.parent.parent / cand
                    if c.is_file():
                        cfg = c
                        break
            register_model(cat, project, names, a.checkpoint, cfg, a.model_name, a.arch,
                           a.best_miou, a.skip_sha)
        elif a.checkpoint and str(a.checkpoint):
            print(f"  WARN: checkpoint not found at {a.checkpoint} — no model attached")
        cat._save_projects()

    tot = max(1, int(hist.sum()))
    print(f"\n{'[dry-run] would import' if a.dry_run else 'imported'} {len(keys)}/{len(items)} "
          f"clouds ({n_bad} skipped)" + (f" into '{project.name}' ({project.id})" if project else ""))
    print("label distribution:")
    for c, n in enumerate(names):
        print(f"  {c:2d}  {n:20} {int(hist[c]):>13,}  ({100 * hist[c] / tot:5.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
