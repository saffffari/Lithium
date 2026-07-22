#!/usr/bin/env python3
"""Build the Deepfield PTv3 workspace inside a 3Photon 1.1 catalog.

The migration is deliberately additive:

* existing catalog entries, projects, labels, and model records are never
  overwritten;
* every pre-existing destination label is compared byte-for-byte before use;
* the verified Gold247 snapshot remains isolated from model-generated seeds;
* incompatible historical taxonomies receive their own project namespaces;
* model checkpoints are registered by reference, not duplicated.

Run without ``--apply`` for a complete audit.  The apply path snapshots all
mutable catalog metadata and label namespaces before creating anything.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from plyfile import PlyData


_default_repo = Path(__file__).resolve().parent.parent
if not (_default_repo / "src").is_dir() and Path("/home/alex/3Photon_1.1/src").is_dir():
    _default_repo = Path("/home/alex/3Photon_1.1")
REPO = Path(os.environ.get("THREEPHOTON_REPO", str(_default_repo))).expanduser().resolve()
SPINELAB_ARCHIVE = Path("/home/alex/Projects/spinelab")
VERSE = Path("/run/media/alex/citadel/data/verse")
GOLD_ROOT = SPINELAB_ARCHIVE / "cloud_models/yamato_gold247_v1"
GOLD_DATASET = GOLD_ROOT / "dataset"

P_GOLD6 = "Deepfield · Gold247 · 6-class [LOCKED]"
P_QUEUE6 = "Deepfield · Polish Queue · 396 teacher seeds"
P_QUARANTINE6 = "Deepfield · Quarantine · 18 rejected or incomplete"
P_C12 = "Deepfield · Cervical C1-C2 · 24 unlabeled"
P_UNLABELED6 = "Deepfield · Unlabeled VerSe · 6 bones"
P_POINTCLUB5 = "Deepfield · Specialist · PointClub 5-class"
P_ENDPLATE2 = "Deepfield · Specialist · Endplate 2-class"
P_ENDPLATE3 = "Deepfield · Specialist · Endplate and Body 3-class"
P_POSTERIOR4 = "Deepfield · Specialist · Posterior 4-class"
P_LEGACY_SEPARATE6 = "Deepfield · Legacy · Separate pedicles 261"
P_LEGACY_SEEDS4 = "Deepfield · Legacy · Separate-pedicle seeds 4"


COLORS = {
    "gray": (128, 128, 128),
    "orange": (214, 94, 0),
    "blue": (0, 115, 178),
    "green": (87, 232, 126),
    "pink": (232, 87, 180),
    "teal": (38, 191, 166),
    "violet": (148, 87, 235),
    "yellow": (240, 200, 30),
}


FULL6 = (
    "Unlabeled",
    "Superior_Endplate",
    "Inferior_Endplate",
    "Pedicle",
    "Body_Wall",
    "Spinous_Process",
)
POINTCLUB5 = FULL6[:5]
ENDPLATE2 = ("NotEndplate", "Endplate")
ENDPLATE3 = ("Rest", "Endplate", "BodyWall")
POSTERIOR4 = ("Other_Bone", "Body_Wall", "Pedicle_Merged", "Spinous_Process")
SEPARATE6 = (
    "Unlabeled",
    "Superior_Endplate",
    "Inferior_Endplate",
    "Pedicle_Left",
    "Pedicle_Right",
    "Body_Wall",
)


@dataclass
class EntryPlan:
    file_key: str
    path: Path
    is_new: bool


@dataclass
class ProjectPlan:
    name: str
    class_names: tuple[str, ...]
    labels: dict[str, np.ndarray]
    ontology_locked: bool = True
    note: str = ""
    colors: tuple[tuple[int, int, int], ...] | None = None

    @property
    def file_keys(self) -> list[str]:
        return list(self.labels)


@dataclass
class ModelPlan:
    name: str
    work_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    class_names: tuple[str, ...]
    target_project: str
    epochs: int
    batch_size: int
    best_miou: float
    created: float
    config_path: Path | None
    archive_only_reason: str = ""

    @property
    def model_id(self) -> str:
        payload = (
            str(self.best_checkpoint.resolve())
            + "|"
            + json.dumps(self.class_names, separators=(",", ":"))
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass
class MigrationPlan:
    library: Path
    entries: dict[str, EntryPlan]
    projects: list[ProjectPlan]
    models: list[ModelPlan]
    source_counts: dict[str, int]
    warnings: list[str] = field(default_factory=list)


def _json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _sanitize_namespace(namespace: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in namespace)


def _compute_file_key(path: Path) -> str:
    absolute = str(path.resolve())
    stat = path.stat()
    payload = f"{absolute}|{stat.st_mtime_ns}|{stat.st_size}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _coord_digest(coord: np.ndarray) -> str:
    arr = np.ascontiguousarray(coord, dtype="<f4")
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _ply_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(path))["vertex"].data
    coord = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float32, copy=False
    )
    if "label" not in vertex.dtype.names:
        labels = np.zeros(len(vertex), dtype=np.int32)
    else:
        labels = np.asarray(vertex["label"], dtype=np.int32).reshape(-1)
    return coord, labels


def _labels_path(library: Path, namespace: str, file_key: str) -> Path:
    return library / "labels" / _sanitize_namespace(namespace) / f"{file_key}.npy"


def _load_project_label(
    library: Path, namespace: str, file_key: str, count: int = 32_000
) -> np.ndarray:
    path = _labels_path(library, namespace, file_key)
    if not path.is_file():
        return np.zeros(count, dtype=np.int32)
    array = np.asarray(np.load(path), dtype=np.int32).reshape(-1)
    if len(array) != count:
        raise RuntimeError(f"{path}: {len(array)} labels, expected {count}")
    return array


def _norm_names(names: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"[^a-z0-9]", "", name.lower()) for name in names)


def _class_colors(names: tuple[str, ...]) -> tuple[tuple[int, int, int], ...]:
    canonical = [
        COLORS["gray"],
        COLORS["orange"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["teal"],
        COLORS["violet"],
        COLORS["yellow"],
        COLORS["pink"],
    ]
    return tuple(canonical[i % len(canonical)] for i in range(len(names)))


def _ontology(
    names: tuple[str, ...], colors: tuple[tuple[int, int, int], ...] | None = None
) -> dict:
    colors = colors or _class_colors(names)
    labels = []
    for idx, (name, rgb) in enumerate(zip(names, colors, strict=True)):
        labels.append(
            {
                "id": idx,
                "name": name,
                "color": [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 0.3 if idx == 0 else 1.0],
                "parent_id": None,
                "visible": True,
                "locked": False,
            }
        )
    return {
        "version": 1,
        "next_id": len(names),
        "palette_index": 0,
        "labels": labels,
    }


def _ontology_names(data: dict | None) -> tuple[str, ...]:
    if not data:
        return ()
    rows = sorted(data.get("labels", []), key=lambda row: int(row.get("id", -1)))
    return tuple(str(row.get("name", "")) for row in rows)


def _project_id(name: str) -> str:
    return "proj:" + hashlib.sha256(("deepfield-ptv3-v1.1|" + name).encode()).hexdigest()[:8]


def _extract_config_names(config: Path) -> tuple[str, ...]:
    if not config.is_file():
        return ()
    text = config.read_text(errors="ignore")
    start = text.find("data = dict(")
    scope = text[start:] if start >= 0 else text
    match = re.search(
        r"\bnames\s*=\s*(\[(?:.|\n)*?\])\s*,\s*(?:train|test|val)\s*=",
        scope,
    )
    if match is None:
        match = re.search(r"\bnames\s*=\s*(\[(?:.|\n)*?\])", scope)
    if match is None:
        return ()
    value = ast.literal_eval(match.group(1))
    return tuple(str(item) for item in value)


def _extract_int(config: Path, key: str, default: int) -> int:
    if not config.is_file():
        return default
    text = config.read_text(errors="ignore")
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([0-9]+)\s*$", text)
    return int(match.group(1)) if match else default


def _best_miou(log: Path) -> float:
    if not log.is_file():
        return 0.0
    values = re.findall(
        r"(?:Currently\s+)?Best\s+mIoU:\s*([0-9]+(?:\.[0-9]+)?)",
        log.read_text(errors="ignore"),
    )
    return max((float(value) for value in values), default=0.0)


def _discover_board_root() -> Path:
    candidates = [
        Path("/run/media/alex/board_rack/3Photon"),
        Path("/run/media/alex/board-rack/3Photon"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("3Photon archive is not mounted under board_rack or board-rack")


def _model_target(names: tuple[str, ...]) -> tuple[str, str]:
    norm = _norm_names(names)
    known = {
        _norm_names(FULL6): (P_GOLD6, ""),
        _norm_names(POINTCLUB5): (P_POINTCLUB5, ""),
        _norm_names(ENDPLATE2): (P_ENDPLATE2, ""),
        _norm_names(ENDPLATE3): (P_ENDPLATE3, ""),
        _norm_names(POSTERIOR4): (P_POSTERIOR4, ""),
        _norm_names(SEPARATE6): (P_LEGACY_SEPARATE6, ""),
    }
    if norm in known:
        return known[norm]
    signature_map = {
        _norm_names(
            ("Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Pedicle_Left", "Pedicle_Right")
        ): "Deepfield · Model Archive · Pedicles 5-class",
        _norm_names(
            ("Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Spinous Process", "Process_Tips")
        ): "Deepfield · Model Archive · Process tips 5-class",
        _norm_names(
            ("Superior_Endplate", "Inferior_Endplate", "Spinous Process", "Process_Tips")
        ): "Deepfield · Model Archive · Process tips 4-class",
        _norm_names(
            ("Unlabeled", "Superior_Endplate", "Inferior_Endplate", "Spinous Process")
        ): "Deepfield · Model Archive · Spinous 4-class",
    }
    target = signature_map.get(norm)
    if target is None:
        digest = hashlib.sha256("|".join(norm).encode()).hexdigest()[:8]
        target = f"Deepfield · Model Archive · Taxonomy {len(names)}c {digest}"
    reason = (
        "historical checkpoint; taxonomy is retained exactly and excluded "
        "from the active training projects"
    )
    if names and _norm_names((names[0],))[0] != "unlabeled":
        reason += "; output channel 0 is anatomical but 3Photon reserves label 0 for the background role"
    return target, reason


def _discover_models(board_root: Path) -> list[ModelPlan]:
    runs: list[Path] = []
    for run in sorted(board_root.glob("dataset*/training_runs/*")):
        if (run / "model/model_best.pth").is_file():
            runs.append(run)
    local = REPO / "dataset_32k_endplate3class/training_runs/endplate3_v11_200ep"
    if (local / "model/model_best.pth").is_file():
        runs.append(local)
    for run in (
        GOLD_ROOT / "run",
        SPINELAB_ARCHIVE / "cloud_models/posterior_gold247_v1/run",
    ):
        if (run / "model/model_best.pth").is_file():
            runs.append(run)

    models: list[ModelPlan] = []
    seen: set[Path] = set()
    for run in runs:
        best = run / "model/model_best.pth"
        best_resolved = best.resolve()
        if best_resolved in seen:
            continue
        seen.add(best_resolved)
        last = run / "model/model_last.pth"
        if not last.is_file():
            last = best
        config = run / "config.py"
        names = _extract_config_names(config)
        if not names:
            classes_candidates = [run / "classes.json", run.parent.parent / "classes.json"]
            for classes in classes_candidates:
                if classes.is_file():
                    names = tuple(_json(classes).get("class_names", ()))
                    if names:
                        break
        if not names:
            raise RuntimeError(f"No frozen class map could be recovered for {run}")
        target, reason = _model_target(names)
        display = run.name
        if run == GOLD_ROOT / "run":
            display = "Yamato Gold247 v1"
        elif run.parent.parent.name == "posterior_gold247_v1":
            display = "Posterior Gold247 v1"
        elif run == local:
            display = "Endplate3 v1.1 200ep"
        models.append(
            ModelPlan(
                name=display,
                work_dir=run,
                best_checkpoint=best,
                last_checkpoint=last,
                class_names=names,
                target_project=target,
                epochs=_extract_int(config, "epoch", 0),
                batch_size=_extract_int(config, "batch_size", 0),
                best_miou=_best_miou(run / "train.log"),
                created=best.stat().st_mtime,
                config_path=config if config.is_file() else None,
                archive_only_reason=reason,
            )
        )
    return models


def _remap(source: dict[str, np.ndarray], fn: Callable[[np.ndarray], np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(fn(labels), dtype=np.int32) for key, labels in source.items()}


def _build_plan(library: Path) -> MigrationPlan:
    index = _json(library / "index.json")
    projects = _json(library / "projects.json")
    combined_items = [
        (pid, project)
        for pid, project in projects.items()
        if project.get("name") == "verse_manual_combined"
    ]
    if len(combined_items) != 1:
        raise RuntimeError(f"Expected one verse_manual_combined project, found {len(combined_items)}")
    combined_id, combined = combined_items[0]

    by_name: dict[str, tuple[str, dict]] = {}
    for file_key, entry in index.items():
        name = Path(entry["file_path"]).name
        if name in by_name and by_name[name][0] != file_key:
            raise RuntimeError(f"Catalog basename is ambiguous: {name}")
        by_name[name] = (file_key, entry)

    gold_manifest = _json(GOLD_DATASET / "manifest.json")
    if len(gold_manifest) != 247:
        raise RuntimeError(f"Gold snapshot has {len(gold_manifest)} scenes, expected 247")
    gold: dict[str, np.ndarray] = {}
    gold_names: set[str] = set()
    for rel, meta in sorted(gold_manifest.items()):
        file_key = str(meta["key"])
        name = str(meta["cloud"])
        if file_key not in index:
            raise RuntimeError(f"Gold key is absent from live catalog: {file_key} {name}")
        labels = np.asarray(np.load(GOLD_DATASET / rel / "segment.npy"), dtype=np.int32).reshape(-1)
        live = _load_project_label(library, combined_id, file_key, len(labels))
        if not np.array_equal(labels, live):
            raise RuntimeError(f"Gold/live label mismatch: {name}")
        gold[file_key] = labels
        gold_names.add(name)

    combined_members = [str(key) for key in combined.get("file_keys", [])]
    manual_keys: list[str] = []
    prelabel_keys: list[str] = []
    for file_key in combined_members:
        parent = Path(index[file_key]["file_path"]).parent.name
        if parent == "exports_32k_manual_combined":
            manual_keys.append(file_key)
        elif parent == "exports_32k_prelabeled_round_6class":
            prelabel_keys.append(file_key)
    if (len(manual_keys), len(prelabel_keys)) != (261, 400):
        raise RuntimeError(
            f"Combined project provenance changed: manual={len(manual_keys)}, prelabel={len(prelabel_keys)}"
        )

    queue: dict[str, np.ndarray] = {}
    cleared_prelabels: dict[str, np.ndarray] = {}
    for file_key in prelabel_keys:
        source = Path(index[file_key]["file_path"])
        live = _load_project_label(library, combined_id, file_key)
        _, embedded = _ply_arrays(source)
        if np.array_equal(live, embedded):
            queue[file_key] = live
        elif not np.any(live) and np.any(embedded):
            cleared_prelabels[file_key] = live
        else:
            raise RuntimeError(
                f"Unclassified manual edit in teacher pool: {source.name}; "
                "preserve and review before migration"
            )
    if (len(queue), len(cleared_prelabels)) != (396, 4):
        raise RuntimeError(
            f"Teacher pool changed: untouched={len(queue)}, cleared={len(cleared_prelabels)}"
        )

    incomplete_manual: dict[str, np.ndarray] = {}
    for file_key in manual_keys:
        name = Path(index[file_key]["file_path"]).name
        if name not in gold_names:
            incomplete_manual[file_key] = _load_project_label(library, combined_id, file_key)
    if len(incomplete_manual) != 14:
        raise RuntimeError(f"Expected 14 incomplete manual clouds, found {len(incomplete_manual)}")
    quarantine = {**incomplete_manual, **cleared_prelabels}

    c12_items = [
        (pid, project)
        for pid, project in projects.items()
        if project.get("name") == "verse_manual_C1-2"
    ]
    if len(c12_items) != 1:
        raise RuntimeError(f"Expected one verse_manual_C1-2 project, found {len(c12_items)}")
    c12_id, c12_source = c12_items[0]
    c12: dict[str, np.ndarray] = {}
    for file_key in c12_source.get("file_keys", []):
        count = int(index[file_key].get("point_count", 32_000)) or 32_000
        c12[file_key] = _load_project_label(library, c12_id, file_key, count)
    if len(c12) != 24:
        raise RuntimeError(f"Expected 24 C1-C2 clouds, found {len(c12)}")

    canonical = sorted((VERSE / "exports_32k").rglob("*.ply"))
    manual_names = {Path(index[key]["file_path"]).name for key in manual_keys}
    c12_names = {Path(index[key]["file_path"]).name for key in c12}
    unlabeled_paths = [path for path in canonical if path.name not in manual_names | c12_names]
    if len(unlabeled_paths) != 6:
        raise RuntimeError(f"Expected six remaining canonical bones, found {len(unlabeled_paths)}")

    prelabel6_names = {Path(index[key]["file_path"]).name for key in prelabel_keys}
    legacy_round = sorted((VERSE / "exports_32k_prelabeled_round").glob("*.ply"))
    legacy_seed_paths = [path for path in legacy_round if path.name not in prelabel6_names]
    if len(legacy_seed_paths) != 4:
        raise RuntimeError(f"Expected four legacy teacher seeds, found {len(legacy_seed_paths)}")

    entries: dict[str, EntryPlan] = {}
    relevant_paths = {
        Path(entry["file_path"]).name: Path(entry["file_path"])
        for key, entry in index.items()
        if Path(entry["file_path"]).name in (
            manual_names | prelabel6_names | c12_names
        )
    }
    for path in unlabeled_paths + legacy_seed_paths:
        relevant_paths[path.name] = path
    for name, path in sorted(relevant_paths.items()):
        if name in by_name:
            file_key = by_name[name][0]
            entries[file_key] = EntryPlan(file_key, path, False)
        else:
            file_key = _compute_file_key(path)
            entries[file_key] = EntryPlan(file_key, path, True)
            by_name[name] = (file_key, {"file_path": str(path)})

    unlabeled = {
        by_name[path.name][0]: np.zeros(len(_ply_arrays(path)[0]), dtype=np.int32)
        for path in unlabeled_paths
    }
    legacy_seeds = {
        by_name[path.name][0]: _ply_arrays(path)[1]
        for path in legacy_seed_paths
    }
    if any(set(np.unique(labels)) - set(range(6)) for labels in legacy_seeds.values()):
        raise RuntimeError("Legacy teacher seeds contain ids outside the separate-pedicle 6-class ontology")

    board_root = _discover_board_root()
    old_dataset = board_root / "dataset_32k"
    digest_to_scene: dict[str, Path] = {}
    for scene in sorted(old_dataset.glob("*/scene_*")):
        digest = _coord_digest(np.load(scene / "coord.npy"))
        if digest in digest_to_scene:
            raise RuntimeError(f"Duplicate old scene geometry digest: {scene}")
        digest_to_scene[digest] = scene
    legacy_separate: dict[str, np.ndarray] = {}
    for path in sorted((VERSE / "exports_32k_manual_combined").glob("*.ply")):
        coord, _ = _ply_arrays(path)
        scene = digest_to_scene.get(_coord_digest(coord))
        if scene is None:
            raise RuntimeError(f"No old separate-pedicle scene matches {path.name}")
        file_key = by_name[path.name][0]
        segment = np.asarray(np.load(scene / "segment.npy"), dtype=np.int32).reshape(-1)
        if len(segment) != len(coord):
            raise RuntimeError(f"Old scene length mismatch: {scene}")
        legacy_separate[file_key] = segment
    if len(legacy_separate) != 261 or len(digest_to_scene) != 261:
        raise RuntimeError(
            "Separate-pedicle geometry map incomplete: "
            f"labels={len(legacy_separate)} scenes={len(digest_to_scene)}"
        )

    pointclub = _remap(gold, lambda a: np.where(a == 5, 0, a))
    endplate2 = _remap(gold, lambda a: np.where(np.isin(a, (1, 2)), 1, 0))
    endplate3 = _remap(
        gold,
        lambda a: np.where(np.isin(a, (1, 2)), 1, np.where(a == 4, 2, 0)),
    )
    posterior = _remap(
        gold,
        lambda a: np.where(a == 4, 1, np.where(a == 3, 2, np.where(a == 5, 3, 0))),
    )

    project_plans: list[ProjectPlan] = [
        ProjectPlan(P_GOLD6, FULL6, gold, True, "frozen human-labelled source of truth"),
        ProjectPlan(
            P_QUEUE6,
            FULL6,
            queue,
            True,
            "model seeds; edit here, never train without review",
        ),
        ProjectPlan(
            P_QUARANTINE6,
            FULL6,
            quarantine,
            True,
            "cleared or incomplete; excluded from training",
        ),
        ProjectPlan(
            P_C12,
            tuple(_ontology_names(c12_source.get("ontology_data"))) or FULL6,
            c12,
            True,
            "cervical anatomy requires a separate review policy",
        ),
        ProjectPlan(
            P_UNLABELED6,
            FULL6,
            unlabeled,
            True,
            "canonical VerSe clouds not in prior projects",
        ),
        ProjectPlan(
            P_POINTCLUB5,
            POINTCLUB5,
            pointclub,
            True,
            "Gold247 remapped for Point Club evaluation",
        ),
        ProjectPlan(P_ENDPLATE2, ENDPLATE2, endplate2, True, "Gold247 binary endplate remap"),
        ProjectPlan(P_ENDPLATE3, ENDPLATE3, endplate3, True, "Gold247 endplate/body remap"),
        ProjectPlan(P_POSTERIOR4, POSTERIOR4, posterior, True, "Gold247 posterior specialist remap"),
        ProjectPlan(
            P_LEGACY_SEPARATE6,
            SEPARATE6,
            legacy_separate,
            True,
            "historical exact labels; archive only",
        ),
        ProjectPlan(
            P_LEGACY_SEEDS4,
            SEPARATE6,
            legacy_seeds,
            True,
            "historical teacher predictions; archive only",
        ),
    ]

    models = _discover_models(board_root)
    project_by_name = {project.name: project for project in project_plans}
    for model in models:
        if model.target_project not in project_by_name:
            archive = ProjectPlan(
                model.target_project,
                model.class_names,
                {},
                True,
                model.archive_only_reason,
            )
            project_plans.append(archive)
            project_by_name[archive.name] = archive
        elif _norm_names(project_by_name[model.target_project].class_names) != _norm_names(model.class_names):
            raise RuntimeError(
                f"Model taxonomy/project mismatch: {model.name} -> {model.target_project}"
            )

    expected_names = {project.name for project in project_plans}
    for pid, current in projects.items():
        name = current.get("name")
        if name not in expected_names:
            continue
        planned = project_by_name[name]
        current_names = _ontology_names(current.get("ontology_data"))
        if current_names and _norm_names(current_names) != _norm_names(planned.class_names):
            raise RuntimeError(
                f"Existing destination project has incompatible ontology: {name}: {current_names}"
            )
        namespace = pid
        for file_key, expected in planned.labels.items():
            path = _labels_path(library, namespace, file_key)
            if path.is_file():
                actual = np.asarray(np.load(path), dtype=np.int32).reshape(-1)
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"Existing destination label conflict: {name} / {file_key}")

    source_counts = {
        "gold_human": len(gold),
        "teacher_seed_queue": len(queue),
        "quarantine": len(quarantine),
        "c1_c2": len(c12),
        "unlabeled": len(unlabeled),
        "legacy_teacher_seeds": len(legacy_seeds),
        "unique_relevant_bones": len(entries),
        "models": len(models),
    }
    return MigrationPlan(library, entries, project_plans, models, source_counts)


def _assert_app_closed(library: Path) -> None:
    lock = library / ".lock"
    if lock.exists():
        raise RuntimeError(f"3Photon catalog lock is present: {lock}")
    result = subprocess.run(
        ["pgrep", "-af", "-i", "3Photon|pointcept"],
        capture_output=True,
        text=True,
        check=False,
    )
    own = Path(__file__).name
    active = [
        line
        for line in result.stdout.splitlines()
        if line
        and "pgrep" not in line
        and own not in line
        and "codex" not in line.lower()
    ]
    if active:
        raise RuntimeError("3Photon/Pointcept appears active:\n" + "\n".join(active))


def _backup(library: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    destination = library / "backups" / f"deepfield_ptv3_migration_{stamp}"
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("index.json", "projects.json"):
        source = library / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for name in ("labels", "preview_labels", "models"):
        source = library / name
        if source.is_dir():
            shutil.copytree(source, destination / name)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_preview_positions(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        for key in ("positions", "coord", "xyz"):
            if key in data.files:
                return np.asarray(data[key], dtype=np.float32)
    raise RuntimeError(f"No coordinate array in preview: {path}")


def _apply(plan: MigrationPlan, skip_cache: bool, skip_preview_labels: bool) -> dict:
    os.environ["THREEPHOTON_LIBRARY_DIR"] = str(plan.library)
    sys.path.insert(0, str(REPO))
    from scipy.spatial import cKDTree
    from src.data import cloud_store
    from src.data.library_catalog import LibraryCatalog, Project
    from src.data.loader import load_point_cloud
    from src.data.model_registry import ProjectModelRegistry, TrainedModel
    from src.data.resampler import voxel_downsample

    _assert_app_closed(plan.library)
    backup = _backup(plan.library)
    catalog = LibraryCatalog()
    counts = Counter()

    for entry_plan in plan.entries.values():
        entry = catalog.entries.get(entry_plan.file_key)
        if entry is None:
            entry = catalog.register_file(str(entry_plan.path))
            if entry is None or entry.file_key != entry_plan.file_key:
                raise RuntimeError(f"Failed to register {entry_plan.path}")
            counts["entries_added"] += 1
        if skip_cache:
            continue
        if not cloud_store.has_cloud_data(entry.file_key):
            cloud = load_point_cloud(str(entry_plan.path))
            cloud_store.save_cloud_data(entry.file_key, cloud, source_path=str(entry_plan.path))
            entry.point_count = int(cloud.point_count)
            entry.bounds_min = cloud.bounds_min.astype(np.float32)
            entry.bounds_max = cloud.bounds_max.astype(np.float32)
            counts["data_cached"] += 1
        preview_path = plan.library / "previews" / f"{entry.file_key}.npz"
        if not preview_path.is_file():
            loaded = cloud_store.load_cloud_data(entry.file_key)
            cloud = loaded[0] if loaded is not None else load_point_cloud(str(entry_plan.path))
            preview = voxel_downsample(cloud, catalog.preview_points)
            tmp = preview_path.with_suffix(".npz.tmp")
            with tmp.open("wb") as handle:
                np.savez_compressed(handle, positions=preview.positions, colors=preview.colors)
            os.replace(tmp, preview_path)
            counts["previews_built"] += 1
    catalog._save_index()

    projects_by_name = defaultdict(list)
    for project in catalog.projects.values():
        projects_by_name[project.name].append(project)
    resolved: dict[str, Project] = {}
    for spec in plan.projects:
        matches = projects_by_name.get(spec.name, [])
        if len(matches) > 1:
            raise RuntimeError(f"Duplicate destination projects named {spec.name}")
        if matches:
            project = matches[0]
        else:
            pid = _project_id(spec.name)
            existing = catalog.projects.get(pid)
            if existing is not None and existing.name != spec.name:
                raise RuntimeError(f"Deterministic project id collision: {pid}")
            project = Project(id=pid, name=spec.name, ontology_locked=spec.ontology_locked)
            catalog.projects[pid] = project
            counts["projects_created"] += 1
        if project.ontology_data is None:
            project.ontology_data = _ontology(spec.class_names, spec.colors)
        elif _norm_names(_ontology_names(project.ontology_data)) != _norm_names(spec.class_names):
            raise RuntimeError(f"Ontology changed during apply: {spec.name}")
        project.ontology_locked = spec.ontology_locked
        members = set(project.file_keys)
        for file_key in spec.file_keys:
            if file_key not in members:
                project.file_keys.append(file_key)
                members.add(file_key)
                counts["project_memberships_added"] += 1
        resolved[spec.name] = project
    catalog._save_projects()

    for spec in plan.projects:
        project = resolved[spec.name]
        for file_key, expected in spec.labels.items():
            path = cloud_store.cloud_labels_path(file_key, project.id)
            if path.is_file():
                actual = np.asarray(np.load(path), dtype=np.int32).reshape(-1)
                if not np.array_equal(actual, expected):
                    raise RuntimeError(f"Label conflict during apply: {spec.name} / {file_key}")
                counts["labels_verified_existing"] += 1
                continue
            error = cloud_store.save_cloud_labels(file_key, expected, namespace=project.id)
            if error:
                raise RuntimeError(error)
            counts["labels_written"] += 1

    if not skip_preview_labels:
        mapping_cache: dict[str, np.ndarray] = {}
        for spec in plan.projects:
            project = resolved[spec.name]
            for file_key, labels in spec.labels.items():
                output = cloud_store.preview_labels_path(file_key, project.id)
                if output.is_file():
                    counts["preview_labels_existing"] += 1
                    continue
                mapping = mapping_cache.get(file_key)
                if mapping is None:
                    loaded = cloud_store.load_cloud_data(file_key)
                    if loaded is None:
                        raise RuntimeError(
                            f"Full catalog data missing for preview-label propagation: {file_key}. "
                            "Re-run without --skip-cache or use --skip-preview-labels."
                        )
                    full_positions = np.asarray(loaded[0].positions, dtype=np.float32)
                    preview_positions = _load_preview_positions(
                        plan.library / "previews" / f"{file_key}.npz"
                    )
                    tree = cKDTree(np.ascontiguousarray(full_positions))
                    _, nearest = tree.query(preview_positions, k=1, workers=4)
                    mapping = np.asarray(nearest, dtype=np.int64)
                    mapping_cache[file_key] = mapping
                cloud_store.save_preview_labels(
                    file_key, labels[mapping].astype(np.int32), namespace=project.id
                )
                counts["preview_labels_written"] += 1

    registry = ProjectModelRegistry(str(plan.library))
    checkpoint_hash_cache: dict[Path, str] = {}
    for model in plan.models:
        project = resolved[model.target_project]
        existing = registry.load(project.id)
        same = [item for item in existing if item.model_id == model.model_id]
        if same:
            item = same[0]
            if Path(item.best_checkpoint) != model.best_checkpoint:
                raise RuntimeError(f"Model id conflict: {model.model_id}")
            counts["models_verified_existing"] += 1
            continue
        checkpoint_hash = checkpoint_hash_cache.get(model.best_checkpoint)
        if checkpoint_hash is None:
            checkpoint_hash = _sha256(model.best_checkpoint)
            checkpoint_hash_cache[model.best_checkpoint] = checkpoint_hash
        config_snapshot = {
            "migration": "deepfield_ptv3_workspace_v1",
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_bytes": model.best_checkpoint.stat().st_size,
            "source_config": str(model.config_path) if model.config_path else "",
            "archive_only_reason": model.archive_only_reason,
        }
        item = TrainedModel(
            model_id=model.model_id,
            name=model.name,
            architecture="PT-v3m1",
            status="completed",
            created=model.created,
            finished=model.best_checkpoint.stat().st_mtime,
            epochs=model.epochs,
            batch_size=model.batch_size,
            device="cuda",
            num_classes=len(model.class_names),
            best_miou=model.best_miou,
            best_checkpoint=str(model.best_checkpoint),
            last_checkpoint=str(model.last_checkpoint),
            work_dir=str(model.work_dir),
            config_snapshot=config_snapshot,
            class_map={str(i): name for i, name in enumerate(model.class_names)},
        )
        registry.add_model(project.id, item)
        counts["models_registered"] += 1

    stamp = time.strftime("%Y%m%d_%H%M%S")
    receipt_dir = plan.library / "migrations" / f"deepfield_ptv3_workspace_{stamp}"
    receipt_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tool": str(Path(__file__).resolve()),
        "backup": str(backup),
        "source_counts": plan.source_counts,
        "write_counts": dict(counts),
        "projects": [
            {
                "id": resolved[spec.name].id,
                "name": spec.name,
                "members": len(spec.file_keys),
                "classes": list(spec.class_names),
                "note": spec.note,
            }
            for spec in plan.projects
        ],
        "models": [
            {
                "id": model.model_id,
                "name": model.name,
                "target_project": model.target_project,
                "checkpoint": str(model.best_checkpoint),
                "best_miou": model.best_miou,
                "classes": list(model.class_names),
            }
            for model in plan.models
        ],
    }
    receipt_path = receipt_dir / "receipt.json"
    tmp = receipt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    os.replace(tmp, receipt_path)
    return receipt


def _print_plan(plan: MigrationPlan) -> None:
    print("Deepfield PTv3 workspace audit")
    print(f"  library: {plan.library}")
    for key, value in plan.source_counts.items():
        print(f"  {key:24s} {value:4d}")
    missing_data = sum(
        not (plan.library / "data" / f"{entry.file_key}.npz").is_file()
        for entry in plan.entries.values()
    )
    missing_previews = sum(
        not (plan.library / "previews" / f"{entry.file_key}.npz").is_file()
        for entry in plan.entries.values()
    )
    print(f"  missing data caches       {missing_data:4d}")
    print(f"  missing previews          {missing_previews:4d}")
    print("\nProjects")
    for project in plan.projects:
        print(
            f"  {len(project.file_keys):4d} clouds  {len(project.class_names):2d} classes  {project.name}"
        )
        if project.note:
            print(f"       {project.note}")
    print("\nModels")
    grouped = Counter(model.target_project for model in plan.models)
    for target, count in sorted(grouped.items()):
        print(f"  {count:2d}  {target}")
    print(f"\nAudit passed: {len(plan.projects)} projects, {len(plan.models)} checkpoints.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--library",
        default=os.environ.get("THREEPHOTON_LIBRARY_DIR", str(Path.home() / ".3photon/library")),
        help="3Photon 1.1 catalog root",
    )
    parser.add_argument("--apply", action="store_true", help="perform the additive migration")
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="do not populate missing data/previews (not recommended for the live migration)",
    )
    parser.add_argument(
        "--skip-preview-labels",
        action="store_true",
        help="skip derived thumbnail labels (useful for a fast sandbox test)",
    )
    args = parser.parse_args()
    library = Path(args.library).expanduser().resolve()
    plan = _build_plan(library)
    _print_plan(plan)
    if not args.apply:
        print("\nDRY RUN: nothing written. Re-run with --apply after reviewing this plan.")
        return 0
    receipt = _apply(plan, args.skip_cache, args.skip_preview_labels)
    print("\nMigration complete")
    print(f"  backup:  {receipt['backup']}")
    print(f"  writes:  {json.dumps(receipt['write_counts'], sort_keys=True)}")
    print(f"  projects: {len(receipt['projects'])}")
    print(f"  models:   {len(receipt['models'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
