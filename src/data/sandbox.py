"""SANDBOX — the always-available experimentation project (1.2).

Projects stay the organising tool: one ontology, one label namespace each.
The sandbox is deliberately different — it owns NO labels. Any cloud can be
sent to it from the gallery, and any registered model (any project, any
class set) can be run on it. Each model's predictions land in a
**cloud-level layer**: the label namespace ``layer:<model_id>`` whose class
ids are the model's own class indices, described by a ``_layer.json``
sidecar (model name, class names, source project, checkpoint). Layers are
per cloud, visible from the sandbox regardless of which project the cloud
also belongs to, and never touch a project's labels.

Cross-project inference for *normal* projects lives here too: a model from
another project is runnable when every class it predicts exists (by name)
in the active project's ontology — see ``models_compatible``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.data import cloud_store
from src.data.labels import DEFAULT_PALETTE, LabelRegistry
from src.data.library_catalog import SANDBOX_NAME, SANDBOX_PROJECT_ID

LAYER_PREFIX = "layer:"
LAYER_META = "_layer.json"
EMPTY_LAYER = "_none"          # namespace shown when no layer is selected


def is_sandbox(project_id: str | None) -> bool:
    return project_id == SANDBOX_PROJECT_ID


def layer_namespace(model_id: str | None) -> str:
    """Label namespace holding one model's cloud-level predictions."""
    return f"{LAYER_PREFIX}{model_id or EMPTY_LAYER}"


def _layers_root() -> Path:
    from src.data import library_paths
    return Path(library_paths.library_dir()) / cloud_store._LABELS_SUBDIR


def layer_dir(model_id: str) -> Path:
    return cloud_store._labels_dir(layer_namespace(model_id))


def write_layer_meta(model_id: str, class_names: list[str], *, model_name: str = "",
                     source_project_id: str = "", source_project_name: str = "",
                     checkpoint: str = "", architecture: str = "") -> dict:
    meta = {
        "model_id": model_id,
        "model_name": model_name,
        "class_names": list(class_names),
        "source_project_id": source_project_id,
        "source_project_name": source_project_name,
        "checkpoint": checkpoint,
        "architecture": architecture,
        "created": time.time(),
    }
    d = layer_dir(model_id)
    existing = read_layer_meta(model_id)
    if existing and existing.get("created"):
        meta["created"] = existing["created"]   # keep first-seen time
    tmp = d / (LAYER_META + ".tmp")
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, d / LAYER_META)
    return meta


def read_layer_meta(model_id: str | None) -> dict | None:
    if not model_id:
        return None
    p = _layers_root() / cloud_store.sanitize_namespace(layer_namespace(model_id)) / LAYER_META
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def list_layers() -> list[dict]:
    """Every layer that exists on disk, newest first."""
    root = _layers_root()
    if not root.exists():
        return []
    out = []
    prefix = cloud_store.sanitize_namespace(LAYER_PREFIX + "x")[:-1]  # "layer_"
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        p = d / LAYER_META
        if not p.is_file():
            continue
        try:
            meta = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if meta.get("model_id"):
            out.append(meta)
    out.sort(key=lambda m: -float(m.get("created", 0)))
    return out


def layers_for_cloud(file_key: str) -> list[dict]:
    return [m for m in list_layers()
            if cloud_store.has_cloud_labels(file_key, namespace=layer_namespace(m["model_id"]))]


def drop_layer(model_id: str) -> None:
    cloud_store.drop_labels_namespace(layer_namespace(model_id))


def class_names_from_model(model) -> list[str]:
    """Ordered class names for a registry model (index == predicted id)."""
    cm = getattr(model, "class_map", None)
    if cm:
        try:
            return [name for _k, name in sorted(cm.items(), key=lambda kv: int(kv[0]))]
        except (ValueError, TypeError):
            pass
    return []


def registry_for_layer(class_names: list[str],
                       palette_source: LabelRegistry | None = None) -> LabelRegistry:
    """A read-only registry whose ids are the model's class indices.

    Colours come from ``palette_source`` by class name when available (so
    a Sonata layer looks like the project that trained it), else from the
    default palette.
    """
    reg = LabelRegistry()  # id 0 = Unlabeled
    by_name = {}
    if palette_source is not None:
        for info in palette_source.all_labels():
            if info.id != 0:
                by_name[info.name.lower()] = tuple(info.color)
    for i, name in enumerate(class_names):
        if i == 0:
            continue
        color = by_name.get(str(name).lower()) or DEFAULT_PALETTE[(i - 1) % len(DEFAULT_PALETTE)]
        reg.add_label_at(i, str(name), tuple(color))
    return reg


def models_compatible(project_label_names, model_class_names: list[str]) -> bool:
    """A model may run in a project when every class it predicts (index 0
    excluded — that is the model's background) exists in the project's
    ontology by name. Missing classes would silently map to Unlabeled."""
    names = {str(n).lower() for n in project_label_names}
    preds = [str(n).lower() for n in model_class_names[1:]]
    return bool(preds) and all(n in names for n in preds)


def ensure_sandbox(catalog):
    """Return the sandbox Project, creating + persisting it on first use."""
    from src.data.library_catalog import Project
    proj = catalog.projects.get(SANDBOX_PROJECT_ID)
    if proj is None:
        proj = Project(id=SANDBOX_PROJECT_ID, name=SANDBOX_NAME,
                       settings={"sandbox": True}, ontology_locked=True)
        catalog.projects[SANDBOX_PROJECT_ID] = proj
        catalog._save_projects()
    return proj
