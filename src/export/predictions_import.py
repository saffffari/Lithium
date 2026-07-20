"""Import PTv3 predictions back into the label registry.

The round-trip loop from the design doc:

    label layer ──export──> train PTv3 ──infer──> per-point .npy
                                                         │
                           import_predictions ◄──────────┘
                                    │
                                    v
                         new labels in the registry

This module owns the "inference output → app-state" half of that
loop. It does NOT decide how the imported labels are displayed or
merged — it reads a ``.npy`` of PTv3 class ids plus the dataset's
``classes.json`` and returns an array of REGISTRY ids at the source
cloud's resolution, creating any missing labels in the registry with
a configurable name suffix so they're distinguishable from ground
truth in the UI.

The caller decides what to do with the result — attach it as a new
layer when the LayerStack lands (see memory: project_training_architecture),
diff it against the active labels, or just overwrite cloud.labels.
"""

import json
import os
import numpy as np

from src.data.labels import LabelRegistry
from src.export.dataset_export import IGNORE_INDEX


def load_classes_json(path: str) -> dict:
    """Read the ``classes.json`` sidecar emitted by ``export_dataset``."""
    with open(path) as f:
        data = json.load(f)
    required = {"class_names", "class_colors", "ignore_index"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"classes.json at {path} missing keys: {missing}")
    return data


def _ensure_label(registry: LabelRegistry, name: str,
                  color: tuple) -> int:
    """Return the registry id for ``name``, creating it if needed.

    Matches by exact name. New labels inherit the given color so the
    imported layer renders with the same palette as the ground truth
    it was trained against.
    """
    for info in registry.all_labels():
        if info.name == name:
            return info.id
    return registry.add_label(name, color=color)


def import_predictions(
    predictions_path: str,
    classes_json_path: str,
    registry: LabelRegistry,
    name_suffix: str = " (pred)",
) -> np.ndarray:
    """Read a PTv3 prediction ``.npy`` and map it back to registry ids.

    Arguments:
        predictions_path: path to an ``.npy`` of int32/int64 class ids,
            shape (N,). Values in ``[0, num_classes)`` are real
            predictions; values equal to ``ignore_index`` are left as
            registry id 0 (Unlabeled) in the output.
        classes_json_path: path to the sibling ``classes.json`` emitted
            by ``export_dataset``. Class ids in the prediction are
            interpreted against its ``class_names`` / ``class_colors``.
        registry: the current ``LabelRegistry``. Missing classes are
            added with a name of ``"<original_name><name_suffix>"`` so
            GT and Pred can coexist in the same registry without name
            collisions. Pass an empty suffix to import directly onto
            the existing labels (only do this when the prediction
            classes exactly match your GT names).

    Returns:
        ``np.ndarray[int32]`` of registry ids with shape ``(N,)``,
        ready to be written into ``cloud.labels`` OR kept as a side
        channel pending the LayerStack feature.
    """
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(predictions_path)
    preds = np.load(predictions_path)
    preds = np.ascontiguousarray(preds).reshape(-1)

    classes = load_classes_json(classes_json_path)
    class_names = classes["class_names"]
    class_colors = classes["class_colors"]
    ignore_index = int(classes.get("ignore_index", IGNORE_INDEX))
    num_classes = len(class_names)

    if len(class_colors) != num_classes:
        raise ValueError(
            f"classes.json: {num_classes} class_names but "
            f"{len(class_colors)} class_colors"
        )

    # Build the class-id → registry-id table once.
    class_to_registry = np.zeros(num_classes, dtype=np.int32)
    for cls_id in range(num_classes):
        name = class_names[cls_id] + name_suffix
        color_u8 = class_colors[cls_id]
        # normalize uint8 back to float for the registry
        color = (
            color_u8[0] / 255.0,
            color_u8[1] / 255.0,
            color_u8[2] / 255.0,
            1.0,
        )
        class_to_registry[cls_id] = _ensure_label(registry, name, color)

    # Apply. Any prediction == ignore_index becomes Unlabeled (0).
    # Out-of-range ids (shouldn't happen, but defensive) also become 0.
    out = np.zeros(preds.shape, dtype=np.int32)
    valid = (preds >= 0) & (preds < num_classes) & (preds != ignore_index)
    # Fancy-index once for speed; np handles the mask correctly.
    out[valid] = class_to_registry[preds[valid]]
    return out


def import_predictions_into_cloud(
    predictions_path: str,
    classes_json_path: str,
    cloud,
    registry: LabelRegistry,
    name_suffix: str = " (pred)",
) -> None:
    """Convenience wrapper — writes directly into ``cloud.labels``.

    Length-checks against ``cloud.point_count`` and raises on mismatch
    rather than silently truncating. Use ``import_predictions`` when
    you want the raw array without mutating the cloud.
    """
    out = import_predictions(
        predictions_path, classes_json_path, registry, name_suffix=name_suffix)
    if out.shape[0] != cloud.point_count:
        raise ValueError(
            f"prediction length {out.shape[0]} != cloud.point_count "
            f"{cloud.point_count} — was the prediction generated from "
            f"a different export?"
        )
    if cloud.labels is None:
        cloud.labels = out.copy()
    else:
        cloud.labels[:] = out
