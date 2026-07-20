"""Per-point label file I/O — companion .npy label arrays.

Handles discovery and application of ``{stem}_labels.npy`` files that
sit next to point cloud files, where each ``.npy`` is an ``(N,) int32``
array of label IDs matching the vertex order.
"""

import os
import numpy as np

from src.data.point_cloud import PointCloudData


_LABELMAP_SUFFIXES = (
    '_labels', '_label', '_seg', '_segmentation', '_mask', '_lm', '_gt',
)


def find_companion_label_array(cloud_path: str) -> str | None:
    """Find a per-point ``.npy`` label file sitting next to a point cloud.

    Checks for ``{stem}_labels.npy`` next to the source file.  This is
    the companion format produced by batch converters (e.g. the VerSe
    converter) where each ``.npy`` is an ``(N,) int32`` array of label IDs
    matching the vertex order.
    """
    parent = os.path.dirname(cloud_path) or '.'
    stem = os.path.splitext(os.path.basename(cloud_path))[0]
    for suffix in _LABELMAP_SUFFIXES:
        candidate = os.path.join(parent, f"{stem}{suffix}.npy")
        if os.path.exists(candidate):
            return candidate
    return None


def apply_label_array_to_cloud(cloud: PointCloudData,
                               label_path: str,
                               registry) -> tuple[int, list[int]]:
    """Load a per-point ``.npy`` label array and apply to *cloud*.

    The ``.npy`` file must contain an ``(N,)`` integer array with the
    same length as ``cloud.positions``.  Values are treated as registry
    label IDs (e.g. VERSE ontology IDs 1-28).  Creates missing registry
    entries as needed.

    Returns ``(labeled_count, created_ids)``.
    """
    raw = np.load(label_path).astype(np.int32)
    if raw.ndim != 1 or len(raw) != cloud.point_count:
        raise ValueError(
            f"Label array length {len(raw)} != cloud point count "
            f"{cloud.point_count}")

    unique_vals = np.unique(raw)
    unique_vals = unique_vals[unique_vals != 0]

    value_to_id: dict[int, int] = {}
    created_ids: list[int] = []
    existing_by_name = {info.name: info.id for info in registry.all_labels()}

    for val in unique_vals:
        iv = int(val)
        if iv in registry:
            value_to_id[iv] = iv
        else:
            name = f"Label {iv}"
            if name in existing_by_name:
                value_to_id[iv] = existing_by_name[name]
            else:
                new_id = registry.add_label(name)
                value_to_id[iv] = new_id
                created_ids.append(new_id)

    # LS-2: merge-not-zero. The previous implementation zeroed the
    # cloud's entire label array before applying the companion .npy,
    # silently wiping any manual paint the user had already done on
    # this cloud (including labels just loaded from the catalog
    # cache on re-open). Now we only overwrite indices where the
    # source array carries a non-zero value — zeros in the source
    # are treated as "no opinion, keep what's already there".
    for val, label_id in value_to_id.items():
        cloud.labels[raw == val] = label_id

    labeled_count = int((cloud.labels != 0).sum())
    return labeled_count, created_ids
