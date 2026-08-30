"""Dataset export for training-ready labeled point clouds.

Supports three formats:
  - NPZ: single .npz per cloud (coord, color, label, feat, label_map)
  - HDF5: single .h5 with scene groups (requires h5py)
  - PTv3: Pointcept-compatible directory tree with .npy files

Handles contiguous label remapping, feature channel stacking, and
train/val/test splitting.

PTv3 contract:
  - class ids are contiguous 0..K-1.
  - Unlabeled (registry id 0) is EXPORTED AS A REAL CLASS at class id 0,
    NOT as IGNORE_INDEX — the model must learn "this is just background"
    as a valid output, or at inference it forces every background point
    into a foreground class (the 2026-05-10 collapse bug). See
    ``build_class_map`` for the full rationale.
  - Only points explicitly filtered out via ``allowed_ids`` (a user
    choosing to drop the background class from a dataset) become
    IGNORE_INDEX (255) so the loss masks them.
  - A dataset-wide ``classes.json`` sidecar carries class_names,
    class_colors (0..255 RGB), and ignore_index so the prediction
    importer can round-trip to the same layer with the same colors.
"""

import os
import glob
import shutil
import json
import numpy as np

from src.data.labels import LabelRegistry
from src.export.flatten import resolve_flatten


# PTv3/Pointcept convention: 255 means "ignore this point in the loss".
# Must fit in uint8 for voxel hash grids downstream; -1 is not safe.
IGNORE_INDEX = 255


def _color_to_u8(color) -> list[int]:
    r, g, b = color[0], color[1], color[2]
    return [
        int(round(max(0.0, min(1.0, r)) * 255)),
        int(round(max(0.0, min(1.0, g)) * 255)),
        int(round(max(0.0, min(1.0, b)) * 255)),
    ]


def build_class_map(
    registry: LabelRegistry,
    allowed_ids: set[int] | None = None,
    flatten_mode: str = "leaves",
    depth: int | None = None,
    subtree_root: int | None = None,
    custom: dict | None = None,
) -> tuple[dict[int, int], list[str], list[list[int]]]:
    """Build a deterministic registry-id → PTv3 class id mapping.

    One global map per dataset. The pipeline:

      1. Resolve the hierarchy flatten mode → ``{old_id: effective_id}``
         where multiple old ids may collapse to one effective id. See
         ``src/export/flatten.py`` for mode semantics.
      2. Intersect with ``allowed_ids`` if given — any old id whose
         effective id isn't allowed becomes IGNORE.
      3. Collect the set of unique effective ids, sort by effective
         id for determinism, assign each a contiguous class id
         starting at 0.
      4. Compose: original old id → effective id → class id.

    Unlabeled (registry id 0) is mapped to class 0 — i.e. it is
    EXPORTED AS A REAL CLASS, not as IGNORE_INDEX. This is essential
    for semantic segmentation: PT-v3's CrossEntropyLoss skips
    IGNORE_INDEX points entirely during loss computation, so if
    Unlabeled were ignored, the model would only learn the foreground/
    foreground discrimination among the small subset of labelled
    points and at inference time would force every background point
    into one of those classes (the bug we hit on 2026-05-10 — the
    model predicted ~90% of every cloud as ``superior endplate``
    because it had never seen "this is just bone" as a valid output).

    Using the registry — not ``np.unique(labels)`` — guarantees that
    class ids are stable across scenes. Per-scene uniques would give
    scene-A's class 2 a different meaning than scene-B's class 2 when
    scenes happen to have different present labels, which is a silent
    training corruption.

    Returns:
        old_to_new:   {registry_id: class_id}, ALWAYS includes {0: 0}
        class_names:  [name for class_id in 0..K-1] — class 0 = "Unlabeled"
        class_colors: [[r, g, b] uint8 for class_id in 0..K-1]
    """
    # 1. flatten
    flatten_map = resolve_flatten(
        registry, mode=flatten_mode, depth=depth,
        subtree_root=subtree_root, custom=custom,
    )

    # 2. filter by allowed_ids (applied to EFFECTIVE id, not original)
    if allowed_ids is not None:
        flatten_map = {
            old: eff for old, eff in flatten_map.items() if eff in allowed_ids
        }

    # 3. unique effective ids → contiguous class ids. Class 0 is
    #    reserved for Unlabeled (registry id 0). The Unlabeled root is
    #    always part of every registry, but ``resolve_flatten('leaves')``
    #    doesn't include the root in flatten_map.values(), so we decide
    #    inclusion from ``allowed_ids`` instead: include Unlabeled as a
    #    class unless the user explicitly filtered it out. This honours
    #    the user's intent — e.g. allowed_ids={1, 2} means "I only want
    #    these two foreground classes in my dataset, no background"
    #    and apply_class_map will write IGNORE_INDEX for points with
    #    registry id 0.
    eff_values = set(flatten_map.values())
    include_unlabeled = allowed_ids is None or 0 in allowed_ids
    non_unlabeled_eff_ids = sorted(eff for eff in eff_values if eff != 0)
    eff_to_class: dict[int, int] = {}
    if include_unlabeled:
        eff_to_class[0] = 0
        next_cls = 1
    else:
        next_cls = 0
    for eff in non_unlabeled_eff_ids:
        eff_to_class[eff] = next_cls
        next_cls += 1

    # 4. compose. Only seed {0: 0} when Unlabeled is part of the export;
    #    otherwise leave it absent so apply_class_map writes IGNORE_INDEX
    #    for any point whose registry id is 0.
    old_to_new: dict[int, int] = {}
    if include_unlabeled:
        old_to_new[0] = 0
    for old, eff in flatten_map.items():
        old_to_new[old] = eff_to_class[eff]

    # 5. class_names / class_colors. Class 0 = "Unlabeled" only when
    #    included; otherwise the contiguous list starts at the first
    #    non-zero effective id with class 0.
    class_names: list[str] = []
    class_colors: list[list[int]] = []
    if include_unlabeled:
        info_0 = registry.get(0)
        if info_0 is not None:
            class_names.append(info_0.name)
            class_colors.append(_color_to_u8(info_0.color))
        else:
            class_names.append("Unlabeled")
            class_colors.append([128, 128, 128])
    for eff in non_unlabeled_eff_ids:
        info = registry.get(eff)
        if info is not None:
            class_names.append(info.name)
            class_colors.append(_color_to_u8(info.color))
        else:
            class_names.append(f"label_{eff}")
            class_colors.append([128, 128, 128])
    return old_to_new, class_names, class_colors


def apply_class_map(labels: np.ndarray,
                    old_to_new: dict[int, int]) -> np.ndarray:
    """Apply a precomputed class map to a label array.

    Any registry id not in the map (e.g. a label that was filtered out
    via ``allowed_ids``) is emitted as IGNORE_INDEX. Registry id 0
    (Unlabeled) maps to class 0 by default — see ``build_class_map``.

    Returns an int64 array (PTv3 segment dtype).
    """
    out = np.full(labels.shape, IGNORE_INDEX, dtype=np.int64)
    for old_id, new_id in old_to_new.items():
        if new_id == IGNORE_INDEX:
            continue
        out[labels == old_id] = new_id
    return out


def remap_labels(labels: np.ndarray, registry: LabelRegistry,
                 allowed_ids: set[int] | None = None,
                 ) -> tuple[np.ndarray, dict[int, str]]:
    """Remap registry label IDs to contiguous PTv3 class ids.

    Returns ``(remapped_labels, id_to_name)``:
      - ``remapped_labels``: int64, with IGNORE_INDEX (255) wherever
        the source was unlabeled or outside ``allowed_ids``.
      - ``id_to_name``: ``{class_id: name}`` for real classes only.
        The ignore index is NOT included — it's not a class.

    Builds the class map from the registry so ids are stable across
    scenes even when individual clouds don't contain every label.
    """
    old_to_new, class_names, _ = build_class_map(registry, allowed_ids)
    remapped = apply_class_map(labels, old_to_new)
    id_to_name = {i: name for i, name in enumerate(class_names)}
    return remapped, id_to_name


def write_classes_json(path: str,
                       class_names: list[str],
                       class_colors: list[list[int]],
                       source_layer: str = "default",
                       flatten_mode: str = "leaves",
                       depth: int | None = None,
                       extra: dict | None = None) -> None:
    """Write the dataset-wide classes.json sidecar.

    Mirrors the layer's display names and colors so a prediction
    importer can reconstruct an identical layer on return. ``extra``
    is merged in verbatim for provenance fields (git sha, timestamp,
    source project, etc.).
    """
    payload = {
        "class_names": class_names,
        "class_colors": class_colors,
        "ignore_index": IGNORE_INDEX,
        "source_layer": source_layer,
        "flatten_mode": flatten_mode,
        "depth": depth,
        "num_classes": len(class_names),
    }
    if extra:
        payload.update(extra)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def stack_features(cloud, feature_keys: list[str] | None = None,
                   normalize: bool = False) -> np.ndarray:
    """Stack selected scalar fields into an (N, F) float32 feature matrix.

    ``feature_keys``: explicit list of scalar keys to include, in order.
        Defaults to ``[]`` — no extra features. RGB and normals already
        live in their own PTv3 channels (color.npy, normal.npy), so
        feat.npy should only carry user-picked extras.

    Passing ``None`` for feature_keys keeps the old auto-dump behavior
    (all scalars) for backwards compatibility with callers that still
    expect it, but NEW exporters should pass an explicit list.

    ``normalize``: min-max per-scene normalization. OFF by default —
    per-scene normalization is a training footgun because the same
    raw intensity can map to different normalized values in different
    scenes, breaking the feature's meaning. Prefer raw values and let
    the training pipeline compute dataset-wide stats.

    Missing keys are silently skipped (no crash). Keys that exist but
    don't match ``cloud.point_count`` are also skipped.
    """
    if feature_keys is None:
        keys = sorted(cloud.scalars.keys())
    else:
        keys = [k for k in feature_keys if k in cloud.scalars]

    feats: list[np.ndarray] = []
    for key in keys:
        v = cloud.scalars[key].astype(np.float32)
        if v.ndim != 1 or v.shape[0] != cloud.point_count:
            continue
        if normalize:
            vmin = float(v.min())
            vmax = float(v.max())
            if vmax - vmin > 1e-8:
                v = (v - vmin) / (vmax - vmin)
            else:
                v = np.zeros_like(v)
        feats.append(v.reshape(-1, 1))

    if not feats:
        return np.zeros((cloud.point_count, 0), dtype=np.float32)
    return np.concatenate(feats, axis=1).astype(np.float32)


def export_npz(cloud, output_path: str, registry: LabelRegistry,
               allowed_ids: set[int] | None = None,
               feature_keys: list[str] | None = None):
    """Export a single cloud as .npz with coord, color, label, feat."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    labels_remapped, id_to_name = remap_labels(cloud.labels, registry, allowed_ids)
    features = stack_features(cloud, feature_keys=feature_keys)
    np.savez_compressed(
        output_path,
        coord=cloud.positions.astype(np.float32),
        color=cloud.colors.astype(np.float32),
        label=labels_remapped.astype(np.int64),
        feat=features,
    )
    # Write label map as sibling JSON
    map_path = os.path.splitext(output_path)[0] + '_label_map.json'
    with open(map_path, 'w') as f:
        json.dump({str(k): v for k, v in id_to_name.items()}, f, indent=2)


def compute_normals(positions: np.ndarray, k: int = 16) -> np.ndarray:
    """Estimate per-point normals via local PCA. Returns (N, 3) float32.

    Uses Open3D if available (fast, robust); falls back to scipy KDTree
    + manual PCA. If neither is available, returns zeros.
    """
    n = len(positions)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    try:
        import open3d as o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(positions.astype(np.float64))
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
        return np.asarray(pc.normals, dtype=np.float32)
    except Exception:
        pass
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(positions)
        k_eff = min(k, n)
        _, idx = tree.query(positions, k=k_eff)
        normals = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            nbrs = positions[idx[i]]
            c = nbrs - nbrs.mean(axis=0)
            _, _, vh = np.linalg.svd(c, full_matrices=False)
            normals[i] = vh[-1]
        return normals
    except Exception:
        return np.zeros((n, 3), dtype=np.float32)


def export_ptv3_scene(cloud, output_dir: str, registry: LabelRegistry,
                      with_normals: bool = True,
                      allowed_ids: set[int] | None = None,
                      class_map: dict[int, int] | None = None,
                      feature_keys: list[str] | None = None) -> bool:
    """Export a single cloud as a PTv3/Pointcept directory.

    Creates output_dir/ with coord.npy, color.npy, normal.npy,
    segment.npy, instance.npy (if present), feat.npy (if present),
    and label_map.json.

    ``class_map`` is an optional precomputed registry-id → class-id
    table (from ``build_class_map``). Passing it in makes the scene's
    class ids consistent with the rest of a dataset-wide export; when
    omitted, a scene-local map is built from the registry (fine for
    single-scene debug exports, unsafe for multi-scene training).

    Returns True if the scene was written; False if it was skipped
    because every point would have been IGNORE under the supplied class
    map (no real labels in the destination ontology). Skipped scenes
    are NOT created on disk — the caller is expected to drop them from
    its split-assignment bookkeeping.
    """
    if class_map is None:
        class_map, class_names, _ = build_class_map(registry, allowed_ids)
    else:
        # Rebuild class_names from the supplied map for label_map.json.
        real_pairs = sorted(
            ((new, old) for old, new in class_map.items() if new != IGNORE_INDEX),
            key=lambda p: p[0],
        )
        class_names = []
        for _new, old in real_pairs:
            info = registry.get(old)
            class_names.append(info.name if info is not None else f"label_{old}")
    labels_remapped = apply_class_map(cloud.labels, class_map)
    # Refuse to write a scene where every point becomes IGNORE. PTv3's
    # CrossEntropyLoss masks IGNORE_INDEX points; an all-IGNORE scene
    # produces an empty [0, num_classes] tensor and crashes the loss with
    # "output with shape [] doesn't match the broadcast shape [0, K]" the
    # first time it hits val. Skip the scene and let the caller drop it
    # from the dataset split. This typically catches clouds whose
    # catalog labels are stale relative to the project's current
    # ontology (every id collapses to IGNORE on remap).
    if not np.any(labels_remapped != IGNORE_INDEX):
        name = os.path.basename(output_dir)
        print(
            f"Export skip: scene {name!r} has no labels in the destination "
            f"ontology (all points → IGNORE). Cloud file: {cloud.file_path}"
        )
        return False
    os.makedirs(output_dir, exist_ok=True)
    id_to_name = {i: name for i, name in enumerate(class_names)}
    # Default to empty feat.npy — color and normals already live in
    # their own channels. Only populated when the caller picks extras.
    features = stack_features(cloud, feature_keys=feature_keys if feature_keys is not None else [])

    np.save(os.path.join(output_dir, 'coord.npy'),
            cloud.positions.astype(np.float32))
    # PTv3 default configs expect color in [0, 255] uint8 range; our
    # clouds store floats in [0, 1], so upscale on export.
    color_u8 = np.clip(cloud.colors * 255.0, 0, 255).astype(np.uint8)
    np.save(os.path.join(output_dir, 'color.npy'), color_u8)

    if with_normals:
        normals = compute_normals(cloud.positions)
        np.save(os.path.join(output_dir, 'normal.npy'), normals)

    # PTv3 uses 'segment' for semantic segmentation labels, int64
    np.save(os.path.join(output_dir, 'segment.npy'),
            labels_remapped.astype(np.int64))

    # Optional instance ID channel, if the cloud carries one
    inst = cloud.scalars.get('instance')
    if inst is not None and len(inst) == cloud.point_count:
        np.save(os.path.join(output_dir, 'instance.npy'),
                inst.astype(np.int32))

    if features.shape[1] > 0:
        np.save(os.path.join(output_dir, 'feat.npy'), features)
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in id_to_name.items()}, f, indent=2)
    # Signal a successful write. Without this the function falls off the
    # end returning None, and the caller's ``if not wrote`` marks every
    # exported scene as skipped — zeroing splits.json's scene counts even
    # though the files are on disk (the total==0 bug).
    return True


def export_hdf5(clouds, output_path: str, registry: LabelRegistry,
                allowed_ids: set[int] | None = None,
                class_map: dict[int, int] | None = None,
                feature_keys: list[str] | None = None):
    """Export multiple clouds to a single HDF5 file.

    Each cloud becomes a group: /scene_000, /scene_001, ...
    Uses the shared ``class_map`` (or builds one from the registry) so
    class ids line up across all scenes in the file.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py required for HDF5 export. Install with: pip install h5py")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if class_map is None:
        class_map, class_names, _ = build_class_map(registry, allowed_ids)
    else:
        real_pairs = sorted(
            ((new, old) for old, new in class_map.items() if new != IGNORE_INDEX),
            key=lambda p: p[0],
        )
        class_names = []
        for _new, old in real_pairs:
            info = registry.get(old)
            class_names.append(info.name if info is not None else f"label_{old}")

    with h5py.File(output_path, 'w') as f:
        label_map = {str(i): name for i, name in enumerate(class_names)}
        f.attrs['label_map'] = json.dumps(label_map)
        f.attrs['ignore_index'] = IGNORE_INDEX

        for i, cloud in enumerate(clouds):
            g = f.create_group(f'scene_{i:03d}')
            g.create_dataset('coord', data=cloud.positions.astype(np.float32))
            g.create_dataset('color', data=cloud.colors.astype(np.float32))
            g.create_dataset('label', data=apply_class_map(cloud.labels, class_map))
            features = stack_features(
                cloud, feature_keys=feature_keys if feature_keys is not None else [])
            if features.shape[1] > 0:
                g.create_dataset('feat', data=features)
            g.attrs['file_path'] = cloud.file_path or ''


def split_indices(n: int, train: float, val: float, test: float,
                  seed: int = 42) -> tuple[list, list, list]:
    """Split n items deterministically into (train_idx, val_idx, test_idx)."""
    assert abs((train + val + test) - 1.0) < 1e-4, "ratios must sum to 1"
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # The rest is test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return train_idx, val_idx, test_idx


def export_dataset(clouds: list, registry: LabelRegistry,
                   output_dir: str, format: str = 'ptv3',
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15,
                   seed: int = 42,
                   allowed_ids: set[int] | None = None,
                   per_cloud_filters: list | None = None,
                   source_layer: str = "default",
                   flatten_mode: str = "leaves",
                   depth: int | None = None,
                   subtree_root: int | None = None,
                   custom_flatten: dict | None = None,
                   feature_keys: list[str] | None = None) -> dict:
    """Export a list of clouds as a training dataset with train/val/test splits.

    format: 'npz' | 'ptv3' | 'h5'
    allowed_ids: global label filter. Labels outside the set become
        IGNORE_INDEX (255) on export.
    per_cloud_filters: optional list[set[int] | None] parallel to clouds.
        A cloud's own filter intersects with ``allowed_ids``; points
        whose registry id falls outside the effective filter become
        IGNORE_INDEX. The global class map is built from the UNION
        of all per-cloud filters (intersected with ``allowed_ids``),
        so class ids stay consistent across every scene.

    Writes a dataset-level ``classes.json`` with class_names,
    class_colors, ignore_index, source_layer, and flatten_mode so a
    prediction importer can round-trip back to the same layer.

    Returns a summary dict with split assignments and paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Wipe stale scenes from any previous export. Without this, a smaller
    # new export rides along with leftover scene_NNN dirs from a larger
    # previous run — the LithiumDataset loader globs scene_* and
    # picks up everything, including incomplete leftovers that crash the
    # transform pipeline (KeyError: 'normal' on scenes that didn't get
    # fully written). Only the per-split dirs are removed; classes.json,
    # splits.json, and training_runs/ at output_dir are preserved.
    if format in ('ptv3', 'npz'):
        for split_name in ('train', 'val', 'test'):
            split_dir = os.path.join(output_dir, split_name)
            if not os.path.isdir(split_dir):
                continue
            for scene_path in glob.glob(os.path.join(split_dir, 'scene_*')):
                try:
                    if os.path.isdir(scene_path):
                        shutil.rmtree(scene_path)
                    elif os.path.isfile(scene_path):
                        os.remove(scene_path)
                except OSError as e:
                    print(f"Stale scene cleanup failed for {scene_path}: {e}")
    n = len(clouds)
    train_idx, val_idx, test_idx = split_indices(n, train_ratio, val_ratio, test_ratio, seed)

    # Resolve the global allowed-id set once. A per-cloud filter of
    # None means "any allowed id"; we take the union of every filter
    # that contributes, intersected with allowed_ids if it's set.
    if per_cloud_filters is None:
        global_allowed = allowed_ids
    else:
        any_open = False
        union: set[int] = set()
        for i in range(n):
            f = per_cloud_filters[i] if i < len(per_cloud_filters) else None
            if f is None:
                any_open = True
                break
            union.update(f)
        if any_open:
            global_allowed = allowed_ids
        elif allowed_ids is None:
            global_allowed = union
        else:
            global_allowed = union & allowed_ids

    class_map, class_names, class_colors = build_class_map(
        registry, global_allowed,
        flatten_mode=flatten_mode, depth=depth,
        subtree_root=subtree_root, custom=custom_flatten,
    )

    def _filter_for(i: int) -> set[int] | None:
        if per_cloud_filters is not None and i < len(per_cloud_filters):
            f = per_cloud_filters[i]
            return f if f is not None else allowed_ids
        return allowed_ids

    def _scene_class_map(i: int) -> dict[int, int]:
        """Per-scene class map — same ids as the global map, but any
        registry id not in this scene's filter is collapsed to IGNORE.
        """
        f = _filter_for(i)
        if f is None:
            return class_map
        return {
            old: (new if (old in f or new == IGNORE_INDEX) else IGNORE_INDEX)
            for old, new in class_map.items()
        }

    split_map = {}
    for i in train_idx:
        split_map[i] = 'train'
    for i in val_idx:
        split_map[i] = 'val'
    for i in test_idx:
        split_map[i] = 'test'

    if format == 'h5':
        train_clouds = [clouds[i] for i in train_idx]
        val_clouds = [clouds[i] for i in val_idx]
        test_clouds = [clouds[i] for i in test_idx]
        if train_clouds:
            export_hdf5(train_clouds, os.path.join(output_dir, 'train.h5'),
                        registry, class_map=class_map, feature_keys=feature_keys)
        if val_clouds:
            export_hdf5(val_clouds, os.path.join(output_dir, 'val.h5'),
                        registry, class_map=class_map, feature_keys=feature_keys)
        if test_clouds:
            export_hdf5(test_clouds, os.path.join(output_dir, 'test.h5'),
                        registry, class_map=class_map, feature_keys=feature_keys)
    else:
        # Indices whose scene was skipped (e.g. all-IGNORE for PTv3).
        # Removed from split_map below so splits.json reflects what
        # actually landed on disk.
        skipped: set[int] = set()
        for i, cloud in enumerate(clouds):
            split = split_map[i]
            cm = _scene_class_map(i)
            if format == 'npz':
                out_path = os.path.join(output_dir, split, f'scene_{i:03d}.npz')
                # NPZ path goes through the legacy remap_labels for back-
                # compat; it builds its own scene-local map from the
                # filter. Kept here because NPZ is debug-only and not
                # training-critical.
                export_npz(cloud, out_path, registry,
                           allowed_ids=_filter_for(i), feature_keys=feature_keys)
            elif format == 'ptv3':
                out_dir = os.path.join(output_dir, split, f'scene_{i:03d}')
                wrote = export_ptv3_scene(cloud, out_dir, registry, class_map=cm,
                                          feature_keys=feature_keys)
                if not wrote:
                    skipped.add(i)
            else:
                raise ValueError(f"Unknown format: {format}")
        # Drop skipped scenes from the split bookkeeping so splits.json
        # only references scenes that were actually written.
        for i in skipped:
            split_map.pop(i, None)
        train_idx = [i for i in train_idx if i not in skipped]
        val_idx = [i for i in val_idx if i not in skipped]
        test_idx = [i for i in test_idx if i not in skipped]

    summary = {
        'format': format,
        'output_dir': output_dir,
        'total': len(split_map) if format != 'h5' else n,
        'train': len(train_idx),
        'val': len(val_idx),
        'test': len(test_idx),
        'split_assignments': {str(i): s for i, s in split_map.items()},
        'num_classes': len(class_names),
        'ignore_index': IGNORE_INDEX,
    }
    with open(os.path.join(output_dir, 'splits.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    write_classes_json(
        os.path.join(output_dir, 'classes.json'),
        class_names=class_names,
        class_colors=class_colors,
        source_layer=source_layer,
        flatten_mode=flatten_mode,
        depth=depth,
        extra={
            'format': format,
            'num_scenes': n,
            'splits': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
            'feature_keys': list(feature_keys) if feature_keys else [],
            'subtree_root': subtree_root,
        },
    )
    return summary
