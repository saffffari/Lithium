"""Catalog-side storage for full-resolution point clouds and their labels.

The 3Photon catalog *is* the dataset. Once a cloud is imported, its
positions, colors, scalars, and labels live under ``~/.3photon/library/``
keyed by ``file_key``. The original source file (PLY/LAS/NIfTI/...) is
only touched once, on first import; every subsequent open reads from
the catalog. This means:

- Closing and reopening 3Photon preserves every label automatically.
- Moving or deleting the source file does not break the catalog entry.
- Any future model can be trained directly off the catalog directory
  without an explicit ``EXPORT`` step (the catalog *is* the export).

Layout:

    ~/.3photon/library/
    ├── data/<file_key>.npz       coord + color + scalars  (write-once on import)
    ├── labels/<file_key>.npy     int32 per-point labels   (rewritten on every stroke)
    └── previews/<file_key>.npz   gallery thumbnail        (existing — separate)

The data file carries metadata (source path, source mtime at import
time, scalar key list) so we can detect a stale source on reload and
warn the user without auto-overwriting their work.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np

# Import the module rather than the symbol so test monkey-patching of
# ``library_paths.library_dir`` lands in the dispatch path here. Going
# through ``library_paths`` (not ``library_catalog``) keeps cloud_store
# free of any back-edge to the catalog module — see AR-2 in REPORT.md.
from src.data import library_paths
from src.data.point_cloud import PointCloudData


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Bumped when the on-disk format changes in a backward-incompatible way.
# `_migrate_meta` is the upgrade hook for older files.
FORMAT_VERSION_CURRENT = 1
_FORMAT_VERSION = FORMAT_VERSION_CURRENT  # legacy alias used inside save_cloud_data
_DATA_SUBDIR = "data"
_LABELS_SUBDIR = "labels"
_PREVIEWS_SUBDIR = "previews"
_MESHES_SUBDIR = "meshes"   # Poisson-reconstructed surfaces, lazy-built
_BACKUPS_SUBDIR = "backups"  # sibling of library/ — held under ~/.3photon/


def _data_dir() -> Path:
    p = Path(library_paths.library_dir()) / _DATA_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _labels_dir() -> Path:
    p = Path(library_paths.library_dir()) / _LABELS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def _meshes_dir() -> Path:
    p = Path(library_paths.library_dir()) / _MESHES_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def cloud_data_path(file_key: str) -> Path:
    return _data_dir() / f"{file_key}.npz"


def cloud_labels_path(file_key: str) -> Path:
    return _labels_dir() / f"{file_key}.npy"


def cloud_mesh_path(file_key: str) -> Path:
    return _meshes_dir() / f"{file_key}.npz"


def has_cloud_data(file_key: str) -> bool:
    return cloud_data_path(file_key).exists()


def has_cloud_labels(file_key: str) -> bool:
    return cloud_labels_path(file_key).exists()


def has_cloud_mesh(file_key: str) -> bool:
    return cloud_mesh_path(file_key).exists()


# ---------------------------------------------------------------------------
# Format-version migration
# ---------------------------------------------------------------------------

def _migrate_meta(meta: dict) -> dict:
    """Upgrade an on-disk metadata blob to the current format version.

    No-op for v1 — this is a scaffold so future format bumps have a
    single place to add migration steps. The function is total: it
    always returns a dict, and the returned dict is guaranteed to
    carry a ``format_version`` key matching ``FORMAT_VERSION_CURRENT``.
    """
    if not isinstance(meta, dict):
        return {"format_version": FORMAT_VERSION_CURRENT}
    out = dict(meta)
    version_raw = out.get("format_version")
    try:
        version = int(version_raw) if version_raw is not None else 0
    except (TypeError, ValueError):
        version = 0
    # Future versions: insert `if version < 2: ...` migration blocks here.
    # For v1 (or unknown versions stamped to current), we just normalize
    # the version field and return.
    out["format_version"] = FORMAT_VERSION_CURRENT
    return out


# ---------------------------------------------------------------------------
# Write verification
# ---------------------------------------------------------------------------

def _verify_write(path: Path, expect_min_bytes: int = 1) -> bool:
    """Confirm a freshly-written file actually exists and is non-empty.

    Used after every os.replace() in the savers so silent disk failures
    (full disk, transient I/O error, antivirus interference) surface
    immediately instead of being noticed only on the next load.
    Returns True if the file passes the check.
    """
    try:
        if not path.exists():
            return False
        return path.stat().st_size >= expect_min_bytes
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Save / load: full cloud (positions + colors + scalars + metadata)
# ---------------------------------------------------------------------------

def save_cloud_data(
    file_key: str,
    cloud: PointCloudData,
    source_path: str | None = None,
    source_mtime: float | None = None,
) -> None:
    """Atomically write the cloud's full-resolution data to the catalog.

    Stores positions, colors, every scalar field, plus a small JSON
    metadata blob (source path, source mtime at write time, scalar
    key list, format version). Atomic via tmp + os.replace. Silent on
    IO error — persistence is best-effort and never breaks the UI.
    """
    if not file_key or cloud is None:
        return
    if cloud.positions is None or cloud.positions.size == 0:
        return
    try:
        target = cloud_data_path(file_key)
        # numpy.savez_compressed unconditionally appends '.npz' if the
        # filename doesn't already end in it. Use a no-suffix temp name
        # so the resulting file lands at exactly tmp_with_npz.
        tmp_base = target.with_suffix("")  # strip the .npz
        tmp_base = tmp_base.parent / (tmp_base.name + "_tmp")
        tmp_with_npz = tmp_base.with_suffix(".npz")

        # Resolve source mtime if the caller didn't pass one but the file
        # exists — useful for stale-source detection on later loads.
        if source_mtime is None and source_path:
            try:
                source_mtime = os.path.getmtime(source_path)
            except OSError:
                source_mtime = None

        scalar_keys = sorted(cloud.scalars.keys())
        # Coordinate metadata — preserve world offset, source unit, and
        # voxel spacing so multi-cloud alignment survives catalog round-trips.
        cm = cloud.metadata if cloud.metadata else {}
        local_origin = cm.get('world_offset', [0.0, 0.0, 0.0])
        source_unit = cm.get('source_unit', 'raw')
        voxel_spacing = cm.get('voxel_spacing', None)
        source_label_id = cm.get('source_label_id', None)
        meta = {
            "format_version": _FORMAT_VERSION,
            "source_path": source_path or cloud.file_path,
            "source_mtime": float(source_mtime) if source_mtime is not None else None,
            "scalar_keys": scalar_keys,
            "point_count": int(cloud.point_count),
            "local_origin": [float(v) for v in local_origin],
            "source_unit": source_unit,
            "voxel_spacing": [float(v) for v in voxel_spacing] if voxel_spacing else None,
            # Segmentation label this cloud was carved from. PolyPose
            # pose-import matches transforms.json bones to clouds via this
            # field, so it MUST survive a catalog cold-load.
            "source_label_id": int(source_label_id) if source_label_id is not None else None,
        }

        np_kwargs = {
            "coord": np.ascontiguousarray(cloud.positions, dtype=np.float32),
            "color": np.ascontiguousarray(cloud.colors, dtype=np.float32),
            "_meta_json": np.frombuffer(json.dumps(meta).encode('utf-8'), dtype=np.uint8).copy(),
        }
        for k in scalar_keys:
            np_kwargs[f"scalar_{k}"] = np.ascontiguousarray(
                cloud.scalars[k], dtype=np.float32
            )
        # PolyPose supine→standing matrix, stored only when present so
        # existing posed-cloud files don't grow a useless identity blob.
        if getattr(cloud, "pose_to_world", None) is not None:
            np_kwargs["pose_to_world"] = np.ascontiguousarray(
                cloud.pose_to_world, dtype=np.float32
            )

        np.savez_compressed(str(tmp_base), **np_kwargs)
        # savez_compressed wrote to tmp_base + ".npz" — promote it to target.
        os.replace(tmp_with_npz, target)
        # Verify the file actually landed. Catches silent disk failures
        # (full disk, antivirus quarantine) before they go unnoticed.
        if not _verify_write(target, expect_min_bytes=64):
            print(
                f"Catalog data verify failed for {file_key} — wrote 0 bytes "
                f"or file vanished. Refusing to mark as saved."
            )
    except (OSError, ValueError) as e:
        print(f"Catalog data save failed for {file_key}: {e}")
        # Best-effort cleanup of any half-written tmp file.
        try:
            if 'tmp_with_npz' in locals() and tmp_with_npz.exists():
                tmp_with_npz.unlink()
        except OSError:
            pass


def load_cloud_data(
    file_key: str,
) -> tuple[PointCloudData, dict] | None:
    """Reload a previously-saved cloud from the catalog.

    Returns ``(cloud, metadata)`` where ``metadata`` is the dict written
    by ``save_cloud_data`` (with at least ``source_path`` and
    ``source_mtime`` keys when present). Returns ``None`` if the file
    is missing or unreadable so callers can fall back to the source
    loader.
    """
    path = cloud_data_path(file_key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            files = set(npz.files)
            if "coord" not in files or "color" not in files:
                return None

            positions = np.asarray(npz["coord"], dtype=np.float32)
            colors = np.asarray(npz["color"], dtype=np.float32)

            meta: dict = {}
            if "_meta_json" in files:
                try:
                    raw = npz["_meta_json"]
                    # New format: UTF-8 byte array; legacy: pickled object
                    if raw.dtype == np.uint8:
                        meta = json.loads(bytes(raw).decode('utf-8'))
                    else:
                        meta = json.loads(str(raw.item()))
                except (TypeError, ValueError, json.JSONDecodeError):
                    meta = {}
            # Run any necessary format-version upgrade.
            meta = _migrate_meta(meta)

            scalars: dict[str, np.ndarray] = {}
            scalar_keys = meta.get("scalar_keys") or [
                f[len("scalar_"):] for f in files if f.startswith("scalar_")
            ]
            for k in scalar_keys:
                key = f"scalar_{k}"
                if key in files:
                    scalars[k] = np.asarray(npz[key], dtype=np.float32)

            # PolyPose pose, if persisted. Must be pulled inside the
            # with-block — npz arrays are lazy-loaded and become unreadable
            # once the file handle closes.
            pose_to_world = None
            if "pose_to_world" in files:
                try:
                    pose_to_world = np.asarray(
                        npz["pose_to_world"], dtype=np.float32
                    ).reshape(4, 4).copy()
                except (ValueError, KeyError):
                    pose_to_world = None

        cloud = PointCloudData(
            positions=positions,
            colors=colors,
            scalars=scalars,
            file_path=meta.get("source_path", "") or "",
        )
        # Restore coordinate metadata that was persisted alongside the cloud.
        if meta.get("local_origin"):
            cloud.metadata['world_offset'] = meta["local_origin"]
        if meta.get("source_unit"):
            cloud.metadata['source_unit'] = meta["source_unit"]
        if meta.get("voxel_spacing"):
            cloud.metadata['voxel_spacing'] = tuple(meta["voxel_spacing"])
        if meta.get("source_label_id") is not None:
            cloud.metadata['source_label_id'] = int(meta["source_label_id"])
        if pose_to_world is not None:
            cloud.pose_to_world = pose_to_world
        return cloud, meta
    except (OSError, ValueError, EOFError, KeyError) as e:
        print(f"Catalog data load failed for {file_key}: {e}")
        return None


# ---------------------------------------------------------------------------
# Save / load: per-cloud labels
# ---------------------------------------------------------------------------

def save_cloud_labels(file_key: str, labels: np.ndarray,
                      catalog=None) -> str | None:
    """Atomically persist a cloud's int32 label array to the catalog.

    Returns ``None`` on success, or a human-readable error string on
    failure. LS-7: callers (App._persist_cloud_labels) flash the
    string through the status banner so save errors stop being
    silent. ``print`` is kept for stdout-aware contexts (CLI / tests).

    CP-4: when ``catalog`` is supplied (typically ``app.catalog``),
    acquires the per-file_key lock for the duration of the write so
    a background worker can't interleave its write with a main-
    thread write on the same key. Per-key locks let writes to
    DIFFERENT keys still run in parallel (the typical batch case).
    Pass ``catalog=None`` to skip locking (single-threaded contexts:
    CLI batch tools, headless tests).
    """
    if not file_key or labels is None or labels.size == 0:
        return None

    def _do_write() -> str | None:
        target = cloud_labels_path(file_key)
        tmp_base = target.parent / f"_tmp_{target.stem}"
        tmp_with_npy = tmp_base.with_suffix(".npy")
        try:
            arr = np.ascontiguousarray(labels, dtype=np.int32)
            np.save(str(tmp_base), arr)
            os.replace(tmp_with_npy, target)
            if not _verify_write(target, expect_min_bytes=64):
                msg = (f"Catalog labels verify failed for {file_key} — "
                       f"wrote 0 bytes or file vanished. Refusing to mark as saved.")
                print(msg)
                return msg
        except (OSError, ValueError) as e:
            msg = f"Catalog labels save failed for {file_key}: {e}"
            print(msg)
            try:
                if tmp_with_npy.exists():
                    tmp_with_npy.unlink()
            except OSError:
                pass
            return msg
        return None

    if catalog is not None and hasattr(catalog, "labels_lock_for"):
        with catalog.labels_lock_for(file_key):
            return _do_write()
    return _do_write()


def snapshot_labels(file_keys: list[str]) -> tuple[Path, int, int]:
    """Copy each cloud's catalog labels file to a timestamped backup dir.

    Used by File > Save Points so the user has a manual recovery point
    they can roll back to. Backups live at
    ``<catalog>/backups/<YYYY-MM-DD_HH-MM-SS>/<file_key>.npy`` so the
    layout mirrors the live catalog and ``copy back`` is a plain file
    copy.

    Returns ``(backup_dir, copied, skipped)``. Skipped count includes
    file_keys with no live labels file and any per-file copy failure.
    """
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = Path(library_paths.library_dir()) / "backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    for fk in file_keys:
        if not fk:
            skipped += 1
            continue
        src = cloud_labels_path(fk)
        if not src.exists():
            skipped += 1
            continue
        try:
            shutil.copy2(src, backup_dir / f"{fk}.npy")
            copied += 1
        except OSError as e:
            print(f"Snapshot copy failed for {fk}: {e}")
            skipped += 1
    return backup_dir, copied, skipped


def prune_old_snapshots(keep_last: int) -> int:
    """Delete snapshot directories beyond the most-recent ``keep_last``.

    Snapshot dirs are sorted by name (timestamp format
    ``YYYY-MM-DD_HH-MM-SS`` sorts chronologically as a string). Returns
    the number of directories pruned.
    """
    if keep_last < 1:
        return 0
    backups_root = Path(library_paths.library_dir()) / "backups"
    if not backups_root.exists():
        return 0
    try:
        dirs = sorted(
            (p for p in backups_root.iterdir() if p.is_dir()),
            key=lambda p: p.name,
        )
    except OSError as e:
        print(f"Snapshot prune scan failed: {e}")
        return 0
    if len(dirs) <= keep_last:
        return 0
    pruned = 0
    for old in dirs[:-keep_last]:
        try:
            shutil.rmtree(old)
            pruned += 1
        except OSError as e:
            print(f"Snapshot prune failed for {old.name}: {e}")
    return pruned


_PREVIEW_LABELS_SUBDIR = "preview_labels"


def _preview_labels_dir() -> Path:
    p = Path(library_paths.library_dir()) / _PREVIEW_LABELS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def preview_labels_path(file_key: str) -> Path:
    return _preview_labels_dir() / f"{file_key}.npy"


def save_preview_labels(file_key: str, labels: np.ndarray) -> None:
    """Persist preview-resolution labels alongside the full ones.

    The catalog ``labels/<file_key>.npy`` file holds FULL-resolution
    labels (what the user painted on full_gpu). The preview gpu has
    its own downsampled labels array — without a separate persisted
    copy, those are lost across restarts and only repopulate when the
    user re-opens each cloud in Light Table. That's why the Train
    tab's "labelled" check looked stale until every cloud was cycled
    through. Writing this file fixes that.
    """
    if not file_key or labels is None or labels.size == 0:
        return
    target = preview_labels_path(file_key)
    tmp_base = target.parent / f"_tmp_{target.stem}"
    tmp_with_npy = tmp_base.with_suffix(".npy")
    try:
        arr = np.ascontiguousarray(labels, dtype=np.int32)
        np.save(str(tmp_base), arr)
        os.replace(tmp_with_npy, target)
    except (OSError, ValueError) as e:
        print(f"Preview labels save failed for {file_key}: {e}")
        try:
            if tmp_with_npy.exists():
                tmp_with_npy.unlink()
        except OSError:
            pass


def load_preview_labels(file_key: str,
                        expect_count: int | None = None) -> np.ndarray | None:
    """Return the persisted preview-resolution labels for ``file_key`` or None.

    When ``expect_count`` is provided, the loaded array's length is
    validated against it — a mismatch returns None and logs the issue
    rather than handing back a wrong-shape array that the caller might
    silently broadcast or truncate. Pass the preview's
    ``point_count`` whenever it's known.
    """
    path = preview_labels_path(file_key)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
    except (OSError, ValueError) as e:
        print(f"Preview labels load failed for {file_key}: {e}")
        return None
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    if expect_count is not None and arr.shape[0] != expect_count:
        print(f"Preview labels for {file_key}: length {arr.shape[0]} "
              f"!= expected {expect_count}; ignoring.")
        return None
    return arr


def load_cloud_labels(file_key: str) -> np.ndarray | None:
    """Return the saved int32 label array for ``file_key`` or None."""
    path = cloud_labels_path(file_key)
    if not path.exists():
        return None
    try:
        arr = np.load(path)
        return np.asarray(arr, dtype=np.int32)
    except (OSError, ValueError) as e:
        print(f"Catalog labels load failed for {file_key}: {e}")
        return None


# ---------------------------------------------------------------------------
# Save / load: Poisson-reconstructed mesh
# ---------------------------------------------------------------------------

def save_cloud_mesh(file_key: str, mesh: dict) -> None:
    """Persist a Poisson-reconstructed mesh for ``file_key``.

    ``mesh`` is the dict returned by ``mesh_builder.build_mesh``:
    ``{verts, faces, normals, vertex_labels}``. Written atomically via
    tmp + os.replace so a crashed write doesn't half-poison the file.
    Silent on IO failure — meshes are derived data, callers always
    have the source cloud to rebuild from.
    """
    if not file_key or mesh is None:
        return
    if "verts" not in mesh or "faces" not in mesh:
        return
    target = cloud_mesh_path(file_key)
    tmp_base = target.with_suffix("")
    tmp_base = tmp_base.parent / (tmp_base.name + "_tmp")
    tmp_with_npz = tmp_base.with_suffix(".npz")
    try:
        np.savez_compressed(
            str(tmp_base),
            verts=np.ascontiguousarray(mesh["verts"], dtype=np.float32),
            faces=np.ascontiguousarray(mesh["faces"], dtype=np.int32),
            normals=np.ascontiguousarray(
                mesh.get("normals", np.zeros_like(mesh["verts"])),
                dtype=np.float32),
            vertex_labels=np.ascontiguousarray(
                mesh.get("vertex_labels",
                         np.zeros(len(mesh["verts"]), dtype=np.int32)),
                dtype=np.int32),
        )
        os.replace(tmp_with_npz, target)
    except (OSError, ValueError) as e:
        print(f"Mesh save failed for {file_key}: {e}")
        try:
            if 'tmp_with_npz' in locals() and tmp_with_npz.exists():
                tmp_with_npz.unlink()
        except OSError:
            pass


def load_cloud_mesh(file_key: str) -> dict | None:
    """Return ``{verts, faces, normals, vertex_labels}`` for ``file_key``
    or None if the mesh hasn't been built yet."""
    path = cloud_mesh_path(file_key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            return {
                "verts": np.asarray(npz["verts"], dtype=np.float32),
                "faces": np.asarray(npz["faces"], dtype=np.int32),
                "normals": np.asarray(npz["normals"], dtype=np.float32),
                "vertex_labels": np.asarray(
                    npz["vertex_labels"], dtype=np.int32),
            }
    except (OSError, ValueError, KeyError) as e:
        print(f"Mesh load failed for {file_key}: {e}")
        return None


def drop_cloud_mesh(file_key: str) -> None:
    """Delete the cached mesh for ``file_key``. Called when the source
    cloud's labels change enough that the mesh's vertex_labels are
    stale; the next render triggers a rebuild."""
    try:
        cloud_mesh_path(file_key).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Cross-project label migration
# ---------------------------------------------------------------------------

def migrate_cloud_labels_to_project(
    file_keys: list[str],
    source_registry,
    dest_project,
) -> dict:
    """Carry a cloud's painted labels into a destination project's ontology.

    Labels are stored per-cloud at ``labels/<file_key>.npy`` as raw int
    IDs that only have meaning under whatever registry was active when
    the paint happened (the source registry — typically ``app.label_registry``
    for the currently active view). When the same cloud is added to a
    different project, those IDs may collide with the destination's
    registry (e.g. id=3 means "Spinous Process" in source, "Pedicle" in
    dest). This helper resolves that:

    For each cloud:
      1. Read its labels file.
      2. For each unique non-zero source id present, look up the source
         registry's name. Then look that name up in the destination
         project's registry:
           - found  → reuse the destination's existing id (the painted
             points join that layer)
           - missing → add a new label to the destination registry with
             the source's name + color, and use that new id
      3. Rewrite the cloud's labels file with the destination ids.
      4. Mirror the same remap on the preview labels file when present
         so the gallery/contact-sheet thumbnails stay accurate.

    If ``dest_project`` has no ontology yet, ``source_registry`` is
    cloned wholesale as the destination's ontology — no per-cloud
    remap is needed in that case (the cloud's existing ids stay valid).

    The destination project's ``ontology_data`` is updated in-place.
    The caller is responsible for persisting projects.json (typically
    via ``LibraryCatalog.add_to_project``, which calls ``_save_projects``).

    Returns ``{"clouds_remapped": int, "labels_added": int}``.
    """
    # Late imports — cloud_store is otherwise free of label/registry deps
    # and several tests monkey-patch label_catalog.library_dir without
    # ever touching LabelRegistry.
    from src.data.labels import LabelRegistry

    summary = {"clouds_remapped": 0, "labels_added": 0}
    if not file_keys or source_registry is None or dest_project is None:
        return summary

    # Fast path: destination project has no ontology. Adopt the source
    # registry as-is so the cloud's existing ids stay valid; no per-cloud
    # rewrites needed.
    if dest_project.ontology_data is None:
        dest_project.ontology_data = source_registry.to_json()
        return summary

    dest_registry = LabelRegistry.from_json(dest_project.ontology_data)
    name_to_dest_id = {info.name: info.id for info in dest_registry.all_labels()}

    for fk in file_keys:
        labels = load_cloud_labels(fk)
        if labels is None or labels.size == 0:
            continue
        unique_ids = np.unique(labels)
        non_zero = unique_ids[unique_ids != 0]
        if non_zero.size == 0:
            continue  # nothing painted, nothing to migrate

        remap: dict[int, int] = {0: 0}
        any_remapped = False
        for src_id in non_zero:
            src_id_i = int(src_id)
            src_info = source_registry.get(src_id_i)
            if src_info is None:
                # The cloud carries an id we can't name. Leave it as-is
                # rather than guessing — the user can sort it out manually.
                remap[src_id_i] = src_id_i
                continue
            dest_id = name_to_dest_id.get(src_info.name)
            if dest_id is None:
                try:
                    dest_id = dest_registry.add_label(
                        name=src_info.name, color=src_info.color)
                except ValueError as e:
                    # Hit MAX_LABELS or similar — preserve the source id
                    # so points aren't silently relabelled to 0.
                    print(f"migrate: add_label({src_info.name}) failed: {e}")
                    remap[src_id_i] = src_id_i
                    continue
                name_to_dest_id[src_info.name] = dest_id
                summary["labels_added"] += 1
            remap[src_id_i] = dest_id
            if dest_id != src_id_i:
                any_remapped = True

        if not any_remapped:
            continue

        # Apply remap to the full-resolution labels and preview labels.
        new_labels = labels.copy()
        for src_id_i, dest_id in remap.items():
            if src_id_i != dest_id:
                new_labels[labels == src_id_i] = dest_id
        save_cloud_labels(fk, new_labels)

        prev = load_preview_labels(fk)
        if prev is not None:
            new_prev = prev.copy()
            for src_id_i, dest_id in remap.items():
                if src_id_i != dest_id:
                    new_prev[prev == src_id_i] = dest_id
            save_preview_labels(fk, new_prev)

        summary["clouds_remapped"] += 1

    # Persist the (possibly extended) destination ontology back onto the
    # project. Caller still needs to flush projects.json via the catalog.
    dest_project.ontology_data = dest_registry.to_json()
    return summary


# ---------------------------------------------------------------------------
# Stale-source detection
# ---------------------------------------------------------------------------

def is_source_stale(meta: dict, source_path: str | None) -> bool:
    """True if the source file's mtime is newer than the saved data.

    Used by callers to warn the user (without auto-overwriting their
    labels) when the original PLY/LAS/NIfTI has been edited externally
    since 3Photon last imported it.
    """
    saved_mtime = meta.get("source_mtime") if meta else None
    if saved_mtime is None or not source_path:
        return False
    try:
        live_mtime = os.path.getmtime(source_path)
    except OSError:
        return False
    # 1-second slop tolerates filesystem timestamp granularity.
    return live_mtime > float(saved_mtime) + 1.0


# ---------------------------------------------------------------------------
# Cleanup helper (manual, not auto-called)
# ---------------------------------------------------------------------------

def drop_cloud(file_key: str) -> None:
    """Remove both data and labels for ``file_key``. Silent on missing files."""
    for p in (cloud_data_path(file_key), cloud_labels_path(file_key)):
        try:
            p.unlink()
        except OSError:
            pass


def drop_preview_labels(file_key: str) -> None:
    """Remove ``preview_labels/<file_key>.npy``. Silent on missing file.

    CP-1: previously orphaned when a cloud was removed via
    ``LibraryCatalog.remove_entry`` because remove_entry didn't drop
    the preview-labels file.
    """
    try:
        preview_labels_path(file_key).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Integrity check (read-only)
# ---------------------------------------------------------------------------

def check_catalog_integrity(catalog=None) -> dict:
    """Cross-reference index entries with on-disk data + labels files.

    Returns a small status dict::

        {
            "entries_count": int,    # number of entries in index.json
            "with_data":     int,    # entries that also have data/<key>.npz
            "with_labels":   int,    # entries that also have labels/<key>.npy
            "orphans":       [keys], # data/labels files NOT referenced by index
            "missing_data":  [keys], # index entries with no data file at all
        }

    Read-only — never modifies anything. The caller decides whether to
    act on orphans / missing files.

    ``catalog`` may be a live ``LibraryCatalog`` instance; if omitted,
    we read the on-disk ``index.json`` directly so this works during
    startup before the catalog has been instantiated.
    """
    lib_dir = Path(library_paths.library_dir())
    data_dir = lib_dir / _DATA_SUBDIR
    labels_dir = lib_dir / _LABELS_SUBDIR

    # Resolve the set of indexed file_keys.
    indexed_keys: set[str] = set()
    if catalog is not None and hasattr(catalog, "entries"):
        indexed_keys = set(catalog.entries.keys())
    else:
        index_path = lib_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    indexed_keys = set(raw.keys())
            except (OSError, json.JSONDecodeError):
                pass

    def _files_in(dir_path: Path, suffix: str) -> set[str]:
        if not dir_path.exists():
            return set()
        try:
            return {
                p.stem for p in dir_path.iterdir()
                if p.is_file() and p.suffix == suffix and not p.name.startswith("_tmp_")
            }
        except OSError:
            return set()

    data_keys = _files_in(data_dir, ".npz")
    label_keys = _files_in(labels_dir, ".npy")

    with_data = indexed_keys & data_keys
    with_labels = indexed_keys & label_keys
    orphans = sorted((data_keys | label_keys) - indexed_keys)
    missing_data = sorted(indexed_keys - data_keys)

    return {
        "entries_count": len(indexed_keys),
        "with_data": len(with_data),
        "with_labels": len(with_labels),
        "orphans": orphans,
        "missing_data": missing_data,
    }


def format_integrity_summary(status: dict) -> str:
    """One-line human summary of ``check_catalog_integrity`` output."""
    parts = [
        f"{status.get('entries_count', 0)} entries",
        f"{status.get('with_data', 0)} cached",
        f"{status.get('with_labels', 0)} labeled",
    ]
    n_orphans = len(status.get("orphans", []))
    n_missing = len(status.get("missing_data", []))
    if n_orphans:
        parts.append(f"{n_orphans} orphans")
    if n_missing:
        parts.append(f"{n_missing} missing")
    return "Catalog: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Wipe (destructive — guarded by an explicit caller-side confirmation)
# ---------------------------------------------------------------------------

def _backups_dir() -> Path:
    """The sibling backups directory under ``~/.3photon/``."""
    return Path(library_paths.library_dir()).parent / _BACKUPS_SUBDIR


def wipe_catalog(backup: bool = True) -> Path | None:
    """Erase the entire library catalog. Returns the backup tarball path.

    Layout after wipe:
        ~/.3photon/library/
        ├── data/         (empty)
        ├── labels/       (empty)
        └── previews/     (empty)

    All catalog top-level files (index.json, schema.json, projects.json,
    .lock) are removed. The directory hierarchy is recreated empty so the
    next session starts cleanly without needing to mkdir on first save.

    With ``backup=True`` (the default), a tar.gz of the entire library is
    written to ``~/.3photon/backups/library_<unix_ts>.tar.gz`` BEFORE the
    wipe runs. The function refuses to wipe (returns None) if the backup
    step fails, so the user never loses data without a recovery path.

    With ``backup=False`` the wipe runs immediately and returns None.

    Caller is responsible for confirming with the user. This function is
    intentionally not gated by any flag — the caller is the firewall.
    """
    lib_dir = Path(library_paths.library_dir())
    if not lib_dir.exists():
        # Nothing to wipe; still recreate the empty subdirs so the
        # rest of the catalog code path doesn't trip on missing dirs.
        for sub in (_DATA_SUBDIR, _LABELS_SUBDIR, _PREVIEWS_SUBDIR):
            (lib_dir / sub).mkdir(parents=True, exist_ok=True)
        return None

    backup_path: Path | None = None
    if backup:
        try:
            backups_dir = _backups_dir()
            backups_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            backup_path = backups_dir / f"library_{ts}.tar.gz"
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(lib_dir, arcname=lib_dir.name)
        except (OSError, tarfile.TarError) as e:
            print(
                f"Wipe aborted: backup to {backup_path} failed: {e}. "
                f"Catalog left untouched."
            )
            return None

    # Wipe the contents (NOT the library/ directory itself, in case
    # something else is watching it).
    for child in list(lib_dir.iterdir()):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except OSError as e:
            print(f"Wipe: failed to remove {child}: {e}")

    # Recreate the empty subdirs.
    for sub in (_DATA_SUBDIR, _LABELS_SUBDIR, _PREVIEWS_SUBDIR):
        (lib_dir / sub).mkdir(parents=True, exist_ok=True)

    return backup_path


def estimate_catalog_size() -> int:
    """Sum the size of every file under the library directory.

    Used by the wipe pre-confirmation prompt to show the user how much
    they're about to back up. Returns 0 if the dir doesn't exist.
    """
    lib_dir = Path(library_paths.library_dir())
    if not lib_dir.exists():
        return 0
    total = 0
    try:
        for root, _dirs, files in os.walk(lib_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return total
