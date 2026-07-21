"""Global library catalog — persistent index of every imported point cloud.

Models the Contact Sheet's backing store on Adobe Lightroom's library:

    FOLDERS      — auto-derived tree of source directories on disk
    PROJECTS     — user-defined, drag-to-organize groupings with ontology
    SMART        — "All Photos" and "Recent Imports" virtual views

The library persists across sessions under ``~/.3photon/library/``:

    index.json        metadata for every known cloud (file_key → entry dict)
    projects.json     user projects (id → {name, file_keys, ontology, settings})
    previews/         per-cloud downsampled .npz previews keyed by file_key

The single instance outlives individual ``load_directory`` calls — every
cloud the user has ever imported stays browsable forever (until they
explicitly remove it), regardless of which folder it originally came from.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from src.data import library_paths
from src.data.loader import load_point_cloud
from src.data.point_cloud import PointCloudData
from src.data.resampler import voxel_downsample
from src.utils.file_hash import compute_file_key


# Smart view identifiers
SMART_ALL = "smart:all"
SMART_RECENT = "smart:recent"
SMART_MISSING = "smart:missing"

# Preview defaults
PREVIEW_POINTS = 50_000
RECENT_LIMIT = 50


def library_dir() -> str:
    """Back-compat shim — delegates to ``library_paths.library_dir``.

    cloud_store, catalog_lock, and tests now go through
    ``library_paths.library_dir`` directly. This shim stays so any
    external tool still importing ``library_catalog.library_dir``
    keeps working.
    """
    return library_paths.library_dir()


class LibraryEntry:
    """A single cloud tracked in the library.

    Holds metadata plus an optional in-memory preview (loaded lazily from
    the .npz on disk). The ``file_key`` is stable across sessions as long
    as the underlying file's path, size and mtime don't change.
    """

    def __init__(
        self,
        file_key: str,
        file_path: str,
        point_count: int = 0,
        bounds_min=None,
        bounds_max=None,
        import_date: float | None = None,
        last_opened: float | None = None,
        # Reserved fields, populated by IMAGING when DICOM metadata is
        # present, otherwise None. Schema-only for now — no UI reads or
        # writes these yet, but having them in place means future code
        # doesn't trigger a catalog migration.
        patient_id: str | None = None,
        study_id: str | None = None,
        region: str | None = None,
        pose: str | None = None,
        paired_entry_key: str | None = None,
        pose_transform: list[float] | None = None,
        display_name: str | None = None,
    ):
        self.file_key = file_key
        self.file_path = os.path.abspath(file_path)
        # CP-9: ``display_name`` is the user-facing name (after rename).
        # ``name`` stays the source basename so file-path-derived
        # heuristics still work; UI reads ``display_name or name``.
        self.display_name: str | None = display_name
        self.name = os.path.basename(self.file_path)
        self.source_folder = os.path.dirname(self.file_path)
        self.point_count = point_count
        self.bounds_min = (
            np.array(bounds_min, dtype=np.float32) if bounds_min is not None else None
        )
        self.bounds_max = (
            np.array(bounds_max, dtype=np.float32) if bounds_max is not None else None
        )
        now = time.time()
        self.import_date = import_date if import_date is not None else now
        self.last_opened = last_opened if last_opened is not None else now
        # Lazy in-memory preview
        self.preview_data: PointCloudData | None = None
        # IMAGING / HOLOGRAM reserved fields (see __init__ signature).
        self.patient_id = patient_id
        self.study_id = study_id
        self.region = region
        self.pose = pose
        self.paired_entry_key = paired_entry_key
        # Stored as a flat 16-float list so JSON round-trip is trivial.
        # None means "no transform applied" (use identity / native pose).
        self.pose_transform = (
            [float(v) for v in pose_transform] if pose_transform is not None else None
        )

    def exists(self) -> bool:
        return os.path.isfile(self.file_path)

    def preview_path(self, lib_dir: str) -> str:
        return os.path.join(lib_dir, "previews", f"{self.file_key}.npz")

    def to_dict(self) -> dict:
        return {
            "file_key": self.file_key,
            "file_path": self.file_path,
            "point_count": int(self.point_count),
            "bounds_min": self.bounds_min.tolist() if self.bounds_min is not None else None,
            "bounds_max": self.bounds_max.tolist() if self.bounds_max is not None else None,
            "import_date": float(self.import_date),
            "last_opened": float(self.last_opened),
            # Reserved fields — only written when non-default so existing
            # entries stay byte-stable until IMAGING populates them.
            "patient_id": self.patient_id,
            "study_id": self.study_id,
            "region": self.region,
            "pose": self.pose,
            "paired_entry_key": self.paired_entry_key,
            "pose_transform": self.pose_transform,
            "display_name": self.display_name,
        }

    @staticmethod
    def from_dict(d: dict) -> "LibraryEntry":
        return LibraryEntry(
            file_key=d["file_key"],
            file_path=d["file_path"],
            point_count=d.get("point_count", 0),
            bounds_min=d.get("bounds_min"),
            bounds_max=d.get("bounds_max"),
            import_date=d.get("import_date"),
            last_opened=d.get("last_opened"),
            patient_id=d.get("patient_id"),
            study_id=d.get("study_id"),
            region=d.get("region"),
            pose=d.get("pose"),
            paired_entry_key=d.get("paired_entry_key"),
            pose_transform=d.get("pose_transform"),
            display_name=d.get("display_name"),
        )


class Project:
    """A user-defined project: an ordered group of clouds with shared
    ontology and display settings.

    Optionally carries a label ontology (``ontology_data``) that is
    loaded into the app's ``LabelRegistry`` whenever this project
    becomes the active view. This ensures every cloud in the project
    shares consistent label IDs, names, and colors.

    Display settings (``settings``) are restored when the project is
    opened so each project remembers its own visual configuration.
    """

    def __init__(
        self,
        id: str,
        name: str,
        file_keys: list[str] | None = None,
        created: float | None = None,
        ontology_data: dict | None = None,
        ontology_preset_name: str | None = None,
        settings: dict | None = None,
        # When True (the default for any new project), live edits to the
        # label registry while this project is active are gated behind
        # an explicit "design ontology" UI affordance — the goal is to
        # prevent silent label-drift mid-study, which would corrupt
        # training sets and downstream measurements. Existing projects
        # loaded from disk without this field default to True too, so
        # the safer behaviour wins on first migration. Researcher mode
        # can override per-project when intentionally evolving an
        # ontology.
        ontology_locked: bool = True,
    ):
        self.id = id
        self.name = name
        self.file_keys: list[str] = list(file_keys or [])
        self.created = created if created is not None else time.time()
        self.ontology_data: dict | None = ontology_data
        self.ontology_preset_name: str | None = ontology_preset_name
        self.settings: dict | None = settings
        self.ontology_locked: bool = bool(ontology_locked)

    # -- ontology helpers --------------------------------------------------

    def set_ontology_preset(self, preset_name: str) -> None:
        """Build a registry from a named preset and store it."""
        from src.data.labels import ONTOLOGY_PRESETS
        builder = ONTOLOGY_PRESETS.get(preset_name)
        if builder is None:
            raise ValueError(f"Unknown ontology preset: {preset_name}")
        registry = builder()
        self.ontology_data = registry.to_json()
        self.ontology_preset_name = preset_name

    def clear_ontology(self) -> None:
        self.ontology_data = None
        self.ontology_preset_name = None

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "file_keys": list(self.file_keys),
            "created": float(self.created),
        }
        if self.ontology_data is not None:
            d["ontology_data"] = self.ontology_data
        if self.ontology_preset_name is not None:
            d["ontology_preset_name"] = self.ontology_preset_name
        if self.settings is not None:
            d["settings"] = self.settings
        d["ontology_locked"] = self.ontology_locked
        return d

    @staticmethod
    def from_dict(d: dict) -> "Project":
        return Project(
            id=d["id"],
            name=d["name"],
            file_keys=d.get("file_keys", []),
            created=d.get("created"),
            ontology_data=d.get("ontology_data"),
            ontology_preset_name=d.get("ontology_preset_name"),
            settings=d.get("settings"),
            # Default True for entries that predate this field — the
            # safer behaviour wins on migration.
            ontology_locked=d.get("ontology_locked", True),
        )


# Keep old name as alias for backward compatibility with any external refs
LibraryCollection = Project


class LibraryCatalog:
    """Global persistent catalog of every imported point cloud."""

    def __init__(self, preview_points: int = PREVIEW_POINTS):
        self.dir = library_dir()
        self.previews_dir = os.path.join(self.dir, "previews")
        self.index_path = os.path.join(self.dir, "index.json")
        self.projects_path = os.path.join(self.dir, "projects.json")
        # Legacy path for migration from collections.json
        self._legacy_collections_path = os.path.join(self.dir, "collections.json")
        self.preview_points = preview_points

        self.entries: dict[str, LibraryEntry] = {}
        self.projects: dict[str, Project] = {}
        # Errors caught during _load_index / _load_projects so the App
        # can surface them through the status banner on first frame.
        # LS-5: catalog corruption used to be silent.
        self._init_errors: list[str] = []

        # CP-4: per-file_key locks for save_cloud_labels (init at the
        # very end of __init__, after _load_index / _load_projects).

        self._executor = ThreadPoolExecutor(max_workers=2)
        self._pending_futures: list[tuple[str, Future]] = []
        # Separate queue for mesh (Poisson surface) builds. Same
        # executor, different result shape — keeping them parallel
        # lists is cleaner than tagging each future with a kind enum.
        self._pending_mesh_futures: list[tuple[str, Future]] = []
        # Lock protects both pending lists and entry mutations from the
        # background thread (preview/mesh builds) vs main-thread polling.
        self._lock = __import__('threading').Lock()

        # Version counter bumped whenever entries/projects mutate.
        # Lets per-frame UI code (folder tree, missing count, project
        # counts) memoise derived state instead of rebuilding every frame.
        self._version: int = 0
        # Memoised folder_tree() result; keyed on _version.
        self._folder_tree_cache: dict | None = None
        self._folder_tree_version: int = -1
        # Bulk exists() cache: {file_key: (timestamp, bool)}. TTL below.
        # A 2-second TTL is short enough that a user dragging the file
        # out of the source folder and back sees the right state, long
        # enough to amortise stat() across ~120 frames.
        self._exists_cache: dict[str, tuple[float, bool]] = {}
        self._exists_ttl: float = 2.0

        self._load_index()
        self._load_projects()

        # v2 label layout: upgrade a 1.0 flat labels/ tree to
        # per-namespace subdirs (idempotent; marker-gated no-op after
        # the first run). Must happen after _load_projects so each
        # project's clouds get their own copy of the baseline labels.
        try:
            from src.data import cloud_store as _cs
            mig = _cs.migrate_labels_layout_v2(
                {pid: list(p.file_keys) for pid, p in self.projects.items()})
            if mig.get("moved") or mig.get("copied"):
                print(f"[library] label layout v2 migration: "
                      f"{mig['moved']} moved, {mig['copied']} copied, "
                      f"{mig['errors']} errors")
            if mig.get("errors"):
                self._init_errors.append(
                    f"Label layout migration hit {mig['errors']} errors — "
                    f"see console; flat files were left in place.")
        except Exception as e:
            self._init_errors.append(f"Label layout migration failed: {e}")

        # CP-4: per-file_key locks initialised last so any failure
        # during _load_index doesn't leave a half-constructed catalog
        # with a populated lock dict.
        import threading as _threading_for_locks
        self._labels_locks: dict[str, _threading_for_locks.Lock] = {}
        self._labels_locks_guard = _threading_for_locks.Lock()

    def labels_lock_for(self, file_key: str):
        """Return the per-file_key threading.Lock, creating it lazily.

        Callers use ``with catalog.labels_lock_for(fk):`` around any
        write to ``labels/<fk>.npy`` so a background worker can't
        interleave with the main-thread write on the SAME key. The
        guard lock serialises only the dict creation; the per-key
        Lock is what callers actually hold during the write.
        """
        with self._labels_locks_guard:
            lk = self._labels_locks.get(file_key)
            if lk is None:
                import threading as _threading_for_locks
                lk = _threading_for_locks.Lock()
                self._labels_locks[file_key] = lk
            return lk

    def _bump_version(self) -> None:
        """Signal that entries/projects changed so derived caches reset."""
        self._version += 1

    def entry_exists(self, entry: "LibraryEntry") -> bool:
        """Cached existence check for a library entry.

        Hits the filesystem at most once per entry per ``_exists_ttl``
        seconds. Any other per-frame code that needs to know whether a
        source file still lives on disk should go through this instead
        of calling ``entry.exists()`` directly — a 500-entry library
        was doing 500 stat() syscalls per frame before this.
        """
        now = time.time()
        hit = self._exists_cache.get(entry.file_key)
        if hit is not None and (now - hit[0]) < self._exists_ttl:
            return hit[1]
        ok = os.path.isfile(entry.file_path)
        self._exists_cache[entry.file_key] = (now, ok)
        return ok

    def missing_count(self) -> int:
        """Number of registered entries whose source file is missing.

        Uses the bulk exists cache so the sidebar's per-frame "Missing
        Files" badge doesn't stat every entry every frame.
        """
        return sum(1 for e in self.entries.values() if not self.entry_exists(e))

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def _load_index(self) -> None:
        """Load index.json; on corruption, rename it aside + record the
        error on ``self._init_errors`` so the App can surface a sticky
        status banner. LS-5 fix: previously the bare except dropped
        the error on the floor and the catalog came up empty, hiding
        the corruption from the user.
        """
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r") as f:
                data = json.load(f)
            for key, info in data.items():
                try:
                    self.entries[key] = LibraryEntry.from_dict(info)
                except (KeyError, ValueError, TypeError):
                    continue
        except (json.JSONDecodeError, OSError) as e:
            # Rename to a timestamped sidecar so the corrupt file is
            # preserved for forensics — the user can still inspect /
            # recover from it. Empty in-memory catalog continues so
            # the app at least starts.
            try:
                ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                corrupt_path = f"{self.index_path}.corrupt-{ts}"
                os.replace(self.index_path, corrupt_path)
                msg = (f"Catalog index.json corrupt ({type(e).__name__}); "
                       f"renamed to {os.path.basename(corrupt_path)}. "
                       f"Past imports may be missing.")
            except OSError as rename_err:
                msg = (f"Catalog index.json corrupt ({type(e).__name__}) "
                       f"and could not be renamed: {rename_err}")
            print(f"[library] {msg}")
            self._init_errors.append(msg)

    def _save_index(self) -> None:
        # Any entries-level mutation flows through here, so this is the
        # single chokepoint for invalidating derived caches. CP-2: hold
        # ``self._lock`` over the snapshot AND the file write so a
        # background worker mutating ``self.entries`` (preview/mesh
        # build completing, batch inference appending a key) can't
        # interleave with the dict comprehension below — pre-fix that
        # raced occasionally with "dictionary changed size during
        # iteration".
        with self._lock:
            self._bump_version()
            data = {k: e.to_dict() for k, e in self.entries.items()}
        tmp = self.index_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.index_path)
        except OSError as e:
            print(f"Library index save failed: {e}")

    def _load_projects(self) -> None:
        # Try projects.json first, fall back to legacy collections.json
        path = self.projects_path
        if not os.path.exists(path):
            path = self._legacy_collections_path
            if not os.path.exists(path):
                return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for pid, info in data.items():
                try:
                    self.projects[pid] = Project.from_dict(info)
                except (KeyError, ValueError, TypeError):
                    continue
        except (json.JSONDecodeError, OSError):
            pass

    def _save_projects(self) -> None:
        # CP-2: same guard as _save_index. The projects dict can be
        # mutated by add_to_project from a background event handler
        # (ST-4 imaging_add_to_project) — though it runs on the main
        # thread today, the lock keeps future producers safe.
        with self._lock:
            self._bump_version()
            data = {k: p.to_dict() for k, p in self.projects.items()}
        tmp = self.projects_path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.projects_path)
        except OSError as e:
            print(f"Library projects save failed: {e}")

    # -----------------------------------------------------------------
    # Registration / metrics
    # -----------------------------------------------------------------

    def register_file(self, file_path: str) -> LibraryEntry | None:
        """Add a file to the library (idempotent). Returns the entry.

        If the file is already known under the same hash, the existing
        entry's ``last_opened`` timestamp is touched and returned.
        """
        if not os.path.isfile(file_path):
            return None
        try:
            file_key = compute_file_key(file_path)
        except OSError:
            return None

        existing = self.entries.get(file_key)
        if existing is not None:
            existing.last_opened = time.time()
            self._save_index()
            return existing

        entry = LibraryEntry(file_key=file_key, file_path=file_path)
        self.entries[file_key] = entry
        self._save_index()
        return entry

    def register_files(self, file_paths: list[str]) -> list["LibraryEntry | None"]:
        """Batch ``register_file`` with a single index save at the end.

        LS-1: 4D sequence load registers every frame so each gets a
        file_key for label persistence — doing that through
        ``register_file`` would rewrite index.json once per frame.
        Returns one entry (or None for unreadable paths) per input, in
        order.
        """
        out: list[LibraryEntry | None] = []
        dirty = False
        for file_path in file_paths:
            if not os.path.isfile(file_path):
                out.append(None)
                continue
            try:
                file_key = compute_file_key(file_path)
            except OSError:
                out.append(None)
                continue
            existing = self.entries.get(file_key)
            if existing is not None:
                existing.last_opened = time.time()
                out.append(existing)
                dirty = True
                continue
            entry = LibraryEntry(file_key=file_key, file_path=file_path)
            self.entries[file_key] = entry
            out.append(entry)
            dirty = True
        if dirty:
            self._save_index()
        return out

    def update_metrics(
        self,
        file_key: str,
        point_count: int,
        bounds_min,
        bounds_max,
    ) -> None:
        entry = self.entries.get(file_key)
        if entry is None:
            return
        entry.point_count = int(point_count)
        if bounds_min is not None:
            entry.bounds_min = np.array(bounds_min, dtype=np.float32)
        if bounds_max is not None:
            entry.bounds_max = np.array(bounds_max, dtype=np.float32)
        self._save_index()

    def touch(self, file_key: str) -> None:
        entry = self.entries.get(file_key)
        if entry is None:
            return
        entry.last_opened = time.time()
        self._save_index()

    def reattach_entry_source(self, file_key: str, new_path: str) -> bool:
        """Point an entry at a moved/renamed source file (CP-5).

        The file_key — and with it every catalog sidecar (data, labels
        in all namespaces, previews, meshes) — is preserved; only the
        source path changes. Returns False when the entry is unknown or
        the new path doesn't exist.
        """
        entry = self.entries.get(file_key)
        if entry is None or not os.path.isfile(new_path):
            return False
        entry.file_path = new_path
        self._exists_cache.pop(file_key, None)
        self._save_index()
        return True

    def rename_entry(self, file_key: str, display_name: str) -> bool:
        """Persist a user-rename to the catalog (CP-9).

        Sets ``LibraryEntry.display_name``; the file-path-derived
        ``name`` stays untouched so source-derived heuristics keep
        working. Returns True iff the entry exists and the new name
        is non-empty.
        """
        entry = self.entries.get(file_key)
        if entry is None:
            return False
        new_name = (display_name or "").strip()
        if not new_name:
            entry.display_name = None
        else:
            entry.display_name = new_name
        self._save_index()
        return True

    # -----------------------------------------------------------------
    # Preview cache
    # -----------------------------------------------------------------

    def get_preview(self, entry: LibraryEntry) -> PointCloudData | None:
        """Return an in-memory preview for an entry, loading from disk if needed."""
        if entry.preview_data is not None:
            return entry.preview_data
        p = entry.preview_path(self.dir)
        if not os.path.exists(p):
            return None
        try:
            data = np.load(p)
            preview = PointCloudData(
                positions=data["positions"],
                colors=data["colors"],
                file_path=p,
            )
            entry.preview_data = preview
            return preview
        except Exception:
            return None

    def queue_preview_build(self, entry: LibraryEntry) -> None:
        """Queue background preview generation if not already pending / cached."""
        if entry.preview_data is not None:
            return
        if os.path.exists(entry.preview_path(self.dir)):
            return
        with self._lock:
            for fk, _ in self._pending_futures:
                if fk == entry.file_key:
                    return
            fut = self._executor.submit(self._build_preview, entry)
            self._pending_futures.append((entry.file_key, fut))

    def queue_mesh_build(
        self,
        entry: LibraryEntry,
        *,
        depth: int = 9,
    ) -> None:
        """Queue Poisson surface reconstruction for an entry's cloud.

        Mesh build is heavyweight (~1-2 sec/cloud, single-threaded inside
        open3d). Runs on the same shared executor as preview builds so
        we don't oversubscribe the CPU when a fresh catalog import is
        thrashing both queues. Tracked separately from preview futures
        because the result shape differs and the main-thread poll
        applies them differently.

        No-op if already cached on disk or already pending.
        """
        from src.data.cloud_store import has_cloud_mesh
        if has_cloud_mesh(entry.file_key):
            return
        with self._lock:
            for fk, _ in self._pending_mesh_futures:
                if fk == entry.file_key:
                    return
            fut = self._executor.submit(
                self._build_mesh_job, entry.file_key, depth)
            self._pending_mesh_futures.append((entry.file_key, fut))

    def _build_mesh_job(
        self,
        file_key: str,
        depth: int,
    ) -> str | None:
        """Background job: load cloud data + labels, run Poisson, persist.

        Returns ``file_key`` on success so ``poll_pending_meshes`` knows
        which entry to mark ready. Returns None on any failure (logged
        to stdout); the GUI will see no event and the user can retry by
        rebuilding from a future label edit or restart.
        """
        try:
            from src.data.cloud_store import (
                load_cloud_data, load_cloud_labels, save_cloud_mesh,
            )
            from src.data.mesh_builder import build_mesh, MeshBuildUnavailable

            loaded = load_cloud_data(file_key)
            if loaded is None:
                # Fall back to the catalog entry's source file if the
                # cloud_store snapshot doesn't exist yet.
                # CP-8: re-check ``self.entries.get(file_key)`` under
                # ``self._lock`` and capture the file_path while
                # holding the lock — without this, a remove_entry on
                # the main thread between the get() and the
                # load_point_cloud could pull the source file from
                # under us, building a mesh for a cloud that's no
                # longer in the catalog.
                with self._lock:
                    entry = self.entries.get(file_key)
                    src_path = entry.file_path if entry is not None else None
                if src_path is None:
                    return None
                cloud = load_point_cloud(src_path)
            else:
                cloud, _meta = loaded
            labels = load_cloud_labels(file_key)
            mesh = build_mesh(
                positions=cloud.positions,
                colors=cloud.colors,
                labels=labels if labels is not None else None,
                depth=depth,
            )
            save_cloud_mesh(file_key, mesh)
            return file_key
        except MeshBuildUnavailable as e:
            print(f"Mesh build skipped for {file_key}: {e}")
            return None
        except Exception as e:
            print(f"Mesh build failed for {file_key}: {e}")
            return None

    def poll_pending_meshes(self) -> list[str]:
        """Drain completed mesh builds. Returns the file_keys of entries
        whose mesh just landed on disk. Caller is expected to invalidate
        any cached GPU mesh state for those keys so the next render
        picks up the new geometry.
        """
        completed: list[str] = []
        with self._lock:
            still: list[tuple[str, Future]] = []
            for fk, fut in self._pending_mesh_futures:
                if fut.done():
                    try:
                        res = fut.result()
                        if res is not None:
                            completed.append(res)
                    except Exception as e:
                        print(f"Mesh future error for {fk}: {e}")
                else:
                    still.append((fk, fut))
            self._pending_mesh_futures = still
        return completed

    def _build_preview(self, entry: LibraryEntry) -> LibraryEntry | None:
        """Load, downsample, cache. Runs on background thread.

        Returns a result tuple rather than mutating the shared entry
        directly — the main thread applies the update atomically in
        poll_pending so no reader can observe a half-written state.
        """
        try:
            cloud = load_point_cloud(entry.file_path)
            preview = voxel_downsample(cloud, self.preview_points)
            p = entry.preview_path(self.dir)
            np.savez_compressed(
                p,
                positions=preview.positions,
                colors=preview.colors,
            )
            # Return data for the main thread to apply atomically
            return (entry, preview,
                    int(cloud.point_count),
                    cloud.bounds_min.astype(np.float32),
                    cloud.bounds_max.astype(np.float32))
        except Exception as e:
            print(f"Library preview build failed for {entry.name}: {e}")
            return None

    def poll_pending(self) -> list[LibraryEntry]:
        """Drain completed preview futures. Returns entries newly available."""
        completed: list[LibraryEntry] = []
        with self._lock:
            still: list[tuple[str, Future]] = []
            for fk, fut in self._pending_futures:
                if fut.done():
                    try:
                        res = fut.result()
                        if res is not None:
                            # Apply the background result atomically on main thread
                            entry, preview, pt_count, bmin, bmax = res
                            entry.preview_data = preview
                            entry.point_count = pt_count
                            entry.bounds_min = bmin
                            entry.bounds_max = bmax
                            completed.append(entry)
                    except Exception as e:
                        print(f"Library preview future error: {e}")
                else:
                    still.append((fk, fut))
            self._pending_futures = still
        if completed:
            self._save_index()
        return completed

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_futures)

    # -----------------------------------------------------------------
    # Folder tree (derived from entries' source_folder)
    # -----------------------------------------------------------------

    def folder_tree(self) -> dict:
        """Return a nested folder tree.

        Each node: ``{"name", "path", "count", "children"}``.
        ``count`` is the recursive number of entries under that node.

        Memoised on ``_version``; rebuilds only when entries change.
        The sidebar calls this every frame — previously that meant
        walking every entry and allocating a nested dict tree per
        frame for a library that almost never changes.
        """
        if (self._folder_tree_cache is not None
                and self._folder_tree_version == self._version):
            return self._folder_tree_cache

        root: dict = {"name": "", "path": "", "count": 0, "children": {}}

        for e in self.entries.values():
            parts = _split_path(e.source_folder)
            node = root
            accum = ""
            for part in parts:
                if not part:
                    continue
                accum = _join_path(accum, part)
                child = node["children"].get(part)
                if child is None:
                    child = {
                        "name": part,
                        "path": accum,
                        "count": 0,
                        "children": {},
                    }
                    node["children"][part] = child
                node = child
                node["count"] += 1
            root["count"] += 1

        def finalize(n: dict) -> None:
            kids = sorted(n["children"].values(), key=lambda c: c["name"].lower())
            for c in kids:
                finalize(c)
            n["children"] = kids

        finalize(root)
        self._folder_tree_cache = root
        self._folder_tree_version = self._version
        return root

    def entries_in_folder(
        self, folder_path: str, recursive: bool = True
    ) -> list[LibraryEntry]:
        norm = os.path.normpath(folder_path)
        out: list[LibraryEntry] = []
        for e in self.entries.values():
            ef = os.path.normpath(e.source_folder)
            if recursive:
                if ef == norm or ef.startswith(norm + os.sep):
                    out.append(e)
            else:
                if ef == norm:
                    out.append(e)
        out.sort(key=lambda e: e.name.lower())
        return out

    # -----------------------------------------------------------------
    # Project views
    # -----------------------------------------------------------------

    def entries_in_project(self, project_id: str) -> list[LibraryEntry]:
        if project_id == SMART_ALL:
            out = list(self.entries.values())
            out.sort(key=lambda e: e.name.lower())
            return out
        if project_id == SMART_RECENT:
            out = sorted(self.entries.values(), key=lambda e: -e.import_date)
            return out[:RECENT_LIMIT]
        if project_id == SMART_MISSING:
            return [e for e in self.entries.values() if not self.entry_exists(e)]
        proj = self.projects.get(project_id)
        if proj is None:
            return []
        return [self.entries[fk] for fk in proj.file_keys if fk in self.entries]

    def create_project(self, name: str) -> Project:
        pid = f"proj:{uuid.uuid4().hex[:8]}"
        proj = Project(id=pid, name=name)
        self.projects[pid] = proj
        self._save_projects()
        return proj

    def duplicate_project(
        self,
        source_project_id: str,
        new_name: str | None = None,
        include_clouds: bool = True,
    ) -> Project | None:
        """Create an independent copy of a project.

        With ``include_clouds=True`` (the 1.1 default) the duplicate
        carries the source's ontology, settings, cloud membership, AND a
        verbatim copy of every label file into its own namespace. This
        is the "retrain with a different class set" workflow: duplicate,
        edit the new project's ontology (drop / merge / add classes),
        repaint the deltas, train — the source project is never touched.

        With ``include_clouds=False`` you get the 1.0 behaviour: same
        ontology + settings, empty ``file_keys`` ("apply this setup to a
        new batch of clouds").

        Trained-model registry entries are copied by a sibling helper
        (see ``ProjectModelRegistry.duplicate_from``); checkpoints stay
        shared on disk so the duplicate inherits the source's inference
        capability for free.

        Returns the new ``Project`` instance, or ``None`` if the source
        project doesn't exist.
        """
        import copy
        src = self.projects.get(source_project_id)
        if src is None:
            return None
        if new_name is None:
            # Make collisions visually obvious — the user usually
            # renames immediately, so "(copy)" is good enough as a
            # placeholder default.
            new_name = f"{src.name} (copy)"
        new_proj = self.create_project(new_name)
        # Deep-copy ontology + settings so later edits in either
        # project don't bleed into the other.
        if src.ontology_data is not None:
            new_proj.ontology_data = copy.deepcopy(src.ontology_data)
        if src.settings is not None:
            new_proj.settings = copy.deepcopy(src.settings)
        if include_clouds and src.file_keys:
            new_proj.file_keys = list(src.file_keys)
            try:
                from src.data import cloud_store as _cs
                _cs.copy_labels_between_namespaces(
                    src.file_keys, source_project_id, new_proj.id)
            except Exception as e:
                print(f"[duplicate_project] label copy failed: {e}")
        self._save_projects()
        return new_proj

    def delete_project(self, project_id: str) -> None:
        if project_id in self.projects:
            del self.projects[project_id]
            # SC-07: drop the per-project model registry file so a
            # deleted project doesn't orphan ``models/<id>.json``.
            try:
                from src.data.model_registry import ProjectModelRegistry
                ProjectModelRegistry(self.dir).drop_project(project_id)
            except Exception as e:
                print(f"[delete_project] models drop failed: {e}")
            # v2: the project's label namespace goes with it. Source
            # clouds and every other project's labels are untouched.
            try:
                from src.data import cloud_store as _cs
                _cs.drop_labels_namespace(project_id)
            except Exception as e:
                print(f"[delete_project] namespace drop failed: {e}")
            self._save_projects()

    def rename_project(self, project_id: str, name: str) -> None:
        proj = self.projects.get(project_id)
        if proj is None:
            return
        proj.name = name
        self._save_projects()

    def set_project_ontology(self, project_id: str,
                             preset_name: str) -> None:
        """Assign a label ontology preset to a project and persist."""
        proj = self.projects.get(project_id)
        if proj is None:
            return
        proj.set_ontology_preset(preset_name)
        self._save_projects()

    def clear_project_ontology(self, project_id: str) -> None:
        """Remove the ontology from a project and persist."""
        proj = self.projects.get(project_id)
        if proj is None:
            return
        proj.clear_ontology()
        self._save_projects()

    def update_project_ontology(self, project_id: str,
                                registry_json: dict) -> None:
        """Write mutated registry JSON back to the project and persist."""
        proj = self.projects.get(project_id)
        if proj is None:
            return
        proj.ontology_data = registry_json
        self._save_projects()

    def add_to_project(self, project_id: str, file_keys: list[str]) -> None:
        proj = self.projects.get(project_id)
        if proj is None:
            return
        existing = set(proj.file_keys)
        for fk in file_keys:
            if fk not in existing and fk in self.entries:
                proj.file_keys.append(fk)
                existing.add(fk)
        self._save_projects()

    def remove_from_project(
        self, project_id: str, file_keys: list[str]
    ) -> None:
        proj = self.projects.get(project_id)
        if proj is None:
            return
        rm = set(file_keys)
        proj.file_keys = [fk for fk in proj.file_keys if fk not in rm]
        self._save_projects()

    # Legacy aliases so existing call sites continue to work during migration
    entries_in_collection = entries_in_project
    create_collection = create_project
    delete_collection = delete_project
    rename_collection = rename_project
    set_collection_ontology = set_project_ontology
    clear_collection_ontology = clear_project_ontology
    update_collection_ontology = update_project_ontology
    add_to_collection = add_to_project
    remove_from_collection = remove_from_project

    @property
    def collections(self) -> dict[str, Project]:
        """Legacy alias — use ``projects`` instead."""
        return self.projects

    def remove_entry(self, file_key: str) -> None:
        """Remove a cloud from the library. The source file is untouched.

        CP-1: drops every per-key catalog sidecar in addition to the
        preview thumbnail — data, labels, preview labels, mesh,
        flags. Previously only the preview was dropped; the other
        files lingered on disk as orphans, growing the catalog
        directory + occasionally re-surfacing when a new entry
        happened to hash to the same file_key.
        """
        entry = self.entries.get(file_key)
        if entry is None:
            return
        # Preview thumbnail (lives directly under previews/)
        p = entry.preview_path(self.dir)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
        # CP-1: drop every catalog-managed sidecar for this key.
        # Post-audit extension: also drop the per-cloud primitives JSON
        # (helper has existed since plane-fitting landed — never wired
        # in) and the LS-9 recovery sidecar (added in 758836b but never
        # cleaned up on entry removal, leading to orphan recovery
        # candidates on next launch).
        try:
            from src.data.cloud_store import (
                drop_cloud, drop_cloud_mesh, drop_preview_labels,
                cloud_labels_path, label_namespaces,
            )
            drop_cloud(file_key)           # data + labels (all namespaces)
            drop_cloud_mesh(file_key)      # poisson mesh
            drop_preview_labels(file_key)  # preview labels (all namespaces)
            # LS-9 recovery sidecars — live next to canonical labels,
            # one per namespace in the v2 layout.
            for ns in label_namespaces():
                recovery_path = cloud_labels_path(file_key, ns).parent / (
                    f"{file_key}.recovery.npy"
                )
                try:
                    recovery_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as e:
                    print(f"[remove_entry] recovery sidecar drop failed "
                          f"for {file_key}: {e}")
        except Exception as e:
            print(f"[remove_entry] sidecar drop failed for {file_key}: {e}")
        del self.entries[file_key]
        self._exists_cache.pop(file_key, None)
        for proj in self.projects.values():
            proj.file_keys = [fk for fk in proj.file_keys if fk != file_key]
        self._save_index()
        self._save_projects()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------
# Path helpers — cross-platform tree splitting
# ---------------------------------------------------------------------


def _split_path(path: str) -> list[str]:
    """Split a filesystem path into ordered components for tree building.

    On Windows, the drive (e.g. ``C:\\``) is emitted as a single leading
    component so ``C:\\data\\scans`` → ``["C:\\", "data", "scans"]``.
    """
    if not path:
        return []
    drive, rest = os.path.splitdrive(path)
    parts: list[str] = []
    if drive:
        parts.append(drive + os.sep)
    rest = rest.replace("\\", "/").strip("/")
    if rest:
        parts.extend(p for p in rest.split("/") if p)
    return parts


def _join_path(accum: str, part: str) -> str:
    if not accum:
        return part
    # If ``part`` ends with a separator it's a drive marker — append directly
    if accum.endswith(os.sep) or accum.endswith("/"):
        return accum + part
    return accum + os.sep + part
