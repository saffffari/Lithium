"""Lithium project bundles (``.li``).

A ``.li`` file is a zip that carries any combination of a project's
point clouds, its label layer and its trained model checkpoints, plus
a ``manifest.json`` describing the project (name, ontology, settings)
and every item inside. It is the unit of exchange between machines:
export on one catalog, drop the file on a drive, import on another.

Layout::

    manifest.json
    clouds/<file_key>.npz        coord + color (+ normal), nothing else
    labels/<file_key>.npy        int32 per-point label ids
    models/<model_id>/model_best.pth
    models/<model_id>/config.py  (when the run directory has one)

Clouds are stripped on the way out: only ``coord``, ``color`` and the
``normal`` scalar survive. Every other per-point scalar is dropped,
deliberately — the catalog npz files carry label-bearing scalars
(``labels_*``, ``tips*``, ``label_seed_*``) that the viewer can show as
a colour ramp, and a bundle exported *without* labels must not leak
them through the back door. ``manifest.json`` records how many scalars
were dropped so the omission is visible.

File keys are preserved across machines. Label files are keyed by
them, and keeping them lets a labels-only bundle land on top of clouds
imported earlier from a clouds-only bundle of the same project.

This module has no GUI dependency: ``tools/`` scripts and tests drive
it directly, ``src/gui/bundle_dialog.py`` wraps it in the modal.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from src.data import cloud_store
from src.data import library_paths

FORMAT_NAME = "lithium-bundle"
FORMAT_VERSION = 1
EXTENSION = ".li"

# The per-point scalars a bundle keeps. ``normal`` feeds the viewer's
# lighting; everything else is a label or a leak.
KEPT_SCALARS = ("normal",)

ProgressFn = Callable[[str, int, int, int], None]
"""progress(stage, done, total, bytes_so_far)"""


class BundleError(Exception):
    pass


class BundleCancelled(BundleError):
    pass


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@dataclass
class ExportSpec:
    clouds: bool = True
    labels: bool = True
    models: bool = False


@dataclass
class ExportPlan:
    """What an export would write, with byte estimates for the dialog."""
    project_id: str
    project_name: str
    cloud_keys: list[str] = field(default_factory=list)
    cloud_bytes: int = 0            # catalog npz bytes (upper bound; bundle is smaller)
    label_keys: list[str] = field(default_factory=list)
    label_bytes: int = 0
    models: list[dict] = field(default_factory=list)   # {model_id, name, files:[(src, arcname, bytes)]}
    model_bytes: int = 0
    missing_clouds: int = 0         # project members with no catalog data

    def total_bytes(self, spec: ExportSpec) -> int:
        return ((self.cloud_bytes if spec.clouds else 0)
                + (self.label_bytes if spec.labels else 0)
                + (self.model_bytes if spec.models else 0))


def plan_export(catalog, project_id: str) -> ExportPlan:
    proj = catalog.projects.get(project_id)
    if proj is None:
        raise BundleError(f"no project {project_id}")
    plan = ExportPlan(project_id=project_id, project_name=proj.name)
    for fk in proj.file_keys:
        p = cloud_store.cloud_data_path(fk)
        if p.exists():
            plan.cloud_keys.append(fk)
            plan.cloud_bytes += p.stat().st_size
        else:
            plan.missing_clouds += 1
        lp = cloud_store.cloud_labels_path(fk, project_id)
        if lp.exists():
            plan.label_keys.append(fk)
            plan.label_bytes += lp.stat().st_size
    try:
        from src.data.model_registry import ProjectModelRegistry
        reg = ProjectModelRegistry(library_paths.library_dir())
        for m in reg.load(project_id):
            if not m.has_checkpoint:
                continue
            files = []
            ck = m.best_checkpoint
            files.append((ck, f"models/{m.model_id}/model_best.pth", os.path.getsize(ck)))
            cfg = os.path.join(m.work_dir, "config.py") if m.work_dir else ""
            if cfg and os.path.isfile(cfg):
                files.append((cfg, f"models/{m.model_id}/config.py", os.path.getsize(cfg)))
            plan.models.append({"model_id": m.model_id, "name": m.name,
                                "record": m.to_dict(), "files": files})
            plan.model_bytes += sum(b for _, _, b in files)
    except Exception as e:  # registry trouble must not block a cloud export
        print(f"[bundle] model registry unavailable: {e}")
    return plan


def _strip_cloud_npz(src: Path) -> tuple[bytes, int]:
    """Return (npz bytes with only coord/color/kept scalars, n_dropped)."""
    import io
    keep: dict[str, np.ndarray] = {}
    dropped = 0
    with np.load(src, allow_pickle=False) as z:
        meta = {}
        if "_meta_json" in z.files:
            try:
                meta = json.loads(bytes(z["_meta_json"]).decode("utf-8"))
            except Exception:
                meta = {}
        for name in z.files:
            if name in ("coord", "color", "pose_to_world"):
                keep[name] = z[name]
            elif name.startswith("scalar_") and name[len("scalar_"):] in KEPT_SCALARS:
                keep[name] = z[name]
            elif name == "_meta_json":
                continue
            else:
                dropped += 1
    if "coord" not in keep or "color" not in keep:
        raise BundleError(f"{src.name}: no coord/color")
    meta = {k: v for k, v in meta.items() if k in (
        "local_origin", "source_unit", "voxel_spacing", "source_label_id", "point_count")}
    meta["format_version"] = cloud_store._FORMAT_VERSION
    meta["source_path"] = None
    meta["source_mtime"] = None
    meta["scalar_keys"] = sorted(k[len("scalar_"):] for k in keep if k.startswith("scalar_"))
    meta["bundle_scalars_dropped"] = dropped
    keep["_meta_json"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8).copy()
    buf = io.BytesIO()
    np.savez_compressed(buf, **keep)
    return buf.getvalue(), dropped


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def export_bundle(catalog, project_id: str, spec: ExportSpec, out_path: str,
                  progress: ProgressFn | None = None,
                  cancel: threading.Event | None = None) -> dict:
    """Write ``out_path`` and return the manifest. Safe to run off the
    main thread: it only reads the catalog."""
    plan = plan_export(catalog, project_id)
    proj = catalog.projects[project_id]
    out = Path(out_path)
    if out.suffix.lower() != EXTENSION:
        out = out.with_suffix(EXTENSION)
    tmp = out.with_name(out.name + ".part")

    total = ((len(plan.cloud_keys) if spec.clouds else 0)
             + (len(plan.label_keys) if spec.labels else 0)
             + (sum(len(m["files"]) for m in plan.models) if spec.models else 0))
    done = 0
    written = 0

    def tick(stage):
        if cancel is not None and cancel.is_set():
            raise BundleCancelled()
        if progress:
            progress(stage, done, total, written)

    settings = dict(proj.settings or {})
    if not spec.models:
        settings.pop("active_model_id", None)
    manifest = {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "created": time.time(),
        "includes": {"clouds": spec.clouds, "labels": spec.labels, "models": spec.models},
        "project": {
            "name": proj.name,
            "source_id": proj.id,
            "ontology_data": proj.ontology_data,
            "ontology_preset_name": proj.ontology_preset_name,
            "ontology_locked": proj.ontology_locked,
            "settings": settings,
        },
        "cloud_order": list(proj.file_keys),
        "clouds": [],
        "labels": [],
        "models": [],
        "scalars_dropped": 0,
    }

    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            if spec.clouds:
                for fk in plan.cloud_keys:
                    tick("clouds")
                    entry = catalog.entries.get(fk)
                    data, dropped = _strip_cloud_npz(cloud_store.cloud_data_path(fk))
                    zf.writestr(f"clouds/{fk}.npz", data)
                    written += len(data)
                    manifest["scalars_dropped"] += dropped
                    manifest["clouds"].append({
                        "file_key": fk,
                        "display_name": (entry.display_name if entry else None) or
                                        (entry.name if entry else fk),
                        "source_name": entry.name if entry else None,
                        "point_count": int(entry.point_count) if entry else None,
                        "region": entry.region if entry else None,
                        "patient_id": entry.patient_id if entry else None,
                        "bytes": len(data),
                        "sha256": _sha256_bytes(data),
                    })
                    done += 1
            if spec.labels:
                for fk in plan.label_keys:
                    tick("labels")
                    p = cloud_store.cloud_labels_path(fk, project_id)
                    data = p.read_bytes()
                    zf.writestr(f"labels/{fk}.npy", data)
                    written += len(data)
                    manifest["labels"].append({"file_key": fk, "bytes": len(data),
                                               "sha256": _sha256_bytes(data)})
                    done += 1
            if spec.models:
                for m in plan.models:
                    rec = dict(m["record"])
                    # Paths are meaningless on the other machine; the
                    # importer rewrites them when it unpacks.
                    for k in ("best_checkpoint", "last_checkpoint", "work_dir"):
                        rec[k] = ""
                    files = []
                    for src, arc, nbytes in m["files"]:
                        tick("models")
                        zf.write(src, arc)
                        written += nbytes
                        files.append({"path": arc, "bytes": nbytes, "sha256": _sha256_file(src)})
                        done += 1
                    manifest["models"].append({"model_id": m["model_id"], "record": rec,
                                               "files": files})
            tick("manifest")
            zf.writestr("manifest.json", json.dumps(manifest, indent=1))
        os.replace(tmp, out)
    except BaseException:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise
    if progress:
        progress("done", total, total, written)
    manifest["path"] = str(out)
    return manifest


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def inspect_bundle(path: str) -> dict:
    """Read and validate a bundle's manifest without unpacking it."""
    try:
        with zipfile.ZipFile(path) as zf:
            try:
                raw = zf.read("manifest.json")
            except KeyError:
                raise BundleError("not a Lithium bundle (no manifest.json)")
            names = set(zf.namelist())
    except zipfile.BadZipFile:
        raise BundleError("not a Lithium bundle (not a zip)")
    m = json.loads(raw.decode("utf-8"))
    if m.get("format") != FORMAT_NAME:
        raise BundleError(f"not a Lithium bundle (format {m.get('format')!r})")
    if int(m.get("version", 0)) > FORMAT_VERSION:
        raise BundleError(f"bundle version {m['version']} is newer than this Lithium "
                          f"(reads up to {FORMAT_VERSION})")
    for c in m.get("clouds", []):
        if f"clouds/{c['file_key']}.npz" not in names:
            raise BundleError(f"manifest lists cloud {c['file_key']} but the file is missing")
    m["bytes"] = os.path.getsize(path)
    m["path"] = path
    return m


@dataclass
class ImportResult:
    project_id: str
    project_name: str
    clouds_new: int = 0
    clouds_known: int = 0
    labels_written: int = 0
    labels_skipped: int = 0
    models_registered: int = 0
    entries: list = field(default_factory=list)      # LibraryEntry objects to register
    model_records: list = field(default_factory=list)  # TrainedModel dicts to register
    active_model_id: str | None = None


def _unique_project_name(catalog, name: str) -> str:
    taken = {p.name for p in catalog.projects.values()}
    if name not in taken:
        return name
    i = 2
    while f"{name} ({i})" in taken:
        i += 1
    return f"{name} ({i})"


def import_bundle(catalog, path: str, *, project_name: str | None = None,
                  into_project_id: str | None = None,
                  overwrite_labels: bool = False,
                  progress: ProgressFn | None = None,
                  cancel: threading.Event | None = None) -> ImportResult:
    """Unpack a bundle into the catalog's data/labels/models stores.

    Returns an ``ImportResult`` whose ``entries`` / ``model_records``
    still have to be registered with the catalog on the main thread
    (``apply_import``) — this function only touches per-cloud files,
    which are atomic, so it can run on a worker thread while the UI
    keeps drawing.

    ``into_project_id`` merges into an existing project (labels-only
    round trips); otherwise a new project is created by ``apply_import``
    under ``project_name`` (default: the bundle's name, de-duplicated).
    """
    from src.data.library_catalog import LibraryEntry
    m = inspect_bundle(path)
    pm = m["project"]
    if into_project_id:
        if into_project_id not in catalog.projects:
            raise BundleError(f"no project {into_project_id}")
        pid = into_project_id
        pname = catalog.projects[pid].name
    else:
        pid = f"proj:{hashlib.sha256((path + str(time.time())).encode()).hexdigest()[:8]}"
        pname = _unique_project_name(catalog, project_name or pm.get("name") or "Imported project")
    res = ImportResult(project_id=pid, project_name=pname)

    total = len(m["clouds"]) + len(m["labels"]) + sum(len(x["files"]) for x in m["models"])
    done = 0
    nbytes = 0

    def tick(stage):
        if cancel is not None and cancel.is_set():
            raise BundleCancelled()
        if progress:
            progress(stage, done, total, nbytes)

    data_dir = cloud_store.cloud_data_path("x").parent
    data_dir.mkdir(parents=True, exist_ok=True)
    models_root = Path(library_paths.library_dir()) / "models" / "imported"

    with zipfile.ZipFile(path) as zf:
        for c in m["clouds"]:
            tick("clouds")
            fk = c["file_key"]
            target = cloud_store.cloud_data_path(fk)
            raw = zf.read(f"clouds/{fk}.npz")
            nbytes += len(raw)
            if c.get("sha256") and _sha256_bytes(raw) != c["sha256"]:
                raise BundleError(f"cloud {fk}: checksum mismatch (corrupt bundle)")
            if target.exists():
                res.clouds_known += 1
            else:
                tmp = target.with_name(target.name + ".part")
                tmp.write_bytes(raw)
                os.replace(tmp, target)
                res.clouds_new += 1
            if fk not in catalog.entries:
                with np.load(target, allow_pickle=False) as z:
                    xyz = z["coord"]
                    n = int(xyz.shape[0])
                    bmin = xyz.min(axis=0) if n else None
                    bmax = xyz.max(axis=0) if n else None
                e = LibraryEntry(file_key=fk, file_path=str(target), point_count=n,
                                 bounds_min=bmin, bounds_max=bmax,
                                 patient_id=c.get("patient_id"), region=c.get("region"),
                                 display_name=c.get("display_name"))
                if c.get("source_name"):
                    e.name = c["source_name"]
                res.entries.append(e)
            done += 1

        for l in m["labels"]:
            tick("labels")
            fk = l["file_key"]
            raw = zf.read(f"labels/{fk}.npy")
            nbytes += len(raw)
            if l.get("sha256") and _sha256_bytes(raw) != l["sha256"]:
                raise BundleError(f"labels {fk}: checksum mismatch (corrupt bundle)")
            if not cloud_store.has_cloud_data(fk) and fk not in catalog.entries:
                res.labels_skipped += 1      # labels for a cloud we don't have
                done += 1
                continue
            if cloud_store.has_cloud_labels(fk, pid) and not overwrite_labels:
                res.labels_skipped += 1
                done += 1
                continue
            import io
            arr = np.load(io.BytesIO(raw), allow_pickle=False)
            err = cloud_store.save_cloud_labels(fk, np.asarray(arr, dtype=np.int32),
                                                catalog=None, namespace=pid)
            if err:
                raise BundleError(f"labels {fk}: {err}")
            res.labels_written += 1
            done += 1

        for mm in m["models"]:
            mdir = models_root / mm["model_id"]
            mdir.mkdir(parents=True, exist_ok=True)
            rec = dict(mm["record"])
            for f in mm["files"]:
                tick("models")
                dest = mdir / Path(f["path"]).name
                with zf.open(f["path"]) as src, open(str(dest) + ".part", "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                os.replace(str(dest) + ".part", dest)
                nbytes += f["bytes"]
                if f.get("sha256") and _sha256_file(str(dest)) != f["sha256"]:
                    raise BundleError(f"model {mm['model_id']}: checksum mismatch")
                if dest.name == "model_best.pth":
                    rec["best_checkpoint"] = str(dest)
                    rec["last_checkpoint"] = str(dest)
                done += 1
            rec["work_dir"] = str(mdir)
            res.model_records.append(rec)
            res.models_registered += 1

    res.active_model_id = (pm.get("settings") or {}).get("active_model_id")
    res._manifest = m  # type: ignore[attr-defined]
    if progress:
        progress("done", total, total, nbytes)
    return res


def apply_import(catalog, res: ImportResult):
    """Main-thread half of an import: register entries, create or
    extend the project, register models, save the index once."""
    from src.data.library_catalog import Project
    from src.data.model_registry import ProjectModelRegistry, TrainedModel
    m = getattr(res, "_manifest", {})
    pm = m.get("project", {})
    for e in res.entries:
        if e.file_key not in catalog.entries:
            catalog.entries[e.file_key] = e
    proj = catalog.projects.get(res.project_id)
    if proj is None:
        proj = Project(id=res.project_id, name=res.project_name,
                       ontology_data=pm.get("ontology_data"),
                       ontology_preset_name=pm.get("ontology_preset_name"),
                       settings=dict(pm.get("settings") or {}),
                       ontology_locked=bool(pm.get("ontology_locked", True)))
        catalog.projects[proj.id] = proj
    order = m.get("cloud_order") or [c["file_key"] for c in m.get("clouds", [])]
    have = set(proj.file_keys)
    for fk in order:
        if fk in catalog.entries and fk not in have:
            proj.file_keys.append(fk)
            have.add(fk)
    if res.model_records:
        reg = ProjectModelRegistry(library_paths.library_dir())
        known = {x.model_id for x in reg.load(proj.id)}
        for rec in res.model_records:
            if rec.get("model_id") in known:
                continue
            reg.add_model(proj.id, TrainedModel.from_dict(rec))
        if res.active_model_id and any(r.get("model_id") == res.active_model_id
                                       for r in res.model_records):
            proj.settings = dict(proj.settings or {})
            proj.settings["active_model_id"] = res.active_model_id
    else:
        if proj.settings:
            proj.settings.pop("active_model_id", None)
    catalog._save_index()
    catalog._save_projects()
    try:
        catalog._bump_version()
    except Exception:
        pass
    return proj


def human_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
