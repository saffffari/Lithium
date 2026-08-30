#!/usr/bin/env python
"""Prune the Lithium library down to one (or more) projects.

Wipes every cloud, label, preview, mesh, primitive set, and project
that is NOT a member of the named projects. Preserves the named
projects' ontology, model registry, and every cloud their file_keys
reference. Tarballs the entire library first so nothing is destroyed
without an undo path.

Typical use case: the catalog has accumulated multiple iteration runs
(re-extractions producing new file_keys for the same anatomy) plus
side-experiment projects. You want the catalog to contain ONLY the
canonical training project — same as the state after the first
successful end-to-end run.

Usage::

    # Dry run first — shows what would be removed without touching anything.
    python tools/prune_catalog_to_project.py --project spinelab_4class --dry-run

    # Real run — backs up to ~/.lithium/backups/ then prunes.
    python tools/prune_catalog_to_project.py --project spinelab_4class

    # Keep multiple projects:
    python tools/prune_catalog_to_project.py \\
        --project spinelab_4class --project spinelab_vertebral_subregions

Project may be either the literal project id (``proj:abc123ef``) or
the human-readable name. Names get fuzzy-matched against
``projects.json`` so a typo with the wrong case still resolves.

Out of scope: this script does NOT touch ``D:\\Lithium\\dataset\\
training_runs\\`` — the trained model checkpoints live there, and the
model registry just stores absolute paths to them. After pruning,
those checkpoints stay reachable for inference / fine-tune. If you
also want to clean up training_runs directories that no project
references anymore, that's a separate manual cleanup.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path


# Add project root to sys.path so src.* imports resolve when this is run
# directly via ``python tools/prune_catalog_to_project.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import library_catalog as lc


# Per-cloud artifact subdirectories under ~/.lithium/library/.
# Each maps a subdir name to the file extension entries use.
_PER_CLOUD_SUBDIRS = {
    "data":             ".npz",
    "labels":           ".npy",
    "previews":         ".npz",
    "meshes":           ".npz",
    "primitives":       ".json",
    "preview_labels":   ".npy",
}

# Derived artifacts that are safe to wipe for kept clouds — they
# regenerate cheaply from data + labels. preview_labels was here
# initially but it broke gallery thumbnail colour rendering: the
# SHEETS tab renders previews, and without their labels the thumbnails
# come up white. Regenerating preview_labels in bulk requires loading
# each full cloud + KD-tree query per cloud, much more expensive than
# their on-disk cost (~few KB each). They stay.
_DERIVED_SUBDIRS = {"primitives", "meshes"}


def _backup_library(library_dir: Path) -> Path:
    """Tar-gz the entire library to ~/.lithium/backups/. Returns path."""
    backups_dir = library_dir.parent / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    backup_path = backups_dir / f"library_pre_prune_{ts}.tar.gz"
    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(library_dir, arcname=library_dir.name)
    return backup_path


def _resolve_project_ids(projects: dict, requested: list[str]) -> list[str]:
    """Map each user-supplied string to a project_id present in projects.

    Accepts exact ids (``proj:abc123``) and case-insensitive name
    matches. Returns the resolved list; raises ValueError on unknown
    targets.
    """
    resolved: list[str] = []
    missing: list[str] = []
    name_to_id = {p.get("name", "").lower(): pid for pid, p in projects.items()}
    for r in requested:
        if r in projects:
            resolved.append(r)
            continue
        lower = r.lower()
        if lower in name_to_id:
            resolved.append(name_to_id[lower])
            continue
        # Last-ditch: substring match
        matches = [pid for pid, p in projects.items()
                   if lower in p.get("name", "").lower()]
        if len(matches) == 1:
            resolved.append(matches[0])
            continue
        if len(matches) > 1:
            names = ", ".join(projects[m].get("name", "?") for m in matches)
            raise ValueError(
                f"'{r}' is ambiguous — matches: {names}. Use the full "
                f"name or the project id (proj:...).")
        missing.append(r)
    if missing:
        raise ValueError(
            f"project(s) not found: {missing}. Known projects:\n  " +
            "\n  ".join(f"{p.get('name', '?'):40s}  {pid}"
                        for pid, p in projects.items())
        )
    return resolved


def _model_filename_to_pid(filename: str) -> str:
    """Reverse the colon-sanitisation ``ProjectModelRegistry._path`` does.

    The registry stores ``proj:abc123.json`` as ``proj_abc123.json`` on
    Windows because the FS rejects ``:``. Map the filename stem back to
    the canonical project id.
    """
    stem = Path(filename).stem
    if stem.startswith("proj_"):
        return "proj:" + stem[len("proj_"):]
    return stem


def prune(project_ids: list[str], *, dry_run: bool,
          drop_models: bool = False,
          drop_derived: bool = False) -> int:
    library_dir = Path(lc.library_dir())
    if not library_dir.is_dir():
        print(f"ERROR: library not found at {library_dir}", file=sys.stderr)
        return 2

    projects_path = library_dir / "projects.json"
    index_path = library_dir / "index.json"
    if not projects_path.is_file() or not index_path.is_file():
        print(f"ERROR: catalog incomplete (missing projects.json or index.json)",
              file=sys.stderr)
        return 2

    with open(projects_path, encoding="utf-8") as f:
        projects = json.load(f)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    try:
        keep_pids = _resolve_project_ids(projects, project_ids)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    keep_pids_set = set(keep_pids)
    keep_file_keys: set[str] = set()
    for pid in keep_pids:
        for fk in projects[pid].get("file_keys", []):
            keep_file_keys.add(fk)

    drop_pids = [pid for pid in projects if pid not in keep_pids_set]
    drop_file_keys = set(index.keys()) - keep_file_keys

    print(f"Library: {library_dir}")
    print(f"Keep projects ({len(keep_pids)}):")
    for pid in keep_pids:
        proj = projects[pid]
        print(f"  {proj.get('name', '?'):40s}  {pid}  "
              f"({len(proj.get('file_keys', []))} clouds)")
    print(f"Drop projects ({len(drop_pids)}):")
    for pid in drop_pids:
        proj = projects[pid]
        print(f"  {proj.get('name', '?'):40s}  {pid}  "
              f"({len(proj.get('file_keys', []))} clouds)")
    print(f"Keep clouds ({len(keep_file_keys)})")
    print(f"Drop clouds ({len(drop_file_keys)})")

    # Count per-subdir files that will be deleted. Walk the disk so
    # orphan files (no index entry) also get cleaned up — those are
    # very common after iterative re-extraction runs.
    delete_plan: dict[str, list[Path]] = {sub: [] for sub in _PER_CLOUD_SUBDIRS}
    for sub, ext in _PER_CLOUD_SUBDIRS.items():
        sub_path = library_dir / sub
        if not sub_path.is_dir():
            continue
        drop_for_kept = drop_derived and (sub in _DERIVED_SUBDIRS)
        for fp in sub_path.iterdir():
            if not fp.is_file() or fp.suffix != ext:
                continue
            stem = fp.stem
            if stem.startswith("_tmp_"):
                # Always remove tmp files left over from atomic writes.
                delete_plan[sub].append(fp)
                continue
            if stem not in keep_file_keys:
                delete_plan[sub].append(fp)
            elif drop_for_kept:
                # --drop-derived: nuke regenerable artifacts (primitives,
                # meshes, preview_labels) for kept clouds too. Leaves
                # the minimal "training-ready" state — points + labels
                # + previews, nothing inferred yet.
                delete_plan[sub].append(fp)
    total_files = sum(len(v) for v in delete_plan.values())
    print(f"Per-cloud files to delete: {total_files}")
    for sub, files in delete_plan.items():
        if files:
            note = ""
            if drop_derived and sub in _DERIVED_SUBDIRS:
                note = "  (incl. kept-cloud derived artifacts)"
            print(f"  {sub}/: {len(files)}{note}")

    # Model registry: drop any that don't belong to a kept project.
    # With --drop-models, also drop the kept project's models so the
    # registry starts empty (useful when you're about to retrain from
    # scratch and don't want stale entries pointing at old runs).
    models_dir = library_dir / "models"
    drop_model_files: list[Path] = []
    if models_dir.is_dir():
        for mf in models_dir.iterdir():
            if not mf.is_file() or mf.suffix != ".json":
                continue
            pid = _model_filename_to_pid(mf.name)
            if pid not in keep_pids_set:
                drop_model_files.append(mf)
            elif drop_models:
                drop_model_files.append(mf)
    note = "  (including kept-project registries)" if drop_models else ""
    print(f"Model registry files to delete: {len(drop_model_files)}{note}")

    print()
    if dry_run:
        print("DRY RUN — no changes made. Re-run without --dry-run to apply.")
        return 0

    # Backup first. _backup_library raises on failure; let the exception
    # propagate so we never half-apply a destructive operation.
    print("Backing up library before pruning...")
    backup = _backup_library(library_dir)
    print(f"  backup at {backup}  ({backup.stat().st_size / 1e6:.1f} MB)")

    # Apply pruning.
    pruned_files = 0
    for sub, files in delete_plan.items():
        for fp in files:
            try:
                fp.unlink()
                pruned_files += 1
            except OSError as e:
                print(f"  skip {fp.name}: {e}")
    for mf in drop_model_files:
        try:
            mf.unlink()
            pruned_files += 1
        except OSError as e:
            print(f"  skip {mf.name}: {e}")

    # Rewrite index.json with only the kept file_keys.
    pruned_index = {fk: entry for fk, entry in index.items()
                    if fk in keep_file_keys}
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(pruned_index, f, indent=2)

    # Rewrite projects.json with only the kept projects, and within
    # each kept project, filter file_keys down to what survives in the
    # index (some referenced clouds may have already been missing).
    pruned_projects: dict = {}
    for pid in keep_pids:
        proj = dict(projects[pid])
        proj["file_keys"] = [fk for fk in proj.get("file_keys", [])
                             if fk in pruned_index]
        pruned_projects[pid] = proj
    with open(projects_path, "w", encoding="utf-8") as f:
        json.dump(pruned_projects, f, indent=2)

    # Move the dropped projects' ontology backups too (projects.json.bak).
    bak_path = library_dir / "projects.json.bak"
    if bak_path.exists():
        try:
            bak_path.unlink()
        except OSError:
            pass

    print(f"Pruned {pruned_files} files.")
    print(f"Index now contains {len(pruned_index)} clouds across "
          f"{len(pruned_projects)} project(s).")
    print()
    print(f"Backup available at: {backup}")
    print(f"To restore: extract the tarball back over {library_dir.parent}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", action="append", required=True,
                    help="project id (proj:abc...) or name. Repeat for "
                         "multiple projects to keep.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be pruned without touching disk.")
    ap.add_argument("--drop-models", action="store_true",
                    help="Also delete the kept project's model registry "
                         "file(s). Use when you're about to retrain "
                         "from scratch and don't want stale entries.")
    ap.add_argument("--drop-derived", action="store_true",
                    help="Also delete derived artifacts (primitives, "
                         "meshes, preview_labels) for kept clouds. "
                         "Regenerable from data + labels, so safe to "
                         "wipe when reducing to a training-ready state.")
    ap.add_argument("--training-ready", action="store_true",
                    help="Shortcut: --drop-models + --drop-derived. "
                         "Result: keeps only data + labels + previews + "
                         "project ontology for the named project(s). "
                         "Ideal for handing the catalog off to a fresh "
                         "training run.")
    args = ap.parse_args()
    drop_models = args.drop_models or args.training_ready
    drop_derived = args.drop_derived or args.training_ready
    return prune(args.project, dry_run=args.dry_run,
                 drop_models=drop_models, drop_derived=drop_derived)


if __name__ == "__main__":
    sys.exit(main())
