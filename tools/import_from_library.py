#!/usr/bin/env python
"""Import clouds/projects from a foreign 1.0/1.1 library into the live one.

Use cases: pulling projects out of a backup (e.g. a home-backup
snapshot of ~/.lithium), merging a LITHIUM_LIBRARY_DIR side
library, or migrating from another machine.

Strictly additive: file_keys already present in the live library are
skipped (reported, never overwritten). Per imported cloud we copy:

    data/<fk>.npz                       full-res points (write-once)
    previews/<fk>.npz                   gallery thumbnail
    meshes/<fk>.npz                     poisson mesh, if built
    labels:   v1 flat labels/<fk>.npy   OR v2 labels/<ns>/<fk>.npy
    preview_labels: same two layouts

Labels land in BOTH the live `_library` baseline namespace and the
destination project's namespace (v2 semantics: the project gets its own
independent copy). If the source library has a project with an
ontology and the destination project has none, the ontology is adopted
verbatim so imported label ids keep their names and colors.

Usage:
  import_from_library.py --source <lib_dir> --into <dest project name> \
      [--project <source project id|name>]   # import that project
      [--clouds <substring>|all]             # or pick by filename
      [--source-namespace <ns>]              # v2 source: which ns to read
      [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data import cloud_store, library_paths               # noqa: E402
from src.data.library_catalog import LibraryCatalog, LibraryEntry  # noqa: E402


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _source_label_path(src: Path, fk: str, ns: str | None) -> Path | None:
    """Find a cloud's labels in a v1-flat or v2-namespaced source."""
    candidates = []
    if ns:
        candidates.append(src / "labels" / cloud_store.sanitize_namespace(ns)
                          / f"{fk}.npy")
    candidates += [
        src / "labels" / cloud_store.LIBRARY_NAMESPACE / f"{fk}.npy",  # v2
        src / "labels" / f"{fk}.npy",                                  # v1
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _source_preview_label_path(src: Path, fk: str, ns: str | None):
    candidates = []
    if ns:
        candidates.append(src / "preview_labels"
                          / cloud_store.sanitize_namespace(ns) / f"{fk}.npy")
    candidates += [
        src / "preview_labels" / cloud_store.LIBRARY_NAMESPACE / f"{fk}.npy",
        src / "preview_labels" / f"{fk}.npy",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="foreign library dir (the one containing index.json)")
    ap.add_argument("--into", required=True,
                    help="destination project name in the live library "
                         "(created if missing)")
    ap.add_argument("--project", default="",
                    help="source project id or name to import")
    ap.add_argument("--clouds", default="",
                    help="import clouds whose file path contains this "
                         "substring ('all' = every cloud)")
    ap.add_argument("--source-namespace", default="",
                    help="v2 source: label namespace to read (default: "
                         "the source project's id, else _library)")
    ap.add_argument("--path-map", action="append", default=[],
                    metavar="OLD=NEW",
                    help="rewrite source file path prefixes (repeatable). "
                         "For backups: --path-map /home/alex=/backup/alex")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path_maps = []
    for m in args.path_map:
        old, _, new = m.partition("=")
        if old and new:
            path_maps.append((old, new))

    def remap(p: str) -> str:
        for old, new in path_maps:
            if p.startswith(old):
                return new + p[len(old):]
        return p

    src = Path(args.source).resolve()
    src_index = _load_json(src / "index.json")
    if not src_index:
        print(f"no readable index.json under {src}")
        return 1
    src_projects = (_load_json(src / "projects.json")
                    or _load_json(src / "collections.json"))

    # ---- resolve the set of file_keys + optional source ontology ----
    src_ontology = None
    src_ns = args.source_namespace or None
    if args.project:
        proj = None
        for p in src_projects.values():
            if args.project in (p.get("id"), p.get("name")):
                proj = p
                break
        if proj is None:
            print(f"source project {args.project!r} not found; "
                  f"available: {[p.get('name') for p in src_projects.values()]}")
            return 1
        file_keys = [fk for fk in proj.get("file_keys", [])
                     if fk in src_index]
        src_ontology = proj.get("ontology_data")
        if src_ns is None:
            src_ns = proj.get("id")
    elif args.clouds:
        needle = "" if args.clouds == "all" else args.clouds.lower()
        file_keys = [fk for fk, e in src_index.items()
                     if needle in e.get("file_path", "").lower()]
    else:
        print("pick --project or --clouds")
        return 1
    if not file_keys:
        print("nothing matched")
        return 1

    live_dir = Path(library_paths.library_dir())
    catalog = LibraryCatalog()
    dest_proj = next((p for p in catalog.projects.values()
                      if p.name == args.into), None)

    print(f"source:  {src}  ({len(src_index)} clouds)")
    print(f"import:  {len(file_keys)} clouds -> project {args.into!r} "
          f"({'existing' if dest_proj else 'new'})")
    if args.dry_run:
        for fk in file_keys:
            e = src_index[fk]
            has_lbl = _source_label_path(src, fk, src_ns) is not None
            dup = fk in catalog.entries
            print(f"  {fk} {Path(e['file_path']).name:38s} "
                  f"{e.get('point_count', 0):>10,} pts "
                  f"{'labels' if has_lbl else '      '} "
                  f"{'SKIP(exists)' if dup else ''}")
        return 0

    if dest_proj is None:
        dest_proj = catalog.create_project(args.into)
    if src_ontology and dest_proj.ontology_data is None:
        dest_proj.ontology_data = src_ontology
        print("adopted source project ontology")

    dest_ns = dest_proj.id
    imported = skipped = labeled = 0
    for fk in file_keys:
        if fk in catalog.entries:
            skipped += 1
            continue
        e = src_index[fk]
        src_data = src / "data" / f"{fk}.npz"
        if src_data.is_file():
            shutil.copy2(src_data, live_dir / "data" / f"{fk}.npz")
        else:
            # 1.0 didn't always cache data/<fk>.npz — fall back to
            # re-reading the source file itself (path-remapped so
            # backup trees resolve).
            source_path = remap(e.get("file_path", ""))
            if not Path(source_path).is_file():
                print(f"  {fk}: no data npz and source file missing "
                      f"({source_path}) — skipped")
                skipped += 1
                continue
            from src.data.loader import load_point_cloud
            try:
                cloud = load_point_cloud(source_path)
            except Exception as ex:
                print(f"  {fk}: source load failed: {ex} — skipped")
                skipped += 1
                continue
            cloud_store.save_cloud_data(fk, cloud, source_path=source_path)
            e = dict(e)
            e["file_path"] = source_path
            if not e.get("point_count"):
                e["point_count"] = cloud.point_count
        for sub in ("previews", "meshes"):
            sp = src / sub / f"{fk}.npz"
            if sp.is_file():
                (live_dir / sub).mkdir(exist_ok=True)
                shutil.copy2(sp, live_dir / sub / f"{fk}.npz")
        lp = _source_label_path(src, fk, src_ns)
        if lp is not None:
            arr = np.load(lp)
            cloud_store.save_cloud_labels(
                fk, arr, namespace=cloud_store.LIBRARY_NAMESPACE)
            cloud_store.save_cloud_labels(fk, arr, namespace=dest_ns)
            labeled += 1
        plp = _source_preview_label_path(src, fk, src_ns)
        if plp is not None:
            parr = np.load(plp)
            cloud_store.save_preview_labels(
                fk, parr, namespace=cloud_store.LIBRARY_NAMESPACE)
            cloud_store.save_preview_labels(fk, parr, namespace=dest_ns)
        catalog.entries[fk] = LibraryEntry.from_dict(e)
        dest_proj.file_keys.append(fk)
        imported += 1
        print(f"  + {Path(e['file_path']).name} "
              f"({e.get('point_count', 0):,} pts"
              f"{', labels' if lp is not None else ''})")

    catalog._save_index()
    catalog._save_projects()
    print(f"done: {imported} imported ({labeled} with labels), "
          f"{skipped} skipped, project {dest_proj.name!r} now has "
          f"{len(dest_proj.file_keys)} clouds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
