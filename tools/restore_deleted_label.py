#!/usr/bin/env python
"""Restore a label class that was removed from the registry, without
losing the points already painted with it.

Why this works: ``LabelRegistry.remove_label`` is a soft delete — it
drops the registry entry but leaves every point's numeric ``label_id``
on disk untouched. The points become "ghost-labelled" (a numeric id
with no registry entry → not rendered with that name/color). If you
re-add a label with the same id, those points reappear, painted under
the new entry's name + color.

This script recovers the original name + color from a pre-prune
backup tarball, then injects the label back into the live project's
ontology at the same id. The on-disk point labels never change.

Usage::

    python tools/restore_deleted_label.py \\
        --project spinelab_vertebral_subregions \\
        --label-id 3

Optional flags:

    --backup <path>     Path to the pre-prune tarball. Defaults to the
                        most-recent ``library_pre_prune_*.tar.gz``
                        under ~/.3photon/backups/.
    --name "<text>"     Override the recovered name (e.g. if you want
                        to rename the restored label).
    --color R,G,B[,A]   Override the recovered color, components in
                        [0, 1].
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data import library_catalog as lc


def _latest_backup(library_dir: Path) -> Path | None:
    backups_dir = library_dir.parent / "backups"
    if not backups_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in backups_dir.iterdir()
         if p.is_file() and p.name.startswith("library_pre_prune_")
         and p.suffix == ".gz"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _read_projects_from_tarball(tarball: Path) -> dict:
    """Extract just projects.json from the backup tarball."""
    with tarfile.open(tarball, "r:gz") as tar:
        for member in tar.getmembers():
            # Members are stored as ``library/projects.json``.
            if member.name.endswith("projects.json") and not member.isdir():
                fh = tar.extractfile(member)
                if fh is None:
                    continue
                return json.loads(fh.read().decode("utf-8"))
    raise FileNotFoundError(f"projects.json not found in {tarball}")


def _find_label_in_backup(projects: dict, project_query: str,
                          label_id: int) -> dict | None:
    """Locate the original LabelInfo entry for ``label_id`` inside the
    backup's ontology for the named project. ``project_query`` is matched
    against id, name (exact), and substring."""
    candidates: list[tuple[str, dict]] = []
    for pid, proj in projects.items():
        name = proj.get("name", "")
        if (pid == project_query
                or name.lower() == project_query.lower()
                or project_query.lower() in name.lower()):
            candidates.append((pid, proj))
    if not candidates:
        raise ValueError(
            f"project '{project_query}' not found in backup. "
            f"Available: {[p.get('name') for p in projects.values()]}"
        )
    if len(candidates) > 1:
        names = [c[1].get("name") for c in candidates]
        raise ValueError(
            f"'{project_query}' is ambiguous in backup — matches {names}. "
            f"Pass the full name or project id.")
    proj = candidates[0][1]
    ont = proj.get("ontology_data") or {}
    for lab in ont.get("labels", []):
        if int(lab.get("id", -1)) == int(label_id):
            return lab
    return None


def _parse_color(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError(f"--color expects 3 or 4 comma-separated floats; got {raw!r}")
    vals = tuple(float(p) for p in parts)
    if any(v < 0.0 or v > 1.0 for v in vals):
        raise ValueError("--color components must be in [0, 1]")
    return vals


def restore(project_query: str, label_id: int, *,
            backup_path: Path | None,
            name_override: str | None,
            color_override: tuple[float, ...] | None) -> int:
    library_dir = Path(lc.library_dir())
    if not library_dir.is_dir():
        print(f"ERROR: library not found at {library_dir}", file=sys.stderr)
        return 2

    if backup_path is None:
        backup_path = _latest_backup(library_dir)
        if backup_path is None:
            print("ERROR: no pre-prune backup found and --backup not given.",
                  file=sys.stderr)
            return 3
    if not backup_path.is_file():
        print(f"ERROR: backup not found: {backup_path}", file=sys.stderr)
        return 3
    print(f"Using backup: {backup_path}")

    try:
        backup_projects = _read_projects_from_tarball(backup_path)
    except (tarfile.TarError, OSError, json.JSONDecodeError,
            FileNotFoundError) as e:
        print(f"ERROR reading backup tarball: {e}", file=sys.stderr)
        return 4

    try:
        original = _find_label_in_backup(
            backup_projects, project_query, label_id)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 5
    if original is None:
        print(f"ERROR: label id={label_id} not found in backup's "
              f"'{project_query}' ontology.", file=sys.stderr)
        return 6

    recovered_name = original.get("name", f"label_{label_id}")
    recovered_color = tuple(float(c) for c in original.get(
        "color", (0.8, 0.8, 0.8, 1.0)))
    if len(recovered_color) == 3:
        recovered_color = recovered_color + (1.0,)
    final_name = name_override or recovered_name
    final_color = color_override if color_override is not None else recovered_color
    if len(final_color) == 3:
        final_color = final_color + (1.0,)
    parent_id = original.get("parent_id")

    print(f"Recovered from backup: id={label_id} name={recovered_name!r} "
          f"color={recovered_color}")
    if name_override:
        print(f"  override name → {final_name!r}")
    if color_override is not None:
        print(f"  override color → {final_color}")
    print()

    # Now load the LIVE projects.json + inject the label entry.
    live_path = library_dir / "projects.json"
    with open(live_path, encoding="utf-8") as f:
        live = json.load(f)

    # Find the live project entry that matches the query.
    live_pid: str | None = None
    for pid, proj in live.items():
        name = proj.get("name", "")
        if (pid == project_query
                or name.lower() == project_query.lower()
                or project_query.lower() in name.lower()):
            live_pid = pid
            break
    if live_pid is None:
        print(f"ERROR: '{project_query}' not in live projects.json. The "
              f"backup has it but the catalog doesn't — restore the "
              f"project from backup first, or use a different name.",
              file=sys.stderr)
        return 7

    proj = live[live_pid]
    ont = proj.get("ontology_data")
    if ont is None:
        print(f"ERROR: live project '{live_pid}' has no ontology_data "
              f"(no labels). Add at least one label via the UI first.",
              file=sys.stderr)
        return 8
    labels = ont.setdefault("labels", [])

    # Refuse to overwrite a different label at the same id.
    for lab in labels:
        if int(lab.get("id", -1)) == int(label_id):
            print(f"ERROR: label id={label_id} already occupied in live "
                  f"registry by '{lab.get('name')!r}'. Either delete it "
                  f"first or pick a different id.", file=sys.stderr)
            return 9

    new_entry = {
        "id": int(label_id),
        "name": final_name,
        "color": list(final_color),
    }
    if parent_id is not None:
        new_entry["parent_id"] = parent_id
    labels.append(new_entry)
    # Keep them sorted by id for tidy diffs.
    labels.sort(key=lambda l: int(l.get("id", 0)))

    # Atomic write.
    tmp_path = live_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(live, f, indent=2)
    os.replace(tmp_path, live_path)

    print(f"Restored label id={label_id} '{final_name}' to "
          f"'{proj.get('name')}'.")
    print()
    print("Restart the app (or reload the project) to see the label "
          "in the LABELS panel. Any points still carrying numeric "
          f"label_id={label_id} on disk reappear automatically.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", required=True,
                    help="Project id (proj:abc...) or name/substring.")
    ap.add_argument("--label-id", type=int, required=True,
                    help="The numeric id of the deleted label.")
    ap.add_argument("--backup", type=Path, default=None,
                    help="Pre-prune tarball path. Defaults to the most "
                         "recent library_pre_prune_*.tar.gz in "
                         "~/.3photon/backups/.")
    ap.add_argument("--name", default=None,
                    help="Override the recovered label name.")
    ap.add_argument("--color", default=None,
                    help="Override color, comma-separated RGB or RGBA "
                         "in [0,1]. Example: --color 0.4,0.6,1.0")
    args = ap.parse_args()
    color_override = _parse_color(args.color) if args.color else None
    return restore(
        args.project, args.label_id,
        backup_path=args.backup,
        name_override=args.name,
        color_override=color_override,
    )


if __name__ == "__main__":
    sys.exit(main())
