#!/usr/bin/env python
"""Export / import / inspect Lithium project bundles (.li) from the shell.

Same code path as File > Export Project / Import Project, minus the
window — for building hand-off kits and for scripted round trips.

    python tools/bundle_cli.py export "My project" \
        --out /tmp/my_clouds.li --clouds                  # clouds only (blinded)
    python tools/bundle_cli.py export proj:1a2b3c4d --out x.li --clouds --labels --models
    python tools/bundle_cli.py inspect x.li
    python tools/bundle_cli.py import x.li [--name "Reader 1"] [--into proj:...] [--overwrite-labels]

Close Lithium first (single-instance catalog lock) when importing.
Set LITHIUM_LIBRARY_DIR to target a catalog other than ~/.lithium/library.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import bundle  # noqa: E402
from src.data.library_catalog import LibraryCatalog  # noqa: E402


def _resolve_project(cat: LibraryCatalog, name_or_id: str):
    if name_or_id in cat.projects:
        return cat.projects[name_or_id]
    for p in cat.projects.values():
        if p.name == name_or_id:
            return p
    low = name_or_id.lower()
    hits = [p for p in cat.projects.values() if low in p.name.lower()]
    if len(hits) == 1:
        return hits[0]
    names = "\n  ".join(f"{p.id}  {p.name}" for p in cat.projects.values())
    sys.exit(f"no unique project matching {name_or_id!r}. Projects:\n  {names}")


def _progress_printer():
    t0 = time.time()
    last = [0.0]

    def prog(stage, done, total, nbytes):
        now = time.time()
        if stage == "done" or now - last[0] > 0.5:
            last[0] = now
            pct = (100.0 * done / total) if total else 0.0
            print(f"\r  {stage:<10} {done:>6}/{total:<6} {pct:5.1f}%  "
                  f"{bundle.human_bytes(nbytes):>9}  {now - t0:5.1f}s", end="", flush=True)
        if stage == "done":
            print()
    return prog


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("export", help="write a .li from a project")
    ex.add_argument("project", help="project id (proj:...) or name")
    ex.add_argument("--out", required=True)
    ex.add_argument("--clouds", action="store_true")
    ex.add_argument("--labels", action="store_true")
    ex.add_argument("--models", action="store_true")

    ins = sub.add_parser("inspect", help="print a bundle's manifest summary")
    ins.add_argument("path")

    im = sub.add_parser("import", help="unpack a .li into this catalog")
    im.add_argument("path")
    im.add_argument("--name", help="new project name (default: the bundle's)")
    im.add_argument("--into", help="merge into an existing project id / name")
    im.add_argument("--overwrite-labels", action="store_true")

    a = ap.parse_args()

    if a.cmd == "inspect":
        m = bundle.inspect_bundle(a.path)
        inc = m["includes"]
        print(f"{os.path.basename(a.path)}  ·  {bundle.human_bytes(m['bytes'])}  ·  format v{m['version']}")
        print(f"  project : {m['project'].get('name')}")
        onto = (m['project'].get('ontology_data') or {}).get('labels') or []
        if onto:
            print(f"  ontology: {len(onto)} classes: " + ", ".join(l.get('name', '?') for l in onto))
        print(f"  clouds  : {len(m['clouds'])}" if inc.get('clouds') else "  clouds  : not included")
        print(f"  labels  : {len(m['labels'])}" if inc.get('labels') else "  labels  : not included")
        print(f"  models  : " + (", ".join(x['record'].get('name', x['model_id']) for x in m['models'])
                                 if m['models'] else "not included"))
        print(f"  scalars stripped from clouds: {m.get('scalars_dropped', 0)}")
        return 0

    cat = LibraryCatalog()
    if a.cmd == "export":
        if not (a.clouds or a.labels or a.models):
            sys.exit("pick at least one of --clouds --labels --models")
        proj = _resolve_project(cat, a.project)
        plan = bundle.plan_export(cat, proj.id)
        spec = bundle.ExportSpec(a.clouds, a.labels, a.models)
        print(f"exporting {proj.name!r}: {len(plan.cloud_keys)} clouds "
              f"({bundle.human_bytes(plan.cloud_bytes)}), {len(plan.label_keys)} label files "
              f"({bundle.human_bytes(plan.label_bytes)}), {len(plan.models)} models "
              f"({bundle.human_bytes(plan.model_bytes)}); including "
              + ", ".join(k for k, on in (("clouds", a.clouds), ("labels", a.labels), ("models", a.models)) if on))
        if plan.missing_clouds:
            print(f"  warning: {plan.missing_clouds} members have no catalog data and are skipped")
        man = bundle.export_bundle(cat, proj.id, spec, a.out, progress=_progress_printer())
        print(f"wrote {man['path']}  ({bundle.human_bytes(os.path.getsize(man['path']))}; "
              f"{man['scalars_dropped']} scalar layers stripped)")
        return 0

    if a.cmd == "import":
        into = _resolve_project(cat, a.into).id if a.into else None
        res = bundle.import_bundle(cat, a.path, project_name=a.name, into_project_id=into,
                                   overwrite_labels=a.overwrite_labels, progress=_progress_printer())
        proj = bundle.apply_import(cat, res)
        print(f"imported into {proj.name!r} ({proj.id}): {res.clouds_new} new clouds, "
              f"{res.clouds_known} already present, {res.labels_written} label files written, "
              f"{res.labels_skipped} skipped, {res.models_registered} models")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
