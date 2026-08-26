#!/usr/bin/env python3
"""Apply one canonical label palette across every project in the catalog.

Colours are assigned by *class name* (normalised: case/space/underscore
insensitive, common aliases folded), so "Superior_Endplate", "Sup Endplate"
and "Endplate" all get the endplate colour and the same structure looks the
same in every project. Alpha is preserved (Unlabeled keeps its 0.3).

    .venv/bin/python tools/apply_label_palette.py --palette A          # audit
    .venv/bin/python tools/apply_label_palette.py --palette A --apply  # write (Lithium closed)

A timestamped backup of projects.json is written before any change.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, sys, time
from pathlib import Path

PALETTES = {
    # role:            unlabeled   sup        inf        pedicle    body_wall  spinous    transverse facet_l    facet_r    ped_left   ped_right  tips       misc
    "A": dict(unlabeled="#6e7480", sup="#ff8c1a", inf="#2e9bff", pedicle="#c8ff3d", body="#8c7bff", spinous="#ff4fa0", transverse="#ffd23f", facet_l="#3df2e6", facet_r="#19b8ad", ped_l="#c8ff3d", ped_r="#5be36a", tips="#fff27a", misc="#ff7a5c"),
    "B": dict(unlabeled="#5c6169", sup="#ffb000", inf="#00b4ff", pedicle="#7cff4f", body="#ff5e3a", spinous="#c77dff", transverse="#ffe770", facet_l="#35e5c4", facet_r="#1fa896", ped_l="#7cff4f", ped_r="#2fd97a", tips="#fff2a8", misc="#ff8fb1"),
    "C": dict(unlabeled="#4b5058", sup="#e8a33d", inf="#4c9be8", pedicle="#8fd36b", body="#d96c5f", spinous="#b48ae0", transverse="#e6d36a", facet_l="#63c7bd", facet_r="#3f9a92", ped_l="#8fd36b", ped_r="#5fae62", tips="#f0e6a0", misc="#e08aa8"),
}

ROLE_RULES = [                       # first match wins, on the normalised name
    (r"^(unlabeled|rest|notendplate|otherbone|other|background|label\d+)$", "unlabeled"),
    (r"^(sup|superior)endplate$|^s1supendplate$", "sup"),
    (r"^(inf|inferior)endplate$", "inf"),
    (r"^endplate$", "sup"),
    (r"^pedicle(merged)?$", "pedicle"),
    (r"^(left|l)pedicle$|^pedicle(left|l)$", "ped_l"),
    (r"^(right|r)pedicle$|^pedicle(right|r)$", "ped_r"),
    (r"^(left|l)facet$|^facet(left|l)$", "facet_l"),
    (r"^(right|r)facet$|^facet(right|r)$", "facet_r"),
    (r"^bodywall$", "body"),
    (r"^spinousprocess$", "spinous"),
    (r"^transverseprocess$", "transverse"),
    (r"^processtips?$", "tips"),
]


def norm(name: str) -> str:
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def role_for(name: str) -> str | None:
    n = norm(name)
    for pat, role in ROLE_RULES:
        if re.match(pat, n):
            return role
    return None


def hex_rgb(h: str):
    h = h.lstrip("#"); return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", choices=sorted(PALETTES), required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--library", default=os.environ.get("THREEPHOTON_LIBRARY_DIR", str(Path.home() / ".3photon" / "library")))
    a = ap.parse_args()
    lib = Path(a.library); pj = lib / "projects.json"
    pal = PALETTES[a.palette]
    data = json.load(open(pj))
    changes, unknown = [], set()
    for pid, proj in data.items():
        if not isinstance(proj, dict): continue
        for row in (proj.get("ontology_data") or {}).get("labels", []):
            role = role_for(row.get("name", ""))
            if role is None:
                unknown.add(row.get("name", "")); continue
            new = hex_rgb(pal[role]); old = list(row.get("color", [0.5, 0.5, 0.5, 1.0]))
            if [round(x, 4) for x in old[:3]] != [round(x, 4) for x in new]:
                changes.append((proj.get("name"), row["name"], role, old[:3], new))
                row["color"] = new + [old[3] if len(old) > 3 else (0.3 if role == "unlabeled" else 1.0)]
    print(f"palette {a.palette}: {len(changes)} colour changes across {len({c[0] for c in changes})} projects")
    for pname, cname, role, old, new in changes[:60]:
        print(f"  {pname[:44]:44s} {cname:20s} -> {role:10s} {pal[role]}")
    if unknown:
        print("UNMAPPED class names (left untouched):", sorted(unknown))
    if not a.apply:
        print("audit only — add --apply (with Lithium closed) to write"); return
    if (lib / ".lock").exists():
        sys.exit(f"Lithium is running (lock {lib/'.lock'}); close it first")
    backup = lib / "backups" / f"projects_before_palette_{a.palette}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    backup.parent.mkdir(exist_ok=True); shutil.copy2(pj, backup)
    tmp = pj.with_suffix(".json.tmp"); json.dump(data, open(tmp, "w"), indent=1); os.replace(tmp, pj)
    print(f"written {pj}  (backup {backup})")


if __name__ == "__main__":
    main()
