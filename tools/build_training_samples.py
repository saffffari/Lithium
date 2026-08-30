#!/usr/bin/env python
"""Wipe the catalog and stage three balanced training-sample groups.

Reads the canonical VerSe per-vertebra PLYs exported under
``E:\\data\\verse\\exports\\{train,val,test}\\``, picks three disjoint
groups of ~100 vertebrae, and copies them into
``E:\\data\\verse\\spinelab_training_samples\\group_{1,2,3}\\`` for
import into Lithium as fresh training batches.

Selection strategy:

  - Filter to **full-spine subjects** (≥ 20 PLYs available). Each such
    subject typically contributes C1–C7 + T1–T12 + L1–L5 + S1 ≈ 25
    vertebrae, so 4 full-spine subjects per group yields ~100 PLYs
    with the natural anatomical mix (~28% cervical, ~48% thoracic,
    ~20% lumbar, ~4% sacral) without any per-class cap that would
    skew the distribution.
  - **Round-robin allocate** subjects to the three groups so every
    group sees the same mix of subject sizes + sources (gl-* vs
    verse-*).
  - Seeded so the selection is reproducible.

The wipe step uses the existing ``cloud_store.wipe_catalog`` with
``backup=False`` (the user has a prior backup from the previous prune;
no need to stack another).

Usage::

    python tools/build_training_samples.py

    # Skip the wipe — keep the current catalog, just stage files.
    python tools/build_training_samples.py --keep-catalog

    # Different output / source roots.
    python tools/build_training_samples.py \\
        --source E:/data/verse/exports \\
        --output E:/data/verse/spinelab_training_samples
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_DEFAULT_SOURCE = Path(r"E:/data/verse/exports")
_DEFAULT_OUTPUT = Path(r"E:/data/verse/spinelab_training_samples")
_FULL_SPINE_MIN_PLYS = 20
_TARGET_PER_GROUP = 100
_N_GROUPS = 3
_SUBJECTS_PER_GROUP = 4  # ~25 plys/subject × 4 = ~100


def _region_of(vertebra: str) -> str:
    if vertebra.startswith("C"): return "cervical"
    if vertebra.startswith("T"): return "thoracic"
    if vertebra.startswith("L"): return "lumbar"
    if vertebra.startswith("S"): return "sacral"
    return "other"


def _enumerate(source: Path) -> dict[str, list[Path]]:
    """Return ``{subject: [ply_path, ...]}`` across all splits under source."""
    subjects: dict[str, list[Path]] = defaultdict(list)
    for split_dir in source.iterdir():
        if not split_dir.is_dir():
            continue
        for ply in split_dir.glob("*.ply"):
            base = ply.name
            sub = base.split("_", 1)[0] if "_" in base else base
            subjects[sub].append(ply)
    return dict(sorted(subjects.items()))


def _select_groups(subjects: dict[str, list[Path]],
                   n_groups: int,
                   subjects_per_group: int,
                   seed: int) -> list[list[Path]]:
    """Round-robin pick ``n_groups × subjects_per_group`` full-spine
    subjects and return their PLY lists per group."""
    full_spine = {s: plys for s, plys in subjects.items()
                  if len(plys) >= _FULL_SPINE_MIN_PLYS}
    if len(full_spine) < n_groups * subjects_per_group:
        print(f"WARNING: only {len(full_spine)} full-spine subjects "
              f"available, requested {n_groups * subjects_per_group}. "
              f"Filling remainder with partial-spine subjects.",
              file=sys.stderr)

    # Mix gl-* and verse-* round-robin so each group gets a similar
    # source-prefix distribution. Within each prefix, sort by PLY
    # count descending so the biggest (most complete) subjects come
    # first.
    gl = sorted([s for s in full_spine if s.startswith("sub-gl")],
                key=lambda s: -len(full_spine[s]))
    verse = sorted([s for s in full_spine if s.startswith("sub-verse")],
                   key=lambda s: -len(full_spine[s]))
    other = sorted([s for s in full_spine
                    if not s.startswith("sub-gl")
                    and not s.startswith("sub-verse")],
                   key=lambda s: -len(full_spine[s]))
    interleaved: list[str] = []
    queues = [gl, verse, other]
    while any(queues):
        for q in queues:
            if q:
                interleaved.append(q.pop(0))

    # Allocate subjects to groups round-robin.
    take = n_groups * subjects_per_group
    chosen_subjects = interleaved[:take]
    groups: list[list[str]] = [[] for _ in range(n_groups)]
    for i, sub in enumerate(chosen_subjects):
        groups[i % n_groups].append(sub)

    # Materialise group → list of PLY paths.
    group_plys: list[list[Path]] = []
    for gsubs in groups:
        plys: list[Path] = []
        for sub in gsubs:
            plys.extend(sorted(full_spine[sub], key=lambda p: p.name))
        group_plys.append(plys)
    return group_plys


def _report_group(group_idx: int, plys: list[Path]) -> None:
    by_sub: dict[str, list[str]] = defaultdict(list)
    region_counts: dict[str, int] = defaultdict(int)
    for p in plys:
        base = p.stem
        sub, _, vert = base.partition("_")
        by_sub[sub].append(vert)
        region_counts[_region_of(vert)] += 1
    print(f"--- group_{group_idx} ({len(plys)} PLYs, "
          f"{len(by_sub)} subjects) ---")
    for sub in sorted(by_sub):
        print(f"  {sub:18s} {len(by_sub[sub]):3d}  "
              f"{','.join(sorted(by_sub[sub]))}")
    print(f"  region distribution: "
          + ", ".join(f"{r}={region_counts[r]}"
                      for r in ("cervical","thoracic","lumbar","sacral")
                      if region_counts[r] > 0))


def _wipe(no_backup: bool) -> None:
    """Wipe the library catalog. Skipped when ``--keep-catalog``."""
    from src.data.cloud_store import wipe_catalog
    print("Wiping catalog (no backup)..." if no_backup else "Wiping catalog...")
    result = wipe_catalog(backup=not no_backup)
    if result is not None:
        print(f"  backup at: {result}")
    print("  catalog wiped.\n")


def _copy_group(plys: list[Path], dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in plys:
        dst = dest_dir / src.name
        try:
            shutil.copy2(src, dst)
            n += 1
        except OSError as e:
            print(f"  copy failed for {src.name}: {e}", file=sys.stderr)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, default=_DEFAULT_SOURCE)
    ap.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    ap.add_argument("--n-groups", type=int, default=_N_GROUPS)
    ap.add_argument("--subjects-per-group", type=int,
                    default=_SUBJECTS_PER_GROUP)
    ap.add_argument("--seed", type=int, default=42,
                    help="kept for reproducibility — current selection "
                         "is deterministic on subject count, but the "
                         "flag is there for future jitter.")
    ap.add_argument("--keep-catalog", action="store_true",
                    help="Skip the catalog wipe step.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the selection without copying or wiping.")
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"ERROR: source not found: {args.source}", file=sys.stderr)
        return 2

    subjects = _enumerate(args.source)
    total_plys = sum(len(v) for v in subjects.values())
    print(f"Source: {args.source}")
    print(f"  {len(subjects)} subjects, {total_plys} PLYs total\n")

    groups = _select_groups(
        subjects,
        n_groups=args.n_groups,
        subjects_per_group=args.subjects_per_group,
        seed=args.seed,
    )
    for i, plys in enumerate(groups, start=1):
        _report_group(i, plys)
        print()

    total_selected = sum(len(g) for g in groups)
    print(f"Total selected: {total_selected} PLYs across "
          f"{args.n_groups} groups")
    print()

    if args.dry_run:
        print("DRY RUN — no wipe + no copy. Re-run without --dry-run.")
        return 0

    if not args.keep_catalog:
        _wipe(no_backup=True)

    print(f"Copying selection to {args.output} ...")
    args.output.mkdir(parents=True, exist_ok=True)
    for i, plys in enumerate(groups, start=1):
        dest = args.output / f"group_{i}"
        n = _copy_group(plys, dest)
        print(f"  group_{i}: copied {n}/{len(plys)} PLYs → {dest}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
