#!/usr/bin/env python
"""Re-stage a project's labelled clouds as a PTv3 training dataset, no GUI.

Lets us trigger the export pipeline from the CLI — necessary for
overnight automation where the GUI isn't running. Uses the catalog
caches (positions, colors, labels) so it doesn't need ModernGL or
window setup.

Usage:
    python tools/restage_project.py \\
        --project-id proj:5a770138 \\
        --output dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.labels import LabelRegistry
from src.data.cloud_store import load_cloud_data, load_cloud_labels
from src.data.point_cloud import PointCloudData
from src.data import library_catalog
from src.export.dataset_export import export_dataset


for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_projects() -> dict:
    p = Path(library_catalog.library_dir()) / "projects.json"
    with open(p) as f:
        return json.load(f)


def build_registry(project: dict) -> LabelRegistry:
    """Reconstruct the project's LabelRegistry from its ontology_data."""
    reg = LabelRegistry()
    for entry in project.get("ontology_data", {}).get("labels", []):
        if int(entry["id"]) == 0:
            continue
        reg.add_label_at(int(entry["id"]),
                         entry["name"],
                         tuple(entry["color"]))
    return reg


def hydrate_cloud(file_key: str) -> PointCloudData | None:
    """Load a cloud's data + labels from the catalog, return PointCloudData
    suitable for export_dataset, or None if the catalog data is missing."""
    cloud, _ = load_cloud_data(file_key)
    if cloud is None:
        return None
    stored = load_cloud_labels(file_key)
    if stored is not None and len(stored) == cloud.point_count:
        cloud.labels[:] = stored
    return cloud


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--output", required=True, help="output dataset dir")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    args = ap.parse_args()

    projects = load_projects()
    if args.project_id not in projects:
        raise KeyError(f"project {args.project_id!r} not in projects.json")
    project = projects[args.project_id]
    # v2 label layout: read this project's own label namespace.
    from src.data import cloud_store
    cloud_store.set_active_label_namespace(args.project_id)
    registry = build_registry(project)
    print(f"Project: {project['name']}")
    print(f"Registry labels (non-zero): "
          f"{[(l.id, l.name) for l in registry.all_labels() if l.id != 0]}")

    file_keys = project.get("file_keys", [])
    print(f"File keys in project: {len(file_keys)}")

    clouds = []
    skipped_no_data = 0
    skipped_unlabelled = 0
    for fk in file_keys:
        cloud = hydrate_cloud(fk)
        if cloud is None:
            skipped_no_data += 1
            continue
        # Skip clouds with zero labelled points — export_dataset would
        # write them as all-IGNORE_INDEX before our patch and as all-
        # class-0 (Unlabeled) after it, which would be a useless training
        # signal either way.
        if cloud.labels is None or not (cloud.labels != 0).any():
            skipped_unlabelled += 1
            continue
        clouds.append(cloud)

    print(f"Skipped (no catalog data): {skipped_no_data}")
    print(f"Skipped (no labels):       {skipped_unlabelled}")
    print(f"Exporting:                 {len(clouds)} clouds")
    if not clouds:
        print("Nothing to export.")
        return

    summary = export_dataset(
        clouds, registry, args.output, format="ptv3",
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(f"Done: {summary['train']}t / {summary['val']}v / {summary['test']}ts "
          f"-> {args.output}")
    cj = Path(args.output) / "classes.json"
    if cj.exists():
        with open(cj) as f:
            data = json.load(f)
        print(f"classes.json: num_classes={data.get('num_classes')}, "
              f"names={data.get('class_names')}")


if __name__ == "__main__":
    main()
