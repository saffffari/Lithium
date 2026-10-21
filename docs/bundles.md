# Project bundles (`.li`)

A `.li` file is one project, packed to move between machines: any
combination of its point clouds, its label layer and its trained model
weights, plus a manifest with the project's name and ontology.

## In the app

**File > Export Project (.li)...** — pick the project, tick what to
include, choose a file. The dialog shows counts and sizes next to each
checkbox and a running total.

| Checkbox | What goes in |
|---|---|
| Point clouds | every cloud in the project, stripped to `coord` + `color` + `normal`. Colour is always included. Every other per-point scalar is dropped — the catalog npz files carry label-bearing layers (`labels_*`, `tips*`, `label_seed_*`) and a bundle exported *without* labels must not leak them through the scalar dropdown. |
| Labels | the project's own label layer, one `int32` file per labelled cloud |
| Model weights | `model_best.pth` (+ `config.py`) for every completed model in the project's registry. Leaving this off also drops the project's `active_model_id`, so INFER on the other machine never points at a checkpoint it doesn't have. |

**File > Import Project (.li)...** — pick the file; it imports whatever
it contains, no further questions:

- a **new project** named after the bundle (de-duplicated with ` (2)`),
  or
- a **merge** into the project the bundle was exported from when that
  project still exists in this catalog — that is a reader's labels
  coming home. Labels never overwrite ones the target already has.

A progress card shows the ring filling; Cancel leaves nothing behind
(the file is written as `.part` and renamed at the end).

## Blinded reader kit

Export with **Point clouds only**. The reader gets the geometry, the
ontology and an empty label layer; no model, no label scalars. When
they are done they export **Labels only** (a few MB) and send that
back; importing it merges their layer into the source project.

## From the shell

```
python tools/bundle_cli.py export "<project name or proj:id>" --out kit.li --clouds
python tools/bundle_cli.py inspect kit.li
python tools/bundle_cli.py import kit.li                 # close Lithium first
python tools/bundle_cli.py import labels.li --into proj:xxxxxxxx
```

`LITHIUM_LIBRARY_DIR` redirects the catalog for both the app and the CLI.

## Layout

```
manifest.json                 format, version, project {name, source_id, ontology, settings},
                              cloud_order, clouds[], labels[], models[], scalars_dropped
clouds/<file_key>.npz         coord, color, scalar_normal, _meta_json
labels/<file_key>.npy         int32 per-point ids, indexed by the project's ontology
models/<model_id>/model_best.pth
models/<model_id>/config.py
```

File keys are preserved across machines: label files are keyed by
them, so a labels-only bundle lands on clouds imported earlier from a
clouds-only bundle of the same project. Every item carries a SHA-256
and the importer refuses a bundle whose contents don't match.

Code: `src/data/bundle.py` (format, no GUI), `src/gui/bundle_dialog.py`
(modal + progress card), `tests/test_bundle.py`.
