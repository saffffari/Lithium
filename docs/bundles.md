# Project bundles (`.li`)

A `.li` file packs one project so you can move it to another computer. It can
hold any mix of the project's point clouds, its labels and its trained models,
plus a manifest with the project's name and label set.

## Export

Choose **File > Export Project (.li)...**, pick the project, tick what to
include, then click **Export...** and choose a file name. The dialog shows the
count and size of each part and a running total.

| Checkbox | What goes in |
|---|---|
| Point clouds | Every cloud in the project, reduced to coordinates, colour and normals. All other per-point data is left out, so a bundle without labels cannot leak them through a stored scalar layer. |
| Labels | The project's labels: one integer array per labelled cloud. |
| Model weights | `model_best.pth`, plus `config.py` when the run has one, for every model in the project that has a checkpoint. Leave it unticked and the bundle also drops the project's active-model setting, so the other computer never points at a checkpoint it does not have. |

An export is written to `<name>.li.part` and renamed when it is complete, so a
cancelled or failed export leaves no file behind.

## Import

Choose **File > Import Project (.li)...** and pick the file. Lithium imports
everything the bundle contains and opens the project:

- If the project the bundle was exported from exists in your catalog (for
  example, you are re-importing your own export), the bundle **merges** into
  it.
- Otherwise Lithium creates a **new project** with the bundle's name, adding
  ` (2)`, ` (3)` and so on if the name is taken.

Imported labels **never overwrite** labels the project already has for a
cloud. The progress card reports how many label files were written and how
many were skipped. Labels for clouds that are not in your catalog are skipped
too.

Every cloud, label and model file in a bundle carries a SHA-256 checksum. Lithium refuses a bundle
whose contents do not match.

## Hand a project to a labeller

1. Export with **Point clouds** only. The labeller gets the clouds and the
   label set, but no labels and no models.
2. The labeller imports it, labels, and exports with **Labels** only. That
   file is usually a few MB. [Annotator onboarding](onboarding.md) walks them
   through it.
3. You import their file. Their labels arrive as a new project, for example
   `My project (2)`, on the same clouds. Their copy of the project has its own
   id, so it does not merge into yours, and labels from different people stay
   apart.

Clouds keep the same internal key on every computer. That is what lets a
labels-only bundle land on clouds that arrived earlier in a clouds-only
bundle.

To merge returned labels into a specific project instead, use the command
line with `--into` (below). It still never overwrites existing labels unless
you add `--overwrite-labels`.

## From the command line

`tools/bundle_cli.py` runs the same code without the window. Run it with the
Lithium environment's Python from the repo root.

```bash
python tools/bundle_cli.py export "<project name or proj:id>" --out kit.li --clouds
python tools/bundle_cli.py export "<project>" --out full.li --clouds --labels --models
python tools/bundle_cli.py inspect kit.li
python tools/bundle_cli.py import kit.li [--name "New name"]
python tools/bundle_cli.py import labels.li --into "<project name or proj:id>" [--overwrite-labels]
```

- Pick at least one of `--clouds`, `--labels` and `--models` when exporting.
- A project name can be partial if it matches only one project.
- **Close Lithium before importing.** The catalog allows one user at a time.
- `LITHIUM_LIBRARY_DIR` points both the app and the tool at a different
  catalog.
- Without `--into`, the command-line import always creates a new project.

## File layout

A bundle is an uncompressed zip:

```
manifest.json                   format, version, project (name, source id, label set,
                                settings), cloud order, and one entry per item
clouds/<file_key>.npz           coord, color, scalar_normal, _meta_json (and pose_to_world if present)
labels/<file_key>.npy           int32 label id per point
models/<model_id>/model_best.pth
models/<model_id>/config.py
```

The code lives in `src/data/bundle.py` (the format, no GUI),
`src/gui/bundle_dialog.py` (the dialog and progress card) and
`tests/test_bundle.py`.
