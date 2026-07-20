# 3Photon 1.1 — Design

Rebuild of 1.0 (imported at commit `da3d6e4`) with the accumulated cruft cut,
the audit bug backlog fixed, and the project/label/model data model reworked.
This document is the working spec for the 1.1 changes; `docs/architecture.md`
still describes the carried-over structure.

## 1. Per-project label sets (the core change)

**1.0 flaw:** one `labels/<file_key>.npy` per cloud, shared by every project
that contains the cloud. Adding a cloud to a second project with a different
ontology *destructively remaps* the single file
(`cloud_store.migrate_cloud_labels_to_project`), corrupting the labels as seen
from the first project.

**1.1 model:** the project is the label namespace.

```
~/.3photon/library/
  labels/<project_id>/<file_key>.npy          # per-project full-res labels
  preview_labels/<project_id>/<file_key>.npy  # per-project preview labels
```

- A cloud in N projects has up to N independent label arrays. Zero-copy until
  first paint in that project (missing file == all-unlabeled).
- `cloud_store` label read/write APIs take a `namespace` (project id).
  The active namespace comes from `app.active_view`.
- Non-project views (session, folder, smart views) use the reserved namespace
  `_library`, which also receives legacy 1.0 label files on migration.
- `migrate_cloud_labels_to_project` becomes **copy-and-remap into the
  destination namespace** — the source project's labels are never touched.
- One-time migration on first 1.1 launch: `labels/*.npy` (flat, 1.0 layout) is
  moved to `labels/_library/` and *copied* into `labels/<project_id>/` for
  every project containing that cloud (remapped through the project ontology
  if names differ). Same for `preview_labels/`. The flat files are left
  in place until migration verifies, then removed; a marker file
  `labels/.v2-migrated` gates re-runs.

## 2. Project duplication for class-set experiments

`duplicate_project` in 1.1 copies: ontology, settings, **file_keys**, label
arrays (into the new namespace), and model registry entries (zero-copy
checkpoint refs, as 1.0's GUI duplicate did). After duplication the user edits
the new project's ontology freely — delete a class (points drop to 0 on that
project's copies only), merge classes, add new ones — then paints the deltas
and trains. This is the "train a variant model with more/fewer classes"
workflow, and it never perturbs the source project.

## 3. Models as first-class citizens

Registry stays per-project (`models/<project_id>.json`) with these additions:

- **`class_map` frozen at launch** — `{label_id: name}` recorded on the
  `TrainedModel` when training starts, written as `classes.json` next to the
  checkpoint AND stored in the registry entry. Inference resolves classes from
  the model entry first; the fragile cwd-relative fallback chain is deleted.
- **Active model per project** — the inference model pick persists in project
  settings (1.0 kept it in a transient dict).
- **Delete with artifacts** — deleting a model offers to remove its
  `work_dir` (with size shown); registry-only delete remains the default.
- **Lineage** — `parent_model_id` shown in the model list (fine-tune chains).

## 4. Light Table INFER button

Bottom of the LIGHT_TABLE sidebar: a full-width INFER button that runs the
active project's model on the current cloud, through the same
checkpoint/class-map resolution as batch inference, results applied through
`apply_label` (undoable, persisted to the active namespace).

While running it renders the **cloud constellation loader**: the widget draws
the actual preview point cloud of the current cloud (downsampled to ~600
points, cached per cloud) as a mini starfield in the button footprint;
a scan plane sweeps through it as the subprocess progresses, points ignite
from dim grey to their predicted class colors behind the sweep, with an
orbiting comet ring during the indeterminate startup phase and a
completion pulse that snaps all points to final colors. Failure state:
points collapse to red ember. All ImGui draw-list, tokens-based colors,
~0 cost when idle.

## 5. GUI pass (imports → labels → train, left to right)

- **CONTACT_SHEETS** = intake + project shaping: import, projects, ontology
  picker (moves up next to PROJECTS), cloud membership, per-project labels
  panel. Batch inference moves out (to Train tab; Light Table gets the
  per-cloud INFER button).
- **LIGHT_TABLE** = painting: tools, labels, brush falloff, display controls,
  INFER button at the bottom.
- **TRAIN** = dataset + models: selection, label filter, staging (collapsed
  by default; direct export stays primary), export, train controls, model
  list, batch inference over project clouds, log.
- Selection sets separated: `_train_selected` (training) vs gallery
  multi-select (inference) no longer share state.
- Switching projects clears stale selection sets.
- Pre-launch training summary line: clouds, points, class histogram.

## 6. Bug + performance pass

Fix list derived from `_audit/REPORT.md` (1.0) findings that still apply,
re-verified against this tree — tracked in `docs/bugfix-1.1.md` with
verdicts. Performance targets: startup catalog scan, gallery redraw budget,
per-frame Python allocations in the frame loop, load-path stalls.
