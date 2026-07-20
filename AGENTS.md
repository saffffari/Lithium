# AGENTS.md — Orientation for AI coding agents

This file is the structured orientation any AI agent should read before
making non-trivial changes in this repository. It pairs with
`CLAUDE.md` (project conventions + tech stack) and the live codebase
maps under `_audit/`. If you're a human, start with `README.md` and
`CLAUDE.md`; this file assumes you can navigate Python + Markdown
without preamble.

> The clinical product surface (**SpineLab** — IMAGING / HOLOGRAM /
> OVERWATCH, CT ingest, PolyPose registration, clinical measurement
> analytics) was split out into its own repository. This repo is
> 3Photon only. Don't reintroduce imaging / registration / clinical
> code here.

---

## Project shape

3Photon is a single-process desktop point-cloud workstation with three
modes sharing one `App` god-object (`src/main.py`):

| Mode | Key | Purpose |
|---|---|---|
| **CONTACT_SHEETS** | 1 | Browse a per-cloud gallery |
| **LIGHT_TABLE** | 2 | Paint / refine per-point labels |
| **AUTOMATION / TRAIN** | 3 | Stage datasets + train Pointcept PT-v3 |

Mode integer values are 1 / 2 / 5 (preserved from the historical
six-mode layout so persisted settings stay valid).

---

## Sources of truth

When a question has a code-level answer, these files are authoritative.
Don't duplicate their contents in new docs; link to them.

- **`src/core/modes.py`** — the canonical mode constants. Import from
  here, never hardcode mode integers.
- **`src/data/labels.py`** — `LabelRegistry` + `DEFAULT_PALETTE`: the
  annotation taxonomy. Label trees, colors, visibility, lock.
- **`src/data/library_catalog.py`** + **`src/data/cloud_store.py`** —
  the catalog *is* the dataset. `library_catalog` owns entries /
  projects / previews / mesh-build queue; `cloud_store` is the
  write-once on-disk store (data / labels / meshes NPZ under
  `~/.3photon/library/`). Both are shared infrastructure — do not
  treat as removable.
- **`src/core/measure_registry.py`** — `Anchor(cloud_key, local_pos)`
  + the measurement registry. Anchors are cloud-local.
- **`src/training/config_gen.py`** — PT-v3 / Pointcept config
  generation (`grid_size=0.5 mm`, `batch_size=2`, the depth-assertion
  constraints). The contract with the training subprocess.
- **`src/data/model_registry.py`** — tracks trained PT-v3 checkpoints
  + their class manifests per project.

---

## Architectural conventions

### Pickers + anchors

- Picking returns a `PickResult(world_pos, cloud_key, local_pos, entry)`.
  Visual feedback uses `world_pos`; stored anchors use `cloud_key +
  local_pos`.
- Measurement anchors are `Anchor(cloud_key, local_pos)` — see
  `src/core/measure_registry.py`. They survive view changes because the
  resolver applies whichever model matrix is current at evaluation time.
- **Cloud-local is the storage frame for everything.** No mutable
  "current world position" stored on data objects; always derived by
  composing cloud-local geometry with the cloud's current model matrix.

### Background work

- Mesh builds queue through `catalog.queue_mesh_build`; results arrive
  via `poll_pending_meshes` in the main loop.
- PT-v3 training/inference runs as a subprocess (Pointcept, conda env
  `3photon-ptv3`) following the PointceptRunner template: Popen +
  daemon reader thread + status state machine + `PYTHONIOENCODING=utf-8`.
- **Subprocess reader threads must NOT mutate `App.*` directly** — they
  push events into a main-thread-drained deque (the Pattern-A queue).
  GL + ImGui + `App` state touches happen only on the main thread.

### Layering

`core ← data ← rendering / training ← gui ← main`. No back-edges, no
cycles. The one acknowledged exception is the contained `gui → main`
pair (panels are an overlay on the App god-object).

---

## When to update this file

- New mode or major subsystem → update the project-shape section.
- New source-of-truth file → add it to the list.
- Architectural convention changes (picker contract, frame discipline,
  background-work pattern) → update the conventions section.

This file should stay short. If a section grows past a screen, extract
it into a dedicated file and link to it.
