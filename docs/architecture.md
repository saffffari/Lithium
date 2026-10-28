# Architecture

A short tour of how Lithium is built. The code is the source of truth;
[AGENTS.md](../AGENTS.md) lists the load-bearing files.

## Overall shape

Lithium is a single-process Python 3.12 desktop app: a GLFW window, a ModernGL
OpenGL 4.1 core context, a Dear ImGui overlay and custom GLSL shaders (all
`#version 410 core`, under `shaders/`).

One `App` class in `src/main.py` owns everything visible. It runs three modes
that share the camera, the catalog, the label registry and the undo stack:

| Mode | Tab | Constant | Value |
|---|---|---|---|
| Contact Sheets | SHEETS | `MODE_CONTACT_SHEETS` | 1 |
| Light Table | LIGHT | `MODE_LIGHT_TABLE` | 2 |
| Train | TRAIN | `MODE_AUTOMATION` | 5 |

The values 1, 2 and 5 are kept from an older six-mode layout so saved settings
stay valid. The constants live in `src/core/modes.py`.

## Module layers

```
core ← data ← rendering / training ← gui ← main
```

| Package | Contents |
|---|---|
| `core/` | Camera, input, undo, selection, tools, measurement registry, modes, resource budgets, label propagation, platform glue. |
| `data/` | Loaders (PLY, LAS/LAZ, NPZ), the library catalog, the cloud store, labels, projects, the model registry, `.li` bundles, the Sandbox, the catalog lock, mesh building. |
| `rendering/` | Point and mesh renderers, overlays, gizmo, framebuffers, SSAO, post-processing. |
| `training/` | PT-v3 config generation (`config_gen.py`), the subprocess runner (`ptv3_runner.py`), the Pointcept dataset shim (`pointcept_ext/`), the inference contract. |
| `gui/` | ImGui panels, label and measurement panels, the bundle dialog, the console, theme and widgets. |
| `export/` | Screenshots, turntables, dataset export, prediction import. |
| `utils/` | Maths, file hashing, preferences. |
| `main.py` | `App`, the GLFW event loop, mode dispatch, key and mouse handling. |
| `cli.py` | Headless `render` and `spin`. |

Lower layers never import higher ones. The one known exception is `gui/` and
`main.py`: `App` draws the panels each frame, and a few panels reach back into
`src.main` for helpers. That pair is deliberate and contained. Scripts in
`tools/` import from `src` as a library and never import `App`.

## Design decisions

- **The catalog is the dataset.** Importing a cloud copies it into
  `~/.lithium/library`. Its labels live at
  `labels/<namespace>/<file_key>.npy`, where the namespace is the project id
  (`_library` outside projects, `layer:<model_id>` for Sandbox layers), made
  filesystem-safe by replacing characters such as `:` with `_`. There
  is no separate "save project" step.
- **Labels are written safely.** Every edit goes through the undo stack and is
  saved by writing a temporary file and swapping it in. Label arrays are checked against
  the cloud's point count when they are loaded. Save failures show
  in a banner, because the packaged build has no console.
- **Measurement anchors are cloud-local.** An anchor stores
  `(cloud_key, local_pos)`, never a world position, so measurements follow the
  cloud when it moves.
- **GPU uploads happen once.** Geometry goes to the GPU at load; per frame,
  only uniforms change.
- **Labels have their own buffer.** Each cloud uses four vertex buffers
  (position, colour, label, selection), so a stroke re-uploads 4 bytes per
  point instead of 32. See `src/rendering/point_cloud_renderer.py`.
- **Caches have budgets.** `src/core/resource_budget.py` sets how many
  full-resolution clouds, previews and thumbnails stay resident. The active
  cloud and the visible gallery are always kept.

## Threads and subprocesses

The main thread owns the frame loop and all OpenGL work. Background threads
load files, build previews and meshes, import and export bundles, and run the
training and inference subprocesses. The main thread polls their results each
frame and never blocks.

Worker threads never change `App` state directly. They post events to a
main-thread queue that the frame loop drains.

Training runs through `PointceptRunner` (`src/training/ptv3_runner.py`):
`Popen`, a daemon thread that reads the output, and a status object the TRAIN
tab displays. Every subprocess gets `PYTHONIOENCODING=utf-8`. The subprocesses
run in a separate conda environment with PyTorch and Pointcept; see
[Training setup](training_setup.md).

## Packaging

`lithium.spec` builds a windowed PyInstaller app that bundles `shaders/` and
`assets/`. `packaging/lithium.iss` wraps the Windows build in a per-user
installer. `.github/workflows/release.yml` builds Windows, Linux and macOS
artifacts for `v*` tags and attaches them to a draft release.
`.github/workflows/test.yml` runs the tests on Windows and Linux for every push
and pull request.

## Other docs

| What | Where |
|---|---|
| Using the app | [user_guide.md](user_guide.md) |
| Labelling a shared project | [onboarding.md](onboarding.md) |
| The `.li` format | [bundles.md](bundles.md) |
| The training environment | [training_setup.md](training_setup.md) |
| macOS | [macos.md](macos.md) |
| Conventions for contributors and coding agents | [CLAUDE.md](../CLAUDE.md), [AGENTS.md](../AGENTS.md) |
