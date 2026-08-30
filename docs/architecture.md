# Lithium Architecture

This file is a brief overview. The **canonical, citation-backed architecture map** lives in `_audit/`, regenerated on demand by the `cartographer` subagent:

- **[_audit/ORIENTATION.md](../_audit/ORIENTATION.md)** — mental model + where-to-make-common-changes (the doc to read first on returning to this code after months)
- **[_audit/ARCHITECTURE.md](../_audit/ARCHITECTURE.md)** — module table with file:line citations + mermaid dep graph
- **[_audit/SPINE.md](../_audit/SPINE.md)** — the load-bearing files
- **[_audit/FLOWS.md](../_audit/FLOWS.md)** — representative user actions traced end-to-end through the code

Refresh by invoking `cartographer` (see `.claude/agents/cartographer.md`).

---

## High-level shape

Lithium is a single-process Python desktop app: GLFW window, ModernGL OpenGL 4.3 context, Dear ImGui overlay, custom GLSL shaders. The entire visible application is one `App` class in `src/main.py` owning a 3-mode view machine. Each mode is a different render path with shared camera, catalog, label registry, and undo stack. Heavy work runs as background threads (preview build, mesh Poisson, training, inference); the main thread polls futures each frame and never blocks.

The three modes are **CONTACT_SHEETS** (gallery of clouds), **LIGHT_TABLE** (single-cloud inspection + labeling + measurement), and **AUTOMATION** (embedded CLI for batch operations + PT-v3 training).

## Module layering

```
core ← data ← rendering / training ← gui ← main
```

- `core/` — camera, input, undo, tools, selection, modes, measure registry
- `data/` — loaders (PLY/LAS/LAZ/NPZ), catalog, cloud_store, library_catalog, labels, project, model_registry, mesh_builder
- `rendering/` — point/mesh renderers, overlays, gizmo, FBO, SSAO, post-process (+ GLSL shaders under `shaders/`)
- `training/` — PT-v3 launch (config_gen, ptv3_runner)
- `gui/` — ImGui panels, label/measure panels, scale, tokens, theme, OP-1 widgets, automation
- `export/` — image and spin render export
- `utils/` — math, file hashing
- `main.py` — App class, GLFW event loop, mode dispatch
- `cli.py` — headless render/spin CLI

No back-edges. No cycles. Mode constants live in `src/core/modes.py` (imported by panels, widgets, and automation so they don't pull in the `App` god-object).

The one acknowledged exception: `gui/` and `main.py` form a tight bidirectional pair. The GUI panels call back into `App` for state, and `App` drives the panels each frame. This `gui → main` cycle is intentional and contained — the GUI layer is an overlay on the App god-object, not an independent module.

## Key design decisions

The decisions that have proven durable are documented in [_audit/ORIENTATION.md](../_audit/ORIENTATION.md) ("What's surprising or non-obvious"). Highlights:

- **The catalog *is* the dataset.** Once a cloud is imported, its labels live under `~/.lithium/library/labels/<namespace>/<file_key>.npy` forever (namespace = project id, or `_library` outside projects — see [design-1.1.md](design-1.1.md)). There is no separate "save project" step.
- **Anchors are cloud-local.** `(cloud_key, local_pos)` — never world coords. Measurements survive model-matrix changes because the resolver applies the current model matrix at eval time.
- **GPU per-stroke label upload is 4 MB, not 32 MB**, via the 4-VBO split in `src/rendering/point_cloud_renderer.py`.
- **GPU uploads happen once.** Cloud geometry is uploaded to VBOs at load time; only uniforms change per frame.
- **All subprocesses go through the PointceptRunner template** — Popen + daemon reader thread + status state machine + `PYTHONIOENCODING=utf-8`. Reader threads push events into a main-thread deque; they never mutate `App.*` directly.

## Threading model

Background threads load files, build previews/meshes, and run the PT-v3 training/inference subprocesses. The main thread owns the GLFW frame loop and all GPU work, polling futures each frame. Nothing that touches GL or `App.*` state runs off the main thread; subprocess reader threads communicate exclusively through a main-thread-drained deque.

## Subprocess venvs

- conda env `lithium-ptv3` — Pointcept PT-v3 training

See [docs/training_setup.md](training_setup.md) for the PT-v3 env recipe.

## Where docs live

| What | Where |
|---|---|
| Project conventions + tech stack | [CLAUDE.md](../CLAUDE.md) |
| AI agent orientation | [AGENTS.md](../AGENTS.md) |
| User-facing workflows | [user_guide.md](user_guide.md) |
| Training env setup | [training_setup.md](training_setup.md) |
| Operational runbooks | [runbooks/](runbooks/) |
| Codebase maps (live) | [_audit/](../_audit/) |
| ADRs | [adr/](adr/) |
