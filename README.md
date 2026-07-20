# 3Photon

**Desktop point cloud workstation. Browse, annotate, and train segmentation models in one app.**

A native Python desktop app for offline processing of point cloud data. GPU-accelerated, workstation-grade, no cloud upload required.

3Photon builds labeled point-cloud datasets and trains PT-v3 segmentation models, all from a single instrument-grade interface.

## Features

- **Three-mode interface** (`1`–`3`): CONTACT_SHEETS / LIGHT_TABLE / AUTOMATION
  - **CONTACT_SHEETS** — browse a per-cloud gallery; jump into any cloud to inspect it
  - **LIGHT_TABLE** — paint and refine per-point labels with the full selection toolkit
  - **AUTOMATION / TRAIN** — stage datasets and train Pointcept PT-v3 models
- **GPU rendering** at interactive frame rates for 1M+ point clouds (ModernGL + custom GLSL)
- **Selection tools**: pick, box, lasso, brush, polygon, curve, measure (line / angle / landmark)
- **Hierarchical labels** with color, visibility, lock — atomic on-disk persistence per stroke
- **Undo/redo** for every label operation, snapshots into recovery sidecars
- **4D time-series** with timeline scrubber and KD-tree label propagation
- **In-app PT-v3 training** via Pointcept subprocess (separate `3photon-ptv3` conda env)
- **Embedded CLI** for scripting bulk operations

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| PLY | .ply | Mesh-derived and generic point clouds |
| LAS / LAZ | .las / .laz | LiDAR and aerial surveys |
| NPZ | .npz | Catalog-internal storage + PT-v3 predictions |

## Quick Start

```bash
git clone <repo-url> 3photon
cd 3photon
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows:
source .venv/Scripts/activate
pip install -r requirements.txt
python -m src.main <path_to_ply_or_las_or_directory>
```

Runs on Linux and Windows. Requires Python 3.12.

For PT-v3 training, install Pointcept in a separate conda env — see [docs/training_setup.md](docs/training_setup.md).

## Dependencies

Core stack: `moderngl`, `glfw`, `imgui[glfw]`, `laspy`, `plyfile`, `numpy`, `scipy`, `Pillow`, `h5py`, `trimesh`, `pyinstaller`.

- **Rendering:** ModernGL (OpenGL 4.3+) with custom GLSL shaders
- **Window:** GLFW
- **GUI:** Dear ImGui (`imgui[glfw]`)
- **Data:** laspy (LAS/LAZ), plyfile (PLY), numpy
- **Packaging:** PyInstaller (windowed standalone executable)

## Documentation

- [docs/user_guide.md](docs/user_guide.md) — workflows, modes, shortcuts, CLI reference
- [docs/architecture.md](docs/architecture.md) — high-level module overview (canonical map lives in [_audit/](_audit/))
- [docs/training_setup.md](docs/training_setup.md) — Pointcept + PT-v3 env setup
- [docs/runbooks/](docs/runbooks/) — operational recipes (multi-agent audit, etc.)
- [docs/adr/](docs/adr/) — architectural decision records
- [_audit/](_audit/) — cartographer-produced codebase maps + bug audit (`ARCHITECTURE.md`, `FLOWS.md`, `SPINE.md`, `ORIENTATION.md`, `REPORT.md`)
- [AGENTS.md](AGENTS.md) — orientation for AI coding agents
- [CLAUDE.md](CLAUDE.md) — project conventions + tech stack

## Performance

Benchmarks on 1M-point clouds (RTX 4090):

| Operation | Time |
|-----------|------|
| Point pick | 56ms |
| Box selection | 54ms |
| Lasso selection | 74ms |
| Brush selection | 11ms |
| Apply label (500K) | 3.6ms |
| Undo / Redo (500K) | 1.1ms |
| 4-VBO per-stroke label upload | ~4 MB (vs 32 MB monolithic) |

## Philosophy

Research tools have historically prioritized function over form. 3Photon rejects that trade-off. The interface matters — it signals that the engineering underneath is modern, it lowers the barrier for new users, and it makes hours-long annotation sessions tolerable. Visual language inspired by Teenage Engineering's OP-1: ultra-dark neutral greys, warm accent colors, precise layout, instrument-grade density.

## Project Status

Active development. Research-mode software.

Shipping today: three-mode workstation (CONTACT_SHEETS / LIGHT_TABLE / AUTOMATION); end-to-end label authoring with atomic per-stroke persistence; PT-v3 model training via Pointcept; embedded CLI for bulk operations.

## License

TBD — pre-commercialization.
