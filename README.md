# Lithium

**Desktop point cloud workstation. Browse, annotate, and train segmentation models in one app.**

Lithium is a native desktop app for offline point cloud work — GPU-accelerated,
workstation-grade, no cloud upload, no account. Import clouds, paint per-point
labels with a full selection toolkit, and train PT-v3 segmentation models on
your own data from inside the app.

*The first element of the [2Photon Elements](https://2photon.io/elements)
portfolio. Free.*

> Formerly known as 3Photon. The 3 lives on: three tabs, three axes,
> element three.

## Features

- **Three-tab workflow** (`1`–`3`), left to right as the pipeline:
  - **SHEETS** — import clouds, organize projects, define the label ontology,
    browse a per-cloud gallery
  - **LIGHT TABLE** — paint and refine per-point labels: pick, box, lasso,
    polygon, brush, curve; measure tools; one-click **INFER** on the current
    cloud with a live per-cloud progress constellation
  - **TRAIN** — stage datasets, launch PT-v3 training, manage trained models,
    run batch inference
- **GPU rendering** at interactive frame rates for 1M+ point clouds
  (ModernGL + custom GLSL); per-monitor DPI aware
- **Projects with independent label sets** — the same cloud can live in many
  projects with different ontologies; painting in one never touches another
- **The catalog is the dataset** — every stroke persists atomically; there is
  no "save" button to forget
- **Undo/redo for every label operation**, including applied inference
- **Sandbox + cross-project models** — send any cloud to the always-present
  SANDBOX and run any registered model on it; results live as cloud-level
  layers. In a project, any model whose classes match the ontology is offered.
- **In-app PT-v3 training** ([Pointcept](https://github.com/Pointcept/Pointcept))
  with model registry, frozen class maps, fine-tune lineage, and
  crash-resilient training
- **4D time-series** with timeline scrubber and label propagation
- **Headless CLI** for scripted rendering and bulk operations

## Supported formats

| Format | Extension | Typical source |
|--------|-----------|----------------|
| PLY | `.ply` | mesh-derived and generic point clouds |
| LAS / LAZ | `.las` / `.laz` | LiDAR and aerial surveys |
| NPZ | `.npz` | catalog-internal + PT-v3 predictions |

## Install

**Installers** (recommended): grab the latest release for your platform from
[Releases](../../releases) — Windows installer, Linux binary, macOS app (unsigned).

**From source** (Linux / Windows / macOS, Python 3.12):

```bash
git clone <repo-url> lithium
cd lithium
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # or requirements-lock.txt for exact versions
python -m src.main               # optionally: <path_to_ply_las_or_directory>
```

Or use the launchers: `./run.sh` (Linux) / `run.bat` (Windows).

**Training** is optional and needs an NVIDIA GPU + a separate Python env with
PyTorch/CUDA — see [docs/training_setup.md](docs/training_setup.md). The
viewer/annotator never imports torch.

## Documentation

- [docs/user_guide.md](docs/user_guide.md) — workflows, modes, shortcuts, CLI
- [docs/architecture.md](docs/architecture.md) — module overview
- [docs/training_setup.md](docs/training_setup.md) — PT-v3 environment recipe
  (verified Linux + Windows paths)
- [docs/design-1.1.md](docs/design-1.1.md) · [docs/perf-1.1.md](docs/perf-1.1.md)
  — design notes for this release and the massive-cloud renderer roadmap

## Performance

Benchmarks on 1M-point clouds (RTX 4090): pick 56 ms · box select 54 ms ·
lasso 74 ms · brush 11 ms · label apply (500 K pts) 3.6 ms · undo 1.1 ms.
Per-stroke GPU upload is 4 MB, not 32 MB, via a split-buffer layout.

The out-of-core renderer in [prototypes/wgpu](prototypes/wgpu) (the next
major milestone) streams **11+ billion point** collections at interactive
rates on a single GPU; see [docs/perf-1.1.md](docs/perf-1.1.md).

## Platform notes

- **Linux** — primary development platform.
- **Windows** — supported; the app was originally Windows-native.
- **macOS** — runs from source on OpenGL 4.1 (Apple Silicon and Intel); see
  [docs/macos.md](docs/macos.md). View/annotate only — training and INFER need
  CUDA. An unsigned `.app` is built by CI for each release.
- Unsigned binaries: Windows SmartScreen and browser warnings are expected
  for now — the code is right here to audit.

## Philosophy

Research tools have historically prioritized function over form. Lithium
rejects that trade-off. The interface matters — it lowers the barrier for new
users and makes hours-long annotation sessions tolerable. Visual language
inspired by Teenage Engineering's OP-1: ultra-dark neutral greys, warm accent
colors, instrument-grade density.

## License

[MIT](LICENSE)
