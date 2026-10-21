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

## Install and launch

### Windows installer

1. Download `LithiumSetup-<version>.exe` from [Releases](https://github.com/saffffari/Lithium/releases), or use the installer supplied in your review kit.
2. Run it, choose the installation location, and optionally create a desktop shortcut. The installer defaults to your user account under `%LOCALAPPDATA%\Programs\Lithium` and does not require administrator rights.
3. Launch **Lithium** from Start or the shortcut. A supplied portable `Lithium.exe` can also run directly.
4. Install a current GPU driver if the window reports an OpenGL error. The viewer needs OpenGL 4.1 or newer. Python, WSL, Hyper-V and CUDA are not needed for the packaged viewer.
5. To update, close Lithium and run the new installer. Uninstall through Windows **Settings > Apps**. Your catalog is stored separately in your user profile; back it up before replacing or removing it.

The Windows builds are unsigned. Check the source and the supplied checksums before allowing an unfamiliar binary through SmartScreen.

### Run from source

Use Python 3.12. In PowerShell:

```powershell
git clone https://github.com/saffffari/Lithium.git
cd Lithium
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m src.main
```

On Linux, install your distribution's C++ compiler and OpenGL/X11 runtime libraries first if pip needs to build pyimgui (for example `g++ libgl1 libxinerama1 libxcursor1 libxi6 libxrandr2` on Ubuntu). On Linux/macOS, create the same environment with `python3 -m venv .venv`, then use `.venv/bin/python` in place of the Windows executable. `run.bat` and `run.sh` are alternative launchers after dependencies are installed.

Mesh reconstruction is optional and additionally needs `open3d` in the viewer environment. Point-only annotation and `.li` project import do not need it. The packaged review build's QA report states whether mesh reconstruction is included.

For a source launch with a file or directory:

```powershell
.\.venv\Scripts\python.exe -m src.main "D:\data\cloud.npz"
.\.venv\Scripts\python.exe -m src.main "D:\data\clouds"
```

## First session / Kevin's review kit

1. Open **File > Import Project (.li)** and select a supplied package. Wait for the import to complete, then close the progress card.
2. Select the imported project in **SHEETS**. Double-click a cloud to open **LIGHT TABLE**.
3. Choose a label in **LABELS**, then choose a paint tool in the viewport toolbar. Tools paint directly; there is no second Apply/Enter step.
4. Use right mouse drag to orbit, middle mouse drag to pan, the wheel to zoom, and **F** to focus. **B/V** or Left/Right visit the previous/next cloud.
5. Use **Ctrl+Z** to undo and **Ctrl+Shift+Z** to redo. Completed edits save immediately.
6. At the end, use **File > Export Project (.li)** to make a portable copy for backup or return.

The recovered Spitfire kit has a 54-cloud starter (50 VerSe plus C1/C2) and a 175-cloud review package. Each cloud has 64,000 points and existing labels. These are selected subsets of the 714-cloud source project. They contain no trained model. Importing a package does not upload it anywhere.

## Projects and portable packages

**SHEETS** manages the cloud library, project membership, label ontology and gallery. Files/folders can be imported from File menus or by drag and drop. The catalog caches the geometry, so later sessions can work without the original file. Keep each project's label names and meanings consistent.

A cloud may belong to several projects. Each project has independent labels and ontology; edits in one project do not replace another project's labels. Gallery context menus provide project membership, rename and Sandbox actions. Ctrl-click toggles gallery/list selection; Shift-click extends a range for batch operations.

**File > Export Project (.li)** lets you choose clouds, labels and models, with size estimates. To transfer a complete review project, include both clouds and labels. A labels-only package needs the matching cloud geometry in the destination catalog. An import creates a project, or merges with its original project when that project is already present. Existing labels are preserved where the importer reports them as skipped; review the import summary.

`.li` is a checksummed project bundle. It preserves cloud membership, ontology and selected model records. It is different from the older automation `save-project` session file. Use `.li` for moving work between computers.

## Annotation tools and labels

Choose an active label first. A tool writes that label as you finish its gesture; Ctrl removes only points carrying the active label. Hidden and locked labels are protected by the painting filters.

| Tool | Use |
|---|---|
| Pick (`P`) | Click a point to paint it. |
| Box (toolbar) | Drag a rectangle around points. |
| Lasso (`O`) | Drag a freehand outline. |
| Polygon (toolbar) | Place polygon vertices and finish the outline using the tool's completion gesture. |
| Brush (`K`) | Drag a spherical brush over the cloud. Ctrl+wheel changes radius; distance falloff limits its effective interior. |
| Curve (`U`) | Draw a curve to paint nearby points. |
| Line / angle measurement (toolbar) | Place point anchors to measure distance or angle; open the measurement panel with `M`. |

**LABELS:** click a row to activate it; `+ ADD` creates a label. Rename and color controls edit the ontology; eye/visibility controls hide points and lock controls prevent repainting. Deleting a populated label asks for confirmation. Label trees support nested classes. **G** cycles active labels, **H** hides/shows the active label, and **C** shows all labels in Light Table. **Alt+click** samples the label under the pointer. The labels list shows up to about six full-sized rows by default; scroll within it for larger label sets. Its handle pins a custom height and double-click resets it.

Measurements use cloud-local anchors, so they follow the cloud's transform. Read the displayed units; numeric distances depend on the source coordinate units. Measurements are inspection aids and are not clinical validation.

## Camera, display and clipping

| Control | Operation |
|---|---|
| Right drag / right hold | Orbit / open the radial camera menu. |
| Middle drag / wheel | Pan / zoom toward the pointer. |
| Double-click / Space | Place the 3D cursor and focus around it. |
| `F`, `T`, `R`, `I` | Focus selected cloud/cursor; top, right, isometric views. |
| Camera cube | Click a face for a preset; double-click for the opposite side; right-click returns to isometric. |
| Projection / viewport icons | Choose perspective/orthographic and one to four views. Planar views use orthographic projection. |
| Point Size / middle button + wheel | Change point sprite size. |
| FOCUS | Depth-of-field effect; zero disables it. |
| DEPTH | Depth brightness falloff. |
| SOFT | Saturation. |
| COLOR | Brightness. |
| POINTS / MESH / BOTH | Choose point and/or reconstructed mesh display. Mesh generation needs the optional Open3D dependency. |
| CLIP R/S/A | Adjust the low/high clipping limits per axis; toggle an axis with its button. |
| `Shift+B`, `Shift+G` | Toggle bounds and ground grid. |
| `L` | Toggle raw point colors / label colors. |

Tone controls support double-click/right-click reset. Display settings persist across launches. If colors look washed out, reset COLOR, SOFT and DEPTH and compare exposure/preferences on both machines; Windows HDR, monitor profiles and saved settings can also change perceived brightness. The scene uses an ACES display pass; sidebar swatches and rendered cloud colors can differ. There is no verified Linux/Windows color-match claim for the current review build.

The cloud-list resize handle can pin its height; double-click returns to filling the space left by the controls. Drag the sidebar's right edge to change its width. `Ctrl/Cmd` + `+`, `-`, `0` changes or resets UI scale. **Shift+H** toggles the full interface; **?** opens keyboard help. **Esc** cancels an active operation or returns toward Sheets/exit, depending on context.

## Saving, autosave and crash recovery

Completed paint gestures, undo and redo atomically save the full-resolution label array. During a long brush drag, a recovery snapshot is written approximately every **15 seconds**, leaving the last completed stroke intact. If a newer, compatible interrupted-stroke snapshot exists when its cloud is encountered, the recovery dialog offers **Restore snapshot** or **Keep last committed labels**. Restoring backs up the old committed labels first. A verified clean save removes the snapshot. Work since the last mid-stroke snapshot may be lost after a crash.

Automatic timestamped label backups run approximately every **10 minutes** for dirty clouds. **File > Save Points** also creates a manual label recovery point. A screenshot is not a label backup. Export `.li` packages and back up the whole catalog for a portable, independent copy.

Default Windows storage is `%USERPROFILE%\.lithium`; Linux/macOS use `~/.lithium`. `prefs.json` holds viewer/training paths. The `library` directory holds the index, projects, `data` geometry, `labels/<project namespace>` arrays, previews, models and backups. Keep these together when backing up. A project namespace prevents cross-project edits from mixing.

To use a different library, set `LITHIUM_LIBRARY_DIR` before launching:

```powershell
$env:LITHIUM_LIBRARY_DIR = 'D:\LithiumLibrary'
& "$env:LOCALAPPDATA\Programs\Lithium\Lithium.exe"
```

Only one app instance should own a catalog; startup checks its lock. For recovery from a historical timestamped backup, close Lithium, back up the current catalog, and restore the matching full-length label array into the same project's namespace. Do not substitute preview arrays or change point order.

## Models, inference and training

The viewer/annotator is torch-free. **Importing a model** and **running it** are separate operations:

1. On the source machine export a `.li` project with **Models** selected. Include geometry/labels if the destination needs them.
2. On the destination use **File > Import Project (.li)**. Import preserves checkpoint bytes, the available `config.py`, the frozen class map and the active-model choice, and rewrites paths to the destination library.
3. Select the imported model in **MODEL** above **INFER**, or in Train's model list. A bare `.pth` alone is not a complete project import: it needs the matching architecture/configuration and training-time class order. The supported portable UI workflow is a model-containing `.li`.
4. For actual inference, install the separate Pointcept/PyTorch environment using [training_setup.md](docs/training_setup.md), then set **TRAIN > POINTCEPT ENV** to that environment's Python executable and Pointcept directory. Model import does not install this environment.
5. Use Light Table **INFER** for the current full-resolution cloud, or Train batch inference for a selection. Wait for completion, inspect predictions, and use undo if needed. Inference translates frozen class names into project label IDs.

Compatible models from other projects appear in the model picker when all their predicted class names exist in the current ontology. **SANDBOX** can run registered models without assigning a project ontology; predictions become separate per-cloud model layers selected through **LAYER**, leaving project labels independent.

To train: stage clouds/labels in **TRAIN**, export a PT-v3 dataset, choose train/validation/test splits and output directory, set epochs/batch/device and optional augmentation/hyperparameters, then launch training. The panel shows progress and stop controls. A completed run registers its checkpoint and frozen class map; fine-tuning records the parent model. Stop training before removing a running model. Removing a registry entry can keep files; the context menu's artifact deletion option also removes unshared run files.

Training and inference require their configured backend and compatible CUDA dependencies; importing a bundle successfully does not prove its weights can execute. The Windows review QA currently verifies model packaging/registration/class maps, not a real training or inference run.

## Export, sequences and automation

**File > Export Screenshot** (`Ctrl+E`) saves an image. **Spin Render** (`Ctrl+Shift+E`) renders a camera rotation. **TRAIN > EXPORT DATASET** exports PT-v3, NPZ or H5 datasets with selected classes/splits. These outputs serve different purposes from `.li` project transfer.

Importing numbered frames as a sequence enables the timeline. Left/Right advances frames, and the timeline can jump directly. `propagate` transfers labels to the next frame with nearest-neighbour settings; `propagate-all` repeats across the remaining frames. Inspect each result and correct failures before continuing.

The embedded automation console accepts quoted paths. Type `help` or `help <command>` for argument details:

| Commands | Purpose |
|---|---|
| `load`, `load-recursive`, `list`, `select`, `unload`, `info` | Import/browse/remove session clouds; recursive import bypasses sequence detection. |
| `label`, `labels` | Create or inspect labels. |
| `export`, `spin`, `batch-export` | Images, rotation sequences and per-cloud images. |
| `export-dataset` | PT-v3/NPZ/H5 training datasets and splits. |
| `propagate`, `propagate-all` | Sequence label transfer. |
| `save-project`, `load-project` | Legacy annotation-session files; use File's `.li` menus for portable project bundles. |
| `train`, `train-status`, `train-stop` | Control the configured training subprocess. |
| `set`, `get`, `camera`, `status`, `catalog-status`, `clear` | Settings, camera, diagnostics and console output. |
| `wipe-catalog` | Destructive catalog reset; back up first and read its help before using it. |

Source-only command-line rendering (no desktop interaction):

```powershell
.\.venv\Scripts\python.exe -m src.cli render cloud.ply -o output --width 1920 --height 1080
.\.venv\Scripts\python.exe -m src.cli spin cloud.ply -o frames --frames 60
```

## Performance on Windows and Linux

**View > Performance > Automatic** selects a cache profile for the detected machine. **Lean** keeps smaller memory caches for integrated graphics, software renderers and lower-memory PCs. **Workstation** keeps more clouds and larger thumbnails warm for fast revisits; your high-end machine keeps this path. You can choose either profile manually if the driver does not expose GPU memory information. Switching profiles does not change labels, annotation point counts, camera controls or saved data.

Automatic uses Workstation with at least 32 GiB RAM and 8 GiB reported dedicated graphics memory, Balanced with at least 16 GiB/4 GiB, and Lean otherwise. NVIDIA's GL memory extension is supported on Windows and Linux; an unknown GPU uses the conservative fallback. See [resource_budget.py](src/core/resource_budget.py) for exact cache limits. These are cache payload limits, not a promise about total process memory; the active cloud and visible working set take priority.

Previews are read and labelled on a bounded background worker; only completed previews upload on the UI thread. Off-screen thumbnails and cold full-resolution clouds can be evicted and loaded again from the catalog. Save failures keep unsaved labels resident. Light Table always annotates the full-resolution cloud; lower-resolution thumbnails affect the gallery only.

Disabled SSAO keeps tiny buffers instead of full-screen effect targets. Minimized windows skip rendering, and unfocused windows refresh less often while background work continues. Focused windows keep normal VSync/high-refresh behaviour. No foreground 60 FPS cap is added.

For limited hardware, start with the 54-cloud package, Points display and a 1080p window. The 175-cloud package can be browsed without retaining all full-resolution clouds in Lean mode. Very large individual clouds, high-resolution windows, meshes and model training have higher requirements. A supported OpenGL 4.1+ driver is still required; there is no claim that every old integrated GPU is supported. Consult the [measured optimization report](docs/optimization-2026-10.md) for the actual hardware, tests and limits of verification.

## Troubleshooting and verification

- **Import seems idle:** check the progress card and console; large projects also need disk space for the extracted geometry/labels.
- **No INFER/model:** Kevin's data-only packages contain no checkpoint. Import a model-containing bundle and configure Pointcept before inference.
- **Mesh missing:** use Points; source installs can add Open3D to the viewer environment and retry.
- **Save-error banner:** stop editing and resolve the reported disk/permission error before closing; export/back up what you can. A failed save is not reported as success.
- **Source disappeared:** cached catalog geometry remains usable. Keep the library, not just the old source paths.
- **Sidebar larger on another screen:** monitor DPI and saved UI scale affect available space. Reset with `Ctrl+0`; resizing must preserve readable controls.

For source verification, install pytest and run `python -m pytest -q`. See [Windows handoff](docs/handoff-windows.md) for the measured test/build status; [architecture](docs/architecture.md) for internals; and [training setup](docs/training_setup.md) for backend installation. The canonical shortcuts are implemented in [src/main.py](src/main.py); the in-app reference is [shortcuts_panel.py](src/gui/shortcuts_panel.py).

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
