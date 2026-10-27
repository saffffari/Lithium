# Lithium user guide

Everything Lithium does, in one place. For the short version, see the [README](../README.md); for people labelling a shared project, see [annotator onboarding](onboarding.md).

## Install and launch

### Windows installer

1. Download `LithiumSetup-<version>.exe` from [Releases](https://github.com/saffffari/Lithium/releases) when one is published, or use an installer you were given.
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

Mesh reconstruction is optional and additionally needs `open3d` in the viewer environment. Point-only annotation and `.li` project import do not need it. The packaged Windows build does not include it.

For a source launch with a file or directory:

```powershell
.\.venv\Scripts\python.exe -m src.main "D:\data\cloud.npz"
.\.venv\Scripts\python.exe -m src.main "D:\data\clouds"
```

## First session

1. Open **File > Import Project (.li)** and select a supplied package. Wait for the import to complete, then close the progress card.
2. Select the imported project in **SHEETS**. Double-click a cloud to open **LIGHT TABLE**.
3. Choose a label in **LABELS** and a tool (`P` pick, `O` lasso, `K` brush, `U` curve, or the viewport toolbar). **Hold Shift and drag** (Shift-click for Pick) to paint; hold **Ctrl** to erase. A plain left drag pans the camera.
4. Use right mouse drag to orbit, middle mouse drag to pan, the wheel to zoom, and **F** to focus. **B/V**, Left/Right or a horizontal thumb wheel visit the previous/next cloud.
5. Use **Ctrl+Z** to undo and **Ctrl+Shift+Z** to redo. Completed edits save immediately.
6. At the end, use **File > Export Project (.li)** to make a portable copy for backup or return.

A `.li` package can carry clouds, labels and trained models. Importing a package does not upload it anywhere.

## Projects and portable packages

**SHEETS** manages the cloud library, project membership, label ontology and gallery. Files/folders can be imported from File menus or by drag and drop. The catalog caches the geometry, so later sessions can work without the original file. Keep each project's label names and meanings consistent.

A cloud may belong to several projects. Each project has independent labels and ontology; edits in one project do not replace another project's labels. Gallery context menus provide project membership, rename and Sandbox actions. Ctrl-click toggles gallery/list selection; Shift-click extends a range for batch operations.

**File > Export Project (.li)** lets you choose clouds, labels and models, with size estimates. To transfer a complete review project, include both clouds and labels. A labels-only package needs the matching cloud geometry in the destination catalog. An import creates a project, or merges with its original project when that project is already present. Existing labels are preserved where the importer reports them as skipped; review the import summary.

`.li` is a checksummed project bundle. It preserves cloud membership, ontology and selected model records. It is different from the older automation `save-project` session file. Use `.li` for moving work between computers.

## Annotation tools and labels

Choose an active label first. Hold **Shift** while using a tool to write that label as you finish the gesture; hold **Ctrl** to erase, which removes only points carrying the active label. A plain left drag pans the camera. Hidden and locked labels are protected by the painting filters.

| Tool | Use |
|---|---|
| Pick (`P`) | Shift-click a point to paint it. |
| Box (toolbar) | Shift-drag a rectangle around points. |
| Lasso (`O`) | Shift-drag a freehand outline. |
| Polygon (toolbar) | Place polygon vertices and finish the outline using the tool's completion gesture. |
| Brush (`K`) | Shift-drag a spherical brush over the cloud. Ctrl+wheel changes radius; distance falloff limits its effective interior. |
| Curve (`U`) | Shift-drag a curve to paint nearby points. |
| Line / angle measurement (toolbar) | Place point anchors to measure distance or angle; open the measurement panel with `M`. |

**LABELS:** click a row to activate it; `+ ADD` creates a label. Rename and color controls edit the ontology; eye/visibility controls hide points and lock controls prevent repainting. Deleting a populated label asks for confirmation. Label trees support nested classes. **G** cycles active labels, **H** hides/shows the active label, and **C** shows all labels in Light Table. **Alt+click** places the 3D cursor without zooming. The labels list shows up to about six full-sized rows by default; scroll within it for larger label sets. Its handle pins a custom height and double-click resets it.

Measurements use cloud-local anchors, so they follow the cloud's transform. Read the displayed units; numeric distances depend on the source coordinate units. Measurements are inspection aids.

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

Tone controls support double-click/right-click reset. Display settings persist across launches. If colors look washed out, reset COLOR, SOFT and DEPTH and compare exposure/preferences on both machines; Windows HDR, monitor profiles and saved settings can also change perceived brightness. The scene uses an ACES display pass; sidebar swatches and rendered cloud colors can differ. Colors have not been verified to match exactly between Linux and Windows.

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
4. For actual inference, install the separate Pointcept/PyTorch environment using [training_setup.md](training_setup.md), then set **TRAIN > POINTCEPT ENV** to that environment's Python executable and Pointcept directory. Model import does not install this environment.
5. Use Light Table **INFER** for the current full-resolution cloud, or Train batch inference for a selection. Wait for completion, inspect predictions, and use undo if needed. Inference translates frozen class names into project label IDs.

Compatible models from other projects appear in the model picker when all their predicted class names exist in the current ontology. **SANDBOX** can run registered models without assigning a project ontology; predictions become separate per-cloud model layers selected through **LAYER**, leaving project labels independent.

To train: stage clouds/labels in **TRAIN**, export a PT-v3 dataset, choose train/validation/test splits and output directory, set epochs/batch/device and optional augmentation/hyperparameters, then launch training. The panel shows progress and stop controls. A completed run registers its checkpoint and frozen class map; fine-tuning records the parent model. Stop training before removing a running model. Removing a registry entry can keep files; the context menu's artifact deletion option also removes unshared run files.

Training and inference require their configured backend and compatible CUDA dependencies; importing a bundle successfully does not prove its weights can execute. The test suite covers model packaging, registration and class maps, not a real training or inference run.

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

**View > Performance > Automatic** selects a cache profile for the detected machine. **Lean** keeps smaller memory caches for integrated graphics, software renderers and lower-memory PCs. **Workstation** keeps more clouds and larger thumbnails warm for fast revisits. You can choose either profile manually if the driver does not expose GPU memory information. Switching profiles does not change labels, annotation point counts, camera controls or saved data.

Automatic uses Workstation with at least 32 GiB RAM and 8 GiB reported dedicated graphics memory, Balanced with at least 16 GiB/4 GiB, and Lean otherwise. NVIDIA's GL memory extension is supported on Windows and Linux; an unknown GPU uses the conservative fallback. See [resource_budget.py](../src/core/resource_budget.py) for exact cache limits. These are cache payload limits, not a promise about total process memory; the active cloud and visible working set take priority.

Previews are read and labelled on a bounded background worker; only completed previews upload on the UI thread. Off-screen thumbnails and cold full-resolution clouds can be evicted and loaded again from the catalog. Save failures keep unsaved labels resident. Light Table always annotates the full-resolution cloud; lower-resolution thumbnails affect the gallery only.

Disabled SSAO keeps tiny buffers instead of full-screen effect targets. Minimized windows skip rendering, and unfocused windows refresh less often while background work continues. Focused windows keep normal VSync/high-refresh behaviour. No foreground 60 FPS cap is added.

For limited hardware, use Points display and a 1080p window; Lean mode browses large projects without keeping every full-resolution cloud in memory. Very large individual clouds, high-resolution windows, meshes and model training have higher requirements. A supported OpenGL 4.1+ driver is still required; there is no claim that every old integrated GPU is supported.

## Troubleshooting and verification

- **Import seems idle:** check the progress card and console; large projects also need disk space for the extracted geometry/labels.
- **No INFER/model:** data-only packages contain no checkpoint. Import a model-containing bundle and configure Pointcept before inference.
- **Mesh missing:** use Points; source installs can add Open3D to the viewer environment and retry.
- **Save-error banner:** stop editing and resolve the reported disk/permission error before closing; export/back up what you can. A failed save is not reported as success.
- **Source disappeared:** cached catalog geometry remains usable. Keep the library, not just the old source paths.
- **Sidebar larger on another screen:** monitor DPI and saved UI scale affect available space. Reset with `Ctrl+0`; resizing must preserve readable controls.

For source verification, install pytest and run `python -m pytest -q`. See [architecture](architecture.md) for internals and [training setup](training_setup.md) for backend installation. The canonical shortcuts are implemented in [src/main.py](../src/main.py); the in-app reference is [shortcuts_panel.py](../src/gui/shortcuts_panel.py).

## Performance

Benchmarks on 1M-point clouds (RTX 4090): pick 56 ms · box select 54 ms ·
lasso 74 ms · brush 11 ms · label apply (500 K pts) 3.6 ms · undo 1.1 ms.
Per-stroke GPU upload is 4 MB, not 32 MB, via a split-buffer layout.

## Platform notes

- **Linux** — primary development platform.
- **Windows** — supported; the app was originally Windows-native.
- **macOS** — runs from source on OpenGL 4.1 (Apple Silicon and Intel); see
  [macos.md](macos.md). View/annotate only — training and INFER need
  CUDA.
- Unsigned binaries: Windows SmartScreen and browser warnings are expected
  for now — the code is right here to audit.

