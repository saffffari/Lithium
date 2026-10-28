# Lithium user guide

This is the full manual. For the short version, see the [README](../README.md).
If someone sent you a project to label, [annotator onboarding](onboarding.md)
is all you need.

- [Install](#install)
- [Start Lithium](#start-lithium)
- [The window](#the-window)
- [Contact Sheets: projects and the gallery](#contact-sheets-projects-and-the-gallery)
- [Light Table: labelling](#light-table-labelling)
- [Camera and display](#camera-and-display)
- [Saving, backups and recovery](#saving-backups-and-recovery)
- [Sharing projects (.li)](#sharing-projects-li)
- [Performance profiles](#performance-profiles)
- [Training and inference](#training-and-inference)
- [Exports, sequences and the console](#exports-sequences-and-the-console)
- [Keyboard and mouse reference](#keyboard-and-mouse-reference)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [Platform notes](#platform-notes)

## Install

### What you need

- A GPU and driver that support **OpenGL 4.1** or newer. Most computers from
  the last decade qualify once the graphics driver is current. No NVIDIA card
  is needed for viewing and labelling.
- **Git**, to clone the repository.
- A **C++ compiler**. One dependency, the Dear ImGui bindings (`imgui`), has
  no prebuilt wheel for Python 3.12 and builds from source during install.
  - Windows: [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
    with the "Desktop development with C++" workload.
  - macOS: `xcode-select --install`.
  - Linux: `g++` from your package manager.

You do not need to install Python yourself. The setup script fetches a private
Python 3.12.

### Setup script (recommended)

```bash
git clone https://github.com/saffffari/Lithium.git
cd Lithium
```

Then, from that folder:

- **Windows (PowerShell):** `.\setup.ps1`
  If PowerShell refuses to run scripts:
  `powershell -ExecutionPolicy Bypass -File .\setup.ps1`
- **macOS / Linux:** `./setup.sh`

The script installs [uv](https://docs.astral.sh/uv/) if it is missing, creates
`.venv` with Python 3.12, installs `requirements-annotator.txt`, and launches
Lithium. Re-running it is safe. Options:

| Windows | macOS / Linux | Effect |
|---|---|---|
| `-Kit <file.li>` | `--kit <file.li>` | Import a project bundle before launching. |
| `-NoLaunch` | `--no-launch` | Install only. |

After the first run, start Lithium with `run.bat` (Windows) or `./run.sh`.

To update, run `git pull` in the repo folder, then run the setup script again.

### Manual install

If you prefer to manage the environment yourself, use Python 3.12:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-annotator.txt
.venv/bin/python -m src.main
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Plain `python -m venv` and `pip` work too.

The requirements files:

| File | Contents |
|---|---|
| `requirements-annotator.txt` | Viewer and labeller only. Pinned. |
| `requirements-lock.txt` | Everything, including PyInstaller and the GLB conversion tools. Pinned. Used by CI. |
| `requirements.txt` | Unpinned minimum versions. |

Mesh display (the MESH and BOTH display modes) also needs `open3d` in the
same environment. Nothing else depends on it.

### Prebuilt builds

When a release is published on the
[Releases page](https://github.com/saffffari/Lithium/releases), it carries a
Windows installer (`LithiumSetup-<version>.exe`) and portable `Lithium.exe`, a
single-file Linux binary, and a zipped macOS app. The Windows installer
installs for your user account only, into `%LOCALAPPDATA%\Programs\Lithium`,
and does not need administrator rights. It can add a desktop shortcut.
Uninstall it from Windows Settings > Apps. Your catalog lives in your user
profile and is not removed.

The builds are unsigned, so Windows SmartScreen and macOS Gatekeeper will warn
you. Training and inference do not work from a prebuilt build (see
[Training and inference](#training-and-inference)).

## Start Lithium

```bash
run.bat                              # Windows
./run.sh                             # macOS / Linux
./run.sh path/to/cloud.ply           # open a file or a folder
./run.sh --project "My project" --cloud 3 --tab light
```

| Flag | Effect |
|---|---|
| `<path>` | Import a file or folder on start. |
| `--project <name or id>` | Open this project. |
| `--cloud <N>` | Select the N-th cloud of that view, counting from 0. |
| `--tab sheets\|light\|train` | Start on this tab. |

### Where your data lives

Lithium keeps everything in a **catalog** under your home folder:

| Path | Contents |
|---|---|
| `~/.lithium/prefs.json` | Viewer settings and the training environment paths. |
| `~/.lithium/library/` | The catalog: `index.json`, `projects.json`, cloud data, previews, labels, models and backups. |
| `~/.lithium/library/labels/<project>/` | One label array per cloud, per project. |
| `~/.lithium/library/backups/` | Timestamped label backups. |

`<project>` is the project's id with `:` replaced by `_`, for example
`proj_1a2b3c4d`. Clouds outside any project use `_library`. On Windows, `~` is
your user folder (`%USERPROFILE%`). Set `LITHIUM_LIBRARY_DIR` before launching
to use a different catalog folder.

Once you import a cloud, the catalog holds its own copy. You can move or
delete the original file and keep working. There is no separate "save
project" step.

Only one Lithium can use a catalog at a time. A second instance refuses to
start while the first is running. A lock left behind by a crash is cleared
automatically.

## The window

Lithium has three modes. Switch with the tabs at the top of the sidebar, the
**View** menu, or the number keys:

| Key | Tab | Mode | What it is for |
|---|---|---|---|
| `1` | SHEETS | Contact Sheets | Import, organise projects, browse the gallery. |
| `2` | LIGHT | Light Table | Label one cloud at a time. |
| `3` | TRAIN | Train | Export datasets, train and run models, use the console. |

The menu bar has **File**, **Edit**, **View** and **Help**:

- **File:** Import File..., Import Folder..., Import Project (.li)...,
  Export Project (.li)..., Save Points, Predict Current Cloud, Export
  Screenshot, Spin Render, Quit.
- **Edit:** Undo, Redo, Clear Labels, Clear 3D Cursor + Fit.
- **View:** Performance, the three modes, Toggle Sidebar, Bounding Box, Ground
  Grid, Measurements, Mouse, and the Top / Right / Iso / Frame View camera
  presets.
- **Help:** Keyboard Shortcuts (`?`).

Press `?` at any time for the shortcut overlay. Its GETTING STARTED block is
the whole labelling loop.

## Contact Sheets: projects and the gallery

### Import clouds

Lithium reads **PLY**, **LAS**, **LAZ** and **NPZ**. To bring clouds in:

- **File > Import File...** or **Import Folder...**, or the **FILE** and
  **FOLDER** buttons under IMPORT. If you cancel the file picker, a folder
  picker opens instead.
- Drag files or folders onto the window.
- Pass a path on the command line.

New clouds join the project that is currently selected, if it is one of your
own projects. Otherwise they appear under **All Photos** and **Recent
Imports** only. To label your own data, create a project first, select it,
then import.

Two import rules to know:

- **Numbered folders open as a sequence.** If every file in a folder has a
  number in its name, importing the folder opens it as one time series with a
  timeline, not as separate clouds. To get separate clouds, select or drag the
  files themselves instead of the folder, or type `load-recursive <folder>` in
  the console on the TRAIN tab. See [Sequences](#sequences).
- **Companion label files.** When you import a single file and a
  `<name>_labels.npy` sits next to it, Lithium applies it as labels. The file
  must be an integer array with one entry per point, in the same order.

### Projects

The **PROJECTS** list holds three automatic views and your own projects:

| View | Shows |
|---|---|
| All Photos | Every cloud in the catalog. |
| Recent Imports | Recently imported clouds. |
| Missing Files | Clouds whose original file is no longer where it was. Appears only when there are some. |
| Your projects | The clouds you added. Newest project first. |

Click **+ NEW PROJECT** to create one. Right-click a project for:

- **Duplicate (labels + models)**: a new project with the same label set and
  model list, and no clouds.
- **Delete project**: removes the project and its label set. The clouds and
  their label files stay in the catalog.

A cloud can belong to several projects. **Each project has its own labels.**
Painting a cloud in one project never changes its labels in another.

Below the project list, the **ONTOLOGY** row (shown for your own projects)
lets you pick a preset label set for the project, or **None**.

**SANDBOX** is a separate area for running models. See
[The Sandbox](#the-sandbox).

### The gallery

The gallery shows every cloud in the current view.

| Action | Effect |
|---|---|
| Click | Select a cloud. |
| Ctrl+click | Add a cloud to the selection, or remove it. |
| Shift+click | Select a range. |
| Double-click | Open the cloud in the Light Table. |
| Right-click | Open the cloud menu. |
| Right-drag on a cell | Turn that preview. |
| Wheel | Scroll. |
| Ctrl+wheel | Change the cell size. |
| Shift+wheel | Zoom the selected preview. |

The same click, Ctrl+click and Shift+click rules apply in the sidebar's
**CLOUDS** list.

The cloud menu acts on the whole selection when you right-click a selected
cloud:

- **Load → Light Table**
- **Send to Sandbox** / **Remove from Sandbox**
- **Add to project**: adds the clouds to another project. Their labels come
  along and are matched to that project's labels by name.
- **Clear labels** (`Ctrl+Shift+X`)
- **Rename...**
- **Re-attach source...**: point a cloud whose source file moved at its new
  location. Shown only when the source is missing.
- **Remove from project**: takes the clouds out of the current project only.
- **Delete from program**: removes the clouds from the catalog and from every
  project. The original files on disk are never touched.

## Light Table: labelling

### The sidebar

From top to bottom: **CLOUDS** (the clouds in the current view), **LABELS**,
**BRUSH DISTANCE FALLOFF** (only while the brush is active), **POINT SIZE**,
**DISPLAY**, **CLIP**, **CAMERA**, then the **MODEL**
picker and the **INFER** button when a model is available.

Click a cloud in CLOUDS to switch to it. Drag the handle under the list to
fix its height. Double-click the handle to let it fill the free space again.

### Labels

Labels belong to the current project. **+ ADD** and **− DEL** work only when
one of your own projects is selected.

| Action | Effect |
|---|---|
| Click a row | Make it the active label. |
| `G` | Make the next label active. |
| **+ ADD** | Add a label and make it active. |
| Double-click the name | Rename. Press Enter to confirm, Esc to cancel. |
| Click the colour swatch | Pick a new colour. |
| Click the eye (right edge) | Hide or show the label's points. |
| `H` | Hide or show the active label. |
| `C` | Show every label again. |
| **− DEL** | Delete the active label. |

Each row shows how many points of the current cloud carry that label. Labels
can be nested; children are indented under their parent. The list shows about
six rows and scrolls beyond that. Drag the handle under it to change its
height; double-click the handle to reset.

**Deleting a label cannot be undone.** If any points carry it, Lithium asks
first, then sets those points to Unlabeled in every cloud of the catalog. To
change only the name or colour, rename it instead.

**Hidden labels are protected.** Tools skip points whose label is hidden, so
you can hide finished parts and paint around them.

**Locked labels.** A label set can mark labels as locked. Lithium will not
paint with a locked label or paint over its points. There is no lock button
in the LABELS list.

### Tools

Pick a tool with its key or from the toolbar at the top left of the viewport.
Press the key again, or `Esc`, to put the tool down.

| Tool | Key | How to use it |
|---|---|---|
| Pick (PCK) | `P` | Shift+click a point to label it. |
| Box (BOX) | toolbar | Shift+drag a rectangle. |
| Lasso (LSO) | `O` | Shift+drag a freehand outline. |
| Polygon (PLY) | toolbar | Shift+click to place corners. Shift+double-click to close the shape (three corners or more). |
| Brush (BRU) | `B` | Shift+drag to paint with a sphere. |
| Curve (CRV) | `U` | Shift+drag a line. Points within a few pixels of it are labelled. |
| Move (MOV) | toolbar | Drag the gizmo arrows to move the cloud. |
| Rotate (ROT) | toolbar | Drag the gizmo to rotate the cloud. |
| Distance (DST) | toolbar | Click two points. |
| Angle (ANG) | toolbar | Click three points: arm, vertex, arm. |
| Landmark (LMK) | toolbar | Click one point. |

The painting rules are the same for every labelling tool:

- **Shift** writes the active label.
- **Ctrl** erases, and only removes the **active** label. Points with other
  labels are left alone.
- A plain left-drag pans the camera, even with a tool selected.
- A stroke is saved when you finish it.

Tool sizes use **Ctrl+wheel**:

- **Brush:** radius. The **BRUSH DISTANCE FALLOFF** slider limits how deep the
  brush reaches. At 0 it paints the whole sphere. Higher values paint only a
  thin layer around the surface under the cursor, so the brush stops painting
  through the cloud.
- **Curve:** how far from the line points are picked up, in pixels.
- **Box and Lasso:** a depth limit, drawn as a plane. Without one, Box,
  Lasso, Polygon and Curve label every point inside the outline, front to
  back.

### Undo and redo

`Ctrl+Z` undoes and `Ctrl+Shift+Z` redoes (also in the Edit menu). Each stroke
is one step, however long. Undo and redo save to disk like any other edit.

Undo history belongs to the cloud you are on. Switching clouds starts a fresh
history.

### Clear labels

**Edit > Clear Labels** (`Ctrl+Shift+X`) sets every point to Unlabeled:

- In the Light Table, it clears the current cloud as one undo step.
- In Contact Sheets, it clears every selected cloud. `Ctrl+Z` in Contact
  Sheets brings them back.

### Move between clouds

- `E` or `Right`: next cloud. `Q` or `Left`: previous cloud.
- A horizontal thumb wheel (on mice that have one) also changes cloud, in the
  Light Table and in Contact Sheets. Turn it off, or reverse it, under
  **View > Mouse**.
- Click a row in CLOUDS.

Your labels are saved before the switch. The thumb wheel does nothing while a
mouse button is held, so you cannot change cloud in the middle of a stroke.

### Measurements

Distance, Angle and Landmark place anchors with a plain left-click; no Shift
needed. Anchors snap to the nearest point. A measurement is stored when it has
all its points. Drag a placed anchor to move it.

Press `M` (or **View > Measurements**) to open the measurement panel once
there is at least one measurement. With the panel open, `Delete` removes the
selected measurements. Distances are in the cloud's own coordinate units.

## Camera and display

### Mouse in the Light Table

| Input | Effect |
|---|---|
| Right-drag | Orbit. Release while moving and it keeps spinning briefly. |
| Right-button hold, no tool | Radial menu: right, top, left or bottom view. |
| Left-drag | Pan. |
| Middle-drag | Depth of field (the FOCUS knob). Drag up for more. |
| Wheel | Zoom toward the point under the cursor. |
| Middle button + wheel | Point size. |
| Alt+click | Make the clicked point's label the active label. With no tool, also place the 3D cursor without moving the camera. |

### Keys

| Key | Effect |
|---|---|
| `F` | Focus on the 3D cursor if one is placed, otherwise frame the selected cloud. The cursor starts at the centroid of each cloud you open. |
| `Space` | Place the 3D cursor under the mouse and zoom to it. |
| `T` / `R` / `I` | Top, right and isometric views. |
| `C` (outside the Light Table) | Clear the 3D cursor and frame everything. |
| `Shift+B` / `Shift+G` | Bounding box / ground grid. |
| `L` | Switch between label colours and the cloud's own colours. |
| `F6` | Ambient occlusion on or off. |

**Edit > Clear 3D Cursor + Fit** removes the cursor and frames the scene.

### CAMERA panel

- **Camera cube:** click a face to look from that side. Double-click a face to
  look from the opposite side. Right-click the cube to return to the
  isometric view.
- **The icon under the cube** switches between perspective (green) and
  orthographic (orange). Top, side and front views are always orthographic.
- **The four small icons** set one to four viewports.
- **The grid button** toggles the ground grid.

### Display controls

| Control | Effect |
|---|---|
| POINT SIZE | Size of each point. |
| DISPLAY: FOCUS | Depth of field. 0 turns it off. |
| DISPLAY: DEPTH | Darkens points with distance. |
| DISPLAY: SHADE | Ambient-occlusion shading in creases. 0 turns it off. |
| DISPLAY: COLOR | Brightness. |
| Display mode (top-right of the viewport) | Points, mesh, or both. Mesh and both need `open3d`. |
| CLIP | One fader per axis (R, S, A) with low and high limits. Click a letter to turn that axis on or off. Double-click a fader to reset it. |

Drag a DISPLAY knob up or down to change it. Double-click or right-click to
reset. Display settings are remembered between sessions.

### Interface size

- Drag the sidebar's right edge to change its width.
- `Ctrl` + `=` / `-` / `0` makes the interface larger, smaller, or resets it.
  On macOS, `Cmd` works too.
- `Shift+H` hides or shows the whole interface. Outside the Light Table, `H`
  does the same.

## Saving, backups and recovery

**Labels save themselves.** Every finished stroke, undo, redo, clear and
inference result is written to disk at once. Files are written to a temporary
name and then swapped in, so a crash cannot leave a half-written label file.

On top of that:

- **During a long brush stroke**, Lithium writes a recovery copy about every
  15 seconds. If Lithium stops in the middle of a stroke, the next time you
  open that cloud it offers **Restore snapshot** or **Keep last committed
  labels**. Restoring keeps a backup of the committed labels first.
- **Every 10 minutes**, if anything changed, the labels of every loaded cloud
  are copied to `~/.lithium/library/backups/<timestamp>/<project>/`. The
  newest 12 backups are kept.
- **File > Save Points** saves every loaded cloud and makes a backup right
  away.

If a save fails, a red banner appears at the top of the window. Stop editing
and fix the cause (disk full, permissions) before closing Lithium. A failed
save is never reported as a success.

For an independent copy, export a `.li` bundle or copy the whole
`~/.lithium/library` folder while Lithium is closed.

To restore an older backup by hand: close Lithium, copy the current catalog
somewhere safe, then copy `backups/<timestamp>/<project>/<file_key>.npy` over
`labels/<project>/<file_key>.npy`. Only use a file from the same cloud and
project.

## Sharing projects (.li)

A `.li` file packs one project for another computer: any mix of its point
clouds, its labels and its trained models.

- **File > Export Project (.li)...**: pick the project and tick **Point
  clouds**, **Labels** and/or **Model weights**. The dialog shows the size of
  each part.
- **File > Import Project (.li)...**: pick the file. Lithium imports
  everything inside and opens the project. Dropping a `.li` file on the
  window, or passing it on the command line, does the same.

An import creates a new project, unless the project it was exported from
exists in your catalog; then it merges into that project. Imported labels
never overwrite labels that are already there. The progress card reports how
many label files were written and how many were skipped.

A labels-only bundle needs the matching clouds in your catalog already, for
example from an earlier clouds-only bundle of the same project.

[Project bundles](bundles.md) covers the format, the merge rules and the
command-line tool.

## Performance profiles

**View > Performance** sets how much memory Lithium uses for caches:

| Setting | Behaviour |
|---|---|
| Automatic (default) | Picks a profile from your hardware. |
| Lean | Small caches. For integrated graphics, software renderers and machines with little memory. |
| Workstation | Large caches. Keeps more full-resolution clouds and larger thumbnails ready. |

Automatic chooses **Workstation** with at least 32 GB of RAM and 8 GB of
graphics memory, **Balanced** with at least 16 GB and 4 GB, and **Lean**
otherwise. Software renderers always get Lean. Lithium can only read graphics
memory on NVIDIA drivers; on other GPUs Automatic chooses Lean, so pick
Workstation by hand if your machine can take it. The menu shows the profile in
use. Exact limits are in
[resource_budget.py](../src/core/resource_budget.py).

The profile changes caching only. Labels, point counts and saved data are the
same in every profile. The Light Table always labels the full-resolution
cloud; thumbnails affect the gallery only.

Lithium also saves work on its own: it stops drawing while minimised and
redraws about 15 times per second while the window is in the background.

On modest hardware, use the POINTS display mode, a 1080p window and the Lean
profile.

## Training and inference

Lithium can train
[Point Transformer V3](https://github.com/Pointcept/Pointcept) segmentation
models on your labels and run them on other clouds. This is optional. It
needs an NVIDIA GPU with CUDA, a source checkout of Lithium, and a separate
Python environment with PyTorch and Pointcept. The viewer itself never
imports PyTorch.

Set that environment up with [Training setup](training_setup.md), then open
**TRAIN > POINTCEPT ENV** and enter its **python exe** and **pointcept dir**.

### Train a model

On the TRAIN tab, with one of your projects selected:

1. **SELECT CLOUDS:** tick the clouds to use. **All**, **None** and
   **Labeled** help.
2. **LABELS TO EXPORT:** choose the classes. Keep Unlabeled ticked so the
   model learns a background class.
3. **EXPORT DATASET:** choose the format (`ptv3` for training), the train and
   val split (test gets the rest), the output folder and the source
   (**selected** clouds or the **staged** list from STAGED DATASET). Click
   **EXPORT**.
4. **TRAIN:** choose **+ New Model** and type a name, or pick an existing
   model to fine-tune from it. Set **epochs**, **batch** and **device**.
   **ADVANCED HYPERPARAMS** holds the learning rate, weight decay, warm-up,
   drop path, class-weight boost and rotation augmentation.
5. Click **LAUNCH TRAINING** (or **FINE-TUNE FROM ...**). Progress, loss and
   mIoU appear below. **STOP TRAINING** ends the run.

Lithium trains on the dataset in the output folder and writes the run to
`<output folder>/training_runs/<model name>_<time>/`. When the run finishes,
the model is added to the project with its best checkpoint and the class list
it was trained on. That class list is fixed at launch.

Right-click a model in the list to remove it: **Delete entry (keep files)**,
or **Delete entry + run dir**. Stop a running model first.

### Run a model

- **Light Table:** pick a model under **MODEL** (or keep **(auto: highest
  mIoU)**) and click **INFER**. **File > Predict Current Cloud** does the same.
- **TRAIN > BATCH INFERENCE:** run the model on the ticked clouds, or on every
  cloud in the project that has no labels yet.

Predictions are written as labels in the current project. Before it
overwrites a cloud's labels, Lithium copies the old ones to
`~/.lithium/library/backups/<time>/inference_pre/`. On the current cloud,
`Ctrl+Z` also undoes a prediction, one step per predicted class.

Lithium matches the model's class names to the project's label names, so a
model from another project is offered only when all its class names exist in
the current project.

### The Sandbox

**SANDBOX** runs any model on any cloud without touching project labels.
Right-click clouds in the gallery and choose **Send to Sandbox**. In the
Sandbox, each model's predictions become a separate layer per cloud. Choose
the layer to show under **LAYER**.

### Moving models between computers

Export the project with **Model weights** ticked. The bundle carries each
model's checkpoint, its `config.py` and its class list. Importing it registers
the models in the new catalog. To run them there, that computer also needs the
training environment. A bare `.pth` file is not enough on its own.

## Exports, sequences and the console

### Images

- **File > Export Screenshot** (`Ctrl+E`) saves `export/screenshot.png`,
  replacing the previous one.
- **File > Spin Render** (`Ctrl+Shift+E`) saves a 360-frame turntable to
  `export/spin/`.

Both folders are created in the directory Lithium was started from: the repo
folder when you use `run.bat` or `run.sh`.

For batch rendering without the window:

```bash
python -m src.cli render <file-or-folder> -o output --width 1920 --height 1080
python -m src.cli spin <file-or-folder> -o frames --frames 60
```

Both accept `--camera`, `--point-size`, `--bg`, `--grid` and `--bbox`. Run
with `--help` for the full list.

### Sequences

A folder in which every file name contains a number opens as a sequence: one
time series with a timeline. `Left` and `Right` (and `Q` / `E`) step through
frames. In the console, `propagate` copies labels from the current frame to
the next by nearest neighbour, and `propagate-all` repeats that through the
remaining frames. Check each frame afterwards.

### The console

The LOG section at the bottom of the TRAIN tab has a command line. Type
`help` to list commands with their arguments, or `help <command>` for one.
Quote paths that contain spaces.

| Commands | Purpose |
|---|---|
| `load`, `load-recursive`, `list`, `select`, `unload`, `info` | Load and browse clouds. `load-recursive` imports every file as its own cloud, skipping sequence detection. |
| `label`, `labels` | Create and list labels. |
| `export`, `spin`, `batch-export` | Images and turntables. |
| `export-dataset` | Training datasets (PT-v3, NPZ, H5). |
| `propagate`, `propagate-all` | Copy labels along a sequence. |
| `train`, `train-status`, `train-stop` | Control training. |
| `set`, `get`, `camera`, `status`, `catalog-status`, `clear` | Settings, camera, status and clearing the log. |
| `save-project`, `load-project` | An older session-file format. Use `.li` bundles to move work between computers. |
| `wipe-catalog` | Empties the catalog. On its own it only reports what would be lost. With `--confirm` it saves a `.tar.gz` backup to `~/.lithium/backups/` and then wipes. |

## Keyboard and mouse reference

Shortcuts are ignored while you type in a text field.

| Key | Effect |
|---|---|
| `1` / `2` / `3` | Contact Sheets / Light Table / Train |
| `?` | Shortcut overlay |
| `Esc` | Close the overlay, put down the tool, clear the selection, or go back to Contact Sheets, in that order. It never quits; use **File > Quit** or close the window. |
| `P` / `O` / `B` / `U` | Pick / Lasso / Brush / Curve tool |
| `G` | Next label (Light Table) |
| `H` | Hide or show the active label (Light Table); hide or show the interface elsewhere |
| `Shift+H` | Hide or show the interface |
| `C` | Show all labels (Light Table); clear the 3D cursor and frame everything elsewhere |
| `L` | Label colours or original colours |
| `E` / `Right` | Next cloud (Light Table) or frame (sequence) |
| `Q` / `Left` | Previous cloud (Light Table) or frame (sequence) |
| `F` | Focus the 3D cursor, or frame the selected cloud |
| `Space` | Place the 3D cursor and zoom to it (Light Table) |
| `T` / `R` / `I` | Top / right / isometric view |
| `Shift+B` / `Shift+G` | Bounding box / ground grid |
| `M` | Measurement panel |
| `Delete` | Delete the selected measurements (panel open) |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |
| `Ctrl+Shift+X` | Clear labels |
| `Ctrl+E` / `Ctrl+Shift+E` | Screenshot / spin render |
| `Ctrl` + `=` / `-` / `0` | Interface larger / smaller / reset (`Cmd` on macOS) |
| `F6` | Ambient occlusion on or off |
| `F5` | Reload shaders from disk (for development) |

| Mouse | Light Table | Contact Sheets |
|---|---|---|
| Left-drag | Pan | |
| Shift + left-drag / click | Paint with the active tool | Shift+click selects a range |
| Ctrl + left-drag / click | Erase the active label | Ctrl+click adds to the selection |
| Right-drag | Orbit | Turn the preview under the cursor |
| Right-button hold | Radial view menu | |
| Right-click | | Cloud menu |
| Middle-drag | Depth of field | |
| Wheel | Zoom | Scroll |
| Ctrl + wheel | Tool size or depth limit | Cell size |
| Shift + wheel | | Zoom the selected preview |
| Middle button + wheel | Point size | |
| Alt+click | Pick the clicked point's label (and place the 3D cursor) | |
| Double-click | Shift+double-click closes a polygon | Open in the Light Table |
| Thumb wheel | Previous / next cloud | Previous / next cloud |

## Environment variables

| Variable | Effect |
|---|---|
| `LITHIUM_LIBRARY_DIR` | Use this folder as the catalog instead of `~/.lithium/library`. Also read by `tools/bundle_cli.py`. |
| `LITHIUM_MSAA` | Multisampling level, default 4. `0` turns it off, which helps on very high-resolution displays. |
| `LITHIUM_HIDDEN=1` | Start with the window hidden. |
| `LITHIUM_EXIT_AFTER=<seconds>` | Close cleanly after this many seconds. |
| `LITHIUM_FPS_LOG=<seconds>` | Print the frame rate at this interval. |
| `LITHIUM_PROFILE=<file>` | Run under cProfile and write the stats to this file on exit. |

The last four are for measuring Lithium on machines you cannot watch.

## Troubleshooting

- **Setup fails while building `imgui`:** install the C++ compiler listed
  under [What you need](#what-you-need), then run the setup script again.
- **`uv` is not on PATH after setup installs it:** open a new terminal and run
  the script again.
- **Black window, crash on launch, or an OpenGL error:** update your graphics
  driver. Lithium needs OpenGL 4.1.
- **"Refusing to start: another Lithium instance is using the catalog":**
  close the other window. To run two copies, give one its own
  `LITHIUM_LIBRARY_DIR`.
- **File dialogs do not open (Linux):** install `python3-tk` (Debian/Ubuntu)
  or `python3-tkinter` (Fedora).
- **A folder opened as one sequence instead of many clouds:** every file name
  contained a number. Import the files instead of the folder, or use
  `load-recursive <folder>` in the console.
- **I can't add labels:** select one of your own projects. All Photos and
  Recent Imports have no label set of their own.
- **A stroke did nothing:** check that you held Shift, that the active label
  is not hidden or locked, and, when erasing, that the points carry the active
  label.
- **No INFER button or no models:** the project has no trained model, or the
  bundle you imported had no model weights. Import a bundle with models, or
  train one.
- **Inference or training fails at once:** set **TRAIN > POINTCEPT ENV** and
  read the LOG. See [Training setup](training_setup.md).
- **MESH shows nothing:** install `open3d` into `.venv`, or use POINTS.
- **Red save-error banner:** stop editing and fix the disk or permission
  problem it names before closing Lithium.
- **The original files moved:** the catalog copy keeps working. Right-click >
  **Re-attach source...** to point Lithium at the new location.
- **The interface is too big or too small on another screen:** press
  `Ctrl+0` to reset the interface size.
- **Colours look washed out:** reset COLOR, SHADE and DEPTH by double-clicking
  them. Windows HDR and monitor colour profiles also change what you see.

## Platform notes

- **Windows and Linux:** fully supported, including training and inference on
  machines with an NVIDIA GPU.
- **macOS (Apple Silicon and Intel):** viewing and labelling, from source. No
  training or inference, because they need CUDA. See [macOS](macos.md).

For internals, see [Architecture](architecture.md).
