# 3Photon User Guide

3Photon is a desktop workstation for point-cloud annotation and vision-model training — all local, no cloud upload. Build labeled datasets, then train Pointcept PT-v3 on them.

## Installation

```bash
git clone <repo-url> 3photon
cd 3photon
python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

Runs on Linux and Windows. Python 3.12.

For PT-v3 training, set up the Pointcept conda env: see [training_setup.md](training_setup.md).

## Launch

```bash
python -m src.main                       # empty session
python -m src.main cloud.ply             # single file
python -m src.main /path/to/directory    # directory or 4D sequence
python -m src.cli render scene.ply out.png            # headless render
python -m src.cli spin scene.ply ./spin --frames 60   # spin sequence
```

A single-instance lock lives at `~/.3photon/library/.lock` — 3Photon refuses to start if another live instance owns it. Override the library dir with `THREEPHOTON_LIBRARY_DIR`.

## The Three Modes

Jump between modes with number keys `1`–`3`. Each mode shares the camera, catalog, and undo stack but is otherwise a separate render path.

| Key | Mode | Purpose |
|-----|------|---------|
| 1 | **CONTACT_SHEETS** | Gallery grid of every imported cloud. Click to focus. |
| 2 | **LIGHT_TABLE** | Single-cloud annotation surface with tools, labels, and the timeline. |
| 3 | **AUTOMATION / TRAIN** | Embedded CLI for bulk operations (batch export, propagation) and Pointcept PT-v3 training. |

## Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| PLY    | .ply      | Mesh-derived and generic point clouds |
| LAS / LAZ | .las / .laz | LiDAR and aerial surveys |
| NPZ    | .npz      | Catalog-internal + PT-v3 predictions |

Drag-and-drop files or folders onto the window, or use the IMPORT buttons.

---

## Annotation workflow (CONTACT_SHEETS + LIGHT_TABLE)

1. **Import** clouds (CONTACT_SHEETS tab — drag-drop or IMPORT button)
2. **Select** a cloud (click in gallery → switches to LIGHT_TABLE)
3. **Create labels** in the LABELS panel (`+ ADD`)
4. **Pick an active label** (coral border indicates active)
5. **Pick a tool** from the TOOLS palette
6. **Select points**, then **Apply** with Enter or APPLY
7. **Toggle LABELS ON** to see color-coded result
8. **Export** the labeled dataset via AUTOMATION tab → `export-dataset`

Labels persist immediately to `~/.3photon/library/labels/<file_key>.npy`. There is no separate "save" step — the catalog *is* the dataset.

### Selection tools

| Tool | Shortcut | Description |
|------|----------|-------------|
| PICK | P | Click a single point |
| BOX | — | Drag a screen-space rectangle |
| LASSO | O | Freeform shape |
| BRUSH | K | Paint a 3D sphere (click+drag) |
| POLYGON | — | Click vertices to define a polygon |
| CURVE | — | Draw a line; points near it select |

**Modifiers:** Shift+click = add · Ctrl+click = remove · Ctrl+scroll = brush radius / depth limit

### Hierarchical labels

The LABELS panel supports nested label trees (Rhino-style). Eye = visibility · Lock = prevent selection · Color swatch = label color.

---

## 4D sequences

Load a directory of frames with numeric suffixes (e.g. `frame_000.ply`, `frame_001.ply` …) → a timeline appears at the bottom.

- `←` / `→` — previous/next frame
- Click timeline — jump to frame
- Green tick = has labels · grey = unlabeled

### Label propagation (4D)

Label the first frame manually, then propagate via the AUTOMATION tab:

```
propagate                       # propagate to next frame
propagate --radius 0.5 --k 5    # custom KD-tree settings
propagate-all                   # all remaining frames
```

Inspect and correct, re-propagate from the corrected frame.

---

## Training (TRAIN tab inside AUTOMATION)

```
train /path/to/dataset --epochs 50 --device cuda
train-status
train-stop
```

Training runs in a sidecar Pointcept subprocess in the `3photon-ptv3` conda env (separate from the renderer). See [training_setup.md](training_setup.md) for env install.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1–3 | Mode jump |
| H | Toggle GUI |
| G | Toggle gallery / light-table view |
| T / F / R / I | Camera presets (top / front / right / iso) |
| B | Toggle bounding box |
| Shift+G | Toggle grid floor |
| Space | Fit camera to view |
| Ctrl+E | Export screenshot |
| Ctrl+Shift+E | Spin render |
| Esc | Back to gallery / quit |

Press `?` (Shift+/) for the shortcut overlay.

---

## AUTOMATION CLI commands

- `help [cmd]` — list / inspect commands
- `load <path>` — load file or directory
- `list` / `select <idx>` / `unload <idx|all>` / `info [idx]`
- `export <path> [--res WxH]` — screenshot
- `spin <dir> [--frames N]` — spin render
- `batch-export <dir>` — screenshot every cloud
- `export-dataset <dir> [--format ptv3|npz|h5] [--split 0.7,0.15,0.15]`
- `label <name>` / `labels`
- `propagate [--radius R] [--k K]` / `propagate-all`
- `save-project <path>` / `load-project <path>`
- `train <data_dir> [--epochs N]` / `train-status` / `train-stop`
- `set <key> <value>` / `get <key>`
- `camera <preset>`
- `clear` / `status`

## License

TBD.
