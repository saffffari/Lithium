# Windows handoff — verification + build

For the agent (or human) working in this repo cloned on the Windows
box. Goal: verify Lithium 1.1 on Windows and produce a local installer
build. All 1.1 development happened on Linux; 1.0 was Windows-native,
so expect small breaks, not structural ones.

## 1. Environment

```bat
python -m venv .venv            & rem Python 3.12 x64
.venv\Scripts\activate
pip install -r requirements-lock.txt
python -m pytest tests -q       & rem expect 210 passed, 3 skipped
python -m src.main              & rem app boots to SHEETS
```

## 2. Verification checklist (things most likely to have rotted)

- [ ] **Boot + fonts** — segoeui.ttf discovery in
      `src/gui/imgui_layer.py:_find_system_font`; title bar dark via
      DWM (`_apply_dark_title_bar`).
- [ ] **DPI** — drag the window between monitors with different scale
      factors; the UI must rebuild (new in 1.1:
      `_on_content_scale` → `ImGuiLayer.rebuild_for_scale`). GLFW on
      Windows fires content-scale changes on monitor move — confirm.
- [ ] **Import** — PLY + LAS via File menu (tkinter dialogs) and
      drag-drop; catalog write to `%USERPROFILE%\.lithium\library`.
- [ ] **Label paint + restart** — paint, quit, relaunch, labels
      persist (atomic os.replace path on NTFS).
- [ ] **v2 label migration** — if this machine has a 1.0 library,
      first launch migrates `labels\` to per-namespace subdirs;
      verify counts + `labels\.v2-migrated` marker.
- [ ] **Projects** — create, duplicate (labels copied), delete.
- [ ] **Light Table INFER button** renders (idle constellation) even
      without a model.
- [ ] **Training env** (optional, needs NVIDIA): follow the Windows
      recipe in docs/training_setup.md; note the 1.1 additions —
      peft + wandb now required, `amp_dtype=bfloat16` default,
      `tools/train_until_done.sh` is bash-only (use WSL or run
      launch_training.py directly; a .ps1 port is welcome).
      Set `LITHIUM_TRAIN_PYTHON` instead of --python-exe.
- [ ] **CLI render** — `python -m src.main render <ply> --output out\`.

Known Windows-specific code paths to eyeball if something breaks:
`src/main.py` `_apply_dark_title_bar` / AppUserModelID block,
`src/data/catalog_lock.py` (file locking), `model_registry._path`
(filename sanitization), `num-worker 2` default in launch_training
(Windows DataLoader IPC deadlock — keep it).

## 3. Build the installer

```bat
pip install pyinstaller
pyinstaller lithium.spec --noconfirm     & rem dist\Lithium.exe (onefile)
rem Install Inno Setup 6 (winget install JRSoftware.InnoSetup), then:
iscc packaging\lithium.iss               & rem dist\LithiumSetup-1.1.0.exe
```

Smoke the installer in a Windows Sandbox if available: install, launch,
import a PLY, uninstall cleanly.

## 4. Report back

Fix what you find (commit with the repo's conventional style), and
record verification results in this file under a "## Results" heading.
CI (.github/workflows/release.yml) builds the same artifacts on tag
push — your local build validates the recipe before the first tag.
