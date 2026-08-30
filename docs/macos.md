# Lithium on macOS (Apple Silicon or Intel)

The renderer needs an OpenGL **4.1 core** context — the last version macOS ships —
and every shader now targets `#version 410`. macOS is a **view / annotate**
platform: training and INFER need CUDA and stay on the Linux workstation.

## Run from source (10 minutes)

```bash
xcode-select --install                      # compilers for the one package without an arm64 wheel
brew install uv git                         # or: curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/saffffari/Lithium.git ~/Lithium && cd ~/Lithium
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -r requirements-lock.txt   # pyimgui builds from source here (~2 min)
./run.sh                                     # or: .venv/bin/python -m src.main
```

If `pyopengl-accelerate` refuses to build, drop it: `uv pip install -p .venv/bin/python
--no-deps pyopengl` is enough — it is an optional speed-up.

Open a bone directly: `./run.sh path/to/vertebra.ply`. Open a project:
`./run.sh --project "Deepfield · Gold247 · 6-class [LOCKED]" --cloud 3 --tab light`.

## Bring the catalog over

The catalog is a directory. From the Linux workstation, with SSH to the Mac working:

```bash
rsync -a --info=progress2 ~/.lithium/ sprinta:~/.lithium/      # ~2 GB
```

Model checkpoints referenced by the registry live on the workstation; on the Mac
the INFER picker simply shows no model. Labels, projects, previews and the
gallery all work.

## What to expect

- Window creation asks for 4.1 core + forward-compatible; Cocoa gives a Retina
  framebuffer (2× the window size) — the app reads the framebuffer size, so
  HiDPI is handled.
- First launch builds gallery previews in the background.
- Unsigned: if you run a packaged `.app` from the CI instead of source,
  right-click → Open the first time, or `xattr -dr com.apple.quarantine Lithium.app`.

## Packaged app

`.github/workflows/release.yml` has a `macos` job (Apple Silicon runner) that
runs the tests and PyInstaller and attaches `Lithium-macos-arm64.zip` to the
GitHub release for every `v*` tag. It is unsigned (no Apple Developer ID yet).
