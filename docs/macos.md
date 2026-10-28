# Lithium on macOS

Lithium runs on Apple Silicon and Intel Macs. macOS ships OpenGL 4.1 core as
its last OpenGL version, and every Lithium shader targets exactly that
(`#version 410 core`).

On a Mac you can view and label. Training and inference need an NVIDIA GPU
with CUDA, so they do not run on macOS. Run them on a Windows or Linux machine
and move the project with a `.li` bundle.

## Install from source

The setup script from the [README](../README.md) works on macOS:

```bash
xcode-select --install          # C++ compiler; the imgui package builds from source
git clone https://github.com/saffffari/Lithium.git
cd Lithium
./setup.sh
```

To do the same by hand with [uv](https://docs.astral.sh/uv/) (installed with
`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-annotator.txt
./run.sh
```

Building `imgui` takes a couple of minutes. If `pyopengl-accelerate` fails to
build, leave it out. It is an optional speed-up, and `pyopengl` works alone.

Open a file or a project straight away:

```bash
./run.sh path/to/cloud.ply
./run.sh --project "My project" --cloud 3 --tab light
```

## Bring your work over

The simplest way is a `.li` bundle: **File > Export Project (.li)...** on the
other machine, then **File > Import Project (.li)...** on the Mac. See
[Project bundles](bundles.md).

To copy a whole catalog instead, close Lithium on both machines and copy the
`~/.lithium` folder, for example:

```bash
rsync -a --info=progress2 <other-machine>:~/.lithium/ ~/.lithium/
```

Model checkpoints that the copied registry points at stay on the other
machine, so the Mac shows no models. Labels, projects, previews and the
gallery all work.

## What to expect

- Lithium asks for an OpenGL 4.1 core, forward-compatible context. On a Retina
  display the framebuffer is twice the window size; Lithium reads the real
  framebuffer size, so the interface stays sharp.
- On very high-resolution displays, starting with `LITHIUM_MSAA=0` turns off
  multisampling and can make drawing noticeably faster.
- The first launch builds gallery previews in the background.
- `Cmd` works in place of `Ctrl` for the interface-size shortcuts
  (`Cmd` + `=` / `-` / `0`). Other shortcuts use `Ctrl`.

## Prebuilt app

The release workflow (`.github/workflows/release.yml`) has a macOS job on an
Apple Silicon runner. It runs the tests, builds the app with PyInstaller and
attaches `Lithium-macos-arm64.zip` to a draft GitHub release for every `v*`
tag. It appears on the Releases page once that release is published.

The app is not signed. The first time, right-click it and choose **Open**, or
run `xattr -dr com.apple.quarantine Lithium.app`.
