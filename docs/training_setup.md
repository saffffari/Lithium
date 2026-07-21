# Training Environment Setup

3Photon's viz app is torch-free on purpose — the main `.venv` only
needs numpy, moderngl, imgui, laspy, plyfile. Training happens in a
**separate conda env** that has PyTorch, CUDA, and Pointcept, and
3Photon subprocesses into it via `src/training/ptv3_runner.py`. This
keeps the viz env from breaking every time a ML dependency drifts.

This document is the reproducible recipe for that training env.

## Linux quick path (verified 2026-07-20, RTX 4090, driver 610)

The system CUDA toolkit may be too new for the pinned torch (nvcc 13.x
vs torch cu12.4 → extension build fails on the major-version check),
so nvcc 12.4 and gcc 13 live inside the env:

```bash
# Miniforge (user-level)
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh
bash /tmp/miniforge.sh -b -p ~/miniforge3

~/miniforge3/bin/conda create -n 3photon-ptv3 python=3.11 -y
ENVP=~/miniforge3/envs/3photon-ptv3

# Pinned known-good torch
$ENVP/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# nvcc 12.4 + gcc 13 inside the env (for the pointops build)
~/miniforge3/bin/conda install -n 3photon-ptv3 \
    -c nvidia/label/cuda-12.4.1 cuda-toolkit -y
~/miniforge3/bin/conda install -n 3photon-ptv3 \
    -c conda-forge gcc_linux-64=13 gxx_linux-64=13 -y

# Remaining deps (peft + wandb are new hard imports in the pinned
# Pointcept clone; SharedArray builds fine on Linux)
$ENVP/bin/pip install torch_scatter torch_cluster torch_sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
$ENVP/bin/pip install spconv-cu124 torch-geometric ninja timm addict \
    yapf termcolor tensorboard tensorboardX ftfy regex tqdm einops \
    h5py open3d SharedArray peft wandb

# Build pointops against the env's nvcc + gcc
cd training/pointcept/libs/pointops
env CUDA_HOME=$ENVP CC=$ENVP/bin/x86_64-conda-linux-gnu-gcc \
    CXX=$ENVP/bin/x86_64-conda-linux-gnu-g++ \
    TORCH_CUDA_ARCH_LIST="8.9" $ENVP/bin/python setup.py install

# Verify (from training/pointcept)
env PYTHONPATH=$PWD $ENVP/bin/python ../../tools/probe_pointcept.py
env PYTHONPATH=$PWD:../.. $ENVP/bin/python ../../tools/test_generated_config.py
```

Then set `train_python_exe` / `train_pointcept_dir` in
`~/.3photon/prefs.json` (or via TRAIN tab → POINTCEPT ENV) to
`$ENVP/bin/python` and `<repo>/training/pointcept`.

The Windows recipe below remains valid for the Windows box.

## Prerequisites

- Windows 10/11 (Linux works too, steps are similar)
- NVIDIA GPU with recent driver (tested on RTX 4090, driver 591.86)
- CUDA toolkit 12.4 installed at
  `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4`
  (needed for compiling the CUDA extensions in `libs/pointops`)
- Miniforge or Miniconda already on PATH
- Visual Studio 2022 Build Tools with MSVC C++ compiler (for pointops)

## Step 1 — Conda env

```cmd
conda create -n 3photon-ptv3 python=3.11 -y
conda activate 3photon-ptv3
```

## Step 2 — PyTorch + CUDA 12.4

```cmd
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Verify:
```cmd
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Should print `2.5.x True NVIDIA GeForce RTX 4090` (or your GPU).

## Step 3 — PyG trio via conda (the Windows-friendly way)

```cmd
conda install pytorch-scatter pytorch-cluster pytorch-sparse -c pyg -y
```

If `pytorch-sparse` isn't available via conda on your platform, fall
back to the PyG wheel index:
```cmd
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

## Step 4 — Remaining pip deps

```cmd
pip install spconv-cu124
pip install torch-geometric
pip install ninja timm addict yapf termcolor tensorboard tensorboardX
pip install ftfy regex tqdm open3d einops h5py
```

`sharedarray` is listed in Pointcept's README but fails to build on
Windows (uses POSIX shared memory). It's optional — Pointcept falls
back to standard caching when it's missing.

## Step 5 — Clone Pointcept

Pointcept is cloned as a sibling inside the project:

```cmd
cd D:\3Photon
git clone https://github.com/Pointcept/Pointcept.git training\pointcept
```

Note: `training/` is in `.gitignore`. The clone is a local-only
sibling — it never enters 3Photon's git history.

## Step 6 — Patch `libs/pointops/setup.py` on Windows

Pointcept's `libs/pointops/setup.py` has a Linux-only assumption that
crashes on Windows:

```python
from distutils.sysconfig import get_config_vars
(opt,) = get_config_vars("OPT")         # returns None on Windows
os.environ["OPT"] = " ".join(           # crashes: None has no split
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)
```

Edit `training/pointcept/libs/pointops/setup.py` to guard the block:

```python
from distutils.sysconfig import get_config_vars
(opt,) = get_config_vars("OPT")
if opt is not None:                     # Windows: OPT is None, skip
    os.environ["OPT"] = " ".join(
        flag for flag in opt.split() if flag != "-Wstrict-prototypes"
    )
```

After an upstream `git pull` in the Pointcept clone this patch will
need to be reapplied until the upstream fix lands (or we submit it).

## Step 7 — Build `pointops` CUDA extension

```cmd
cd D:\3Photon\training\pointcept\libs\pointops
python setup.py install
```

This compiles the CUDA kernels against the current PyTorch's
`torch_cuda_arch_list`. Takes 1-3 minutes on a warm VS Build Tools
cache. You should see `Successfully installed pointops-1.0` at the end.

**Note:** Pointcept bundles a few other CUDA extensions under `libs/`
(`pointgroup_ops`, `pointops2`, `pointrope`, `pointseg`). We only need
`pointops` for PT-v3m1. The others are only required if you train
different models. `pointrope` missing is harmless — Pointcept falls
back to a pure-Python implementation and logs a one-line warning.

## Step 8 — Verify Pointcept imports

Pointcept has no `setup.py` at its root; their install pattern is
`PYTHONPATH=./ python tools/train.py ...`. Our `ptv3_runner.py` sets
this automatically, but to verify the env manually:

```cmd
cd D:\3Photon\training\pointcept
set PYTHONPATH=%CD%
python D:\3Photon\tools\probe_pointcept.py
```

Expected output:
```
pointcept module path: D:\3Photon\training\pointcept\pointcept\__init__.py
MODELS registered: 60
DATASETS registered: 31
PT-v3m1 registered: True
torch CUDA: True
device: NVIDIA GeForce RTX 4090
OK: pointcept env ready
```

## Step 9 — End-to-end smoke test

Generates a fake 3Photon export, runs our config generator on it,
feeds the config to Pointcept, and instantiates PT-v3m1:

```cmd
cd D:\3Photon\training\pointcept
set PYTHONPATH=%CD%
python D:\3Photon\tools\test_generated_config.py
```

Expected final line:
```
PT-v3m1 built OK: 46,178,467 parameters
SUCCESS
```

## Running a real training run

Once the env is set up, the normal flow is:

1. Label point clouds in 3Photon.
2. Export a dataset via `export_dataset(...)` — writes the
   `train/val/test/scene_*/` tree plus `classes.json`.
3. Generate a Pointcept config:
   ```python
   from src.training.config_gen import PTv3TrainParams, generate_ptv3_config
   params = PTv3TrainParams(data_root="D:/3Photon/exports/my_run",
                            num_classes=0,  # auto-load from classes.json
                            batch_size=32, epochs=200)
   generate_ptv3_config(params,
                        r"D:\3Photon\src\training\pointcept_ext",
                        r"D:\3Photon\training\runs\my_run\config.py")
   ```
4. Launch:
   ```python
   from src.training.ptv3_runner import PointceptRunner, PointceptLaunchConfig
   cfg = PointceptLaunchConfig(
       python_exe=r"C:\Users\<you>\miniforge3\envs\3photon-ptv3\python.exe",
       pointcept_dir=r"D:\3Photon\training\pointcept",
       config_file=r"D:\3Photon\training\runs\my_run\config.py",
       work_dir=r"D:\3Photon\training\runs\my_run",
   )
   PointceptRunner().launch(cfg)
   ```
5. After training, take the best checkpoint + a prediction `.npy`
   from a test run and feed them to `import_predictions()` back in
   3Photon to close the loop.

## Known issues

- **pointops `setup.py` Windows patch** — see Step 6. Document this
  or upstream it.
- **pointrope CUDA missing** — Pointcept falls back to PyTorch.
  Harmless for PT-v3m1 since the model doesn't use it. Ignore the
  warning on startup.
- **flash-attn not installed** — optional speed-up. PT-v3m1 works
  fine without it; `enable_flash=False` is the `config_gen.py`
  default.
- **sharedarray not installed** — expected on Windows. Pointcept's
  default datasets have a fallback. Our `ThreePhotonDataset` doesn't
  use sharedarray at all.

## Required launch parameters (regression-prone)

Per `project_ptv3_training_realities` memory and the wave-2 audit
(`_audit/REPORT.md` Theme 5):

- **`pointcept_ext_dir`** must be passed to `PointceptLaunchConfig`. If
  omitted, `_build_command` at `ptv3_runner.py:458` skips the
  `python -u -c <bootstrap>` block and the dataloader fails with
  `KeyError('ThreePhotonDataset')`. The TRAIN-tab button,
  `tools/launch_training.py`, and the CLI `train` subcommand all pass
  it correctly.
- **`grid_size=0.5`** mm (NOT meters — point clouds here are ~50-80 mm
  scale and Pointcept's default 0.001 m fails the depth ≤ 16 assertion).
- **`batch_size=2`** fits VRAM with the standard config.
- **`extent / grid_size < 65536`** or Pointcept's depth assertion fires.
- **`PYTHONIOENCODING=utf-8`** in subprocess env — default cp1252 chokes
  on tqdm em-dashes and crashes the reader thread.
- Conda env name: **`3photon-ptv3`**.
