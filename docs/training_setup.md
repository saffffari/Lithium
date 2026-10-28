# Training setup

Lithium can train Point Transformer V3 (PT-v3) segmentation models with
[Pointcept](https://github.com/Pointcept/Pointcept) and run them on your
clouds. This page sets up the environment that does that work. You only need
it for training and inference; viewing and labelling never use it.

## How it fits together

The viewer's `.venv` has no PyTorch on purpose. Training and inference run in
a **separate conda environment** with PyTorch, CUDA and Pointcept. Lithium
starts that environment's Python as a subprocess
(`src/training/ptv3_runner.py` for training, `tools/infer_single.py` for
inference) and reads its output. Machine-learning packages can change versions
without ever breaking the viewer.

You need:

- An NVIDIA GPU with a current driver. A 24 GB card trains the default model
  at batch size 2; smaller cards may need a smaller batch.
- A **source checkout** of Lithium. Prebuilt builds do not include the helper
  scripts the subprocesses run.
- [Miniforge](https://github.com/conda-forge/miniforge) (or Miniconda).
- A C++ toolchain to build Pointcept's `pointops` CUDA extension. On Linux the
  recipe installs one into the environment; on Windows you install Visual
  Studio Build Tools.

The recipes below were last checked with Python 3.11, PyTorch 2.5.1 and
CUDA 12.4. Pointcept's main branch moves; if a newer clone breaks something,
the verification step at the end will show it.

## Where Pointcept goes

Clone Pointcept to **`<repo>/training/pointcept`**, where `<repo>` is your
Lithium checkout. `training/` is in `.gitignore`, so the clone never enters
Lithium's history.

Use this exact location. Inference (`tools/infer_single.py`, behind the INFER
button and **File > Predict Current Cloud**) looks for Pointcept there no
matter what you enter in the app, and `tools/launch_training.py` and
`tools/test_generated_config.py` default to it.

```bash
cd <repo>
git clone https://github.com/Pointcept/Pointcept.git training/pointcept
```

## Linux

The system CUDA toolkit may be newer than the one PyTorch was built with, and
the `pointops` build refuses a major-version mismatch. So this recipe puts
`nvcc` 12.4 and `gcc` 13 inside the environment.

```bash
# Miniforge, user-level (skip if you already have conda)
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh
bash miniforge.sh -b -p ~/miniforge3

~/miniforge3/bin/conda create -n lithium-ptv3 python=3.11 -y
ENVP=~/miniforge3/envs/lithium-ptv3

# PyTorch
$ENVP/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# nvcc 12.4 and gcc 13 inside the environment, for the pointops build
~/miniforge3/bin/conda install -n lithium-ptv3 \
    -c nvidia/label/cuda-12.4.1 cuda-toolkit -y
~/miniforge3/bin/conda install -n lithium-ptv3 \
    -c conda-forge gcc_linux-64=13 gxx_linux-64=13 -y

# The rest of Pointcept's dependencies
$ENVP/bin/pip install torch_scatter torch_cluster torch_sparse \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
$ENVP/bin/pip install spconv-cu124 torch-geometric ninja timm addict \
    yapf termcolor tensorboard tensorboardX ftfy regex tqdm einops \
    h5py open3d SharedArray peft wandb
```

Build `pointops` with the environment's compilers. Set
`TORCH_CUDA_ARCH_LIST` to your GPU's compute capability, for example `8.6` for
an RTX 30-series card or `8.9` for an RTX 40-series card.
`$ENVP/bin/python -c "import torch; print(torch.cuda.get_device_capability())"`
prints it.

```bash
cd <repo>/training/pointcept/libs/pointops
env CUDA_HOME=$ENVP \
    CC=$ENVP/bin/x86_64-conda-linux-gnu-gcc \
    CXX=$ENVP/bin/x86_64-conda-linux-gnu-g++ \
    TORCH_CUDA_ARCH_LIST="8.9" \
    $ENVP/bin/python setup.py install
```

The Python to give Lithium is `~/miniforge3/envs/lithium-ptv3/bin/python`.

## Windows

You need, before you start:

- The [CUDA Toolkit 12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive),
  installed in its default location.
- Visual Studio 2022 Build Tools with the MSVC C++ compiler.
- Miniforge or Miniconda on `PATH`.

Run these in a Miniforge Prompt (or any shell where `conda` works):

```bat
conda create -n lithium-ptv3 python=3.11 -y
conda activate lithium-ptv3

conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install pytorch-scatter pytorch-cluster pytorch-sparse -c pyg -y

pip install spconv-cu124 torch-geometric
pip install ninja timm addict yapf termcolor tensorboard tensorboardX
pip install ftfy regex tqdm open3d einops h5py peft wandb
```

If conda cannot find `pytorch-sparse`, install it from the PyG wheel index
instead:

```bat
pip install torch_sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

Leave out `SharedArray`. It does not build on Windows, and Pointcept and
Lithium's dataset class work without it.

### Patch `pointops` for Windows

`training\pointcept\libs\pointops\setup.py` assumes Linux and fails on Windows
with `'NoneType' object has no attribute 'split'`. If you see that, guard the
`OPT` block:

```python
from distutils.sysconfig import get_config_vars
(opt,) = get_config_vars("OPT")
if opt is not None:                     # None on Windows
    os.environ["OPT"] = " ".join(
        flag for flag in opt.split() if flag != "-Wstrict-prototypes"
    )
```

Reapply the patch after each `git pull` in the Pointcept clone until upstream
fixes it.

### Build `pointops`

```bat
cd <repo>\training\pointcept\libs\pointops
python setup.py install
```

This takes a few minutes and ends with `Successfully installed pointops`.

The Python to give Lithium is
`<miniforge>\envs\lithium-ptv3\python.exe`.

## Other Pointcept extensions

Pointcept ships more CUDA extensions under `libs/` (`pointgroup_ops`,
`pointops2`, `pointrope`, `pointseg`). PT-v3 only needs `pointops`. If
`pointrope` is missing, Pointcept prints a one-line warning and falls back to a
pure-Python version.

`flash-attn` is optional too. Lithium's generated configs turn it off
(`enable_flash=False`).

## Verify

Both checks run from the Pointcept folder with it on `PYTHONPATH`, using the
training environment's Python.

Linux:

```bash
cd <repo>/training/pointcept
PYTHONPATH=$PWD $ENVP/bin/python ../../tools/probe_pointcept.py
PYTHONPATH=$PWD $ENVP/bin/python ../../tools/test_generated_config.py
```

Windows:

```bat
cd <repo>\training\pointcept
set PYTHONPATH=%CD%
python ..\..\tools\probe_pointcept.py
python ..\..\tools\test_generated_config.py
```

`probe_pointcept.py` checks that Pointcept imports, that the `PT-v3m1` model
is registered and that CUDA works. Look for these lines; both values must be
`True`:

```
PT-v3m1 registered: True
torch CUDA: True
device: <your GPU>
```

It prints `OK: pointcept env ready` at the end even when CUDA is missing, so
check the lines above it.

`test_generated_config.py` builds a tiny fake Lithium export, generates a
config from it, has Pointcept parse it and builds the model. It ends with:

```
PT-v3m1 built OK: <N> parameters
SUCCESS
```

## Point Lithium at the environment

In Lithium, open **TRAIN > POINTCEPT ENV** and fill in:

- **python exe:** the environment's Python, from the Linux or Windows section
  above.
- **pointcept dir:** `<repo>/training/pointcept`.

Lithium saves both in `~/.lithium/prefs.json` as `train_python_exe` and
`train_pointcept_dir`. You can also edit that file directly while Lithium is
closed.

Then follow [Training and inference](user_guide.md#training-and-inference) in
the user guide.

## Train without the window

`tools/launch_training.py` runs the same launch as the **LAUNCH TRAINING**
button and waits for it to finish:

```bash
python tools/launch_training.py --dataset <exported dataset folder> \
    --python-exe <training env python> --epochs 200
```

Instead of `--python-exe`, you can set `LITHIUM_TRAIN_PYTHON`. Run it with
`--help` for the other options (batch size, learning rate, fine-tuning from a
checkpoint, and so on).

The console on the TRAIN tab has a `train` command too:
`train <dataset folder> [--epochs N] [--batch B]`.

To drive it from Python:

```python
from src.training.config_gen import PTv3TrainParams, generate_ptv3_config
from src.training.ptv3_runner import PointceptRunner, PointceptLaunchConfig

run_dir = "<dataset>/training_runs/my_run"
params = PTv3TrainParams(data_root="<dataset>",
                         num_classes=0,      # 0 = read classes.json
                         batch_size=2, epochs=200)
generate_ptv3_config(params, "<repo>/src/training/pointcept_ext",
                     f"{run_dir}/config.py")

cfg = PointceptLaunchConfig(
    python_exe="<training env python>",
    pointcept_dir="<repo>/training/pointcept",
    config_file=f"{run_dir}/config.py",
    work_dir=run_dir,
    pointcept_ext_dir="<repo>/src/training/pointcept_ext",
)
PointceptRunner().launch(cfg)
```

## Launch parameters that break runs

Each of these has broken a run before:

- **`pointcept_ext_dir` must be set** on `PointceptLaunchConfig`. Without it,
  `PointceptRunner._build_command` skips the bootstrap that registers
  Lithium's dataset class, and the data loader fails with
  `KeyError('LithiumDataset')`. The LAUNCH TRAINING button, the console `train`
  command and `tools/launch_training.py` all set it.
- **`grid_size` is in the cloud's own units.** The default, `0.5`, suits
  clouds stored in millimetres. Pointcept's upstream default assumes metres
  and fails its depth assertion on small objects.
- **Keep `extent / grid_size` below 65536**, where extent is the cloud's
  largest dimension. Otherwise Pointcept's depth assertion fails.
- **`batch_size=2`** is the default because larger batches run out of memory
  on a 24 GB card with the default model.
- **`PYTHONIOENCODING=utf-8`** is set for every subprocess. Without it,
  Windows' default code page cannot decode tqdm's output and the reader thread
  crashes.
- **Environment name:** the tools and docs assume `lithium-ptv3`. Any name
  works as long as POINTCEPT ENV points at its Python.
