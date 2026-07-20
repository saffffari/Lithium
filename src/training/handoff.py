"""Export → hand off to a remote GPU box for training.

Mode B from the design: when training doesn't run inside 3Photon
(shared cluster, borrowed workstation, cloud instance), the user
still wants a one-shot "scp this directory and run" package. This
module drops ``train.sh`` + ``train.bat`` + ``README.md`` next to
an exported dataset so that starting training on any Pointcept-
equipped machine is a two-command affair.

Zero 3Photon involvement during the training run itself — the
scripts invoke the user's own Pointcept install and leave
checkpoints and predictions on disk where ``import_predictions``
can find them when the user comes back.

No torch, no pointcept, no subprocesses. Pure file writes.
"""

import os
import json


_README_TEMPLATE = """# 3Photon → PTv3 handoff

This directory is a PTv3/Pointcept-compatible dataset exported from
3Photon, plus a self-contained launcher for training.

## Layout

```
.
├── classes.json        # num_classes, class_names, ignore_index (255)
├── splits.json         # which scene went into train/val/test
├── train/
│   └── scene_000/
│       ├── coord.npy
│       ├── color.npy
│       ├── normal.npy
│       └── segment.npy
├── val/
├── test/
├── three_photon_dataset.py   # custom Pointcept DATASETS entry
├── ptv3_config.py      # generated Pointcept config
├── train.sh            # Linux / WSL / macOS launcher
├── train.bat           # Windows launcher
└── runs/               # checkpoints + logs land here
```

## Prerequisites on the training machine

- Python 3.10+ env with Pointcept, PyTorch (CUDA build), torch_scatter,
  torch_cluster, torch_sparse, spconv-cu12x, timm.
- An NVIDIA GPU with ≥ 8 GB VRAM recommended.

## Run

Linux / WSL / macOS:
```
./train.sh /path/to/python/with/pointcept /path/to/pointcept/repo
```

Windows (cmd / PowerShell):
```
train.bat C:\\path\\to\\python.exe C:\\path\\to\\pointcept
```

Both scripts:

1. Copy ``three_photon_dataset.py`` onto ``PYTHONPATH`` so Pointcept's
   registry picks up our custom dataset.
2. Invoke ``tools/train.py`` with ``ptv3_config.py``.
3. Write checkpoints and logs into ``runs/{num_classes}class-{timestamp}/``.

## After training

Copy the best checkpoint (``runs/.../model_best.pth``) and, after
inference, the per-point prediction ``predictions.npy`` back to your
3Photon project. In 3Photon, use **Import predictions** pointing at
the ``.npy`` and this directory's ``classes.json`` — the registry
will pick up matching class names and colors automatically.

## Dataset metadata

{metadata_block}
"""


_TRAIN_SH_TEMPLATE = """#!/usr/bin/env bash
#
# 3Photon → PTv3 training launcher (Linux / WSL / macOS).
# Usage:   ./train.sh <python_exe_with_pointcept> <pointcept_repo_dir>
#
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <python_exe> <pointcept_dir>"
    echo ""
    echo "Example:"
    echo "  ./train.sh /path/to/conda_env/bin/python \\\\"
    echo "             /path/to/Pointcept"
    exit 1
fi

PY="$1"
PTC="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
RUN_NAME="{run_name}"
WORK_DIR="$HERE/runs/$RUN_NAME"
mkdir -p "$WORK_DIR"

# Put our custom dataset module on PYTHONPATH so the config can
# register it with Pointcept's DATASETS registry at import time.
export PYTHONPATH="$HERE:${{PYTHONPATH:-}}"

cd "$PTC"
"$PY" -u tools/train.py \\\\
    --config-file "$HERE/ptv3_config.py" \\\\
    --options save_path="$WORK_DIR"
"""


_TRAIN_BAT_TEMPLATE = """@echo off
REM 3Photon -> PTv3 training launcher (Windows).
REM Usage:  train.bat <python_exe_with_pointcept> <pointcept_repo_dir>

setlocal enabledelayedexpansion

if "%~2"=="" (
    echo usage: train.bat ^<python_exe^> ^<pointcept_dir^>
    echo.
    echo Example:
    echo   train.bat C:\\path\\to\\conda_env\\python.exe C:\\path\\to\\Pointcept
    exit /b 1
)

set "PY=%~1"
set "PTC=%~2"
set "HERE=%~dp0"
set "RUN_NAME={run_name}"
set "WORK_DIR=%HERE%runs\\%RUN_NAME%"
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"

REM Custom dataset on PYTHONPATH so Pointcept picks up the registry entry.
set "PYTHONPATH=%HERE%;%PYTHONPATH%"

pushd "%PTC%"
"%PY%" -u tools\\train.py --config-file "%HERE%ptv3_config.py" --options save_path="%WORK_DIR%"
popd
"""


def _load_classes(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, "classes.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _format_metadata_block(dataset_dir: str) -> str:
    classes = _load_classes(dataset_dir)
    if not classes:
        return "(no classes.json found)"
    lines = []
    lines.append(f"- num_classes: {classes.get('num_classes', '?')}")
    lines.append(f"- ignore_index: {classes.get('ignore_index', 255)}")
    lines.append(f"- source_layer: {classes.get('source_layer', 'default')}")
    lines.append(f"- flatten_mode: {classes.get('flatten_mode', 'leaves')}")
    names = classes.get("class_names", [])
    if names:
        preview = ", ".join(names[:8])
        more = "" if len(names) <= 8 else f" (+{len(names) - 8} more)"
        lines.append(f"- classes: {preview}{more}")
    return "\n".join(lines)


def write_handoff_scripts(dataset_dir: str,
                          run_name: str = "run_001") -> dict:
    """Write ``train.sh``, ``train.bat``, and ``README.md`` into the export dir.

    Also copies ``three_photon_dataset.py`` from ``pointcept_ext/``
    next to the scripts so the handoff bundle is fully self-contained
    — the user doesn't need 3Photon itself installed on the GPU box,
    just Pointcept.

    Returns a dict of the paths written for the caller to log.
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")

    # Copy the custom dataset module into the handoff bundle.
    src_dataset = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pointcept_ext", "three_photon_dataset.py",
    )
    dst_dataset = os.path.join(dataset_dir, "three_photon_dataset.py")
    if os.path.isfile(src_dataset):
        with open(src_dataset) as f_in, open(dst_dataset, "w") as f_out:
            f_out.write(f_in.read())

    # Templates use literal ``{metadata_block}`` / ``{run_name}`` as
    # placeholders and plain ``str.replace`` rather than ``.format``
    # because the surrounding text contains markdown code blocks with
    # their own literal braces that would otherwise collide with
    # format spec parsing.
    metadata = _format_metadata_block(dataset_dir)
    readme = _README_TEMPLATE.replace("{metadata_block}", metadata)
    train_sh = _TRAIN_SH_TEMPLATE.replace("{run_name}", run_name)
    train_bat = _TRAIN_BAT_TEMPLATE.replace("{run_name}", run_name)

    written: dict[str, str] = {}

    readme_path = os.path.join(dataset_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    written["readme"] = readme_path

    sh_path = os.path.join(dataset_dir, "train.sh")
    with open(sh_path, "w", newline="\n", encoding="utf-8") as f:
        f.write(train_sh)
    try:
        # Mark executable on POSIX; silently ignore on Windows where
        # chmod is a no-op on NTFS.
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
    written["train_sh"] = sh_path

    bat_path = os.path.join(dataset_dir, "train.bat")
    with open(bat_path, "w", newline="\r\n", encoding="utf-8") as f:
        f.write(train_bat)
    written["train_bat"] = bat_path

    if os.path.isfile(dst_dataset):
        written["dataset_module"] = dst_dataset

    return written
