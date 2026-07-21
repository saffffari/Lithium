#!/usr/bin/env python
"""Headless PT-v3 training launcher — same path as the GUI's LAUNCH
TRAINING button, just driveable from a script for overnight automation.

Generates a config from the dataset's classes.json + the training params
defaults (with auto class_weights), spawns the Pointcept subprocess via
``PointceptRunner``, blocks until it exits or a timeout hits.

Usage:
    python tools/launch_training.py \\
        --dataset D:/3Photon/dataset \\
        --epochs 200
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.training.config_gen import PTv3TrainParams, generate_ptv3_config


for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-worker", type=int, default=2,
                    help="DataLoader workers. Default 2 on Windows to "
                         "avoid the multi-worker IPC deadlock that "
                         "freezes long-running runs at random iterations. "
                         "Set to 0 for absolute reliability at ~15%% "
                         "throughput cost.")
    ap.add_argument("--run-name", default=None,
                    help="custom run dir name; default uses timestamp")
    ap.add_argument("--python-exe",
                    default=r"C:/Users/alex/miniforge3/envs/3photon-ptv3/python.exe")
    ap.add_argument("--pointcept-dir", default=str(ROOT / "training" / "pointcept"))
    ap.add_argument("--pointcept-ext-dir",
                    default=str(ROOT / "src" / "training" / "pointcept_ext"))
    # Advanced hyperparameters — match the TRAIN tab's ADVANCED section
    # so headless runs can use the same recipe as interactive ones.
    ap.add_argument("--base-lr", type=float, default=0.006)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=10)
    ap.add_argument("--drop-path", type=float, default=0.3)
    ap.add_argument("--minority-boost", type=float, default=1.0,
                    help="Multiply the LAST 2 class weights (pedicles "
                         "in the 5/6-class spinal ontology). 1.0 = no boost.")
    ap.add_argument("--weight", default="",
                    help="Checkpoint to warm-start from (fine-tune / "
                         "crash recovery). Same field the TRAIN tab's "
                         "fine-tune uses.")
    ap.add_argument("--no-rotate-so3", action="store_true",
                    help="Disable full SO(3) rotation augmentation. "
                         "Falls back to z-axis-only rotation so absolute "
                         "Z=up is preserved as a model cue. Right for "
                         "consistently-oriented supine CT data.")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    classes_json = dataset / "classes.json"
    with open(classes_json) as f:
        meta = json.load(f)
    num_classes = int(meta["num_classes"])
    print(f"Dataset: {dataset}  num_classes={num_classes} "
          f"({meta.get('class_names', [])})")

    # Auto class_weights — down-weight class 0 (Unlabeled) 10x, then
    # apply minority_boost to the last 2 classes if requested.
    class_weights = (0.1,) + (1.0,) * (num_classes - 1) if num_classes >= 2 else ()
    if args.minority_boost != 1.0 and num_classes >= 3:
        k = min(2, num_classes - 1)
        boosted = list(class_weights)
        for i in range(len(boosted) - k, len(boosted)):
            boosted[i] = boosted[i] * args.minority_boost
        class_weights = tuple(boosted)

    run_name = args.run_name or f"auto_{int(time.time())}"
    work_dir = dataset / "training_runs" / run_name
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "config.py"
    params = PTv3TrainParams(
        data_root=str(dataset),
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        weight=args.weight,
        class_weights=class_weights,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        drop_path=args.drop_path,
        random_rotate_so3=(not args.no_rotate_so3),
    )
    generate_ptv3_config(
        params, args.pointcept_ext_dir, str(config_path))
    print(f"Config: {config_path}")
    print(f"Work:   {work_dir}")
    print(f"Class weights: {class_weights}")

    # Build the bootstrap command exactly the way ptv3_runner does, so we
    # benefit from the registry shim + the worker_init_fn monkey patch.
    plain_args = ["tools/train.py",
                  "--config-file", str(config_path),
                  "--options", f"save_path={work_dir}"]
    ext = args.pointcept_ext_dir.replace("\\", "/")
    argv_repr = ", ".join(repr(a) for a in plain_args)
    bootstrap = "\n".join([
        "import sys, runpy",
        f"sys.path.insert(0, r'{ext}')",
        "import three_photon_dataset",
        "import pointcept.engines.defaults as _pcd",
        "from worker_init import photon_worker_init_fn as _3p_wif",
        "_pcd.worker_init_fn = _3p_wif",
        f"sys.argv = [{argv_repr}]",
        "runpy.run_path(r'tools/train.py', run_name='__main__')",
    ])
    cmd = [args.python_exe, "-u", "-c", bootstrap]

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        args.pointcept_dir + os.pathsep + args.pointcept_ext_dir
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    # UTF-8 in the child so tqdm progress bars don't crash the reader
    # loop below on Windows (cp1252 default).
    env["PYTHONIOENCODING"] = "utf-8"

    # Tee subprocess output to <work_dir>/run.log AND our stdout, so we
    # can leave this hanging in the background overnight and get a
    # complete log either via this script's stdout or via the file the
    # ptv3_runner already writes when launched from the GUI.
    log_path = work_dir / "launch.log"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        logf.write(f"$ {' '.join(repr(c) for c in cmd[:3])}\n\n")
        proc = subprocess.Popen(
            cmd, cwd=args.pointcept_dir, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace",
            bufsize=1, env=env,
        )
        try:
            for line in proc.stdout:
                line = line.rstrip()
                logf.write(line + "\n")
                logf.flush()
                # Mirror only "interesting" lines to stdout to keep the
                # background launcher's output tractable.
                if any(s in line for s in (
                        "Train: [", "Train result", "Val result",
                        "Eval result", "miou", "Best validation",
                        "Error", "Traceback", "AssertionError",
                        "Start Training", "Building", "epoch:")):
                    print(line)
        except Exception as e:
            print(f"[launcher] reader exception: {e}")
        finally:
            proc.wait()
    print(f"\n[launcher] subprocess exited with code {proc.returncode}")
    print(f"[launcher] full log: {log_path}")
    # Propagate the training exit code — retry wrappers and CI need a
    # non-zero exit when the run died. (A crashed run used to exit 0
    # here, which made train_until_done.sh declare victory on failure.)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
