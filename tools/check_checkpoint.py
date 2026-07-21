#!/usr/bin/env python
"""Sanity-check a PTv3 checkpoint: loadable, finite, sane magnitudes.

Exit 0 = clean, 1 = corrupt/suspect. Used by train_until_done.sh to
refuse warm-starting from a poisoned checkpoint (a crash mid-step can
leave model_last.pth with NaN/Inf weights — training warm-started from
one collapses to single-class predictions and never recovers).

Usage: check_checkpoint.py <ckpt.pth> [--python-exe PATH]
The torch env differs from the viz venv; this script re-execs itself
under --python-exe when torch isn't importable.
"""

import argparse
import os
import subprocess
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--python-exe", default="")
    args = ap.parse_args()

    try:
        import torch
    except ModuleNotFoundError:
        if not args.python_exe:
            print("torch unavailable and no --python-exe given")
            return 2
        r = subprocess.run([args.python_exe, os.path.abspath(__file__),
                            args.ckpt])
        return r.returncode

    try:
        blob = torch.load(args.ckpt, map_location="cpu",
                          weights_only=False)
    except Exception as e:
        print(f"BAD {args.ckpt}: unreadable: {type(e).__name__}: {e}")
        return 1
    sd = blob.get("state_dict", blob) if isinstance(blob, dict) else None
    if not isinstance(sd, dict) or not sd:
        print(f"BAD {args.ckpt}: no state_dict")
        return 1
    n_params = 0
    worst = 0.0
    worst_stat = 0.0
    for k, v in sd.items():
        if not torch.is_tensor(v) or not v.is_floating_point():
            continue
        n_params += v.numel()
        if not torch.isfinite(v).all():
            print(f"BAD {args.ckpt}: non-finite values in {k}")
            return 1
        # BN running stats track raw activation statistics and can be
        # legitimately huge (mm-scale coords); magnitude-gate only the
        # learned parameters.
        if k.endswith(("running_mean", "running_var")):
            worst_stat = max(worst_stat, float(v.abs().max()))
        else:
            worst = max(worst, float(v.abs().max()))
    if n_params == 0:
        print(f"BAD {args.ckpt}: zero float params")
        return 1
    if worst > 1e4:
        print(f"BAD {args.ckpt}: exploded weights (max |w| = {worst:.3g})")
        return 1
    epoch = blob.get("epoch", "?") if isinstance(blob, dict) else "?"
    best = blob.get("best_metric_value", "?") if isinstance(blob, dict) else "?"
    print(f"OK {os.path.basename(args.ckpt)}: {n_params:,} params, "
          f"max|w| {worst:.3g}, epoch {epoch}, best {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
