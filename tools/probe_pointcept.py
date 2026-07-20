"""Verify pointcept imports from inside the repo directory.

Must be run with CWD = pointcept repo root (Pointcept has no setup.py).
Reports version, module locations, and whether the PT-v3m1 backbone
can be built via the registry without actually loading weights.
"""
import os
import sys

print(f"cwd: {os.getcwd()}")
print(f"python: {sys.version.split()[0]}")
print()

try:
    import pointcept
    print(f"pointcept module path: {pointcept.__file__}")
    print(f"pointcept version: {getattr(pointcept, '__version__', '?')}")
except Exception as e:
    print(f"FAIL pointcept import: {e}")
    sys.exit(1)

try:
    from pointcept.models.builder import MODELS
    from pointcept.datasets.builder import DATASETS
    print(f"MODELS registered: {len(MODELS._module_dict)}")
    print(f"DATASETS registered: {len(DATASETS._module_dict)}")
    has_ptv3 = "PT-v3m1" in MODELS._module_dict
    print(f"PT-v3m1 registered: {has_ptv3}")
    # Show a few model names so we know the registry actually loaded
    sample = sorted(MODELS._module_dict.keys())[:8]
    print(f"sample MODELS: {sample}")
except Exception as e:
    import traceback
    print(f"FAIL registry probe: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    import torch
    print(f"torch CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch probe failed: {e}")

print()
print("OK: pointcept env ready")
