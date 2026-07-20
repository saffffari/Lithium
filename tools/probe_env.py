"""Probe a Python env for PTv3 dependencies.

Writes the result to stdout so the caller can pipe it wherever.
Never raises — always reports a status per module.
"""
import sys
print(f"python: {sys.version.split()[0]}")
mods = [
    'torch', 'torchvision',
    'torch_scatter', 'torch_cluster', 'torch_sparse',
    'spconv', 'pointcept', 'timm', 'open3d', 'einops',
    'h5py', 'addict', 'yapf', 'termcolor',
    'tensorboard', 'tensorboardX', 'ninja',
    'ftfy', 'regex', 'tqdm', 'sharedarray',
]
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, '__version__', '?')
        print(f'  OK       {m:20s} {v}')
    except Exception as e:
        print(f'  MISSING  {m:20s} ({type(e).__name__})')
try:
    import torch
    print(f"torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch CUDA device:    {torch.cuda.get_device_name(0)}")
        print(f"torch CUDA version:   {torch.version.cuda}")
except Exception as e:
    print(f"torch probe failed: {e}")
