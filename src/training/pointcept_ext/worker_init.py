"""DataLoader worker init shim for Windows spawn-mode workers.

Windows ``torch.utils.data.DataLoader`` spawns each worker as a fresh
Python interpreter. The main process registers ``ThreePhotonDataset``
via the ``-c`` bootstrap before tools/train.py runs, but workers don't
inherit any of that — they pickle/unpickle the dataset class and then
try to look it up in their own (empty) DATASETS registry.

The runner installs ``photon_worker_init_fn`` (this module) as
Pointcept's ``worker_init_fn`` so each worker re-imports our dataset
module on startup, which fires the ``@DATASETS.register_module()``
side effect. The function lives in a real module (not ``__main__``)
so PyTorch can pickle it across the spawn boundary by qualified name.

The original Pointcept worker_init handles seed-setting; we delegate
to it after our own import to preserve that behaviour.
"""

from __future__ import annotations


def photon_worker_init_fn(worker_id: int, num_workers: int,
                          rank: int, seed) -> None:
    """Re-import the ThreePhotonDataset class then run Pointcept's seed init."""
    # Importing this module already implies pointcept_ext is on
    # ``sys.path`` (otherwise the worker couldn't have unpickled this
    # function in the first place). So the import below is a plain
    # named import — no path mutation needed.
    import three_photon_dataset  # noqa: F401 — registers ThreePhotonDataset
    from pointcept.utils.env import set_seed
    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)
