"""Custom Pointcept dataset for Lithium's exported directory layout.

Pointcept's default datasets (ScanNetDataset, S3DISDataset, etc.) all
assume their own specific preprocessing output. Rather than pretending
we're one of them, we register our own dataset class that reads the
tree produced by ``src/export/dataset_export.py``:

    data_root/
        classes.json
        splits.json
        train/
            scene_000/
                coord.npy      (N, 3)  float32
                color.npy      (N, 3)  uint8
                normal.npy     (N, 3)  float32   [optional]
                segment.npy    (N,)    int64     [IGNORE_INDEX=255 for unlabeled]
                feat.npy       (N, F)  float32   [optional]
        val/
        test/

This file is imported ONLY by the training subprocess (the viz app's
main venv has no torch / pointcept). At import time it registers
``LithiumDataset`` with Pointcept's ``DATASETS`` registry, so any
generated config with ``type="LithiumDataset"`` will instantiate
it. Side-effect imports are the convention Pointcept uses throughout
its own datasets.

Not imported by Lithium's viz code. Not covered by the main test
suite (would require installing torch + pointcept in the main venv).
Tested end-to-end once an actual training env exists.
"""

import glob
import os

import numpy as np

# Pointcept imports. These will fail in the viz env — that's the
# whole point of keeping this file out of the main import path.
# The training env is responsible for having pointcept on PYTHONPATH
# before this module is imported.
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.defaults import DefaultDataset
from pointcept.datasets.transform import TRANSFORMS


@TRANSFORMS.register_module()
class RelabelSegment(object):
    """Remap integer class IDs in the segment array.

    Used for ablations that merge or drop classes without rewriting the
    on-disk dataset. ``mapping`` is a dict of {old_id: new_id}; any id
    not in the mapping is left unchanged.

    Example: ``mapping={4: 3}`` merges class 4 into class 3.
    Example: ``mapping={4: 255}`` drops class 4 to ignore_index.
    """

    def __init__(self, mapping=None):
        self.mapping = {int(k): int(v) for k, v in (mapping or {}).items()}

    def __call__(self, data_dict):
        if "segment" not in data_dict or not self.mapping:
            return data_dict
        seg = data_dict["segment"]
        for old, new in self.mapping.items():
            if old == new:
                continue
            seg = np.where(seg == old, new, seg)
        data_dict["segment"] = seg.astype(np.int32, copy=False)
        return data_dict


@DATASETS.register_module()
class LithiumDataset(DefaultDataset):
    """Reads Lithium's per-scene directory layout.

    Each scene is a folder containing ``coord.npy`` / ``color.npy``
    / ``segment.npy`` and optionally ``normal.npy`` and ``feat.npy``.
    Missing optional files are simply not added to the returned dict;
    Pointcept's Collect transform filters on ``feat_keys`` so any
    feature that isn't listed in the generated config is never touched.

    ``ignore_index`` is configurable to match whatever was used at
    export time — Lithium writes 255 by default (see
    ``src/export/dataset_export.py::IGNORE_INDEX``) but the dataset
    class doesn't hard-code it so an old export with a different
    convention still loads.
    """

    class2id = None  # populated lazily from classes.json
    VALID_ASSETS = ["coord", "color", "normal", "segment", "feat"]

    def __init__(self,
                 split="train",
                 data_root=None,
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1,
                 ignore_index=255,
                 **kwargs):
        self.ignore_index = int(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_data_list(self):
        """Return absolute paths to every scene directory in the split."""
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.isdir(split_dir):
            # Return an empty list rather than raising so an empty
            # test split (valid!) doesn't kill training at startup.
            return []
        scene_dirs = sorted(glob.glob(os.path.join(split_dir, "scene_*")))
        return [d for d in scene_dirs if os.path.isdir(d)]

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)])

    def get_data(self, idx):
        """Load one scene and return a dict matching Pointcept's contract.

        Pointcept's transforms expect keys:
          coord (N, 3) float32, optionally color/normal/segment/grid_coord.
        We read everything the scene has on disk and let the generated
        transform pipeline's ``Collect`` step pick which ones end up
        as the model's ``feat`` tensor.
        """
        scene_dir = self.data_list[idx % len(self.data_list)]

        coord = np.load(os.path.join(scene_dir, "coord.npy"))
        coord = np.ascontiguousarray(coord, dtype=np.float32)

        result = {
            "coord": coord,
            "name": os.path.basename(scene_dir),
        }

        color_path = os.path.join(scene_dir, "color.npy")
        if os.path.isfile(color_path):
            color = np.load(color_path)
            # Lithium writes uint8 [0,255]. Pointcept's NormalizeColor
            # transform expects float in that range, so pass through
            # as float32 — the scale happens inside Normalize.
            result["color"] = np.ascontiguousarray(color, dtype=np.float32)

        normal_path = os.path.join(scene_dir, "normal.npy")
        if os.path.isfile(normal_path):
            result["normal"] = np.ascontiguousarray(
                np.load(normal_path), dtype=np.float32)

        segment_path = os.path.join(scene_dir, "segment.npy")
        if os.path.isfile(segment_path):
            seg = np.load(segment_path)
            # PTv3 training expects int32 segments; most operator
            # code casts to long anyway but we normalize here so the
            # dtype is consistent across scenes.
            result["segment"] = np.ascontiguousarray(seg, dtype=np.int32)
        else:
            # Test-only scenes may not have segments. Create a filler
            # filled with ignore_index so loss masks everything out.
            result["segment"] = np.full(
                coord.shape[0], self.ignore_index, dtype=np.int32)

        feat_path = os.path.join(scene_dir, "feat.npy")
        if os.path.isfile(feat_path):
            result["feat"] = np.ascontiguousarray(
                np.load(feat_path), dtype=np.float32)

        return result
