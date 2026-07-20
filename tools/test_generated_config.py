"""End-to-end config generation + Pointcept parse test.

1. Build a fake 3Photon export with classes.json + a scene.
2. Run our config_gen on it.
3. Ask Pointcept to parse that config.
4. Try building the MODEL from the parsed config (no training).

This is the smallest possible test that proves the generator's
output actually works with the installed Pointcept version.
"""
import os
import sys
import tempfile
import shutil
import traceback

# Need 3Photon on sys.path so we can import our generator
sys.path.insert(0, r"D:\3Photon")

import numpy as np

from src.data.labels import LabelRegistry
from src.data.point_cloud import PointCloudData
from src.export.dataset_export import export_dataset
from src.training.config_gen import PTv3TrainParams, generate_ptv3_config


POINTCEPT_DIR = r"D:\3Photon\training\pointcept"
EXT_DIR = r"D:\3Photon\src\training\pointcept_ext"


def make_tiny_export(root: str):
    """Minimal labeled dataset so Pointcept has something to parse."""
    reg = LabelRegistry()
    for i in range(3):
        reg.add_label(f"Part{i}")

    clouds = []
    for _ in range(4):
        n = 1024
        positions = np.random.rand(n, 3).astype(np.float32)
        colors = (np.random.rand(n, 3) * 255).astype(np.float32)
        cloud = PointCloudData(positions=positions, colors=colors)
        # Give each point some label
        cloud.labels[:340] = 1
        cloud.labels[340:680] = 2
        cloud.labels[680:] = 3
        clouds.append(cloud)

    export_dataset(
        clouds, reg, root, format="ptv3",
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        source_layer="Test",
    )


def main():
    tmp = tempfile.mkdtemp(prefix="3photon_cfg_test_")
    try:
        export_dir = os.path.join(tmp, "dataset")
        os.makedirs(export_dir)
        make_tiny_export(export_dir)
        print(f"fake export at: {export_dir}")

        cfg_path = os.path.join(tmp, "test_config.py")
        params = PTv3TrainParams(
            data_root=export_dir,
            num_classes=0,
            has_color=True,
            has_normal=True,
            batch_size=2,
            epochs=1,
        )
        source = generate_ptv3_config(params, EXT_DIR, cfg_path)
        print(f"config written to: {cfg_path}")
        print(f"config size: {len(source)} bytes")

        # Now ask Pointcept to parse it. Pass options=None so the
        # parser skips the merge_from_dict step (it expects a dict
        # not a list).
        from pointcept.engines.defaults import default_config_parser
        cfg = default_config_parser(cfg_path, None)
        print(f"config parsed OK, num_classes={cfg.model.num_classes}")
        print(f"backbone type={cfg.model.backbone.type}")
        print(f"in_channels={cfg.model.backbone.in_channels}")

        # And try building the model
        from pointcept.models.builder import build_model
        model = build_model(cfg.model)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"PT-v3m1 built OK: {n_params:,} parameters")
        print("SUCCESS")
    except Exception:
        print("FAIL:")
        traceback.print_exc()
        sys.exit(1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
