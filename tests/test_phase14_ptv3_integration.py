"""Phase 14: PTv3 integration — runner, config generator, handoff.

Covers every part of the Pointcept pipeline that does NOT require
torch / pointcept / a GPU: stdout parsing, config generation, handoff
file emission. The custom dataset class in pointcept_ext is not
tested here because it imports pointcept.
"""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.labels import LabelRegistry
from src.data.point_cloud import PointCloudData
from src.export.dataset_export import export_dataset, IGNORE_INDEX
from src.training.ptv3_runner import (
    parse_pointcept_line, ParsedLine,
    PointceptLaunchConfig, PointceptRunner, PointceptStatus,
)
from src.training.config_gen import (
    PTv3TrainParams, generate_ptv3_config, load_num_classes,
)
from src.training.handoff import write_handoff_scripts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _fake_export(tmp: str, num_scenes: int = 4, num_labels: int = 3) -> str:
    reg = LabelRegistry()
    for i in range(num_labels):
        reg.add_label(f"Part{i}")
    clouds = []
    for _ in range(num_scenes):
        n = 500
        positions = np.random.rand(n, 3).astype(np.float32)
        colors = np.random.rand(n, 3).astype(np.float32)
        cloud = PointCloudData(positions=positions, colors=colors)
        for i in range(num_labels):
            lo = int(n * i / (num_labels + 1))
            hi = int(n * (i + 1) / (num_labels + 1))
            cloud.labels[lo:hi] = i + 1
        clouds.append(cloud)
    export_dataset(
        clouds, reg, tmp, format="ptv3",
        train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
        source_layer="Anatomy",
    )
    return tmp


# ---------------------------------------------------------------------------
# Stdout parser
# ---------------------------------------------------------------------------

def test_parse_pointcept_train_step_line():
    line = ("[2024-01-15 14:23:45 main INFO] Train: [12/200][50/150] "
            "Loss 0.3421 (0.3812) Acc 0.8921 (0.8745) Batch 0.123")
    p = parse_pointcept_line(line)
    assert p.kind == "train_step"
    assert p.epoch == 12 and p.total_epochs == 200
    assert p.step == 50 and p.total_steps == 150
    assert abs(p.loss - 0.3421) < 1e-6
    assert abs(p.loss_avg - 0.3812) < 1e-6
    assert abs(p.acc - 0.8921) < 1e-6
    print("PASS: parse train step")


def test_parse_pointcept_val_result_line():
    line = "[2024-01-15 14:30:00 main INFO] Val result: mIoU/mAcc/allAcc 0.6834/0.7123/0.8921"
    p = parse_pointcept_line(line)
    assert p.kind == "val_result"
    assert abs(p.miou - 0.6834) < 1e-6
    assert abs(p.macc - 0.7123) < 1e-6
    assert abs(p.all_acc - 0.8921) < 1e-6
    print("PASS: parse val result")


def test_parse_legacy_in_house_epoch_line():
    """Our existing train_script.py format still matches."""
    line = "epoch: 12/200 loss: 0.4210 miou: 0.5503"
    p = parse_pointcept_line(line)
    assert p.kind == "epoch"
    assert p.epoch == 12 and p.total_epochs == 200
    assert abs(p.loss - 0.4210) < 1e-6
    assert abs(p.miou - 0.5503) < 1e-6
    print("PASS: parse legacy epoch")


def test_parse_checkpoint_save_line():
    line = "[info] Saved checkpoint to runs/scannet/model_best.pth"
    p = parse_pointcept_line(line)
    assert p.kind == "checkpoint"
    assert p.checkpoint_path == "runs/scannet/model_best.pth"
    print("PASS: parse checkpoint")


def test_parse_best_miou_line():
    line = "[info] Best val mIoU: 0.7421"
    p = parse_pointcept_line(line)
    assert p.kind == "best"
    assert abs(p.best_miou - 0.7421) < 1e-6
    print("PASS: parse best miou")


def test_parse_raw_garbage():
    p = parse_pointcept_line("some random stderr from a dependency")
    assert p.kind == "raw"
    assert p.loss is None and p.miou is None
    print("PASS: parse raw garbage")


# ---------------------------------------------------------------------------
# Status application via the runner's internal parser
# ---------------------------------------------------------------------------

def test_runner_applies_train_step_to_status():
    runner = PointceptRunner()
    runner.status = PointceptStatus(running=True)
    parsed = parse_pointcept_line(
        "Train: [3/50][10/20] Loss 0.12 (0.15) Acc 0.94")
    runner._apply_parsed(parsed)
    assert runner.status.current_epoch == 3
    assert runner.status.total_epochs == 50
    assert runner.status.current_step == 10
    assert runner.status.total_steps == 20
    assert abs(runner.status.last_loss - 0.12) < 1e-6
    print("PASS: runner apply train step")


def test_runner_tracks_best_miou_monotonically():
    runner = PointceptRunner()
    runner.status = PointceptStatus(running=True)
    # First val result
    runner._apply_parsed(parse_pointcept_line(
        "Val result: mIoU/mAcc/allAcc 0.45/0.50/0.80"))
    assert abs(runner.status.best_miou - 0.45) < 1e-6
    # Improvement → best updates
    runner._apply_parsed(parse_pointcept_line(
        "Val result: mIoU/mAcc/allAcc 0.60/0.62/0.85"))
    assert abs(runner.status.best_miou - 0.60) < 1e-6
    # Regression → best does NOT update
    runner._apply_parsed(parse_pointcept_line(
        "Val result: mIoU/mAcc/allAcc 0.50/0.55/0.82"))
    assert abs(runner.status.best_miou - 0.60) < 1e-6
    assert abs(runner.status.last_miou - 0.50) < 1e-6
    print("PASS: runner best miou tracking")


def test_runner_log_ring_buffer_trims():
    runner = PointceptRunner()
    runner.status = PointceptStatus(running=True)
    for i in range(1200):
        runner._append_log(f"line {i}")
    assert 800 <= len(runner.status.log_lines) <= 1000
    # Most recent line is preserved
    assert runner.status.log_lines[-1] == "line 1199"
    print("PASS: runner log ring buffer")


def test_runner_launch_with_nonexistent_python_fails_gracefully():
    runner = PointceptRunner()
    cfg = PointceptLaunchConfig(
        python_exe="definitely_not_a_real_interpreter_xyz.exe",
        pointcept_dir=os.path.dirname(os.path.abspath(__file__)),
        config_file="nope.py",
        work_dir=tempfile.mkdtemp(),
    )
    assert not runner.launch(cfg)
    assert runner.status.error  # populated, no crash
    assert not runner.status.running
    print("PASS: runner graceful launch failure")


# ---------------------------------------------------------------------------
# Config generator
# ---------------------------------------------------------------------------

def test_config_gen_reads_num_classes_from_export():
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = _fake_export(tmp, num_scenes=4, num_labels=5)
        n = load_num_classes(export_dir)
        assert n == 6  # 5 real labels + Unlabeled@0
    print("PASS: config_gen load_num_classes")


def test_config_gen_generates_valid_python():
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = _fake_export(tmp, num_scenes=4, num_labels=3)
        ext_dir = os.path.join(tmp, "pointcept_ext")
        os.makedirs(ext_dir)
        out_path = os.path.join(tmp, "ptv3_config.py")

        params = PTv3TrainParams(
            data_root=export_dir,
            num_classes=0,  # auto from classes.json
            has_color=True,
            has_normal=True,
        )
        source = generate_ptv3_config(params, ext_dir, out_path)

        assert os.path.isfile(out_path)
        # Config should be parseable as Python AST — that's a cheap
        # proxy for "not a syntax mess" without actually running it.
        import ast
        tree = ast.parse(source)
        # Must contain the key Pointcept symbols
        assert "DefaultSegmentorV2" in source
        assert "PT-v3m1" in source
        assert 'ignore_index=255' in source
        assert "ThreePhotonDataset" in source
        # num_classes was injected from classes.json (3 real + Unlabeled@0)
        assert "num_classes=4" in source
        # Dataset path is forward-slashed for cross-platform friendliness
        assert "\\\\" not in source.split('"""', 2)[2]  # no backslash paths in body
    print("PASS: config_gen generate_ptv3_config")


def test_config_gen_injects_so3_rotation_for_vertebrae():
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = _fake_export(tmp, num_scenes=2, num_labels=2)
        ext_dir = os.path.join(tmp, "pointcept_ext")
        os.makedirs(ext_dir)
        out_path = os.path.join(tmp, "cfg.py")
        params = PTv3TrainParams(
            data_root=export_dir,
            random_rotate_so3=True,
        )
        source = generate_ptv3_config(params, ext_dir, out_path)
        # Must rotate around all three axes
        assert 'axis="x"' in source
        assert 'axis="y"' in source
        assert 'axis="z"' in source
    print("PASS: config_gen SO3 augmentation")


def test_config_gen_input_channels_match_feature_availability():
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = _fake_export(tmp, num_scenes=2, num_labels=2)
        ext_dir = os.path.join(tmp, "pointcept_ext")
        os.makedirs(ext_dir)

        # xyz only → in_channels=3
        params = PTv3TrainParams(
            data_root=export_dir, has_color=False, has_normal=False)
        src = generate_ptv3_config(params, ext_dir, os.path.join(tmp, "a.py"))
        assert "in_channels=3" in src

        # xyz + color → 6
        params = PTv3TrainParams(
            data_root=export_dir, has_color=True, has_normal=False)
        src = generate_ptv3_config(params, ext_dir, os.path.join(tmp, "b.py"))
        assert "in_channels=6" in src

        # xyz + color + normal → 9
        params = PTv3TrainParams(
            data_root=export_dir, has_color=True, has_normal=True)
        src = generate_ptv3_config(params, ext_dir, os.path.join(tmp, "c.py"))
        assert "in_channels=9" in src
    print("PASS: config_gen in_channels")


# ---------------------------------------------------------------------------
# Handoff scripts
# ---------------------------------------------------------------------------

def test_handoff_writes_all_files():
    with tempfile.TemporaryDirectory() as tmp:
        export_dir = _fake_export(tmp, num_scenes=4, num_labels=3)
        written = write_handoff_scripts(export_dir, run_name="vtbr_001")

        assert os.path.isfile(written["readme"])
        assert os.path.isfile(written["train_sh"])
        assert os.path.isfile(written["train_bat"])
        assert os.path.isfile(written["dataset_module"])

        with open(written["readme"]) as f:
            readme = f.read()
        assert "num_classes: 4" in readme  # 3 real + Unlabeled@0
        assert "source_layer: Anatomy" in readme
        assert "ignore_index: 255" in readme

        with open(written["train_sh"]) as f:
            sh = f.read()
        assert "vtbr_001" in sh
        assert "tools/train.py" in sh
        assert "PYTHONPATH" in sh
        # Make sure we didn't accidentally CRLF the shell script
        assert "\r\n" not in sh

        with open(written["train_bat"], "rb") as f:
            bat_bytes = f.read()
        # .bat needs CRLF line endings so cmd.exe parses it
        assert b"\r\n" in bat_bytes
    print("PASS: handoff write_handoff_scripts")


def test_handoff_missing_dataset_raises():
    try:
        write_handoff_scripts("/nonexistent/path/xyz")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
    print("PASS: handoff missing dataset raises")


if __name__ == "__main__":
    test_parse_pointcept_train_step_line()
    test_parse_pointcept_val_result_line()
    test_parse_legacy_in_house_epoch_line()
    test_parse_checkpoint_save_line()
    test_parse_best_miou_line()
    test_parse_raw_garbage()
    test_runner_applies_train_step_to_status()
    test_runner_tracks_best_miou_monotonically()
    test_runner_log_ring_buffer_trims()
    test_runner_launch_with_nonexistent_python_fails_gracefully()
    test_config_gen_reads_num_classes_from_export()
    test_config_gen_generates_valid_python()
    test_config_gen_injects_so3_rotation_for_vertebrae()
    test_config_gen_input_channels_match_feature_availability()
    test_handoff_writes_all_files()
    test_handoff_missing_dataset_raises()
    print("\n=== Phase 14 ALL TESTS PASSED ===")
