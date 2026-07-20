"""Phase 7 verification: Dataset export in PTv3/NPZ/HDF5 formats."""

import sys
import os
import json
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.point_cloud import PointCloudData
from src.data.labels import LabelRegistry
from src.export.dataset_export import (
    remap_labels, stack_features, export_npz, export_ptv3_scene,
    export_hdf5, export_dataset, split_indices,
    build_class_map, apply_class_map, IGNORE_INDEX,
)


def make_labeled_cloud(n=1000):
    positions = np.random.rand(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    scalars = {
        'intensity': np.random.rand(n).astype(np.float32),
        'height': positions[:, 2].copy(),
    }
    cloud = PointCloudData(positions=positions, colors=colors, scalars=scalars)
    # Use non-contiguous label IDs (1, 5, 10 - simulates deletions)
    cloud.labels[0 : n // 10 * 3] = 1       # 30%
    cloud.labels[n // 10 * 3 : n // 10 * 6] = 5    # 30%
    cloud.labels[n // 10 * 6 : n // 10 * 8] = 10   # 20%
    # rest remain 0 (unlabeled, 20%)
    return cloud


def make_registry():
    reg = LabelRegistry()
    # Add enough dummies to get non-contiguous IDs
    reg.add_label("Background")  # 1
    reg.add_label("_dummy2")     # 2
    reg.add_label("_dummy3")     # 3
    reg.add_label("_dummy4")     # 4
    reg.add_label("Vertebra")    # 5
    for i in range(4):
        reg.add_label(f"_d{i}")  # 6-9
    reg.add_label("Disc")        # 10
    return reg


def test_remap_labels():
    labels = np.array([0, 1, 5, 5, 10, 0, 1], dtype=np.int32)
    reg = make_registry()
    remapped, id_to_name = remap_labels(labels, reg)
    # Class map is built from the full registry, not scene uniques.
    # make_registry has 10 real labels (1..10) plus Unlabeled (0), which
    # is now EXPORTED AS A REAL CLASS at id 0 (not IGNORE) — so class_names
    # has 11 entries and the class ids are the identity of the registry ids.
    assert len(id_to_name) == 11
    assert id_to_name[0] == "Unlabeled"       # registry id 0 -> class 0
    assert id_to_name[1] == "Background"      # registry id 1 -> class 1
    assert id_to_name[5] == "Vertebra"        # registry id 5 -> class 5
    assert id_to_name[10] == "Disc"           # registry id 10 -> class 10
    # Unlabeled → class 0 (a real class now), never IGNORE
    assert (remapped[labels == 0] == 0).all()
    # Real labels map consistently (identity here: leaves + Unlabeled@0)
    assert (remapped[labels == 1] == 1).all()
    assert (remapped[labels == 5] == 5).all()
    assert (remapped[labels == 10] == 10).all()
    # With no allowed_ids filter nothing is dropped to IGNORE_INDEX
    assert not (remapped == IGNORE_INDEX).any()
    print(f"PASS: Remap labels (classes={len(id_to_name)})")


def test_build_class_map_stable_across_scenes():
    """Class ids must be stable even when scenes have different uniques."""
    reg = make_registry()
    class_map, class_names, class_colors = build_class_map(reg)
    # Registry ids 0..10 -> contiguous 0..10 (Unlabeled included at 0)
    assert class_map[0] == 0
    for reg_id in range(0, 11):
        assert class_map[reg_id] == reg_id
    assert len(class_names) == 11
    assert len(class_colors) == 11
    # Colors are 0..255 ints
    for c in class_colors:
        assert len(c) == 3 and all(0 <= x <= 255 for x in c)
    print(f"PASS: build_class_map ({len(class_names)} classes)")


def test_stack_features():
    cloud = make_labeled_cloud(500)
    feats = stack_features(cloud)
    assert feats.shape == (500, 2), f"expected (500,2), got {feats.shape}"
    assert feats.dtype == np.float32
    assert feats.min() >= 0.0
    assert feats.max() <= 1.0
    print(f"PASS: Stack features {feats.shape}")


def test_split_indices():
    train, val, test = split_indices(100, 0.7, 0.15, 0.15, seed=42)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    all_indices = set(train) | set(val) | set(test)
    assert len(all_indices) == 100, "splits must be disjoint and cover all"
    print(f"PASS: Split indices ({len(train)}/{len(val)}/{len(test)})")


def test_export_npz():
    cloud = make_labeled_cloud(500)
    reg = make_registry()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'out.npz')
        export_npz(cloud, path, reg)
        assert os.path.exists(path)
        with np.load(path) as data:
            assert data['coord'].shape == (500, 3)
            assert data['label'].shape == (500,)
            unique = np.unique(data['label'])
        # label_map.json sibling
        map_path = os.path.splitext(path)[0] + '_label_map.json'
        assert os.path.exists(map_path)
        with open(map_path) as f:
            label_map = json.load(f)
        # Every unique class label must exist in the map; IGNORE_INDEX
        # (unlabeled points) is allowed to show up in the array but is
        # not a class and must NOT appear in the map.
        for u in unique:
            u = int(u)
            if u == IGNORE_INDEX:
                continue
            assert str(u) in label_map, f"class {u} missing from map"
        assert str(IGNORE_INDEX) not in label_map
        print(f"PASS: NPZ export (labels={unique.tolist()}, map={len(label_map)})")


def test_export_ptv3():
    cloud = make_labeled_cloud(500)
    reg = make_registry()
    with tempfile.TemporaryDirectory() as tmp:
        scene_dir = os.path.join(tmp, 'scene_000')
        export_ptv3_scene(cloud, scene_dir, reg)
        assert os.path.exists(os.path.join(scene_dir, 'coord.npy'))
        assert os.path.exists(os.path.join(scene_dir, 'color.npy'))
        assert os.path.exists(os.path.join(scene_dir, 'segment.npy'))
        # feat.npy is only emitted when the caller explicitly picks
        # feature columns; default is empty so the file doesn't exist.
        assert not os.path.exists(os.path.join(scene_dir, 'feat.npy'))
        assert os.path.exists(os.path.join(scene_dir, 'label_map.json'))

        coord = np.load(os.path.join(scene_dir, 'coord.npy'))
        segment = np.load(os.path.join(scene_dir, 'segment.npy'))
        assert coord.shape == (500, 3)
        assert segment.dtype == np.int64
        # Cloud has ~20% unlabeled points — those are now class 0
        # (Unlabeled is a real class), NOT IGNORE_INDEX.
        assert (segment == 0).any()
        assert not (segment == IGNORE_INDEX).any()
        with open(os.path.join(scene_dir, 'label_map.json')) as f:
            label_map = json.load(f)
        # Every segment class must appear in the map, class 0 included.
        for u in np.unique(segment):
            assert str(int(u)) in label_map
        assert str(IGNORE_INDEX) not in label_map
        print(f"PASS: PTv3 scene export (labels={np.unique(segment).tolist()})")


def test_export_hdf5():
    clouds = [make_labeled_cloud(500) for _ in range(3)]
    reg = make_registry()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'out.h5')
        export_hdf5(clouds, path, reg)
        import h5py
        with h5py.File(path, 'r') as f:
            assert 'scene_000' in f
            assert 'scene_001' in f
            assert 'scene_002' in f
            assert f['scene_000/coord'].shape == (500, 3)
            label_map = json.loads(f.attrs['label_map'])
            assert "0" in label_map
            assert f.attrs['ignore_index'] == IGNORE_INDEX
            # Unlabeled points in each scene → class 0 (a real class now),
            # not IGNORE_INDEX.
            seg = np.array(f['scene_000/label'])
            assert (seg == 0).any()
            assert not (seg == IGNORE_INDEX).any()
    print("PASS: HDF5 export")


def test_export_dataset_ptv3():
    clouds = [make_labeled_cloud(500) for _ in range(10)]
    reg = make_registry()
    with tempfile.TemporaryDirectory() as tmp:
        summary = export_dataset(clouds, reg, tmp, format='ptv3',
                                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                 source_layer="Anatomy")
        assert summary['total'] == 10
        assert summary['train'] == 7
        assert summary['val'] == 2  # rounding
        assert summary['test'] == 1  # rounding
        assert summary['ignore_index'] == IGNORE_INDEX
        assert summary['num_classes'] == 11  # 10 real labels + Unlabeled@0
        # Verify directories + sidecars exist
        assert os.path.exists(os.path.join(tmp, 'train'))
        assert os.path.exists(os.path.join(tmp, 'val'))
        assert os.path.exists(os.path.join(tmp, 'test'))
        assert os.path.exists(os.path.join(tmp, 'splits.json'))
        classes_path = os.path.join(tmp, 'classes.json')
        assert os.path.exists(classes_path)
        with open(classes_path) as f:
            classes = json.load(f)
        assert classes['ignore_index'] == IGNORE_INDEX
        assert classes['num_classes'] == 11
        assert classes['source_layer'] == "Anatomy"
        assert classes['flatten_mode'] == "leaves"
        assert len(classes['class_names']) == 11
        assert len(classes['class_colors']) == 11
        # Count scene dirs
        train_scenes = [d for d in os.listdir(os.path.join(tmp, 'train')) if d.startswith('scene_')]
        assert len(train_scenes) == 7
        # Every scene's segment.npy must use the SAME class ids (stable
        # across scenes — the whole point of building one global map).
        for scene in train_scenes:
            seg = np.load(os.path.join(tmp, 'train', scene, 'segment.npy'))
            unique = set(int(x) for x in np.unique(seg))
            # Every non-ignore id must be < num_classes
            for u in unique:
                if u == IGNORE_INDEX:
                    continue
                assert 0 <= u < 11, f"scene {scene}: class id {u} out of range"
    print(f"PASS: Full dataset export (7/2/1, ignore={IGNORE_INDEX})")


if __name__ == '__main__':
    test_remap_labels()
    test_build_class_map_stable_across_scenes()
    test_stack_features()
    test_split_indices()
    test_export_npz()
    test_export_ptv3()
    test_export_hdf5()
    test_export_dataset_ptv3()
    print("\n=== Phase 7 ALL TESTS PASSED ===")
