"""Phase 13: PTv3 pipeline — flatten modes, feature picker, prediction import."""

import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data.labels import LabelRegistry
from src.data.point_cloud import PointCloudData
from src.export.dataset_export import (
    build_class_map, apply_class_map, stack_features, export_dataset,
    IGNORE_INDEX,
)
from src.export.flatten import (
    flatten_leaves, flatten_depth, flatten_subtree, flatten_custom,
    resolve_flatten, _depth_of,
)
from src.export.predictions_import import (
    import_predictions, import_predictions_into_cloud, load_classes_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_hierarchical_registry():
    """Build a 3-level hierarchy:

        Vertebrae (1)
          Cervical (2)
            C1 (3)
            C2 (4)
          Thoracic (5)
            T1 (6)
            T2 (7)
        Soft (8)
    """
    reg = LabelRegistry()
    reg.add_label("Vertebrae")       # 1
    reg.add_label("Cervical", parent_id=1)  # 2
    reg.add_label("C1", parent_id=2)        # 3
    reg.add_label("C2", parent_id=2)        # 4
    reg.add_label("Thoracic", parent_id=1)  # 5
    reg.add_label("T1", parent_id=5)        # 6
    reg.add_label("T2", parent_id=5)        # 7
    reg.add_label("Soft")                    # 8
    return reg


def make_cloud(n=200, reg=None):
    positions = np.random.rand(n, 3).astype(np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)
    scalars = {
        'intensity': np.random.rand(n).astype(np.float32),
        'junk_scalar': np.arange(n, dtype=np.float32),  # should NOT land in feat
    }
    cloud = PointCloudData(positions=positions, colors=colors, scalars=scalars)
    # 20 unlabeled, 40 C1, 40 C2, 40 T1, 40 T2, 20 Soft
    cloud.labels[20:60] = 3   # C1
    cloud.labels[60:100] = 4  # C2
    cloud.labels[100:140] = 6 # T1
    cloud.labels[140:180] = 7 # T2
    cloud.labels[180:200] = 8 # Soft
    return cloud


# ---------------------------------------------------------------------------
# Flatten modes
# ---------------------------------------------------------------------------

def test_depth_of():
    reg = make_hierarchical_registry()
    assert _depth_of(reg, 1) == 0   # Vertebrae: root
    assert _depth_of(reg, 2) == 1   # Cervical
    assert _depth_of(reg, 3) == 2   # C1
    assert _depth_of(reg, 8) == 0   # Soft: root
    print("PASS: _depth_of")


def test_flatten_leaves():
    reg = make_hierarchical_registry()
    m = flatten_leaves(reg)
    # Identity for non-unlabeled; unlabeled never included
    assert 0 not in m
    for info in reg.all_labels():
        if info.id == 0:
            continue
        assert m[info.id] == info.id
    print("PASS: flatten_leaves")


def test_flatten_depth_rolls_up():
    reg = make_hierarchical_registry()
    m = flatten_depth(reg, depth=1)
    # At depth 1: Cervical (2) and Thoracic (5) are the target classes
    # under Vertebrae; Vertebrae itself (depth 0) stays as itself;
    # Soft (depth 0) stays as itself.
    assert m[1] == 1  # Vertebrae stays
    assert m[2] == 2  # Cervical stays
    assert m[3] == 2  # C1 rolls up to Cervical
    assert m[4] == 2  # C2 rolls up to Cervical
    assert m[5] == 5  # Thoracic stays
    assert m[6] == 5  # T1 rolls up to Thoracic
    assert m[7] == 5  # T2 rolls up to Thoracic
    assert m[8] == 8  # Soft stays
    print("PASS: flatten_depth(1)")


def test_flatten_depth_zero_rolls_to_roots():
    reg = make_hierarchical_registry()
    m = flatten_depth(reg, depth=0)
    # Every label rolls up to its root
    assert m[3] == 1  # C1 → Vertebrae
    assert m[7] == 1  # T2 → Vertebrae
    assert m[8] == 8  # Soft is already root
    print("PASS: flatten_depth(0)")


def test_flatten_subtree():
    reg = make_hierarchical_registry()
    m = flatten_subtree(reg, root_id=1)  # Vertebrae subtree
    # Every vertebra label collapses to 1
    for lid in [1, 2, 3, 4, 5, 6, 7]:
        assert m[lid] == 1
    # Soft is NOT in the subtree → not in mapping → treated as IGNORE
    assert 8 not in m
    print("PASS: flatten_subtree")


def test_flatten_custom():
    reg = make_hierarchical_registry()
    m = flatten_custom(reg, {
        "C1": 1,
        "C2": 1,
        "T1": 1,
        "T2": 1,
        "Soft": 8,
    })
    # Four vertebra leaves all collapse to Vertebrae (1)
    for lid in [3, 4, 6, 7]:
        assert m[lid] == 1
    assert m[8] == 8
    # Other labels (Cervical, Thoracic, Vertebrae) are absent → IGNORE
    assert 2 not in m
    print("PASS: flatten_custom")


def test_build_class_map_depth_mode():
    reg = make_hierarchical_registry()
    class_map, names, colors = build_class_map(
        reg, flatten_mode="depth", depth=1)
    # Unlabeled is class 0; effective ids after depth=1: {1, 2, 5, 8}
    # take classes 1..4 in sorted order.
    assert class_map[0] == 0  # Unlabeled — a real class now
    assert class_map[1] == 1  # Vertebrae
    assert class_map[2] == 2  # Cervical
    assert class_map[3] == 2  # C1 → Cervical
    assert class_map[4] == 2  # C2 → Cervical
    assert class_map[5] == 3  # Thoracic
    assert class_map[6] == 3  # T1 → Thoracic
    assert class_map[7] == 3  # T2 → Thoracic
    assert class_map[8] == 4  # Soft
    assert names == ["Unlabeled", "Vertebrae", "Cervical", "Thoracic", "Soft"]
    assert len(colors) == 5
    print("PASS: build_class_map depth mode")


def test_build_class_map_subtree_mode():
    reg = make_hierarchical_registry()
    class_map, names, colors = build_class_map(
        reg, flatten_mode="subtree", subtree_root=1)
    # Unlabeled (class 0) plus the one subtree class: Vertebrae. Soft is
    # outside the subtree → IGNORE.
    assert names == ["Unlabeled", "Vertebrae"]
    assert class_map[0] == 0
    assert class_map[1] == 1
    assert class_map[7] == 1  # T2 inside subtree → the Vertebrae class
    assert 8 not in class_map  # Soft outside → IGNORE on apply
    print("PASS: build_class_map subtree mode")


# ---------------------------------------------------------------------------
# Feature column picker
# ---------------------------------------------------------------------------

def test_stack_features_default_empty():
    """Default (feature_keys=[]) emits 0 columns — RGB/normals handled elsewhere."""
    reg = make_hierarchical_registry()
    cloud = make_cloud(reg=reg)
    feats = stack_features(cloud, feature_keys=[])
    assert feats.shape == (200, 0)
    print("PASS: stack_features default empty")


def test_stack_features_explicit_keys():
    reg = make_hierarchical_registry()
    cloud = make_cloud(reg=reg)
    feats = stack_features(cloud, feature_keys=['intensity'])
    assert feats.shape == (200, 1)
    # Without normalize=True, raw values pass through
    assert feats.min() >= 0.0 and feats.max() <= 1.0  # random() range
    # junk_scalar was NOT requested → not included
    feats2 = stack_features(cloud, feature_keys=['intensity', 'missing_key'])
    assert feats2.shape == (200, 1)
    print("PASS: stack_features explicit keys")


def test_stack_features_legacy_auto_dump():
    """Passing None still auto-dumps all scalars for back-compat."""
    reg = make_hierarchical_registry()
    cloud = make_cloud(reg=reg)
    feats = stack_features(cloud, feature_keys=None)
    assert feats.shape[1] == 2  # intensity + junk_scalar
    print("PASS: stack_features legacy auto-dump")


# ---------------------------------------------------------------------------
# Full dataset export with depth flattening
# ---------------------------------------------------------------------------

def test_export_dataset_depth_mode_produces_classes_json():
    reg = make_hierarchical_registry()
    clouds = [make_cloud(reg=reg) for _ in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        summary = export_dataset(
            clouds, reg, tmp, format='ptv3',
            train_ratio=0.5, val_ratio=0.25, test_ratio=0.25,
            flatten_mode="depth", depth=1,
            source_layer="Anatomy",
        )
        classes_path = os.path.join(tmp, 'classes.json')
        assert os.path.exists(classes_path)
        with open(classes_path) as f:
            classes = json.load(f)
        assert classes['flatten_mode'] == "depth"
        assert classes['depth'] == 1
        assert classes['class_names'] == [
            "Unlabeled", "Vertebrae", "Cervical", "Thoracic", "Soft"]
        assert summary['num_classes'] == 5
        assert summary['ignore_index'] == IGNORE_INDEX

        # Verify actual segments match the depth-flattened class map
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(tmp, split)
            if not os.path.isdir(split_dir):
                continue
            for scene in os.listdir(split_dir):
                seg = np.load(os.path.join(split_dir, scene, 'segment.npy'))
                unique = set(int(x) for x in np.unique(seg))
                # Every non-ignore id must be in 0..4 (Unlabeled + 4 classes)
                for u in unique:
                    if u == IGNORE_INDEX:
                        continue
                    assert 0 <= u <= 4
        # feat.npy should NOT be written with default feature_keys (None)
        # None still triggers auto-dump for back-compat. Verify explicit []
        # path below instead.
    print("PASS: depth mode export + classes.json")


def test_export_dataset_with_feature_picker():
    reg = make_hierarchical_registry()
    clouds = [make_cloud(reg=reg) for _ in range(2)]
    with tempfile.TemporaryDirectory() as tmp:
        export_dataset(
            clouds, reg, tmp, format='ptv3',
            train_ratio=0.5, val_ratio=0.5, test_ratio=0.0,
            feature_keys=['intensity'],
        )
        # Find any scene directory
        found_feat = False
        for split in ['train', 'val']:
            split_dir = os.path.join(tmp, split)
            if not os.path.isdir(split_dir):
                continue
            for scene in os.listdir(split_dir):
                feat_path = os.path.join(split_dir, scene, 'feat.npy')
                if os.path.exists(feat_path):
                    feat = np.load(feat_path)
                    assert feat.shape == (200, 1)
                    found_feat = True
        assert found_feat, "feat.npy should be written when feature_keys is set"

        # Empty feature_keys → no feat.npy emitted (the file only
        # writes when features.shape[1] > 0)
    with tempfile.TemporaryDirectory() as tmp:
        export_dataset(
            clouds, reg, tmp, format='ptv3',
            train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
            feature_keys=[],
        )
        for scene in os.listdir(os.path.join(tmp, 'train')):
            feat_path = os.path.join(tmp, 'train', scene, 'feat.npy')
            assert not os.path.exists(feat_path), \
                f"feat.npy should be absent when feature_keys=[], got {feat_path}"
        # classes.json records the empty feature list
        with open(os.path.join(tmp, 'classes.json')) as f:
            classes = json.load(f)
        assert classes['feature_keys'] == []
    print("PASS: feature picker emission")


# ---------------------------------------------------------------------------
# Prediction importer (round-trip)
# ---------------------------------------------------------------------------

def test_prediction_roundtrip_leaves_mode():
    """Export → synthesize prediction → import → verify labels match."""
    reg = make_hierarchical_registry()
    cloud = make_cloud(reg=reg)
    with tempfile.TemporaryDirectory() as tmp:
        export_dataset(
            [cloud], reg, tmp, format='ptv3',
            train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
            flatten_mode="leaves",
        )
        # Read the ground-truth segment as our "prediction"
        scene_dir = os.path.join(tmp, 'train', 'scene_000')
        gt_segment = np.load(os.path.join(scene_dir, 'segment.npy'))
        pred_path = os.path.join(tmp, 'predictions.npy')
        np.save(pred_path, gt_segment)

        # Fresh registry to simulate import into a different project
        target_reg = LabelRegistry()
        imported = import_predictions(
            pred_path, os.path.join(tmp, 'classes.json'),
            target_reg, name_suffix=" (pred)",
        )
        assert imported.shape == (200,)
        # Unlabeled is now a real exported class (class 0), so the source's
        # unlabeled points round-trip to an 'Unlabeled (pred)' entry, not to
        # registry 0. They still all share one consistent registry id.
        names_by_id = {info.id: info.name for info in target_reg.all_labels()}
        assert names_by_id[int(imported[0])] == "Unlabeled (pred)"
        assert (imported[:20] == imported[0]).all()
        # Labeled points map to distinct, non-Unlabeled entries.
        assert (imported[20:] != imported[0]).all()
        # New registry should have real class entries with ' (pred)' suffix
        names = set(names_by_id.values())
        assert "C1 (pred)" in names
        assert "Soft (pred)" in names
    print("PASS: prediction round-trip (leaves)")


def test_prediction_roundtrip_depth_mode():
    """Depth-flattened export should still round-trip to matching labels."""
    reg = make_hierarchical_registry()
    cloud = make_cloud(reg=reg)
    with tempfile.TemporaryDirectory() as tmp:
        export_dataset(
            [cloud], reg, tmp, format='ptv3',
            train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
            flatten_mode="depth", depth=1,
        )
        scene_dir = os.path.join(tmp, 'train', 'scene_000')
        gt_segment = np.load(os.path.join(scene_dir, 'segment.npy'))
        pred_path = os.path.join(tmp, 'predictions.npy')
        np.save(pred_path, gt_segment)

        target_reg = LabelRegistry()
        imported = import_predictions(
            pred_path, os.path.join(tmp, 'classes.json'),
            target_reg, name_suffix="",  # import directly onto existing names
        )
        # At depth=1 the only classes are Vertebrae / Cervical / Thoracic / Soft,
        # so the imported registry should have just those (plus Unlabeled).
        names = {info.name for info in target_reg.all_labels()}
        assert names == {"Unlabeled", "Vertebrae", "Cervical", "Thoracic", "Soft"}
        # All 40 C1 points should now carry the Cervical label
        cervical_id = next(i.id for i in target_reg.all_labels() if i.name == "Cervical")
        assert (imported[20:60] == cervical_id).all()
    print("PASS: prediction round-trip (depth)")


def test_prediction_into_cloud_length_check():
    """Length mismatch must raise, not silently truncate."""
    reg = make_hierarchical_registry()
    cloud = make_cloud(reg=reg)
    with tempfile.TemporaryDirectory() as tmp:
        export_dataset(
            [cloud], reg, tmp, format='ptv3',
            train_ratio=1.0, val_ratio=0.0, test_ratio=0.0,
        )
        classes_path = os.path.join(tmp, 'classes.json')
        pred_path = os.path.join(tmp, 'wrong_length.npy')
        np.save(pred_path, np.zeros(100, dtype=np.int64))  # wrong N

        try:
            import_predictions_into_cloud(pred_path, classes_path, cloud, reg)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "!= cloud.point_count" in str(e)
    print("PASS: prediction length mismatch raises")


if __name__ == '__main__':
    test_depth_of()
    test_flatten_leaves()
    test_flatten_depth_rolls_up()
    test_flatten_depth_zero_rolls_to_roots()
    test_flatten_subtree()
    test_flatten_custom()
    test_build_class_map_depth_mode()
    test_build_class_map_subtree_mode()
    test_stack_features_default_empty()
    test_stack_features_explicit_keys()
    test_stack_features_legacy_auto_dump()
    test_export_dataset_depth_mode_produces_classes_json()
    test_export_dataset_with_feature_picker()
    test_prediction_roundtrip_leaves_mode()
    test_prediction_roundtrip_depth_mode()
    test_prediction_into_cloud_length_check()
    print("\n=== Phase 13 ALL TESTS PASSED ===")
