"""Phase 1 verification test: Label buffer foundation.

Tests:
1. PointCloudData has labels field, correct shape/dtype
2. LabelRegistry CRUD operations
3. Hierarchical label tree
4. JSON serialization roundtrip
5. Color LUT generation
6. Label preservation through voxel downsample
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data.point_cloud import PointCloudData
from src.data.labels import LabelRegistry, LabelInfo, MAX_LABELS, DEFAULT_PALETTE
from src.data.resampler import voxel_downsample


def test_point_cloud_has_labels():
    """PointCloudData should have a labels field with correct shape and dtype."""
    positions = np.random.rand(1000, 3).astype(np.float32)
    colors = np.random.rand(1000, 3).astype(np.float32)
    cloud = PointCloudData(positions=positions, colors=colors)

    assert cloud.labels is not None, "labels should be auto-initialized"
    assert cloud.labels.shape == (1000,), f"expected (1000,), got {cloud.labels.shape}"
    assert cloud.labels.dtype == np.int32, f"expected int32, got {cloud.labels.dtype}"
    assert (cloud.labels == 0).all(), "all labels should default to 0"
    print("PASS: PointCloudData labels field")


def test_label_registry_basic():
    """LabelRegistry starts with unlabeled, supports add/remove/get."""
    reg = LabelRegistry()
    assert len(reg) == 1, "registry should start with unlabeled"
    assert 0 in reg, "id 0 should exist"
    assert reg.get(0).name == "Unlabeled"

    # Add labels
    id1 = reg.add_label("Background")
    id2 = reg.add_label("Vertebra", color=(1.0, 0.0, 0.0, 1.0))
    assert id1 == 1, f"first added id should be 1, got {id1}"
    assert id2 == 2, f"second added id should be 2, got {id2}"
    assert len(reg) == 3

    # Retrieve
    v = reg.get(id2)
    assert v.name == "Vertebra"
    assert v.color == (1.0, 0.0, 0.0, 1.0)

    # Remove
    assert reg.remove_label(id1) is True
    assert reg.remove_label(0) is False, "cannot remove unlabeled"
    assert reg.remove_label(999) is False, "non-existent label"
    assert len(reg) == 2
    print("PASS: LabelRegistry basic operations")


def test_label_hierarchy():
    """Hierarchical labels with depth tracking."""
    reg = LabelRegistry()
    vertebrae = reg.add_label("Vertebrae")
    thoracic = reg.add_label("Thoracic", parent_id=vertebrae)
    t7 = reg.add_label("T7", parent_id=thoracic)
    spinous = reg.add_label("Spinous Process", parent_id=t7)

    hierarchy = reg.get_hierarchy()
    # Expected: [(Unlabeled, 0), (Vertebrae, 0), (Thoracic, 1), (T7, 2), (Spinous Process, 3)]
    names_depths = [(l.name, d) for l, d in hierarchy]
    assert ("Unlabeled", 0) in names_depths
    assert ("Vertebrae", 0) in names_depths
    assert ("Thoracic", 1) in names_depths
    assert ("T7", 2) in names_depths
    assert ("Spinous Process", 3) in names_depths

    children = reg.get_children(vertebrae)
    assert len(children) == 1
    assert children[0].name == "Thoracic"
    print("PASS: Label hierarchy")


def test_json_roundtrip():
    """JSON serialization preserves all label data."""
    reg = LabelRegistry()
    id1 = reg.add_label("Label1", color=(0.1, 0.2, 0.3, 1.0))
    id2 = reg.add_label("Label2", parent_id=id1)
    reg.set_visible(id2, False)
    reg.set_locked(id1, True)

    data = reg.to_json()
    reg2 = LabelRegistry.from_json(data)

    assert len(reg2) == len(reg)
    assert reg2.get(id1).name == "Label1"
    assert reg2.get(id1).color == (0.1, 0.2, 0.3, 1.0)
    assert reg2.get(id1).locked is True
    assert reg2.get(id2).parent_id == id1
    assert reg2.get(id2).visible is False
    print("PASS: JSON roundtrip")


def test_color_lut():
    """Color LUT should be (256, 4) float32 with labels at correct indices."""
    reg = LabelRegistry()
    id1 = reg.add_label("Red", color=(1.0, 0.0, 0.0, 1.0))
    id2 = reg.add_label("Blue", color=(0.0, 0.0, 1.0, 1.0))

    lut = reg.get_color_lut()
    assert lut.shape == (256, 4), f"expected (256,4), got {lut.shape}"
    assert lut.dtype == np.float32
    assert np.allclose(lut[id1], [1.0, 0.0, 0.0, 1.0])
    assert np.allclose(lut[id2], [0.0, 0.0, 1.0, 1.0])

    # Hidden label should have alpha 0
    reg.set_visible(id1, False)
    lut = reg.get_color_lut()
    assert lut[id1, 3] == 0.0, "hidden label should have alpha 0"
    print("PASS: Color LUT")


def test_resample_preserves_labels():
    """Voxel downsampling should preserve labels at selected indices."""
    np.random.seed(42)
    positions = np.random.rand(10000, 3).astype(np.float32) * 100
    colors = np.random.rand(10000, 3).astype(np.float32)
    labels = np.zeros(10000, dtype=np.int32)
    labels[0:3000] = 1
    labels[3000:6000] = 2
    labels[6000:10000] = 3

    cloud = PointCloudData(positions=positions, colors=colors, labels=labels)

    downsampled = voxel_downsample(cloud, target_points=1000)

    assert downsampled.labels is not None, "downsampled cloud should have labels"
    assert downsampled.labels.shape == (downsampled.point_count,)
    assert downsampled.labels.dtype == np.int32
    # Should have a mix of labels 1, 2, 3 (roughly proportional)
    unique = np.unique(downsampled.labels)
    assert len(unique) >= 2, f"expected multiple label classes, got {unique}"
    print(f"PASS: Resample preserves labels ({downsampled.point_count} pts, labels={unique})")


def test_max_labels():
    """Registry should reject adding more than MAX_LABELS."""
    reg = LabelRegistry()
    for i in range(MAX_LABELS):
        reg.add_label(f"label_{i}")
    assert len(reg) == MAX_LABELS + 1  # +1 for unlabeled
    try:
        reg.add_label("overflow")
        assert False, "should have raised ValueError"
    except ValueError:
        pass
    print("PASS: Max labels enforced")


if __name__ == '__main__':
    test_point_cloud_has_labels()
    test_label_registry_basic()
    test_label_hierarchy()
    test_json_roundtrip()
    test_color_lut()
    test_resample_preserves_labels()
    test_max_labels()
    print("\n=== Phase 1 ALL TESTS PASSED ===")
