"""Data-side helpers of the Light Table INFER constellation widget.

The draw path needs a live ImGui context (covered by manual smoke),
but the sampling/normalization/color-mapping helpers are pure.
"""

import numpy as np

from src.data.labels import LabelRegistry
from src.data.point_cloud import PointCloudData
from src.gui import infer_widget


class _FakeGPU:
    def __init__(self, cloud):
        self.cloud_data = cloud


class _FakeEntry:
    def __init__(self, cloud, file_key="fk_test"):
        self.full_gpu = _FakeGPU(cloud)
        self.preview_gpu = None
        self.file_key = file_key
        self.file_path = "/tmp/fake.ply"


def _make_cloud(n=5000):
    rng = np.random.default_rng(7)
    pos = rng.uniform(-50.0, 50.0, size=(n, 3)).astype(np.float32)
    cloud = PointCloudData(
        positions=pos,
        colors=np.full((n, 3), 0.5, dtype=np.float32),
        file_path="/tmp/fake.ply",
    )
    return cloud


def test_constellation_samples_and_normalizes():
    infer_widget._cache.clear()
    entry = _FakeEntry(_make_cloud(5000))
    con = infer_widget._constellation_for(entry)
    assert con is not None
    pts = con["pts"]
    assert len(pts) <= infer_widget._MAX_PTS
    # Centered unit box.
    assert float(np.abs(pts).max()) <= 1.0 + 1e-5
    center = (pts.min(axis=0) + pts.max(axis=0)) * 0.5
    assert np.all(np.abs(center) < 0.35)
    # Cached: same object on second call.
    assert infer_widget._constellation_for(entry) is con


def test_constellation_cache_invalidates_on_point_count_change():
    infer_widget._cache.clear()
    entry = _FakeEntry(_make_cloud(5000))
    con1 = infer_widget._constellation_for(entry)
    entry.full_gpu = _FakeGPU(_make_cloud(3000))
    con2 = infer_widget._constellation_for(entry)
    assert con2 is not con1
    assert con2["point_count"] == 3000


def test_sampled_label_colors_map_through_registry():
    infer_widget._cache.clear()
    cloud = _make_cloud(1000)
    entry = _FakeEntry(cloud)
    con = infer_widget._constellation_for(entry)

    reg = LabelRegistry()
    rid = reg.add_label(name="bone", color=(0.9, 0.1, 0.2, 1.0))
    cloud.labels[:] = 0
    cloud.labels[::2] = rid

    colors = infer_widget._sampled_label_colors(entry, con["sample_idx"], reg)
    assert colors is not None and colors.shape == (len(con["pts"]), 3)
    lbl = cloud.labels[con["sample_idx"]]
    # Labeled points carry the registry color; unlabeled the neutral grey.
    assert np.allclose(colors[lbl == rid][0], (0.9, 0.1, 0.2), atol=1e-5)
    assert np.allclose(colors[lbl == 0][0], (0.45, 0.45, 0.45), atol=1e-5)
