"""Point cloud data structure."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class PointCloudData:
    """Container for point cloud data loaded from disk."""
    positions: np.ndarray          # (N, 3) float32
    colors: np.ndarray             # (N, 3) float32, range [0, 1]
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    bounds_min: np.ndarray = field(default=None)
    bounds_max: np.ndarray = field(default=None)
    center: np.ndarray = field(default=None)
    point_count: int = 0
    file_path: str = ""
    model_transform: np.ndarray = field(default=None)
    labels: np.ndarray = field(default=None)  # (N,) int32, 0 = unlabeled
    metadata: dict = field(default_factory=dict)  # free-form (spacing, source, etc.)
    # Raw supine→standing 4x4 from PolyPose (mm, original CT frame).
    # None = no pose imported yet. Distinct from ``model_transform``,
    # which is the user-driven gizmo edit in LIGHT TABLE; this one
    # comes from PolyPose and drives the HOLOGRAM multi-vertebra scene.
    pose_to_world: np.ndarray = field(default=None)
    labels_version: int = 0  # bumped on every in-place label mutation; render
                              # caches that key on cloud identity (e.g. the
                              # gallery thumbnail cache) include this so they
                              # invalidate when labels change without the
                              # cloud_data object being replaced.

    def __post_init__(self):
        self.point_count = len(self.positions)
        if self.point_count == 0:
            self.bounds_min = np.zeros(3, dtype=np.float32)
            self.bounds_max = np.zeros(3, dtype=np.float32)
            self.center = np.zeros(3, dtype=np.float32)
        else:
            if self.bounds_min is None:
                self.bounds_min = self.positions.min(axis=0)
            if self.bounds_max is None:
                self.bounds_max = self.positions.max(axis=0)
            if self.center is None:
                self.center = (self.bounds_min + self.bounds_max) / 2.0
        if self.model_transform is None:
            self.model_transform = np.eye(4, dtype=np.float32)
        if self.labels is None:
            self.labels = np.zeros(self.point_count, dtype=np.int32)
        elif self.labels.dtype != np.int32:
            self.labels = self.labels.astype(np.int32)
