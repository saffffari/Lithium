"""Phase 11 verification: Project persistence (save/load)."""

import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from plyfile import PlyData, PlyElement

from src.data.labels import LabelRegistry
from src.data.project import AnnotationProject, PROJECT_EXT, resolve_cloud_path


class MockCamera:
    def __init__(self):
        self.target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.distance = 10.0
        self.azimuth = 0.5
        self.elevation = 0.7
        self.fov = np.radians(60.0)

    def _snap(self):
        pass


class MockCloudEntry:
    def __init__(self, path: str, labels: np.ndarray):
        self.file_path = path
        self.name = os.path.basename(path)
        self.full_gpu = MockGPU(labels)
        self.preview_gpu = None


class MockGPU:
    def __init__(self, labels: np.ndarray):
        self.point_count = len(labels)
        self.cloud_data = type('Cloud', (), {'labels': labels})()


class MockApp:
    def __init__(self):
        self.entries = []
        self.gpu_clouds = []
        self.selected_index = 0
        self.sequence = None
        self.label_registry = LabelRegistry()
        self.active_label_id = 0
        self.point_size = 3.0
        self.point_sharpness = 0.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.bg_color = (0.1, 0.2, 0.3)
        self.label_blend = 1.0
        self.show_bbox = True
        self.show_grid = False
        # Dual per-axis clip fractions (matches App.__init__) — project
        # load mutates these in place, so the mock must carry them.
        self.clip_fraction_lo = np.zeros(3, dtype=np.float32)
        self.clip_fraction_hi = np.zeros(3, dtype=np.float32)
        self.camera = MockCamera()
        self.label_texture = None


def test_save_project_basic():
    """Save creates expected files and structure."""
    with tempfile.TemporaryDirectory() as tmp:
        app = MockApp()
        # Add labels
        v_id = app.label_registry.add_label("Vertebra")
        d_id = app.label_registry.add_label("Disc")
        app.active_label_id = v_id

        # Add clouds
        labels1 = np.zeros(100, dtype=np.int32)
        labels1[0:30] = v_id
        labels1[30:60] = d_id
        ply_path = os.path.join(tmp, 'cloud1.ply')
        positions = np.random.rand(100, 3).astype(np.float32)
        vertex = np.array([(p[0], p[1], p[2]) for p in positions],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        PlyData([PlyElement.describe(vertex, 'vertex')]).write(ply_path)
        app.entries.append(MockCloudEntry(ply_path, labels1))

        proj = AnnotationProject(app)
        save_path = os.path.join(tmp, 'myproject')
        proj.save(save_path)

        expected = save_path + PROJECT_EXT
        assert os.path.isdir(expected)
        assert os.path.exists(os.path.join(expected, 'project.json'))
        assert os.path.exists(os.path.join(expected, 'registry.json'))
        assert os.path.exists(os.path.join(expected, 'camera.json'))
        assert os.path.exists(os.path.join(expected, 'labels', 'cloud_000.npy'))

        # Verify project.json contents
        with open(os.path.join(expected, 'project.json')) as f:
            data = json.load(f)
        assert data['version'] == 1
        assert len(data['clouds']) == 1
        assert data['settings']['point_size'] == 3.0
        assert data['settings']['bg_color'] == [0.1, 0.2, 0.3]
        assert data['active_label_id'] == v_id

        # Verify labels round-trip
        saved_labels = np.load(os.path.join(expected, 'labels', 'cloud_000.npy'))
        assert (saved_labels == labels1).all()

        print("PASS: Save project creates correct structure")


def test_load_project_round_trip():
    """Save then load preserves all settings and labels."""
    with tempfile.TemporaryDirectory() as tmp:
        # Save
        app1 = MockApp()
        v_id = app1.label_registry.add_label("Vertebra",
                                              color=(0.8, 0.2, 0.3, 1.0))
        app1.active_label_id = v_id
        app1.point_size = 7.5
        app1.brightness = -0.3
        app1.bg_color = (0.2, 0.5, 0.8)
        app1.show_bbox = True

        labels = np.zeros(50, dtype=np.int32)
        labels[10:30] = v_id
        ply_path = os.path.join(tmp, 'cloud.ply')
        positions = np.random.rand(50, 3).astype(np.float32)
        vertex = np.array([(p[0], p[1], p[2]) for p in positions],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        PlyData([PlyElement.describe(vertex, 'vertex')]).write(ply_path)
        app1.entries.append(MockCloudEntry(ply_path, labels))

        proj1 = AnnotationProject(app1)
        save_path = os.path.join(tmp, 'p')
        proj1.save(save_path)
        actual_path = save_path + PROJECT_EXT

        # Load into a fresh app
        app2 = MockApp()
        proj2 = AnnotationProject(app2)
        project_data = proj2.load(actual_path)

        # Verify settings
        assert app2.point_size == 7.5
        assert app2.brightness == -0.3
        assert tuple(app2.bg_color) == (0.2, 0.5, 0.8)
        assert app2.show_bbox is True
        assert app2.active_label_id == v_id

        # Verify registry
        vertebra_info = app2.label_registry.get(v_id)
        assert vertebra_info is not None
        assert vertebra_info.name == "Vertebra"
        assert abs(vertebra_info.color[0] - 0.8) < 1e-4

        # Verify camera
        assert abs(app2.camera.distance - 10.0) < 1e-4
        assert abs(app2.camera.azimuth - 0.5) < 1e-4

        # Verify clouds in project data
        assert len(project_data['clouds']) == 1
        cloud_info = project_data['clouds'][0]
        resolved = resolve_cloud_path(cloud_info, actual_path)
        assert resolved is not None
        assert os.path.exists(resolved)

        print("PASS: Full round-trip preserves all state")


def test_relative_path_fallback():
    """If absolute path doesn't exist, use relative path."""
    with tempfile.TemporaryDirectory() as tmp:
        cloud_info = {
            'path': '/nonexistent/absolute/cloud.ply',
            'relative_path': '../cloud.ply',
        }
        # Create a project dir with a sibling cloud file
        project_dir = os.path.join(tmp, 'proj.3photon')
        os.makedirs(project_dir)
        cloud_path = os.path.join(tmp, 'cloud.ply')
        open(cloud_path, 'w').close()

        resolved = resolve_cloud_path(cloud_info, project_dir)
        assert resolved is not None
        assert os.path.exists(resolved)
        print("PASS: Relative path fallback works")


if __name__ == '__main__':
    test_save_project_basic()
    test_load_project_round_trip()
    test_relative_path_fallback()
    print("\n=== Phase 11 ALL TESTS PASSED ===")
