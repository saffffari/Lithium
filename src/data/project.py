"""Annotation project persistence: save, load, and autosave.

Project directory layout (name.3photon/):
    project.json      — metadata, cloud paths, settings
    registry.json     — label registry (hierarchy, colors, flags)
    labels/
        cloud_000.npy — per-point int32 labels for each cloud
        cloud_001.npy
    camera.json       — camera state snapshot
"""

import os
import json
import time
import shutil
import numpy as np

from src.data.labels import LabelRegistry


def _atomic_write_json(path: str, payload: dict) -> None:
    """Write JSON to ``path`` via tmp + replace so partial writes never
    corrupt the existing file.
    """
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _atomic_save_npy(path: str, array: np.ndarray) -> None:
    """np.save with tmp + replace for per-cloud label files."""
    tmp = path + ".tmp"
    np.save(tmp, array)
    # np.save appends .npy if missing — resolve what actually landed.
    actual_tmp = tmp if os.path.exists(tmp) else tmp + ".npy"
    os.replace(actual_tmp, path)

PROJECT_VERSION = 1
PROJECT_EXT = ".3photon"


class AnnotationProject:
    """Serializable annotation project bound to an App instance."""

    def __init__(self, app):
        self.app = app
        self.project_path: str | None = None

    def save(self, path: str):
        """Save the project to a directory (created if needed)."""
        if not path.endswith(PROJECT_EXT):
            path = path + PROJECT_EXT
        os.makedirs(path, exist_ok=True)
        labels_dir = os.path.join(path, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        # project.json
        project_data = {
            'version': PROJECT_VERSION,
            'name': os.path.basename(path).replace(PROJECT_EXT, ''),
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'modified': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'clouds': [
                {
                    'path': entry.file_path,
                    'relative_path': os.path.relpath(entry.file_path, path) if entry.file_path else '',
                    'name': entry.name,
                }
                for entry in self.app.entries
            ],
            'is_sequence': self.app.sequence is not None,
            'active_label_id': self.app.active_label_id,
            'settings': {
                'point_size': float(self.app.point_size),
                'sharpness': float(self.app.point_sharpness),
                'point_softness': float(getattr(self.app, 'point_softness', 0.0)),
                'brightness': float(self.app.brightness),
                'contrast': float(self.app.contrast),
                'saturation': float(self.app.saturation),
                'bg_color': [float(c) for c in self.app.bg_color],
                'label_blend': float(self.app.label_blend),
                'show_bbox': bool(self.app.show_bbox),
                'show_grid': bool(self.app.show_grid),
                'export_width': int(getattr(self.app, 'export_width', 1920)),
                'export_height': int(getattr(self.app, 'export_height', 1080)),
                'clip_fraction_lo': [float(v) for v in getattr(self.app, 'clip_fraction_lo', [0.0, 0.0, 0.0])],
                'clip_fraction_hi': [float(v) for v in getattr(self.app, 'clip_fraction_hi', [0.0, 0.0, 0.0])],
                'camera_projection': str(getattr(self.app.camera, 'projection', 'perspective')),
                'camera_fov': float(self.app.camera.fov),
            },
        }
        _atomic_write_json(os.path.join(path, 'project.json'), project_data)

        # registry.json
        _atomic_write_json(
            os.path.join(path, 'registry.json'),
            self.app.label_registry.to_json(),
        )

        # Per-cloud labels
        for i, entry in enumerate(self.app.entries):
            gpu = entry.full_gpu or entry.preview_gpu
            if gpu is not None:
                label_path = os.path.join(labels_dir, f'cloud_{i:03d}.npy')
                _atomic_save_npy(label_path, gpu.cloud_data.labels)

        # camera.json
        camera_data = {
            'target': [float(x) for x in self.app.camera.target],
            'distance': float(self.app.camera.distance),
            'azimuth': float(self.app.camera.azimuth),
            'elevation': float(self.app.camera.elevation),
            'fov': float(self.app.camera.fov),
        }
        _atomic_write_json(os.path.join(path, 'camera.json'), camera_data)

        self.project_path = path

    def load(self, path: str) -> dict:
        """Load a project. Returns the project metadata dict.

        The app is expected to load the referenced cloud files separately,
        then this method restores labels, settings, camera, and registry.
        """
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Project directory not found: {path}")

        # project.json
        with open(os.path.join(path, 'project.json'), 'r') as f:
            project_data = json.load(f)

        # registry.json
        reg_path = os.path.join(path, 'registry.json')
        if os.path.exists(reg_path):
            with open(reg_path, 'r') as f:
                registry_data = json.load(f)
            self.app.label_registry = LabelRegistry.from_json(registry_data)
            # Update GPU texture
            if self.app.label_texture is not None:
                from src.rendering.label_texture import update_label_color_texture
                update_label_color_texture(self.app.label_texture, self.app.label_registry)

        # Settings
        settings = project_data.get('settings', {})
        self.app.point_size = settings.get('point_size', 3.0)
        self.app.point_sharpness = settings.get('sharpness', 0.0)
        self.app.brightness = settings.get('brightness', 0.0)
        self.app.contrast = settings.get('contrast', 1.0)
        self.app.saturation = settings.get('saturation', 1.0)
        bg = settings.get('bg_color', [0.0, 0.0, 0.0])
        self.app.bg_color = tuple(bg)
        self.app.label_blend = settings.get('label_blend', 0.0)
        self.app.show_bbox = settings.get('show_bbox', False)
        self.app.show_grid = settings.get('show_grid', False)
        self.app.export_width = int(settings.get('export_width', 1920))
        self.app.export_height = int(settings.get('export_height', 1080))
        if hasattr(self.app, 'point_softness'):
            self.app.point_softness = float(settings.get('point_softness', 0.0))
        # Dual clip fractions (new format)
        cf_lo = settings.get('clip_fraction_lo')
        cf_hi = settings.get('clip_fraction_hi')
        if cf_lo is not None and len(cf_lo) == 3:
            try:
                self.app.clip_fraction_lo[:] = [float(v) for v in cf_lo]
            except (TypeError, ValueError):
                pass
        if cf_hi is not None and len(cf_hi) == 3:
            try:
                self.app.clip_fraction_hi[:] = [float(v) for v in cf_hi]
            except (TypeError, ValueError):
                pass
        # Backward compat: old format had clip_fraction + clip_invert
        if cf_lo is None and cf_hi is None:
            cf = settings.get('clip_fraction')
            ci = settings.get('clip_invert')
            if cf is not None and len(cf) == 3:
                try:
                    for axis in range(3):
                        frac = float(cf[axis])
                        inv = bool(ci[axis]) if ci and len(ci) > axis else False
                        if inv:
                            self.app.clip_fraction_lo[axis] = frac
                        else:
                            self.app.clip_fraction_hi[axis] = frac
                except (TypeError, ValueError):
                    pass
        proj_mode = settings.get('camera_projection')
        if proj_mode in ('perspective', 'orthographic') and hasattr(self.app.camera, 'projection'):
            self.app.camera.projection = proj_mode
        cam_fov = settings.get('camera_fov')
        if cam_fov is not None:
            self.app.camera.fov = float(cam_fov)
        self.app.active_label_id = project_data.get('active_label_id', 0)
        # Push the loaded clip + softness state into the GPU
        if hasattr(self.app, '_sync_clip_box'):
            self.app._sync_clip_box()
        if (getattr(self.app, 'falloff_texture', None) is not None
                and hasattr(self.app, 'point_softness')):
            from src.rendering.falloff_texture import update_falloff_texture
            from src.core.envelope import envelope_from_softness
            update_falloff_texture(
                self.app.falloff_texture,
                envelope_from_softness(self.app.point_softness),
            )

        # Camera
        cam_path = os.path.join(path, 'camera.json')
        if os.path.exists(cam_path):
            with open(cam_path, 'r') as f:
                cam = json.load(f)
            self.app.camera.target = np.array(cam['target'], dtype=np.float32)
            self.app.camera.distance = cam['distance']
            self.app.camera.azimuth = cam['azimuth']
            self.app.camera.elevation = cam['elevation']
            self.app.camera.fov = cam['fov']
            # Snap goal state to the loaded pose so the camera doesn't ease
            # from the previous pivot when a project is opened.
            self.app.camera._snap()

        self.project_path = path
        return project_data

    def apply_labels_to_clouds(self, path: str):
        """After loading clouds, apply saved per-cloud labels."""
        labels_dir = os.path.join(path, 'labels')
        if not os.path.isdir(labels_dir):
            return
        for i, entry in enumerate(self.app.entries):
            label_path = os.path.join(labels_dir, f'cloud_{i:03d}.npy')
            if not os.path.exists(label_path):
                continue
            try:
                labels = np.load(label_path)
            except (OSError, ValueError) as e:
                print(f"Project load: failed to read {label_path}: {e}")
                continue
            gpu = entry.full_gpu or entry.preview_gpu
            if gpu is None:
                print(
                    f"Project load: cloud {i} '{entry.name}' has no GPU buffer — "
                    f"skipping saved labels ({len(labels)} pts)."
                )
                continue
            if len(labels) != gpu.point_count:
                # Point count changed since save (cloud re-exported, trimmed,
                # etc.). Silently accepting a truncated labels array would
                # mislabel points; skip and warn loudly.
                print(
                    f"Project load: label count mismatch on cloud {i} "
                    f"'{entry.name}' — saved {len(labels)} labels but cloud "
                    f"has {gpu.point_count} points. Labels not applied."
                )
                continue
            gpu.cloud_data.labels[:] = labels
            from src.rendering.point_cloud_renderer import re_upload_labels
            re_upload_labels(gpu)
        # Drop any stale label-count cache entries
        if hasattr(self.app, 'label_count_cache'):
            self.app.label_count_cache.clear()


def resolve_cloud_path(cloud_entry: dict, project_path: str) -> str | None:
    """Find a cloud file: try absolute path, then relative to project."""
    abs_path = cloud_entry.get('path', '')
    if abs_path and os.path.exists(abs_path):
        return abs_path
    rel_path = cloud_entry.get('relative_path', '')
    if rel_path:
        candidate = os.path.normpath(os.path.join(project_path, rel_path))
        if os.path.exists(candidate):
            return candidate
    return None
