"""Unified point cloud loader — dispatches by file extension or directory type."""

import os

from src.data.point_cloud import PointCloudData
from src.data.ply_loader import load_ply
from src.data.las_loader import load_las
from src.data.npz_loader import load_npz


SUPPORTED_EXTENSIONS = {'.ply', '.las', '.laz', '.npz'}


def load_point_cloud(path: str, **kwargs) -> PointCloudData:
    """Load a point cloud file.

    Raises ValueError for unsupported formats.
    """
    if os.path.isdir(path):
        raise ValueError(
            f"Directory loading not supported here: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == '.ply':
        cloud = load_ply(path)
    elif ext in ('.las', '.laz'):
        cloud = load_las(path)
    elif ext == '.npz':
        cloud = load_npz(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    # Ensure height scalar exists for colormapping
    if 'height' not in cloud.scalars:
        cloud.scalars['height'] = cloud.positions[:, 2].copy()

    return cloud


def scan_directory(directory: str) -> list[str]:
    """Return sorted list of supported point cloud files in a directory."""
    files = []
    for name in sorted(os.listdir(directory)):
        ext = os.path.splitext(name.lower())[1]
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(directory, name))
    return files


def scan_directory_recursive(root: str, max_files: int = 50000) -> list[str]:
    """Walk ``root`` and return every supported point cloud file underneath.

    Recurses into every subdirectory. Used for hierarchical datasets
    where each leaf directory is a subject and each file inside is an
    independent sample (e.g. VerSe: ``verse_points/<split>/sub-XXX/NNN.ply``,
    141 subjects x ~10 vertebrae each).

    Returns paths sorted by their full path string so the ordering is
    stable across platforms. Caps at ``max_files`` to protect against
    accidentally pointing at the filesystem root.

    Skips:
      - Hidden directories (names starting with ``.``)
      - ``__pycache__`` and ``.git`` trees
      - Files with unsupported extensions
    """
    collected: list[str] = []
    skip_dirs = {"__pycache__", ".git", ".hg", ".svn", "node_modules"}
    for cur_root, dirs, files in os.walk(root):
        # Prune unwanted directories in-place so os.walk doesn't descend
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for name in files:
            ext = os.path.splitext(name.lower())[1]
            if ext in SUPPORTED_EXTENSIONS:
                collected.append(os.path.join(cur_root, name))
                if len(collected) >= max_files:
                    collected.sort()
                    return collected
    collected.sort()
    return collected
