"""PLY file writer with optional per-point label support.

Writes binary little-endian PLY files via plyfile. Labels are stored as
an int32 ``label`` vertex property so the file is self-contained and can
be inspected in other viewers.
"""

import numpy as np
from plyfile import PlyData, PlyElement

from src.data.point_cloud import PointCloudData


def save_ply(cloud: PointCloudData,
             path: str,
             include_labels: bool = True) -> None:
    """Write a PointCloudData to a binary PLY file.

    Args:
        cloud: source data (positions, colors, optionally labels).
        path:  output file path (should end in ``.ply``).
        include_labels: if True and ``cloud.labels`` has any non-zero
            entries, include a ``label`` vertex property (int32).
    """
    n = cloud.point_count
    has_labels = (
        include_labels
        and cloud.labels is not None
        and len(cloud.labels) == n
        and n > 0
        and int(cloud.labels.max()) > 0
    )

    # Build the numpy structured array
    dtypes = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ]
    if has_labels:
        dtypes.append(('label', 'i4'))

    vertex = np.empty(n, dtype=dtypes)
    vertex['x'] = cloud.positions[:, 0]
    vertex['y'] = cloud.positions[:, 1]
    vertex['z'] = cloud.positions[:, 2]

    # Colors: stored as [0, 1] float in PointCloudData → scale to 0-255
    vertex['red'] = np.clip(cloud.colors[:, 0] * 255.0, 0, 255).astype(np.uint8)
    vertex['green'] = np.clip(cloud.colors[:, 1] * 255.0, 0, 255).astype(np.uint8)
    vertex['blue'] = np.clip(cloud.colors[:, 2] * 255.0, 0, 255).astype(np.uint8)

    if has_labels:
        vertex['label'] = cloud.labels

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], byte_order='<').write(path)
