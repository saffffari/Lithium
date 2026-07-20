"""Voxel-based point cloud downsampling."""

import numpy as np

from src.data.point_cloud import PointCloudData


def voxel_downsample(cloud: PointCloudData, target_points: int = 50000) -> PointCloudData:
    """Downsample a point cloud to approximately target_points using voxel grid.

    Selects one representative point per voxel cell. The voxel size is computed
    from the cloud's bounding box and the target point count.
    """
    if cloud.point_count <= target_points:
        return cloud

    # Compute voxel size from cloud volume and target count
    extent = cloud.bounds_max - cloud.bounds_min
    volume = max(np.prod(extent), 1e-10)
    voxel_size = (volume / target_points) ** (1.0 / 3.0)

    # Drop non-finite positions before bucketing. A single NaN row casts to
    # int32(-2147483648), hashing into a garbage voxel and contaminating the
    # downsampled output.
    finite_mask = np.isfinite(cloud.positions).all(axis=1)
    if not finite_mask.all():
        positions = cloud.positions[finite_mask]
    else:
        positions = cloud.positions

    # Quantize positions to voxel grid
    voxel_indices = np.floor((positions - cloud.bounds_min) / voxel_size).astype(np.int32)

    # Use structured array for unique voxel lookup
    # Pack 3 ints into a single hashable view
    dt = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32)])
    structured = np.empty(len(voxel_indices), dtype=dt)
    structured['x'] = voxel_indices[:, 0]
    structured['y'] = voxel_indices[:, 1]
    structured['z'] = voxel_indices[:, 2]

    _, unique_indices = np.unique(structured, return_index=True)
    unique_indices.sort()  # preserve spatial ordering

    # If still too many, randomly subsample
    if len(unique_indices) > target_points * 1.5:
        rng = np.random.default_rng(42)
        unique_indices = rng.choice(unique_indices, target_points, replace=False)
        unique_indices.sort()

    if not finite_mask.all():
        # Translate indices back into the original cloud's coordinate space
        # so per-point attributes line up with the kept rows.
        orig_indices = np.nonzero(finite_mask)[0][unique_indices]
    else:
        orig_indices = unique_indices

    new_positions = cloud.positions[orig_indices]
    new_colors = cloud.colors[orig_indices]
    new_scalars = {k: v[orig_indices] for k, v in cloud.scalars.items()}
    new_labels = cloud.labels[orig_indices] if cloud.labels is not None else None

    return PointCloudData(
        positions=new_positions,
        colors=new_colors,
        scalars=new_scalars,
        file_path=cloud.file_path,
        labels=new_labels,
    )
