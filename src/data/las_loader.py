"""LAS/LAZ file loader."""

import numpy as np
import laspy

from src.data.point_cloud import PointCloudData


def load_las(path: str) -> PointCloudData:
    """Load a LAS or LAZ file and return a PointCloudData instance."""
    try:
        las = laspy.read(path)
    except (OSError, ValueError, EOFError, laspy.errors.LaspyException) as e:
        raise IOError(f"Failed to read LAS/LAZ file {path}: {e}") from e

    # Work in float64 first so georeferenced coordinates (UTM / state plane)
    # don't lose sub-meter precision. Center on the cloud centroid, then cast
    # to float32 — this keeps the shader math cheap while preserving detail.
    try:
        x64 = np.asarray(las.x, dtype=np.float64)
        y64 = np.asarray(las.y, dtype=np.float64)
        z64 = np.asarray(las.z, dtype=np.float64)
    except (ValueError, AttributeError) as e:
        raise IOError(
            f"LAS file {path} has unreadable x/y/z fields: {e}"
        ) from e

    # Extract format info and all needed arrays from laspy BEFORE releasing it
    point_format_names = [dim.name for dim in las.point_format.dimensions]

    # Extract colors while laspy handle is alive
    colors = None
    if 'red' in point_format_names and 'green' in point_format_names and x64.size > 0:
        r = np.array(las.red, dtype=np.float32)
        g = np.array(las.green, dtype=np.float32)
        b = np.array(las.blue, dtype=np.float32)
        # LAS stores 16-bit colors
        max_val = max(float(r.max()), float(g.max()), float(b.max()), 1.0)
        if max_val > 255.0:
            r /= 65535.0
            g /= 65535.0
            b /= 65535.0
        elif max_val > 1.0:
            r /= 255.0
            g /= 255.0
            b /= 255.0
        colors = np.column_stack([r, g, b])
        del r, g, b

    # Extract scalars while laspy handle is alive
    scalars = {}
    if 'intensity' in point_format_names:
        scalars['intensity'] = np.array(las.intensity, dtype=np.float32)
    if 'return_number' in point_format_names:
        scalars['return_number'] = np.array(las.return_number, dtype=np.float32)
    if 'classification' in point_format_names:
        scalars['classification'] = np.array(las.classification, dtype=np.float32)

    # Release laspy's internal buffers — no further access to `las` after this
    del las

    if x64.size > 0:
        centroid = np.array(
            [x64.mean(), y64.mean(), z64.mean()], dtype=np.float64
        )
    else:
        centroid = np.zeros(3, dtype=np.float64)

    # Center and cast to float32, freeing float64 arrays as soon as
    # possible to cut peak memory from ~4.8 GB to ~2.4 GB for 100M points.
    x32 = (x64 - centroid[0]).astype(np.float32); del x64
    y32 = (y64 - centroid[1]).astype(np.float32); del y64
    # Keep z64 alive — needed for the 'height' scalar below
    z32 = (z64 - centroid[2]).astype(np.float32)
    positions = np.column_stack([x32, y32, z32])
    del x32, y32, z32

    if colors is None:
        colors = np.ones((len(positions), 3), dtype=np.float32)

    # Height (Z) as a scalar for colormapping — use the original (pre-centering)
    # Z so any color ramp keyed on height still makes sense absolutely.
    scalars['height'] = z64.astype(np.float32, copy=False)
    del z64

    return PointCloudData(
        positions=positions,
        colors=colors,
        scalars=scalars,
        file_path=path,
        metadata={'world_offset': centroid.tolist(), 'source_unit': 'm'},
    )
