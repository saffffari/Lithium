"""PLY file loader."""

import numpy as np
from plyfile import PlyData

from src.data.point_cloud import PointCloudData


def load_ply(path: str) -> PointCloudData:
    """Load a PLY file and return a PointCloudData instance."""
    try:
        ply = PlyData.read(path)
    except (OSError, ValueError, EOFError) as e:
        raise IOError(f"Failed to read PLY file {path}: {e}") from e

    try:
        vertex = ply['vertex']
    except (KeyError, IndexError) as e:
        raise IOError(f"PLY file {path} has no 'vertex' element: {e}") from e

    try:
        positions = np.column_stack([
            np.asarray(vertex['x'], dtype=np.float32),
            np.asarray(vertex['y'], dtype=np.float32),
            np.asarray(vertex['z'], dtype=np.float32),
        ])
    except (KeyError, ValueError) as e:
        raise IOError(
            f"PLY file {path} is missing x/y/z vertex properties: {e}"
        ) from e

    # Drop any vertices that are NaN / Inf. A single bad value poisons
    # bounds computation and the per-frame clip AABB, which can make an
    # otherwise valid cloud render invisible (everything clipped) even
    # though pick tests against the raw positions still hit points.
    finite_mask = np.isfinite(positions).all(axis=1)
    if not finite_mask.all():
        dropped = int((~finite_mask).sum())
        print(f"PLY {path}: dropping {dropped} non-finite vertices")
        positions = positions[finite_mask]

    # Try to extract colors
    prop_names = {p.name for p in vertex.properties}
    if {'red', 'green', 'blue'} <= prop_names:
        r = np.asarray(vertex['red'], dtype=np.float32)
        g = np.asarray(vertex['green'], dtype=np.float32)
        b = np.asarray(vertex['blue'], dtype=np.float32)
        # Normalize: if max > 1.0, assume 0-255 range
        if r.max() > 1.0 or g.max() > 1.0 or b.max() > 1.0:
            r /= 255.0
            g /= 255.0
            b /= 255.0
        if not finite_mask.all():
            r = r[finite_mask]
            g = g[finite_mask]
            b = b[finite_mask]
        colors = np.column_stack([r, g, b])
        # Degenerate color arrays (common in CT-derived / segmentation
        # PLYs where color is either uninitialized or a near-black
        # placeholder) render as black on black. Fall back to white if
        # either the whole array is very dim or it has essentially no
        # dynamic range.
        cmax = float(colors.max())
        cmin = float(colors.min())
        if cmax < 0.15 or (cmax - cmin) < 0.02:
            colors = np.ones_like(colors)
    else:
        colors = np.ones((len(positions), 3), dtype=np.float32)

    # Extract scalars if available
    scalars = {}
    if 'intensity' in prop_names:
        scalars['intensity'] = np.array(vertex['intensity'], dtype=np.float32)
    if 'scalar_field' in prop_names:
        scalars['scalar_field'] = np.array(vertex['scalar_field'], dtype=np.float32)

    # PLY 'label' / 'class' properties are NOT our semantic label ids —
    # they mean whatever the source tool intended (tree id, segment id,
    # ASPRS-style class). Feeding them into cloud.labels sends every
    # point to an unregistered LUT slot (alpha=0), which the fragment
    # shader discards and the cloud renders as completely invisible.
    # Keep the raw values available as scalars so the user can inspect
    # them, but leave cloud.labels at the default (all zeros = Unlabeled)
    # so freshly-imported points show up immediately. If the catalog has
    # prior labels for this file_key, main.py overlays them after load.
    if 'label' in prop_names:
        scalars['label'] = np.array(vertex['label'], dtype=np.float32)
    if 'class' in prop_names:
        scalars['class'] = np.array(vertex['class'], dtype=np.float32)

    if not finite_mask.all():
        for k, v in list(scalars.items()):
            scalars[k] = v[finite_mask]

    return PointCloudData(
        positions=positions,
        colors=colors,
        scalars=scalars,
        file_path=path,
        metadata={'source_unit': 'raw'},
    )
