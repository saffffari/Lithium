"""Point cloud → triangulated mesh via Poisson surface reconstruction.

Used by the IMAGING ingest pipeline and the HOLOGRAM display mode to
produce a watertight surface representation of each per-vertebra cloud.
For our anatomical clouds (~25-40k points, surface-sampled, reasonably
uniform density) Poisson at ``depth=9`` gives clinically-recognisable
surfaces in ~1-2 seconds per cloud.

Per-vertex label mapping: each mesh vertex is assigned the label of its
nearest source point via cKDTree. Lets the mesh shader tint by class
exactly like the point shader does, so the visual transition between
POINTS and MESH display modes is purely a representation change, not a
data change.

Pipeline:
    points (N,3) + colors (N,3) + labels (N,)
       └→ open3d PointCloud with estimated normals
           └→ Poisson reconstruction
               └→ mesh verts (M,3), faces (T,3), normals (M,3),
                  vertex_labels (M,)

If open3d is unavailable, ``build_mesh`` raises ``MeshBuildUnavailable``
with an actionable message. Callers should catch and queue a graceful
"build pending" fallback rather than crashing the UI.
"""

from __future__ import annotations

import numpy as np


class MeshBuildUnavailable(RuntimeError):
    """Raised when open3d (or its alternatives) aren't installed."""


def _have_open3d() -> bool:
    try:
        import open3d  # noqa: F401
        return True
    except ImportError:
        return False


def build_mesh(
    positions: np.ndarray,
    colors: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    *,
    depth: int = 9,
    normal_knn: int = 16,
) -> dict:
    """Reconstruct a triangulated surface from a point cloud.

    Returns a dict with keys::

        verts          (M, 3) float32  — mesh vertex positions
        faces          (T, 3) int32    — triangle indices into verts
        normals        (M, 3) float32  — per-vertex outward normals
        vertex_labels  (M,)   int32    — label id per vertex (0 if no
                                          source labels were provided)

    Parameters
    ----------
    positions
        Source point cloud, shape ``(N, 3)``.
    colors
        Optional per-point RGB in ``[0, 1]``, shape ``(N, 3)``. Used only
        to seed normal orientation in pathological cases; not stored.
    labels
        Optional per-point int label ids, shape ``(N,)``. Each mesh
        vertex is tagged with its nearest source point's label, so
        shader-side label tinting reads the right class without needing
        the original point cloud at render time.
    depth
        Poisson octree depth. 9 produces clean detail for vertebra-scale
        clouds; 7 for sparse / noisy inputs; 11 only if you really need
        sub-millimetre detail and have the patience for it.
    normal_knn
        KNN for the surface normal estimator that feeds Poisson. 16 is
        a sane default for VerSe-density clouds.

    Raises
    ------
    MeshBuildUnavailable
        If the open3d backend isn't installed.
    ValueError
        If ``positions`` is empty or malformed.
    """
    if not _have_open3d():
        raise MeshBuildUnavailable(
            "Mesh build requires open3d. Install with `pip install open3d` "
            "into the same venv that runs 3Photon."
        )
    import open3d as o3d

    if positions is None or len(positions) == 0:
        raise ValueError("Cannot build a mesh from an empty point cloud.")
    pos = np.ascontiguousarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N, 3); got {pos.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=int(normal_knn))
    )
    # Orient normals consistently using a tangent-plane voting pass.
    # Cheaper alternatives (orient_normals_towards_camera) flip half the
    # vertebra when the camera is anywhere but dead-on, which is most of
    # the time. Tangent-plane voting is robust on closed-ish surfaces.
    try:
        pcd.orient_normals_consistent_tangent_plane(k=int(normal_knn))
    except RuntimeError:
        # Pathological clouds (e.g. all-coplanar) can throw; the mesh
        # will still build, just with some normals flipped. Better than
        # failing the whole build.
        pass

    mesh, _densities = (
        o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=int(depth), linear_fit=False)
    )
    # Poisson over-extends past the point hull. Crop to the cloud's
    # bounding box with a small margin so we don't keep the skin that
    # extrapolates into empty space.
    bb_min = pos.min(axis=0)
    bb_max = pos.max(axis=0)
    extent = float(np.linalg.norm(bb_max - bb_min))
    margin = max(extent * 0.02, 1e-3)
    crop = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=bb_min - margin, max_bound=bb_max + margin,
    )
    mesh = mesh.crop(crop)

    # Recompute vertex normals after crop (Poisson's are stale).
    mesh.compute_vertex_normals()

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    # Map each mesh vertex to its nearest source-point label.
    if labels is not None and len(labels) == len(positions):
        from scipy.spatial import cKDTree
        # cKDTree on the source points; query each mesh vertex.
        tree = cKDTree(np.asarray(positions, dtype=np.float32))
        # workers=4 keeps the GUI responsive on a workstation without
        # pinning every core; we hit cKDTree from a background thread
        # anyway, but be polite.
        _, idx = tree.query(verts, k=1, workers=4)
        vertex_labels = np.asarray(labels, dtype=np.int32)[idx]
    else:
        vertex_labels = np.zeros(len(verts), dtype=np.int32)

    return {
        "verts": np.ascontiguousarray(verts, dtype=np.float32),
        "faces": np.ascontiguousarray(faces, dtype=np.int32),
        "normals": np.ascontiguousarray(normals, dtype=np.float32),
        "vertex_labels": np.ascontiguousarray(vertex_labels, dtype=np.int32),
    }


def build_mesh_from_cloud(cloud, **kwargs) -> dict:
    """Convenience wrapper that accepts a ``PointCloudData`` instance.

    Pulls positions, colors, labels directly off the cloud. Keeps the
    builder's primary surface positional-array-only for testability.
    """
    return build_mesh(
        positions=cloud.positions,
        colors=getattr(cloud, "colors", None),
        labels=getattr(cloud, "labels", None),
        **kwargs,
    )
