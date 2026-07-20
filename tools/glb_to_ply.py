"""Convert GLB 3D models to colored point cloud PLY files.

Handles KHR_draco_mesh_compression (Draco) which is common in NASA 3D models.
Samples points uniformly on mesh surfaces, resolves colors from PBR materials,
and writes PLY with XYZ + RGB.
"""

import argparse
import os
import sys
import time

import numpy as np
import pygltflib
import DracoPy


def decode_glb(glb_path: str) -> list[dict]:
    """Load a GLB, decode Draco-compressed meshes, apply node transforms.

    Returns list of dicts with keys: vertices, faces, color (RGB uint8 tuple).
    """
    glb = pygltflib.GLTF2().load(glb_path)
    blob = glb.binary_blob()

    # Build node transform tree
    node_transforms = _compute_world_transforms(glb)

    meshes = []
    for node_idx, node in enumerate(glb.nodes):
        if node.mesh is None:
            continue
        world_tf = node_transforms[node_idx]
        mesh_def = glb.meshes[node.mesh]

        for prim in mesh_def.primitives:
            verts, faces = _decode_primitive(glb, blob, prim)
            if verts is None or len(verts) == 0:
                continue

            # Apply node world transform
            ones = np.ones((len(verts), 1), dtype=np.float32)
            verts_h = np.hstack([verts, ones])
            verts = (world_tf @ verts_h.T).T[:, :3]

            # Get material color
            color = (160, 160, 160)  # fallback
            if prim.material is not None and prim.material < len(glb.materials):
                mat = glb.materials[prim.material]
                if mat.pbrMetallicRoughness and mat.pbrMetallicRoughness.baseColorFactor:
                    f = mat.pbrMetallicRoughness.baseColorFactor
                    color = (int(f[0] * 255), int(f[1] * 255), int(f[2] * 255))

            meshes.append({
                'vertices': verts.astype(np.float32),
                'faces': faces,
                'color': color,
            })

    return meshes


def _decode_primitive(glb, blob, prim):
    """Decode a mesh primitive, handling Draco compression."""
    if prim.extensions and 'KHR_draco_mesh_compression' in prim.extensions:
        draco_ext = prim.extensions['KHR_draco_mesh_compression']
        bv_idx = draco_ext['bufferView']
        bv = glb.bufferViews[bv_idx]
        draco_data = blob[bv.byteOffset:bv.byteOffset + bv.byteLength]

        try:
            mesh_obj = DracoPy.decode(draco_data)
            verts = np.array(mesh_obj.points, dtype=np.float32).reshape(-1, 3)
            faces = np.array(mesh_obj.faces, dtype=np.int32).reshape(-1, 3)
            return verts, faces
        except Exception:
            return None, None

    # Non-Draco: read from standard accessors
    if prim.attributes.POSITION is None:
        return None, None

    verts = _read_accessor(glb, blob, prim.attributes.POSITION)
    if verts is None:
        return None, None
    verts = verts.reshape(-1, 3)

    if prim.indices is not None:
        indices = _read_accessor(glb, blob, prim.indices)
        if indices is not None:
            faces = indices.reshape(-1, 3)
        else:
            faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)
    else:
        faces = np.arange(len(verts), dtype=np.int32).reshape(-1, 3)

    return verts.astype(np.float32), faces.astype(np.int32)


def _read_accessor(glb, blob, acc_idx):
    """Read data from a glTF accessor."""
    acc = glb.accessors[acc_idx]
    if acc.bufferView is None:
        return None

    bv = glb.bufferViews[acc.bufferView]
    comp_type = acc.componentType
    # 5126=FLOAT, 5123=UNSIGNED_SHORT, 5125=UNSIGNED_INT, 5121=UNSIGNED_BYTE
    dtype_map = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8}
    dtype = dtype_map.get(comp_type, np.float32)

    type_count = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
    count = type_count.get(acc.type, 1)

    item_size = np.dtype(dtype).itemsize * count
    stride = bv.byteStride or item_size
    offset = bv.byteOffset + (acc.byteOffset or 0)

    if stride == item_size:
        data = np.frombuffer(blob, dtype=dtype, count=acc.count * count, offset=offset)
    else:
        data = np.empty(acc.count * count, dtype=dtype)
        for i in range(acc.count):
            start = offset + i * stride
            chunk = np.frombuffer(blob, dtype=dtype, count=count, offset=start)
            data[i * count:(i + 1) * count] = chunk

    return data


def _compute_world_transforms(glb) -> list[np.ndarray]:
    """Compute world transform for each node by traversing the scene graph."""
    n = len(glb.nodes)
    local_tfs = []
    for node in glb.nodes:
        if node.matrix:
            tf = np.array(node.matrix, dtype=np.float64).reshape(4, 4).T  # glTF is column-major
        else:
            tf = np.eye(4, dtype=np.float64)
            if node.scale:
                s = node.scale
                tf[:3, :3] *= np.array([s[0], s[1], s[2]])
            if node.rotation:
                q = node.rotation  # [x, y, z, w]
                tf[:3, :3] = tf[:3, :3] @ _quat_to_mat3(q)
            if node.translation:
                tf[:3, 3] = node.translation
        local_tfs.append(tf)

    # Build parent map
    parent = [-1] * n
    for i, node in enumerate(glb.nodes):
        if node.children:
            for c in node.children:
                parent[c] = i

    # Compute world transforms
    world_tfs = [None] * n
    for i in range(n):
        _resolve_world(i, local_tfs, world_tfs, parent)

    return world_tfs


def _resolve_world(idx, local_tfs, world_tfs, parent):
    if world_tfs[idx] is not None:
        return
    p = parent[idx]
    if p < 0:
        world_tfs[idx] = local_tfs[idx].copy()
    else:
        _resolve_world(p, local_tfs, world_tfs, parent)
        world_tfs[idx] = world_tfs[p] @ local_tfs[idx]


def _quat_to_mat3(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def sample_mesh_colored(vertices: np.ndarray, faces: np.ndarray,
                        color: tuple, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly on a triangle mesh surface.

    Returns (points [N,3], colors [N,3] uint8).
    """
    if len(faces) == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Triangle areas
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area < 1e-12:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # Sample faces proportional to area
    probs = areas / total_area
    face_indices = np.random.choice(len(faces), size=count, p=probs)

    # Random barycentric coordinates
    r1 = np.random.random(count).astype(np.float32)
    r2 = np.random.random(count).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    sampled_v0 = v0[face_indices]
    sampled_v1 = v1[face_indices]
    sampled_v2 = v2[face_indices]
    points = u[:, None] * sampled_v0 + v[:, None] * sampled_v1 + w[:, None] * sampled_v2

    colors = np.tile(np.array(color, dtype=np.uint8), (count, 1))
    return points.astype(np.float32), colors


def boost_colors(colors: np.ndarray) -> np.ndarray:
    """Brighten PBR base colors with a cool-shifted tone.

    Applies per-channel gamma: blue gets the most boost, red the least.
    This shifts dark greys/browns toward steely blues while preserving
    warm accents like gold mirrors.
    """
    c = colors.astype(np.float32) / 255.0
    c[:, 0] = np.power(np.maximum(c[:, 0], 0), 1.0 / 2.0)   # red: moderate
    c[:, 1] = np.power(np.maximum(c[:, 1], 0), 1.0 / 2.3)   # green: moderate
    c[:, 2] = np.power(np.maximum(c[:, 2], 0), 1.0 / 3.0)   # blue: strong boost
    return np.clip(c * 255.0, 0, 255).astype(np.uint8)


def glb_to_point_cloud(glb_path: str, target_points: int = 500_000) -> tuple[np.ndarray, np.ndarray]:
    """Convert a GLB to a colored point cloud.

    Returns (points [N,3] float32, colors [N,3] uint8).
    """
    meshes = decode_glb(glb_path)
    if not meshes:
        raise ValueError(f"No meshes found in {glb_path}")

    # Calculate surface area per mesh for proportional sampling
    areas = []
    valid_meshes = []
    for m in meshes:
        v0 = m['vertices'][m['faces'][:, 0]]
        v1 = m['vertices'][m['faces'][:, 1]]
        v2 = m['vertices'][m['faces'][:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross, axis=1).sum()
        if area > 1e-12:
            areas.append(area)
            valid_meshes.append(m)

    if not valid_meshes:
        raise ValueError(f"No valid meshes with area in {glb_path}")

    total_area = sum(areas)
    counts = [max(1, int(target_points * a / total_area)) for a in areas]

    all_points = []
    all_colors = []
    for m, count in zip(valid_meshes, counts):
        pts, cols = sample_mesh_colored(
            m['vertices'], m['faces'], m['color'], count
        )
        if len(pts) > 0:
            all_points.append(pts)
            all_colors.append(cols)

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    colors = boost_colors(colors)

    # Scale model so max extent ~ 100 units for good point cloud rendering
    extent = points.max(axis=0) - points.min(axis=0)
    max_extent = extent.max()
    if max_extent > 0:
        scale = 100.0 / max_extent
        center = (points.max(axis=0) + points.min(axis=0)) / 2.0
        points = (points - center) * scale

    return points, colors


def write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Write a binary PLY with XYZ + RGB."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                          ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        data = np.empty(n, dtype=dtype)
        data['x'] = points[:, 0]
        data['y'] = points[:, 1]
        data['z'] = points[:, 2]
        data['r'] = colors[:, 0]
        data['g'] = colors[:, 1]
        data['b'] = colors[:, 2]
        f.write(data.tobytes())


def convert_single(glb_path: str, output_path: str, target_points: int = 500_000):
    """Convert a single GLB to PLY."""
    t0 = time.time()
    print(f"  Loading: {os.path.basename(glb_path)}")
    points, colors = glb_to_point_cloud(glb_path, target_points)
    print(f"  Sampled: {len(points):,} points ({time.time() - t0:.1f}s)")

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    write_ply(output_path, points, colors)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved:   {output_path} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Convert GLB 3D models to colored point cloud PLY files')
    parser.add_argument('input', help='GLB file or directory containing GLB files')
    parser.add_argument('-o', '--output', default='./data/nasa_spacecraft',
                        help='Output directory for PLY files')
    parser.add_argument('-n', '--points', type=int, default=500_000,
                        help='Target points per model (default: 500000)')
    args = parser.parse_args()

    if os.path.isfile(args.input):
        name = os.path.splitext(os.path.basename(args.input))[0]
        out = os.path.join(args.output, f"{name}.ply")
        convert_single(args.input, out, args.points)
    elif os.path.isdir(args.input):
        glb_files = []
        for root, dirs, files in os.walk(args.input):
            for f in files:
                if f.lower().endswith('.glb'):
                    glb_files.append(os.path.join(root, f))
        glb_files.sort()
        print(f"Found {len(glb_files)} GLB files")
        for i, glb in enumerate(glb_files):
            name = os.path.splitext(os.path.basename(glb))[0]
            out = os.path.join(args.output, f"{name}.ply")
            print(f"\n[{i+1}/{len(glb_files)}] {name}")
            try:
                convert_single(glb, out, args.points)
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print(f"Path not found: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
