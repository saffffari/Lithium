"""Generate a test PLY file with a colored sphere point cloud."""

import numpy as np
import os

def generate_sphere_ply(path: str, n_points: int = 100000):
    """Generate a sphere point cloud with gradient colors."""
    # Random points on a unit sphere
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    cos_theta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(cos_theta)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = cos_theta

    # Color by height (z): blue at bottom, red at top
    t = (z + 1) / 2  # normalize to [0, 1]
    r = (t * 255).astype(np.uint8)
    g = ((1 - abs(2 * t - 1)) * 255).astype(np.uint8)
    b = ((1 - t) * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n_points):
            f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {r[i]} {g[i]} {b[i]}\n")

    print(f"Generated {n_points:,} points -> {path}")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", default="test_sphere.ply")
    ap.add_argument("--points", type=int, default=100000)
    args = ap.parse_args()
    generate_sphere_ply(args.output, n_points=args.points)
