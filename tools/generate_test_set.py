"""Generate a set of test PLY files for gallery testing."""

import numpy as np
import os


def write_ply(path, positions, colors):
    n = len(positions)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        header = (
            f"ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\nproperty float y\nproperty float z\n"
            f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"end_header\n"
        )
        f.write(header.encode())
        for i in range(n):
            f.write(np.float32(positions[i, 0]).tobytes())
            f.write(np.float32(positions[i, 1]).tobytes())
            f.write(np.float32(positions[i, 2]).tobytes())
            f.write(np.uint8(colors[i, 0]).tobytes())
            f.write(np.uint8(colors[i, 1]).tobytes())
            f.write(np.uint8(colors[i, 2]).tobytes())
    print(f"  {os.path.basename(path)}: {n:,} points")


def sphere(n=50000):
    phi = np.random.uniform(0, 2 * np.pi, n)
    ct = np.random.uniform(-1, 1, n)
    st = np.sqrt(1 - ct**2)
    pos = np.column_stack([st * np.cos(phi), st * np.sin(phi), ct])
    t = (ct + 1) / 2
    col = np.column_stack([t * 255, (1 - abs(2*t-1)) * 255, (1-t) * 255]).astype(np.uint8)
    return pos, col


def cube(n=50000):
    pos = np.random.uniform(-0.5, 0.5, (n, 3)).astype(np.float32)
    col = ((pos + 0.5) * 255).astype(np.uint8)
    return pos, col


def torus(n=50000, R=1.0, r=0.3):
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.random.uniform(0, 2 * np.pi, n)
    x = (R + r * np.cos(phi)) * np.cos(theta)
    y = r * np.sin(phi)
    z = (R + r * np.cos(phi)) * np.sin(theta)
    pos = np.column_stack([x, y, z]).astype(np.float32)
    col = np.column_stack([
        ((np.cos(theta) + 1) / 2 * 200 + 55).astype(np.uint8),
        ((np.sin(phi) + 1) / 2 * 200 + 55).astype(np.uint8),
        np.full(n, 140, dtype=np.uint8),
    ])
    return pos, col


def helix(n=50000):
    t = np.linspace(0, 8 * np.pi, n)
    x = np.cos(t)
    y = t / (8 * np.pi) * 3 - 1.5
    z = np.sin(t)
    pos = np.column_stack([x, y, z]).astype(np.float32)
    # Add noise
    pos += np.random.normal(0, 0.03, pos.shape).astype(np.float32)
    t_norm = t / t.max()
    col = np.column_stack([
        (t_norm * 255).astype(np.uint8),
        np.full(n, 100, dtype=np.uint8),
        ((1 - t_norm) * 255).astype(np.uint8),
    ])
    return pos, col


if __name__ == '__main__':
    out = 'D:/3Photon/data/test_gallery'
    print(f"Generating test set in {out}/")
    os.makedirs(out, exist_ok=True)

    shapes = [
        ('sphere', sphere),
        ('cube', cube),
        ('torus', torus),
        ('helix', helix),
    ]
    for name, fn in shapes:
        pos, col = fn(60000)
        write_ply(os.path.join(out, f'{name}.ply'), pos, col)

    print("Done!")
