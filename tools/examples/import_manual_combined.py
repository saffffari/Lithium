"""Import YOUR hand-labeled bones (dataset_32k scenes) into a 3Photon project with the
COMBINED-pedicle + EXTENDED ontology, so you only add Spinous (and optionally Transverse).

Source of truth: dataset_32k/<split>/scene_*/  (coord = stacker geometry, color, segment =
your hand subregion labels 0-5). Bone NAMES are recovered by matching each scene's coords to
the named exports_32k stacker PLYs (verified 261/261). Labels are remapped:
    0 Unlabeled, 1 Sup_EP, 2 Inf_EP, 3 Pedicle_L, 4 Pedicle_R, 5 Body_Wall   (hand)
 -> 0 Unlabeled, 1 Sup_EP, 2 Inf_EP, 3 Pedicle(L+R merged), 4 Body_Wall, 5 Spinous, 6 Transverse
Pedicles merged because SO(3)-augmented PTv3 can't learn L/R anyway (recovered geometrically).
Spinous(5)+Transverse(6) start empty -> you paint them.

PLYs are written named to exports_32k_manual_combined/ (a reusable, correctly-labeled source),
then imported. Additive + idempotent (content-hash; reuse project by name). NO model attached
(these are already hand-labeled; ultramax's L/R pedicles wouldn't map to the merged class).

Run (app closed): /run/media/alex/board_rack/3Photon/.venv/bin/python tools/import_manual_combined.py [--dry-run]
"""
import sys, argparse, os, glob
from pathlib import Path
import numpy as np
from plyfile import PlyData

TP = Path("/run/media/alex/board_rack/3Photon")
sys.path.insert(0, str(TP))
from src.data import library_paths
from src.data.library_catalog import LibraryCatalog
from src.data.ply_loader import load_ply
from src.data.cloud_store import save_cloud_data, save_cloud_labels
from src.data.labels import LabelRegistry

DS = TP / "dataset_32k"
EXPORTS = "/run/media/alex/citadel/data/verse/exports_32k"
OUTDIR = Path("/run/media/alex/citadel/data/verse/exports_32k_manual_combined")
PROJECT_NAME = "verse_manual_combined"

REMAP = np.array([0, 1, 2, 3, 3, 4], dtype=np.int32)  # hand {0..5} -> combined {0..4}
CLASSES = [  # (id, name, rgb 0-255)
    (1, "Superior_Endplate", (214, 94, 0)),
    (2, "Inferior_Endplate", (0, 115, 178)),
    (3, "Pedicle", (87, 232, 126)),          # L+R merged
    (4, "Body_Wall", (38, 191, 166)),
    (5, "Spinous_Process", (148, 87, 235)),  # empty -> user paints
    (6, "Transverse_Process", (240, 200, 30)),  # empty -> optional
]
NAMES = ["Unlabeled"] + [c[1] for c in CLASSES]
PLY_DT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'),
                   ('b', 'u1'), ('intensity', '<f4'), ('label', '<i4')])
PLY_HDR = (b"ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
           b"property float x\nproperty float y\nproperty float z\n"
           b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
           b"property float intensity\nproperty int label\nend_header\n")


def build_ontology():
    reg = LabelRegistry()
    for lid, name, (r, g, b) in CLASSES:
        reg.add_label_at(lid, name, (r / 255.0, g / 255.0, b / 255.0, 1.0))
    return reg.to_json()


def _xyz(ply):
    with open(ply, 'rb') as f:
        h = b''
        while b'end_header' not in h:
            h += f.readline()
        a = np.frombuffer(f.read(), dtype=PLY_DT)
    return np.stack([a['x'], a['y'], a['z']], 1)


def _sig(xyz):
    c = np.round(xyz.mean(0), 2); bb = np.round(xyz.max(0) - xyz.min(0), 2)
    return (len(xyz),) + tuple(c.tolist()) + tuple(bb.tolist())


def name_map():
    idx = {}
    for p in glob.glob(f"{EXPORTS}/**/*.ply", recursive=True):
        idx[_sig(_xyz(p))] = os.path.basename(p)[:-4]
    return idx


def write_ply(path, coord, color, intensity, label):
    n = len(coord)
    arr = np.empty(n, dtype=PLY_DT)
    arr['x'], arr['y'], arr['z'] = coord[:, 0], coord[:, 1], coord[:, 2]
    arr['r'], arr['g'], arr['b'] = color[:, 0], color[:, 1], color[:, 2]
    arr['intensity'] = intensity
    arr['label'] = label
    with open(path, 'wb') as f:
        f.write(PLY_HDR % n)
        f.write(arr.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scenes = sorted(glob.glob(f"{DS}/*/scene_*/"))
    print(f"{len(scenes)} hand-labeled scenes; building name map from {EXPORTS} ...")
    nm = name_map()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cat = None if args.dry_run else LibraryCatalog()
    proj = None
    if not args.dry_run:
        proj = next((p for p in cat.projects.values() if p.name == PROJECT_NAME), None)
        if proj is None:
            proj = cat.create_project(PROJECT_NAME)
            cat.update_project_ontology(proj.id, build_ontology())
            print(f"  created project '{PROJECT_NAME}' ({proj.id}) — 7-class combined ontology")
        else:
            print(f"  reusing project '{PROJECT_NAME}' ({proj.id})")
        # v2 label layout: label writes land in this project's namespace.
        from src.data import cloud_store
        cloud_store.set_active_label_namespace(proj.id)

    keys, n_named, hist = [], 0, np.zeros(len(NAMES), dtype=np.int64)
    for i, d in enumerate(scenes):
        coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(d, "color.npy")).astype(np.uint8)
        seg = np.load(os.path.join(d, "segment.npy")).astype(np.int64)
        seg = np.clip(seg, 0, 5)
        lab = REMAP[seg].astype(np.int32)
        for c in range(len(NAMES)):
            hist[c] += int((lab == c).sum())
        nme = nm.get(_sig(coord))
        if nme:
            n_named += 1
        nme = nme or f"scene_{i:03d}"
        intensity = color[:, 0].astype(np.float32)  # greyscale; not a model feature
        ply = OUTDIR / f"{nme}.ply"
        if args.dry_run:
            continue
        write_ply(ply, coord, color, intensity, lab)
        e = cat.register_file(str(ply))
        if e is None:
            print(f"  SKIP register {ply.name}"); continue
        cloud = load_ply(str(ply))
        cat.update_metrics(e.file_key, cloud.point_count, cloud.bounds_min, cloud.bounds_max)
        save_cloud_data(e.file_key, cloud, source_path=str(ply))
        e.region = "spine"
        err = save_cloud_labels(e.file_key, lab, catalog=None)
        if err:
            print(f"  LABEL ERR {ply.name}: {err}"); continue
        keys.append(e.file_key)
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(scenes)}")

    if not args.dry_run:
        cat.add_to_project(proj.id, keys)
        cat._save_index()

    tot = max(1, int(hist.sum()))
    print(f"\n{'[dry-run] ' if args.dry_run else ''}bones: {len(scenes)}  named-matched: {n_named}/{len(scenes)}"
          + (f"  imported: {len(keys)} -> '{PROJECT_NAME}' ({proj.id})" if proj else ""))
    print("combined-ontology label distribution (your hand labels, pedicles merged):")
    for c in range(len(NAMES)):
        print(f"  {c}  {NAMES[c]:18} {hist[c]:>11,}  ({100 * hist[c] / tot:5.1f}%)")
    if not args.dry_run:
        print(f"\nPLYs -> {OUTDIR}\nOpen 3Photon, pick '{PROJECT_NAME}', paint Spinous (5) [+ Transverse (6)].")


if __name__ == "__main__":
    main()
