# 1.1 performance overhaul — massive clouds

Goal: handle clouds as big as this hardware allows (RTX 4090 24 GB,
128 GB RAM, 24 cores, NVMe) while keeping annotation ergonomics.
Proof point: the full 541-tile Palisades post-fire collection
(~13–14 B points, 52 GB LAZ) at `/run/media/alex/citadel/data/
los_angeles_points/palisades_postfire_C25/`.

Validated by the wgpu spike (`prototypes/wgpu/`): 316 M points
brute-forced at ~29 fps @1080p on Vulkan, 5 GB VRAM upload in 1.5 s.
Brute force tops out around there — the full collection is ~40× that,
so the architecture below is about *never needing* all points resident.

## Ceilings (this machine)

| Path | Limit | Bound by |
|---|---|---|
| 1.0 renderer (32 B/pt resident) | ~500–600 M pts | VRAM |
| wgpu brute force (16 B/pt resident) | ~1.2 B pts, fps sinking | VRAM + raster |
| Out-of-core LOD (below) | disk-bound (100 B+ pts) | none interactive |

## Keep

- **4-buffer split idea** — becomes vertex buffers (pos/color) +
  storage buffers (label/selection) per chunk in wgpu.
- **FOCUS (DOF) + tone controls** — screen-space post passes; cost is
  resolution-, not point-count-dependent. Ported, not redesigned.
- **HDR → SSAO/EDL → tonemap chain**, gallery preview cache, catalog +
  v2 label namespaces, ImGui GUI, per-stroke label save-through
  (chunk-local — see below).
- mm units + PTv3 training flow (with tiled export, below).

## Remove / demote

- **npz-compressed monolithic cloud storage** for big clouds — zlib
  decompress of multi-GB blobs on open. Replace with the chunked store;
  small clouds (< ~30 M pts) keep the legacy path untouched.
- **Full-array label rewrite per stroke** — a stroke on a 500 M-pt
  cloud would rewrite 2 GB. Becomes per-chunk dirty writes.
- **CPU KD-tree paths at full res** (propagation, preview→full
  projection) — voxel-hash / GPU variants for chunked clouds.
- **Poisson mesh build** gated by point count (unbounded RAM).
- **CPU numpy picking** — GPU compute over visible chunks only.

## Add — the chunked LOD store ("c-store")

Potree-style, but tile-grid based (matches how massive data arrives):

```
<cloud>.3pc/
  index.json                 tiles, bounds, per-level counts, quantization
  L0/<tile>.bin              full-res points, 12 B/pt quantized
  L1/<tile>.bin              1/4 density (voxel-decimated)
  L2..Ln/                    1/16, 1/64, ... down to ~100 k pts total
  labels/<namespace>/<tile>.u8   per-tile label bytes (chunk-local writes)
```

- **12 B/pt**: uint16×3 position quantized to tile bounds (≤ 1.2 cm at
  750 m tiles — below sensor noise), uint16 pad, unorm8×4 color.
  Decoded in the vertex shader. 25 % of 1.0's 48 B/pt (CPU+GPU).
- **Renderer**: frustum-cull tiles → pick level by projected
  screen-space error → stream missing buffers via a background IO
  thread → draw within a **point budget** (default ~150 M, slider).
  LRU-evict GPU buffers over budget. Camera stills refine to L0.
- **Labels**: uint8/uint16 per point per tile per namespace; a brush
  stroke marks only touched tiles dirty; save-through writes ~10 MB,
  not gigabytes. Undo stores per-tile sparse diffs with a RAM cap.
- **Picking/brush**: compute pass over resident tiles (the only ones
  the user can see to click) writing hit ids to a small readback
  buffer. Sub-ms vs 11–74 ms CPU today.
- **Import**: parallel LAZ/PLY → c-store conversion (workers = cores),
  ~GB/min class throughput; progress in the IMPORT section.
- **Training export**: spatial tiling straight from c-store tiles —
  also fixes the `extent/grid_size < 65536` ceiling for big scenes.

## Sequencing

1. `prototypes/wgpu/build_lod.py` — LAZ → c-store converter (now).
2. `prototypes/wgpu/demo_lod.py` — streaming budget renderer over the
   full Palisades collection (now).
3. Renderer seam in `main.py` (App submits scenes, owns no GL), then
   `src/render2/` production port of the spike + ImGui-on-wgpu
   migration (pyimgui → imgui-bundle).
4. c-store as an import option in the catalog; labels + picking; LOD
   export to training.

Steps 1–2 prove the architecture on real data before any of 3Photon
proper changes; 3–4 are the integration milestones.
