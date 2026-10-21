# Windows and Linux optimization verification — 2026-10-21

The viewer now selects a resource profile instead of assuming workstation hardware. Active annotation stays at full point resolution in every profile. No foreground FPS cap or smaller UI controls were introduced.

## Changes

- Automatic, Lean and Workstation profiles under View > Performance; automatic checks RAM, renderer and reported dedicated GPU memory, with conservative defaults and a manual override for unknown drivers.
- Bounded cold-cloud residency and thumbnail bytes. Active/visible data stays pinned; failed saves prevent eviction. Workstation retains larger warm caches.
- Preview decompression and label projection on a bounded background worker; main-thread uploads and thumbnail rendering use profile-specific frame budgets.
- Superseded async loads are cancelled/discarded and tied to their project namespace. Gallery textures reset when projects change.
- Disabled SSAO targets stay 1x1. The unused selection-outline target is no longer allocated. Minimized windows skip rendering; unfocused windows throttle refresh while processing background work.
- Linux UI backend selection follows the actual GLFW GLX/EGL context before PyOpenGL imports. This fixes mixed X11/Wayland environments such as the tested WSL setup.
- Failed optional mesh builds stop retrying each frame and fall back to visible points with a status message.
- Windows single-row header, original control sizes, a scrollable labels list, and a cloud list that fills spare sidebar height. Short windows can scroll the outer sidebar.
- Interrupted-stroke recovery offers restore/keep-committed, backs up the previous commit, cleans resolved snapshots and preserves project isolation.

## Measured Windows resource test

Real 175-cloud review package; exactly 64,000 points per cloud. Windows, i9-12900K/128 GB, RTX 4090; test process limited to two logical CPUs and below-normal priority. Hidden framebuffer 1626x971; VSync still applies. All 175 thumbnails and full clouds were visited. Labels were compared with disk after every load; an evicted cloud was reloaded, painted and undone.

| Measurement | Lean | Automatic -> Workstation |
|---|---:|---:|
| Project metadata open | 19 ms | 12 ms |
| Full-resolution clouds retained | 3 | 175 |
| Retained full-cloud CPU + GPU payload | 13.9 MiB | 811.8 MiB |
| Working set after browsing | 271 MiB | 1110 MiB |
| Process private memory after browsing | 485 MiB | 1793 MiB |
| Gallery frame median / p95 | 16.6 / 26.5 ms | 16.6 / 17.4 ms |
| Light Table frame median / p95 | 16.7 / 17.3 ms | 16.7 / 17.6 ms |
| Cold cloud load median / p95 | 33 / 45 ms | 47 / 52 ms |
| Paint 256 points, including save | 5.9 ms | 2.7 ms |

These are one controlled run per profile, not a statistical hardware ranking. Lean releases memory for smaller machines; Workstation retains hot data for revisits. Two-core affinity does not emulate a slower CPU or GPU. No 8 GB RAM machine was physically tested, and cache payload limits are not total-process memory limits.

## Linux and integrated graphics

Ubuntu 22.04 under WSL, Python 3.12.13, Mesa 23.2.1, OpenGL 4.1, **D3D12 Intel UHD Graphics 770**, two logical CPUs. The hidden production app loaded a real 64k vertebra, painted 256 points, verified disk persistence and undid the edit. Automatic selected Lean. Twenty-five frames at 1280x720 measured median **5.6 ms**, p95 **7.4 ms**, with no GL errors. This verifies that the point workflow runs on the integrated GPU through WSL's Linux graphics path. Hidden rendering timings do not promise visible-window FPS, and this does not certify native CachyOS display/DPI behaviour.

## Correctness and packaging coverage

- **250 passed on Windows; 250 passed on Linux**, including actual GL 4.1 label-buffer/render tests on both hosts.
- Production hidden-App checks: pick, box, lasso, polygon, curve, brush; GPU/disk/undo/redo; label locks; erase-only-active-label; autosave; timestamped snapshots; edited `.li` round trip; measurements; camera/display controls; layout; still/spin export. **15 passed**.
- Forced process exit during an unfinished 64k-point stroke: canonical labels stayed intact; a fresh process restored the 714 captured points and backed up the previous commit.
- The recovered 54/175-cloud bundles passed SHA-256, array-length, ontology, membership and ZIP CRC validation. Model bundles preserve checkpoint/config bytes, frozen classes, registration and active selection.
- Hidden native Windows resize checks at four sizes returned no GL errors. Larger label sets scroll within the list; short sidebars may scroll, and tall sidebars expand the cloud list without bottom padding.

## Limits

Actual Pointcept training/inference was not run because its separate backend is not configured. Kevin's recovered data packages contain no model weights. Open3D reconstruction is optional and absent from this viewer build; cached meshes can display, while unavailable reconstruction reports a message and keeps points visible. Cross-monitor DPI dragging and Windows HDR/monitor-profile matching were not tested while the user needed the desktop. No color/shader grading changes were made without a native Linux visual baseline.

The README contains installation, operations, model transfer, backup/recovery and troubleshooting instructions. The Windows installer is per-user and includes the README and Markdown documentation. The handoff kit supplies both original data packages and the tested installer; dataset files are not committed to GitHub.
