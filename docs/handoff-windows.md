# Windows handoff — 2026-10-21

Native Windows viewer, built from the same source used for Linux verification. No WSL, Python or CUDA is required for the packaged annotation app.

## Verification

- Windows and Linux suites: **250 passed each**, including GL label rendering.
- Hidden production App workflow: **15 checks passed** (annotation tools, label protection, GPU/disk updates, undo/redo, autosave/snapshots, portable projects, measurements, display/camera, sidebar layout, PNG/spin export).
- Forced-exit recovery tested on 64k points, restoring 714 interrupted-stroke points with a backup of the previous commit.
- Real packages: 54-cloud starter and 175-cloud review set, every cloud 64k points, existing labels. Import hashes, ontology and persisted membership verified. Earlier packaged GUI import of the 175-cloud set was inspected; the final binary is also checked through a hidden timed startup.
- Adaptive profiles retain workstation cache capacity and use bounded caches for lower-memory machines. Full results and limits: [optimization report](optimization-2026-10.md).
- Windows single-row header: logo and fixed product title at the left; menus and window controls in the same row. Original sidebar control sizes remain; large label sets and short windows scroll, and the cloud list absorbs extra height.

## Delivery

`LithiumSetup-1.1.0.exe` is built with PyInstaller and Inno Setup. It installs into the current user's LocalAppData Programs folder without administrator rights and includes README/documentation. The review kit ZIP also includes the two original `.li` packages, original selection notes, checksums, and import/QA results. It is prepared for the user to upload and send; no email or dataset upload is performed automatically.

## Remaining environment requirements

Pointcept training/inference needs its separate configured Python/CUDA environment and a model; it was not executed in this review environment. The model-containing `.li` import path is tested. Open3D mesh reconstruction is optional and absent in this build; failure is reported and the point view remains available. Native CachyOS visual comparison, cross-monitor DPI movement and Windows HDR/profile matching require later desktop inspection.
