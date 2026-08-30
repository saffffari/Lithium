# Lithium — open-source, licensing and publication brief (2026-08-29)

## 1. License: where we actually stand
- **Code: MIT** (`LICENSE`, © 2026 Alexander Saffari). Right choice for a free research tool.
- **One real blocker: `plyfile` is GPL-3** and Lithium imports it in `src/data/ply_loader.py`,
  `src/data/ply_writer.py` and three tools. An MIT app that links a GPL library — and above all the
  PyInstaller binaries that *bundle* it — is a GPL-compatibility problem. Two fixes:
  (a) replace plyfile with an in-house PLY reader/writer (binary + ASCII little-endian PLY is
  ~150 lines of numpy) and keep MIT — **recommended, one day of work**; or
  (b) relicense Lithium GPL-3 (consistent with the 2Photon "GPLv3 — compile it yourself" strip,
  but it costs adoption in labs that ship internal tooling).
- **Everything else is permissive:** ModernGL MIT · GLFW zlib · pyimgui BSD · laspy BSD-2 ·
  lazrs MIT/Apache · numpy/scipy BSD · Pillow MIT-CMU · PyOpenGL BSD · h5py BSD · trimesh MIT ·
  DracoPy Apache-2 · pygltflib MIT · wgpu BSD-2 · PyInstaller GPL *with the bootloader exception*
  (bundling any license is allowed) · Pointcept MIT (cloned by the user, never vendored — keep it so).
- **Data: VerSe'19/'20 are CC BY-SA 4.0.** Gold247 (surfaces sampled from VerSe segmentations + our
  labels) is a derivative → release it **CC BY-SA 4.0 with VerSe attribution**. Patient cohorts
  (EPIC; any Cedars-sourced part of the incoming 15k set) never leave the building.
- **Weights:** *Yamato* (PT-v3m1 trained from scratch on VerSe-derived data) is clean — release it.
  Whether share-alike propagates from training data into weights is legally unsettled; the
  defensible, simple line is **weights + dataset CC BY-SA 4.0, code MIT**. *Sonata fine-tune*: derived
  from `facebook/sonata`, whose Hugging Face card declares **no license tag** → do **not**
  redistribute Sonata-derived weights until Meta's terms are confirmed; publish the fine-tuning
  recipe/config instead (users fetch Meta's weights themselves). A *next-class* model trained from
  scratch (or from Yamato) on the 15k cohort side-steps this entirely and is the better headline.
- **Employer/IP:** Lithium time is billed as Cedars scope, so Cedars-Sinai's IP policy almost
  certainly covers it; the repo is public under MIT as of today. Before *promoting* a release or
  submitting a paper, get the written OK from Cedars Technology Transfer (open-source release of
  research software is routine but usually needs a disclosure form) and settle the copyright line
  (personal vs. "© Cedars-Sinai Medical Center / A. Saffari"). Same flag as xtracer.

## 2. How to release it properly (in order)
1. Fix `plyfile` (swap or relicense); add `THIRD_PARTY_LICENSES.md` (pip-licenses output).
2. Repo hygiene JOSS reviewers check: `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
   `CHANGELOG.md`, install docs, tests (217 pass), issue templates.
3. `pyproject.toml` + PyPI: the name `lithium` is **taken** (a Django package) → publish as
   `lithium-workstation` (console entry point `lithium`). Keep the product name.
4. Enable GitHub Actions on the repo (no run fired for the v1.1 tag) → CI builds Linux binary +
   Windows installer on every `v*` tag; cut **v1.2** with the sandbox.
5. Zenodo ↔ GitHub integration → DOI per release; DOI into README, `CITATION.cff`, paper.
6. Hugging Face: `safffari/lithium-yamato-gold247` — weights-only (strip optimizer state:
   530 MB → ~200 MB; safetensors), model card with class map, val numbers, data license.
   Dataset: `Gold247` on Zenodo or HF Datasets (npz + manifest + classes + labels, CC BY-SA 4.0).
7. Fresh-clone test on a second machine (the Windows laptop via the F-35 stick) before announcing.
8. Announce: arXiv link + GitHub release + a 60-second screen capture; 2photon.io tile → repo.

## 3. Highest-impact realistic venue
- **For the *combined* method + tool + benchmark paper: Medical Image Analysis** (IF ≈ 10) or
  **Radiology: Artificial Intelligence** (IF ≈ 8; TotalSegmentator's home; rewards released tools
  with clinical validation). IEEE TMI third. Reachable if the paper carries (i) a subject-split
  public benchmark with released weights, (ii) a downstream measurement validation (frames →
  angles vs. annotator variability), (iii) the timing study.
- **For Lithium alone: JOSS** (fast, reviewers run the software, citable) + arXiv; **CMPB** (IF ≈ 5)
  if an impact factor matters. *Nature Methods* is the ceiling for tool papers (ilastik, napari)
  but needs a demonstrated user community and a broad-biology story — not v1.
- **Gold247 alone: Scientific Data** (IF ≈ 6) — a separate, citable unit.
- Spine-clinical angle (SRS frames, Lenke): Spine Deformity / European Spine Journal — lower IF,
  right audience for Cedars.

## 4. One paper or two? — two, plus a dataset descriptor, in this order
1. **Now — Lithium tools paper** (this draft): arXiv cs.CV as soon as plyfile is fixed and Cedars
   says yes; submit to JOSS. Add the timing study when it exists (JOSS does not require it).
2. **Next — the impact paper: "Point Transformers for vertebral surface anatomy"**: Gold247
   benchmark (subject split), Yamato vs. Sonata vs. next-class model, **released weights**, frames
   from labels → SRS/Lenke measurements validated against annotator variability, generalisation to
   the 15k cohort / EPIC (aggregate only if PHI). Target MedIA / Radiology:AI. Lithium is cited as
   the platform in one paragraph, not re-described.
3. **Gold247 dataset descriptor** (Scientific Data), or fold it into paper 2 as the benchmark.

Why not one combined paper: tool and method papers are reviewed by different people with different
demands; combined, the tool gets buried under "pedicle precision is 0.63" and the method gets
attacked for the UI pages. Two short papers = two citable units, and tool citations compound.

Pre-conditions before paper 2 can hit MedIA: a real fix for the pedicle→lamina boundary (label
convention, then retrain); a downstream-measurement validation; and more held-out subjects than the
current 2 val / 2 test (report 5-fold subject CV, or add the 15k cohort's public part).
