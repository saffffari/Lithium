# Lithium: An End-to-End Desktop Platform for Point Cloud Annotation and Vision Model Training

**Authors:** [TBD]
**Affiliation:** [TBD]
**Status:** Draft

## Abstract

Point cloud semantic segmentation models such as PointNet++, Point Transformer, and PTv3 have achieved strong performance on benchmark datasets, but creating labeled training data for new domains remains a manual bottleneck. Existing annotation tools fall into two categories: cloud-based commercial platforms that prohibit use on sensitive data (medical imaging under IRB, proprietary industrial scans), and research prototypes with interfaces that present substantial learning barriers. We present Lithium, a free desktop application that unifies the full workflow: import from point clouds (PLY, LAS/LAZ), medical volumes (TIF z-stacks), and time-series sequences; precise interactive selection via six complementary tools (pick, box, lasso, brush, polygon, curve) with depth limiting; hierarchical label layers; per-frame label propagation for 4D sequences; direct export to PTv3-compatible dataset formats; and in-app training via a sidecar subprocess architecture. The GPU-accelerated rendering supports millions of points at interactive frame rates, and all processing remains local to the user's machine. We demonstrate the tool on a representative spinal anatomy labeling task and report baseline segmentation metrics trained end-to-end from within the application.

## 1. Introduction

The 3D computer vision community has produced increasingly capable point cloud models, from PointNet (Qi et al. 2017) through Point Transformer V3 (Wu et al. 2024), achieving state-of-the-art results on ScanNet, S3DIS, and similar indoor datasets. Applying these models to new domains — medical imaging, forestry, industrial metrology, cultural heritage — requires labeled training data in the target domain, which must be created manually.

The tooling for manual annotation has not kept pace with the models. Common workflows involve:

1. **Open3D scripts** or **PPTK** viewers with minimal selection capabilities.
2. **CloudCompare** with a steep learning curve and a dated interface.
3. **Cloud platforms** (Segments.ai, Labelbox, Pointly) that are expensive, browser-based, and prohibit uploading sensitive data.
4. **Custom one-off Python scripts** written per project.

None of these tools cover the full workflow: import, annotate, export, train, iterate. Every lab reinvents the pipeline. This paper presents Lithium, which fills this gap with a single desktop application.

## 2. Related Work

[TODO: expand]

- **ilastik** (Berg et al. 2019): the closest analogue in the voxel/image world. Interactive machine learning for biological image segmentation. We take inspiration from its interactive correction loop but apply it to point clouds.
- **nnInteractive** (Isensee et al. 2024): similar for medical volumes, but voxel-based.
- **Pointcept** (Wu et al. 2024): the framework for training PTv3 and related models. Lithium exports directly to Pointcept's expected format.
- **PPTK**: a viewer with lasso selection, last released in 2020.
- **Segments.ai**: commercial cloud platform with model-assisted labeling.
- **BasicAI / Kognic / Deepen**: commercial 3D annotation platforms focused on autonomous driving.

## 3. System Design

### 3.1 Architecture Overview

Lithium is implemented in Python 3.12 with ModernGL for rendering, GLFW for windowing, and Dear ImGui for GUI. The application is organized into three tabs (Figure 1):

- **Contact Sheets**: gallery view showing all loaded clouds
- **Light Table**: single-cloud annotation view
- **Automation**: embedded CLI for scripting and bulk operations

### 3.2 GPU Rendering Pipeline

Point clouds are uploaded to GPU memory as interleaved vertex buffers containing position (3×float32), color (3×float32), label (1×int32), and selection state (1×float32) — 32 bytes per vertex. A custom GLSL shader samples a 256×1 RGBA lookup table for label colors, allowing the user to toggle between raw point cloud colors and semantic label overlays without re-uploading data.

The label texture supports:
- Per-label RGBA colors
- Visibility control (hidden labels discard the fragment)
- Up to 255 concurrent labels

### 3.3 Selection Tools

Six complementary tools let users express the shape of the region they want to label:

- **Pick**: nearest-neighbor screen-space point within a pixel radius
- **Box**: axis-aligned rectangle
- **Lasso**: freeform polygon via winding-number test
- **Brush**: 3D sphere via KD-tree radius query
- **Polygon**: click-to-place polygon vertices
- **Curve**: polyline with perpendicular distance threshold — novel, for edge annotation

All screen-space tools support optional depth limiting: points farther from the camera than a user-adjustable threshold are excluded. This solves the common pain point of accidentally labeling points on the far side of a structure.

Selection is stored as a boolean mask over all points, with three modes (replace, add, subtract) mapped to modifier keys.

### 3.4 Hierarchical Label System

Labels are integer IDs (1-255) with associated metadata: name, RGBA color, parent ID, visibility, lock state. The parent ID enables nested hierarchies (e.g., `Vertebrae > Thoracic > T7 > Spinous Process`) rendered as a tree in the UI.

The default color palette uses Okabe-Ito colorblind-safe colors.

### 3.5 Undo/Redo

Every label operation is captured as a `LabelCommand` storing the modified point indices and their previous labels. Undo is a constant-time index-assign. The stack is memory-capped (default 500MB) with FIFO eviction.

### 3.6 4D Time Series Support

Directories containing numbered frames (e.g., `frame_000.ply` through `frame_099.ply`) are automatically detected as time-series sequences. A timeline scrubber appears at the bottom of the viewport for frame navigation. An LRU cache holds up to 3 decoded frames in RAM.

### 3.7 Label Propagation for 4D

For time-series annotation, users label the first frame manually and then propagate to subsequent frames using KD-tree spatial proximity:

```
For each target point:
    Find up to K nearest labeled source points within radius R
    Inverse-distance weighted majority vote
    Confidence = 1 - (nearest_dist / R)
```

Low-confidence points are visually highlighted, directing the user's attention to regions that need manual correction. Iterative refinement (correct frame N, propagate from corrected labels to N+1) keeps per-frame correction effort low.

### 3.8 Dataset Export

Labeled clouds export to three formats:

- **Pointcept PTv3 format**: directory of `scene_NNN/{coord,color,segment,feat}.npy` + `label_map.json`
- **NPZ**: single compressed file per scene
- **HDF5**: single file with scene groups

Labels are remapped to contiguous 0..K-1 indices (required by most training code). Per-point scalar features (intensity, height, etc.) are normalized and stacked into a feature matrix. Train/val/test splitting uses a deterministic seed.

### 3.9 Sidecar Training

Training launches as a separate Python subprocess to avoid GPU memory contention with the main renderer. The parent process reads stdout and parses progress metrics (epoch, loss, mIoU) via regex. The training script loads the exported dataset, constructs a PyTorch model, and periodically checkpoints.

The sidecar pattern decouples training from the GUI: the renderer stays responsive, training can be stopped cleanly, and model code can be swapped without affecting the main application.

### 3.10 Design Philosophy

Lithium deliberately rejects the utilitarian aesthetic of most research tools. The GUI follows an OP-1-inspired design language: ultra-dark neutral grey backgrounds, warm accent colors (coral, orange, amber, teal), custom-drawn widgets with font-metric based layout, and visible feedback for every action.

This is not decoration. Research shows that tool aesthetics correlate with adoption (Nielsen 2000). A polished interface signals careful engineering and attracts users who would otherwise bounce off a dated-looking prototype. Design is a trust signal, and in Lithium's case it is an honest one — the underlying engineering (GPU rendering, KD-tree acceleration, contiguous label remapping) is the reason the interface can afford to prioritize feel.

## 4. Evaluation

[TODO: User study and benchmark section]

- **User study**: N annotators × M scenes × 3 tools (Lithium vs CloudCompare vs custom scripts). Metrics: time-to-label, mIoU of resulting labels vs ground truth.
- **Benchmark**: export ScanNet and S3DIS via Lithium, train PTv3, compare mIoU to published numbers.
- **Performance**: frame rates and interaction latencies on clouds of 100K, 1M, 10M points.

Preliminary numbers from the Lithium test suite on 1M-point synthetic clouds:

| Operation | Time |
|-----------|------|
| Point pick | 56ms |
| Box select | 54ms |
| Lasso select | 74ms |
| Brush select (r=1.0) | 11ms |
| Apply label (500K pts) | 3.6ms |
| Undo (500K pts) | 1.1ms |

## 5. Spinal Anatomy Case Study

[TODO: demonstrate the full workflow on spinal CT data]

- Load a DICOM/TIF z-stack of a cervical spine CT
- Threshold-sample bone voxels to a point cloud
- Manually label C1-C7 vertebrae, intervertebral discs, and spinal canal
- Export a labeled dataset
- Train a segmentation model end-to-end in-app
- Report mIoU on a held-out scan

## 6. Limitations and Future Work

- The PointNet-style baseline included is a proof of concept. Full PTv3 integration requires bundling or referencing the Pointcept library.
- GPU picking could scale to 50M+ points; the current NumPy projection handles up to ~10M comfortably.
- Real-time collaborative annotation is not supported.
- DICOM import is stubbed but not implemented; TIF z-stacks work end-to-end.

## 7. Conclusion

Lithium demonstrates that a modern, polished, offline desktop tool for the full point cloud annotation and training workflow is achievable in a modest codebase. We hope the combination of fast rendering, precise selection tools, and end-to-end integration lowers the barrier for researchers applying point cloud vision models to new domains — particularly those with data that cannot leave a local machine.

The codebase is released under [LICENSE] at [REPO URL].

## Acknowledgments

[TBD]

## References

[TBD — full bibliography]

- Qi et al. PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. CVPR 2017.
- Wu et al. Point Transformer V3: Simpler, Faster, Stronger. CVPR 2024.
- Berg et al. ilastik: interactive machine learning for (bio)image analysis. Nature Methods 2019.
- Isensee et al. nnInteractive. 2024.
- Dai et al. ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes. CVPR 2017.
- Armeni et al. 3D Semantic Parsing of Large-Scale Indoor Spaces (S3DIS). CVPR 2016.
- Nielsen. Designing Web Usability. New Riders, 2000.
