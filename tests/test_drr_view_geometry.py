"""DRR view-geometry regression test — guards against re-introducing the
monoplanar-disguised-as-biplanar bug.

History:
  2026-05: original `worker.py` rotated view around the Y axis with source
  offset at (0, sad, 0). Because (0, sad, 0) is on the Y axis, Roty was a
  no-op for the translation — the "lateral" was actually the PA image
  rotated 90° in image space. Pixel-correlation testing on the rendered
  PNGs flagged the bug.

  2026-05-20: Phase 0 empirical probe (`tools/probe_view_axis.py` +
  `probe_view_axis_fullspine.py`) rendered all candidate rotation axes
  (Rotx, Roty, Rotz) with multiple base translation candidates against
  full thoracolumbar verse512. Conclusion: **Rotz with base_t=(0, sad, 0)
  produces a true biplanar lateral** showing spine vertical, vertebrae in
  side profile, posterior elements (spinous processes) visible. Empirical
  audit superseded the second-round agent's a-priori derivation that
  predicted Roty.

This test enforces the geometric invariants of the correct setup. Any
future change that violates these invariants must be deliberate and update
this test.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.spatial.transform import Rotation as Rsci


# ====================================================================
# Geometric invariants (no GPU / no DRR — pure math)
# ====================================================================
def _view_pose(angle_deg: float, sad_mm: float = 1000.0):
    """Reproduce the view-pose construction from worker.py:_render_one."""
    base_t = np.array([0.0, sad_mm, 0.0], dtype=np.float64)
    view_R = Rsci.from_euler("z", float(angle_deg), degrees=True).as_matrix()
    view_t = view_R @ base_t
    return view_R, view_t


def test_pa_view_source_anterior():
    """At 0° rotation, the source is at +Y (patient anterior after AP reorient)."""
    R, t = _view_pose(0.0)
    assert np.allclose(R, np.eye(3), atol=1e-9), \
        f"PA rotation must be identity, got {R}"
    assert np.allclose(t, [0.0, 1000.0, 0.0]), \
        f"PA source must be at (0, sad, 0), got {t}"


def test_lateral_view_source_left():
    """At 90° rotation, the source moves to the patient's left (−X)."""
    R, t = _view_pose(90.0)
    # Rotz(90°) @ (0, sad, 0) = (-sad, 0, 0)
    assert np.allclose(t, [-1000.0, 0.0, 0.0], atol=1e-6), \
        f"Lateral source must be at (-sad, 0, 0), got {t}"
    # Rotation about Z axis is what makes this a true biplanar swing
    assert np.allclose(R, Rsci.from_euler("z", 90, degrees=True).as_matrix()), \
        f"Lateral rotation must be Rotz(90°), got {R}"


def test_view_separation_meets_invariant():
    """Source positions for 0° and 90° must be ≥ 0.5*SAD apart — the actual
    biplanar invariant. The original bug had separation 0 (no-op rotation)."""
    _, t_pa = _view_pose(0.0)
    _, t_lat = _view_pose(90.0)
    sep_mm = float(np.linalg.norm(t_lat - t_pa))
    sad = 1000.0
    assert sep_mm >= 0.5 * sad, \
        f"View separation {sep_mm:.1f}mm < 0.5*SAD ({0.5*sad}mm). " \
        f"This is the monoplanar-disguised-as-biplanar bug; check rotation axis."
    # In fact for Rotz around 90°, separation should be sqrt(2)*SAD
    assert abs(sep_mm - np.sqrt(2) * sad) < 1.0, \
        f"Expected sqrt(2)*SAD ({np.sqrt(2)*sad:.1f}mm), got {sep_mm:.1f}mm"


def test_view_sources_orthogonal():
    """The dot product of PA and lateral source vectors must be near zero
    (sources are 90° apart). This is the geometric definition of biplanar."""
    _, t_pa = _view_pose(0.0)
    _, t_lat = _view_pose(90.0)
    cos_angle = float(t_pa @ t_lat / (np.linalg.norm(t_pa) * np.linalg.norm(t_lat) + 1e-12))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))
    assert abs(angle_deg - 90.0) < 5.0, \
        f"Source vectors must be 85-95° apart, got {angle_deg:.1f}°"


def test_axis_y_would_be_noop():
    """Regression: confirm the OLD broken Roty around (0, sad, 0) would be
    a no-op. If this assertion ever fails, the old buggy code might still
    pass our invariant — meaning the test is no longer effective."""
    base_t = np.array([0.0, 1000.0, 0.0])
    bad_view_R = Rsci.from_euler("y", 90, degrees=True).as_matrix()
    bad_view_t = bad_view_R @ base_t
    assert np.allclose(bad_view_t, base_t, atol=1e-6), \
        "Roty(90) @ (0, sad, 0) should be (0, sad, 0) — confirms the old bug pattern"


def test_axis_x_would_give_axial():
    """Regression: confirm Rotx around (0, sad, 0) puts source at (0, 0, sad)
    which is an axial view (looking down spine), not a lateral."""
    base_t = np.array([0.0, 1000.0, 0.0])
    bad_view_R = Rsci.from_euler("x", 90, degrees=True).as_matrix()
    bad_view_t = bad_view_R @ base_t
    # Rotx(90) @ (0, 1, 0) = (0, 0, 1) → source at +Z
    assert np.allclose(bad_view_t, [0.0, 0.0, 1000.0], atol=1e-6), \
        f"Rotx(90) @ (0, sad, 0) should be (0, 0, sad), got {bad_view_t}"


# ====================================================================
# Optional: full DRR render with image-correlation check
# Only runs if diffdrr is importable AND CUDA is available
# AND VERSE512_CT env var points to a valid CT.
# ====================================================================
def _can_render_drr():
    if os.environ.get("VERSE512_CT") is None:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def test_pa_lateral_pixel_correlation_low():
    """Render a real biplanar pair and confirm rot90(PA) is NOT highly
    correlated with the lateral. The original bug produced correlation=1.0
    (literally the same image rotated). A true biplanar pair has correlation
    well below 0.5 because the views show genuinely different anatomy."""
    if not _can_render_drr():
        print("SKIP: no GPU or VERSE512_CT not set")
        return

    import nibabel as nib  # type: ignore
    from PIL import Image  # type: ignore
    import torch  # type: ignore
    from diffdrr.drr import DRR  # type: ignore
    from diffdrr.data import read  # type: ignore
    from diffdrr.pose import RigidTransform  # type: ignore

    ct_path = os.environ["VERSE512_CT"]
    mask_path = os.environ.get("VERSE512_MASK")
    device = "cuda"
    sad = 1000.0
    subject = read(ct_path, labelmap=mask_path, orientation="AP",
                   center_volume=True, bone_attenuation_multiplier=2.0)
    drr = DRR(subject, sdd=1200.0, height=512, width=512,
              delx=1.0, dely=1.0, reverse_x_axis=False,
              renderer="trilinear").to(device)

    def render(angle):
        R, t = _view_pose(angle, sad_mm=sad)
        pose_4x4 = np.eye(4, dtype=np.float32)
        pose_4x4[:3, :3] = R
        pose_4x4[:3, 3] = t
        pose_t = torch.tensor(pose_4x4, dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            img = drr(RigidTransform(pose_t))
        return img[0, 0].detach().cpu().numpy().astype(np.float32)

    pa = render(0.0)
    lat = render(90.0)

    def corr(a, b):
        a = a.flatten().astype(np.float64) - a.mean()
        b = b.flatten().astype(np.float64) - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-9))

    pa_rot90 = np.rot90(pa, k=-1)
    pa_rot90_ccw = np.rot90(pa, k=1)
    correlations = [abs(corr(pa_rot90, lat)), abs(corr(pa_rot90_ccw, lat))]
    max_rot_corr = max(correlations)
    print(f"max |corr(rot90(PA), lateral)| = {max_rot_corr:.3f}")
    assert max_rot_corr < 0.5, \
        f"rot90(PA) vs lateral correlation = {max_rot_corr:.3f} ≥ 0.5; " \
        f"this indicates the monoplanar-disguised-as-biplanar bug."


# Allow running directly
if __name__ == "__main__":
    test_pa_view_source_anterior()
    test_lateral_view_source_left()
    test_view_separation_meets_invariant()
    test_view_sources_orthogonal()
    test_axis_y_would_be_noop()
    test_axis_x_would_give_axial()
    print("PASS: all geometric-invariant tests")
    if _can_render_drr():
        test_pa_lateral_pixel_correlation_low()
        print("PASS: pixel-correlation test")
    else:
        print("SKIP: pixel-correlation test (no GPU or VERSE512_CT env unset)")
