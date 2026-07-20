"""Point cloud GPU representation and rendering.

Vertex data is split across four VBOs so per-point updates only have to
re-upload the bytes that actually changed:

- ``pos_vbo``       (``3f`` in_position) — written once at upload, never again.
- ``col_vbo``       (``3f`` in_color)    — written once at upload, never again.
- ``label_vbo``     (``1i`` in_label)    — 4 bytes/pt, rewritten on label edit.
- ``selected_vbo``  (``1f`` in_selected) — 4 bytes/pt, rewritten on selection.

The old design packed everything into one 32-byte-per-vertex VBO, so a
brush stroke writing a single label had to re-push the full positions +
colours + labels + selection blob — 32 MB/stroke on a 1M-pt cloud. The
split drops that to 4 MB (labels only) or 4 MB (selection only), and
positions + colours never round-trip the PCIe bus again after upload.

ModernGL's ``vertex_array`` happily accepts multiple (buffer, fmt, attr)
tuples and wires each attribute to its own VBO — the shader side doesn't
care whether the attributes share a buffer.
"""

from dataclasses import dataclass
import numpy as np
import moderngl

from src.data.point_cloud import PointCloudData


@dataclass
class GPUCloud:
    """A point cloud uploaded to the GPU across four attribute VBOs.

    ``vbo`` is kept as an alias for ``pos_vbo`` for backward compat —
    earlier code paths called ``gpu.vbo.write(...)`` directly, but all
    of those now go through ``re_upload_labels`` / ``update_selection``.
    """
    vao: moderngl.VertexArray
    pos_vbo: moderngl.Buffer
    col_vbo: moderngl.Buffer
    label_vbo: moderngl.Buffer
    selected_vbo: moderngl.Buffer
    point_count: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    center: np.ndarray
    cloud_data: PointCloudData  # reference to source data
    # Reusable float32 selection staging to avoid an astype() allocation
    # on every brush tick. Shape (N,), dtype float32.
    _sel_staging: np.ndarray | None = None
    # Reusable int32 label staging (keeps a contiguous copy for upload).
    _lbl_staging: np.ndarray | None = None

    @property
    def vbo(self) -> moderngl.Buffer:
        """Backward-compat alias for the position VBO."""
        return self.pos_vbo

    def release(self):
        self.vao.release()
        self.pos_vbo.release()
        self.col_vbo.release()
        self.label_vbo.release()
        self.selected_vbo.release()


def _ensure_contig_f32(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a contiguous float32 ndarray without copying if possible."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return a


def _ensure_contig_i32(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a contiguous int32 ndarray without copying if possible."""
    return np.ascontiguousarray(arr, dtype=np.int32)


def _selection_to_float32(selection_mask: np.ndarray | None, n: int,
                           staging: np.ndarray | None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Materialise ``selection_mask`` into the GPUCloud's reusable float32
    staging buffer so selection uploads don't allocate.
    """
    if staging is None or staging.shape != (n,) or staging.dtype != np.float32:
        staging = np.empty(n, dtype=np.float32)
    if selection_mask is None or len(selection_mask) != n:
        staging.fill(0.0)
    else:
        # np.multiply with out= avoids allocating; the source may be bool.
        np.copyto(staging, selection_mask, casting='unsafe')
    return staging, staging


def upload_cloud(cloud: PointCloudData, ctx: moderngl.Context,
                 program: moderngl.Program,
                 selection_mask: np.ndarray = None) -> GPUCloud:
    """Upload a PointCloudData to the GPU as four attribute VBOs."""
    n = cloud.point_count

    pos = _ensure_contig_f32(cloud.positions)
    col = _ensure_contig_f32(cloud.colors)
    lbl_staging = _ensure_contig_i32(cloud.labels)
    sel_staging, _ = _selection_to_float32(selection_mask, n, None)

    pos_vbo = ctx.buffer(pos.tobytes())
    col_vbo = ctx.buffer(col.tobytes())
    label_vbo = ctx.buffer(lbl_staging.tobytes())
    selected_vbo = ctx.buffer(sel_staging.tobytes())

    vao = ctx.vertex_array(
        program,
        [
            (pos_vbo,      '3f', 'in_position'),
            (col_vbo,      '3f', 'in_color'),
            (label_vbo,    '1i', 'in_label'),
            (selected_vbo, '1f', 'in_selected'),
        ],
    )

    return GPUCloud(
        vao=vao,
        pos_vbo=pos_vbo,
        col_vbo=col_vbo,
        label_vbo=label_vbo,
        selected_vbo=selected_vbo,
        point_count=n,
        bounds_min=cloud.bounds_min.copy(),
        bounds_max=cloud.bounds_max.copy(),
        center=cloud.center.copy(),
        cloud_data=cloud,
        _sel_staging=sel_staging,
        _lbl_staging=lbl_staging,
    )


def re_upload_labels(gpu_cloud: GPUCloud, selection_mask: np.ndarray = None):
    """Upload just the label column (and selection if provided).

    Called on every brush/box/lasso commit and every undo/redo. With the
    split VBO layout this is a 4-bytes/pt write rather than the 32 B/pt
    round-trip the old interleaved layout required — positions and
    colours never change, so they stay on the GPU.
    """
    cloud = gpu_cloud.cloud_data
    n = gpu_cloud.point_count

    lbl = _ensure_contig_i32(cloud.labels)
    # Keep a reference on the GPUCloud so Python doesn't recycle the
    # backing memory between frames if we're lucky and `cloud.labels`
    # already happens to be contiguous int32.
    gpu_cloud._lbl_staging = lbl
    gpu_cloud.label_vbo.write(lbl.tobytes())

    if selection_mask is not None:
        sel_staging, _ = _selection_to_float32(
            selection_mask, n, gpu_cloud._sel_staging)
        gpu_cloud._sel_staging = sel_staging
        gpu_cloud.selected_vbo.write(sel_staging.tobytes())


def update_selection(gpu_cloud: GPUCloud, selection_mask: np.ndarray):
    """Upload just the selection column.

    Hot path during brush painting — called every tick. Writing the
    dedicated 4-bytes/pt selection VBO instead of the full 32-bytes/pt
    interleaved blob is the single biggest brush-throughput win.
    """
    sel_staging, _ = _selection_to_float32(
        selection_mask, gpu_cloud.point_count, gpu_cloud._sel_staging)
    gpu_cloud._sel_staging = sel_staging
    gpu_cloud.selected_vbo.write(sel_staging.tobytes())


# Reusable buffer for column-major matrix conversion — avoids 3 allocations
# per call (astype + transpose + ascontiguousarray).
_gl_mat4_buf = np.empty((4, 4), dtype=np.float32)


def _gl_mat4(mat: np.ndarray) -> bytes:
    """Convert a row-major numpy matrix to column-major bytes for OpenGL."""
    np.copyto(_gl_mat4_buf, mat.astype(np.float32, copy=False).T)
    return _gl_mat4_buf.tobytes()


def set_shared_point_uniforms(program: moderngl.Program,
                               brightness: float = 0.0, contrast: float = 1.0,
                               saturation: float = 1.0,
                               r_gain: float = 1.0, g_gain: float = 1.0,
                               b_gain: float = 1.0,
                               depth_falloff: float = 1.0,
                               label_blend: float = 0.0,
                               label_texture: moderngl.Texture = None,
                               falloff_texture: moderngl.Texture = None,
                               dof_strength: float = 0.0,
                               dof_focus_dist: float = 1.0,
                               clip_min: np.ndarray = None,
                               clip_max: np.ndarray = None) -> None:
    """Write the 'shared' point-cloud uniforms once before drawing many
    cells (gallery mode) or many clouds. Any subsequent ``draw_cloud``
    calls with ``skip_shared=True`` only touch the per-draw model/view/
    projection/point_size — the globals stay bound from this call.
    """
    program['u_brightness'].value = brightness
    program['u_contrast'].value = contrast
    program['u_saturation'].value = saturation
    program['u_rgb_gain'].value = (r_gain, g_gain, b_gain)
    program['u_depth_falloff'].value = depth_falloff
    # Force label_blend to 0 when no label_texture is bound — otherwise
    # a prior frame's u_label_blend > 0 plus the missing texture means
    # the shader's discard test reads unit 0's stale binding (could be
    # the falloff LUT, an SSAO output, anything). UNI-002 in REPORT.md.
    program['u_label_blend'].value = (
        float(label_blend) if label_texture is not None else 0.0
    )
    program['u_dof_strength'].value = float(dof_strength)
    program['u_dof_focus_dist'].value = float(max(dof_focus_dist, 1e-3))
    if clip_min is not None and clip_max is not None:
        program['u_clip_min'].value = tuple(float(v) for v in clip_min)
        program['u_clip_max'].value = tuple(float(v) for v in clip_max)
    else:
        program['u_clip_min'].value = (-1e9, -1e9, -1e9)
        program['u_clip_max'].value = (1e9, 1e9, 1e9)
    # Clip-proximity point tinting disabled — plane visibility is
    # communicated only via the viewport overlay, not point color,
    # so it can't be confused with label colors.
    program['u_clip_near_zone'].value = (0.0, 0.0, 0.0)
    if label_texture is not None:
        label_texture.use(location=0)
        program['u_label_colors'].value = 0
    if falloff_texture is not None:
        falloff_texture.use(location=1)
        program['u_falloff_lut'].value = 1
        try:
            program['u_has_falloff'].value = 1
        except KeyError:
            pass  # older shader without the gating flag
    else:
        try:
            program['u_has_falloff'].value = 0
        except KeyError:
            pass


def draw_cloud(gpu_cloud: GPUCloud, program: moderngl.Program,
               model: np.ndarray, view: np.ndarray, projection: np.ndarray,
               point_size: float = 5.0, sharpness: float = 0.0,
               depth_falloff: float = 1.0,
               brightness: float = 0.0, contrast: float = 1.0,
               saturation: float = 1.0,
               r_gain: float = 1.0, g_gain: float = 1.0, b_gain: float = 1.0,
               label_blend: float = 0.0,
               label_texture: moderngl.Texture = None,
               falloff_texture: moderngl.Texture = None,
               clip_min: np.ndarray = None, clip_max: np.ndarray = None,
               dof_strength: float = 0.0, dof_focus_dist: float = 1.0,
               skip_shared: bool = False):
    """Render a GPU cloud with the given matrices and settings.

    The `sharpness` parameter is kept for backward compatibility but is
    no longer used by the shader — the falloff curve is driven entirely
    by the `falloff_texture` LUT.

    clip_min / clip_max are world-space (pre-model-transform) AABB
    extents. Points outside are discarded. Pass None for no clipping.

    ``skip_shared=True`` skips the shader-wide uniforms (color grading,
    label/falloff textures, DOF, clip box). The caller is expected to
    have already set them via ``set_shared_point_uniforms``. Only the
    per-draw bits (model, view, projection, point_size) are written.
    """
    program['u_model'].write(_gl_mat4(model))
    program['u_view'].write(_gl_mat4(view))
    program['u_projection'].write(_gl_mat4(projection))
    program['u_point_size'].value = point_size
    if not skip_shared:
        program['u_depth_falloff'].value = depth_falloff
        program['u_brightness'].value = brightness
        program['u_contrast'].value = contrast
        program['u_saturation'].value = saturation
        program['u_rgb_gain'].value = (r_gain, g_gain, b_gain)
        # Same UNI-002 guard as set_shared_point_uniforms — never let
        # u_label_blend stay > 0 when label_texture is None.
        program['u_label_blend'].value = (
            float(label_blend) if label_texture is not None else 0.0
        )
        program['u_dof_strength'].value = float(dof_strength)
        program['u_dof_focus_dist'].value = float(max(dof_focus_dist, 1e-3))
        if clip_min is not None and clip_max is not None:
            program['u_clip_min'].value = tuple(float(v) for v in clip_min)
            program['u_clip_max'].value = tuple(float(v) for v in clip_max)
        else:
            program['u_clip_min'].value = (-1e9, -1e9, -1e9)
            program['u_clip_max'].value = (1e9, 1e9, 1e9)
        # Clip-proximity point tinting disabled — plane visibility is
        # communicated only via the viewport overlay, not point color,
        # so it can't be confused with label colors.
        program['u_clip_near_zone'].value = (0.0, 0.0, 0.0)
        if label_texture is not None:
            label_texture.use(location=0)
            program['u_label_colors'].value = 0
        if falloff_texture is not None:
            falloff_texture.use(location=1)
            program['u_falloff_lut'].value = 1
            try:
                program['u_has_falloff'].value = 1
            except KeyError:
                pass
        else:
            try:
                program['u_has_falloff'].value = 0
            except KeyError:
                pass

    gpu_cloud.vao.render(moderngl.POINTS)
