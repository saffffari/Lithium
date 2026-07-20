"""Mesh GPU representation and rendering.

Parallels point_cloud_renderer.py for the triangulated surface display
mode used in HOLOGRAM (clinician-facing) and optionally in LIGHT TABLE
(researcher). Each mesh uploads three VBOs (position / normal / label)
+ one index buffer, all written once at upload and never re-written
during normal interaction (label edits trigger a full rebuild via the
cloud_store mesh cache rather than partial updates — meshes aren't
edited per-vertex the way clouds are).

Shading: ``shaders/mesh.frag`` does view-space Lambert against an
interpolated per-vertex normal with a camera-locked light direction
and a fixed neutral-grey bone surface. Anatomical class is communicated
via the datum overlays (planes, axes, frames, centroids) drawn by the
primitive renderer on top of the mesh, not by per-face colouring.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import moderngl

from src.rendering.shaders import load_shader


@dataclass
class GPUMesh:
    """A triangulated mesh uploaded to the GPU as 3 VBOs + 1 IBO.

    Two VertexArrays are created at upload time:
      - ``vao``: bound to the main matcap mesh program (position +
        normal + label attributes).
      - ``mask_vao``: bound to the minimal selection-mask program
        (position only — the mask frag writes a constant 1.0).
    Both share the same VBOs/IBO; only attribute → program bindings
    differ. VAOs in OpenGL are essentially attribute-binding state, a
    few bytes — the per-mesh duplication is negligible.
    """
    vao: moderngl.VertexArray
    mask_vao: moderngl.VertexArray
    pos_vbo: moderngl.Buffer
    normal_vbo: moderngl.Buffer
    label_vbo: moderngl.Buffer
    index_buffer: moderngl.Buffer
    triangle_count: int
    vertex_count: int
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    center: np.ndarray

    def release(self) -> None:
        for r in (
            self.vao, self.mask_vao, self.pos_vbo, self.normal_vbo,
            self.label_vbo, self.index_buffer,
        ):
            try:
                r.release()
            except Exception:
                pass


class MeshRenderer:
    """Owns the mesh shader program + the selection-mask program.

    One instance per app. Uploaded meshes hold their own VAO/VBO/IBO;
    this class owns state shared across every mesh:

    - **``program``**: the body shader (``mesh.vert`` + ``mesh.frag``)
      that does view-space Lambert against an interpolated per-vertex
      normal. Camera-locked light, neutral grey surface, subtle
      silhouette rim. Produces the diagrammatic atlas look without a
      matcap lookup (a matcap version lived here briefly but read as
      "glow halo + minimal shading"; live Lambert recovers the crisp
      directional shading the look depends on).
    - **``mask_program``**: minimal pipeline that writes 1.0 to a
      single-channel mask buffer wherever a selected mesh is visible.
      Used by the ``pp_outline`` post-process to draw the orange
      silhouette stroke around selected meshes without re-running the
      full body shader.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.program = ctx.program(
            vertex_shader=load_shader("mesh.vert"),
            fragment_shader=load_shader("mesh.frag"),
        )
        # Selection-mask program: minimal pipeline that just writes 1.0
        # to a single-channel mask buffer wherever a selected mesh is
        # visible. Used by the outline post-process to find silhouettes
        # without re-running the full mesh shader.
        self.mask_program = ctx.program(
            vertex_shader=load_shader("mesh_mask.vert"),
            fragment_shader=load_shader("mesh_mask.frag"),
        )

    def release(self) -> None:
        for r in (self.program, self.mask_program):
            try:
                r.release()
            except Exception:
                pass


def upload_mesh(mesh: dict, ctx: moderngl.Context,
                program: moderngl.Program,
                mask_program: moderngl.Program | None = None) -> GPUMesh:
    """Upload a mesh dict to the GPU. Mesh dict matches the shape
    returned by ``mesh_builder.build_mesh``.

    Two VAOs are created: one bound to the main matcap mesh program
    (full attribute set), one bound to the minimal selection-mask
    program (position only). Both reference the same VBOs/IBO — only
    attribute → program bindings differ. The mask program is required;
    older callers passing only the main program get a ``ValueError``.
    """
    if mask_program is None:
        raise ValueError(
            "upload_mesh now requires mask_program (selection outline "
            "pipeline). Pass mesh_renderer.mask_program."
        )
    verts = np.ascontiguousarray(mesh["verts"], dtype=np.float32)
    faces = np.ascontiguousarray(mesh["faces"], dtype=np.int32)
    normals = np.ascontiguousarray(mesh["normals"], dtype=np.float32)
    labels = np.ascontiguousarray(
        mesh.get("vertex_labels", np.zeros(len(verts), dtype=np.int32)),
        dtype=np.int32,
    )

    pos_vbo = ctx.buffer(verts.tobytes())
    normal_vbo = ctx.buffer(normals.tobytes())
    label_vbo = ctx.buffer(labels.tobytes())
    index_buffer = ctx.buffer(faces.tobytes())

    # Build VAO binding lists tolerantly. The GLSL compiler/linker is
    # free to strip attributes the shader doesn't actually read —
    # the mask program only reads ``in_position`` so its binding list
    # naturally drops in_normal / in_label without needing a separate
    # code path. moderngl would otherwise raise on the
    # ctx.vertex_array call if we tried to bind an absent attribute.
    candidate_bindings = [
        (pos_vbo,    "3f", "in_position"),
        (normal_vbo, "3f", "in_normal"),
        (label_vbo,  "1i", "in_label"),
    ]

    def _filter_bindings(prog: moderngl.Program) -> list:
        out = []
        for vbo, fmt, name in candidate_bindings:
            try:
                prog[name]
            except KeyError:
                continue
            out.append((vbo, fmt, name))
        return out

    vao = ctx.vertex_array(
        program,
        _filter_bindings(program),
        index_buffer=index_buffer,
        index_element_size=4,
    )
    mask_vao = ctx.vertex_array(
        mask_program,
        _filter_bindings(mask_program),
        index_buffer=index_buffer,
        index_element_size=4,
    )

    bmin = verts.min(axis=0) if len(verts) else np.zeros(3, dtype=np.float32)
    bmax = verts.max(axis=0) if len(verts) else np.zeros(3, dtype=np.float32)
    return GPUMesh(
        vao=vao, mask_vao=mask_vao,
        pos_vbo=pos_vbo, normal_vbo=normal_vbo,
        label_vbo=label_vbo, index_buffer=index_buffer,
        triangle_count=int(len(faces)),
        vertex_count=int(len(verts)),
        bounds_min=bmin, bounds_max=bmax,
        center=((bmin + bmax) * 0.5).astype(np.float32),
    )


# Module-level reusable matrix buffer mirrors point_cloud_renderer's
# trick so per-frame matrix updates don't allocate.
_gl_mat4_buf = np.empty((4, 4), dtype=np.float32)


def _gl_mat4(mat: np.ndarray) -> bytes:
    np.copyto(_gl_mat4_buf, mat.astype(np.float32, copy=False).T)
    return _gl_mat4_buf.tobytes()


def set_shared_mesh_uniforms(
    renderer: MeshRenderer,
    *,
    brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0,
    r_gain: float = 1.0, g_gain: float = 1.0, b_gain: float = 1.0,
    label_blend: float = 0.0,
    label_texture: moderngl.Texture | None = None,
    clip_min: np.ndarray | None = None,
    clip_max: np.ndarray | None = None,
) -> None:
    """Write the shared (per-frame, not per-mesh) mesh shader uniforms.

    Mirrors set_shared_point_uniforms in point_cloud_renderer so the
    same per-frame "set once, draw N" pattern applies.

    Most kwargs are accepted for backward-compatibility but ignored —
    the mesh shader doesn't apply point-cloud-style colour grading.
    Anatomical class colour belongs to the datum overlays, not the
    mesh faces. The mesh body is a fixed neutral grey driven by
    Lambert against a camera-locked light direction. Per-frame state
    set here is just the clip-box bounds.

    Selection isn't a per-mesh body uniform either — selected meshes
    get an orange silhouette outline drawn by the ``pp_outline``
    post-process pass, which reads a mask buffer populated by
    ``draw_mesh_mask`` (called by the caller for each selected mesh).
    """
    program = renderer.program
    if clip_min is not None and clip_max is not None:
        program["u_clip_min"].value = tuple(float(v) for v in clip_min)
        program["u_clip_max"].value = tuple(float(v) for v in clip_max)
    else:
        program["u_clip_min"].value = (-1e9, -1e9, -1e9)
        program["u_clip_max"].value = (1e9, 1e9, 1e9)


def draw_mesh(
    gpu_mesh: GPUMesh,
    renderer: MeshRenderer,
    model: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
    *,
    selected: bool = False,
    tint: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Render a single uploaded mesh into the bound FBO.

    ``tint`` multiplies the surface colour so dual-pose HOLOGRAM renders
    can distinguish supine (default ``(1, 1, 1)``) from standing (e.g.
    ``(0.55, 0.75, 1.2)`` for a blue tint). Applied to the SURFACE
    colour before Lambert shading so curvature still reads correctly
    — not a post-multiply on the lit colour, which would wash out
    the directional shading.

    Assumes the caller has set the shared uniforms via
    ``set_shared_mesh_uniforms`` once per frame. Per-mesh state here is
    the MVP matrices + the tint.

    ``selected`` is retained for back-compat but the mesh shader no
    longer reads a selection flag — the orange outline is drawn by a
    separate post-process pass that detects silhouettes via a mask
    buffer (see ``draw_mesh_mask`` + ``shaders/pp_outline.frag``).
    """
    del selected  # selection is handled by the outline pass, not here
    program = renderer.program
    program["u_model"].write(_gl_mat4(model))
    program["u_view"].write(_gl_mat4(view))
    program["u_projection"].write(_gl_mat4(projection))
    try:
        program["u_pose_tint"].value = tuple(float(c) for c in tint)
    except KeyError:
        # Shader didn't declare u_pose_tint (e.g. mid-edit hot reload).
        # The default in the shader is (1,1,1) so missing this just
        # produces the un-tinted grey.
        pass
    gpu_mesh.vao.render(moderngl.TRIANGLES)


def draw_mesh_mask(
    gpu_mesh: GPUMesh,
    renderer: MeshRenderer,
    model: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
) -> None:
    """Render a single mesh into the currently-bound selection mask FBO.

    The mask FBO shares scene_rt's depth attachment, so depth testing
    against the already-rendered scene keeps occluded parts of this
    mesh from contributing to the mask — only the visible silhouette
    is captured for the outline pass to find.
    """
    program = renderer.mask_program
    program["u_model"].write(_gl_mat4(model))
    program["u_view"].write(_gl_mat4(view))
    program["u_projection"].write(_gl_mat4(projection))
    gpu_mesh.mask_vao.render(moderngl.TRIANGLES)
