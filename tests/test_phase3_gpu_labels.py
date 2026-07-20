"""Phase 3 verification: GPU label rendering pipeline.

Uses a hidden GLFW window to create a real GL context, uploads a
synthetic cloud with labels, and verifies:
1. VBO upload succeeds with new format (3f 3f 1i)
2. re_upload_labels() works
3. Shaders compile with label uniforms
4. Rendering into an FBO produces a non-blank image
5. label_blend=1 produces different colors than label_blend=0
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import glfw
import moderngl
import pytest

from src.data.point_cloud import PointCloudData
from src.data.labels import LabelRegistry
from src.rendering.shaders import create_point_cloud_program
from src.rendering.point_cloud_renderer import upload_cloud, draw_cloud, re_upload_labels
from src.rendering.label_texture import create_label_color_texture, update_label_color_texture
from src.rendering.framebuffer import ExportFramebuffer
from src.utils.math_utils import orbit_camera, perspective


# These tests need a real GPU context. On a headless CI box (no display
# / no GLX) moderngl.create_context() raises "cannot detect OpenGL
# context"; setup_context() turns that into a per-test skip so the suite
# stays green on machines without a display, while still exercising the
# GPU path wherever one exists.


def setup_context():
    if not glfw.init():
        pytest.skip("glfw init failed (headless environment)")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(1, 1, "test", None, None)
    if not window:
        pytest.skip("could not create GL window (headless environment)")
    glfw.make_context_current(window)
    try:
        ctx = moderngl.create_context()
    except Exception as e:
        pytest.skip(f"no OpenGL context (headless environment): {e}")
    ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    return window, ctx


def make_grid_cloud(n_side: int = 20) -> PointCloudData:
    """Create a dense 3D grid of points with half labeled as 1, half as 2."""
    x = np.linspace(-1, 1, n_side, dtype=np.float32)
    y = np.linspace(-1, 1, n_side, dtype=np.float32)
    z = np.linspace(-1, 1, n_side, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    # Vertex colors: all white
    colors = np.ones_like(positions)
    cloud = PointCloudData(positions=positions, colors=colors)
    # Assign half the points label 1, half label 2
    n = cloud.point_count
    cloud.labels[0:n // 2] = 1
    cloud.labels[n // 2:] = 2
    return cloud


def render_to_fbo(ctx, program, gpu_cloud, label_texture, width, height,
                  label_blend, bg=(0.0, 0.0, 0.0)):
    fbo = ExportFramebuffer(ctx, width, height)
    fbo.use()
    fbo.clear(*bg)
    ctx.enable_only(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    ctx.viewport = (0, 0, width, height)

    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _, view = orbit_camera(target, distance=5.0, azimuth=0.8, elevation=0.5)
    proj = perspective(np.radians(60.0), width / height, 0.01, 100.0)
    model = np.eye(4, dtype=np.float32)

    draw_cloud(gpu_cloud, program, model, view, proj,
               point_size=3.0, sharpness=0.0,
               label_blend=label_blend, label_texture=label_texture)

    img = fbo.read_image()
    arr = np.array(img)  # (H, W, 4) uint8
    fbo.release()
    return arr


def test_vbo_upload_with_labels():
    window, ctx = setup_context()
    try:
        cloud = make_grid_cloud(10)
        program = create_point_cloud_program(ctx)
        gpu = upload_cloud(cloud, ctx, program)
        assert gpu.vao is not None
        assert gpu.point_count == 1000
        print("PASS: VBO upload with labels")
    finally:
        glfw.terminate()


def test_re_upload_labels():
    window, ctx = setup_context()
    try:
        cloud = make_grid_cloud(10)
        program = create_point_cloud_program(ctx)
        gpu = upload_cloud(cloud, ctx, program)
        # Change labels and re-upload
        cloud.labels[:] = 5
        re_upload_labels(gpu)
        # Read back the VBO to verify (Phase 4: 32 bytes/vertex = 8 int32 slots)
        vbo_bytes = gpu.vbo.read()
        vbo_arr = np.frombuffer(vbo_bytes, dtype=np.int32).reshape(-1, 8)
        labels_from_vbo = vbo_arr[:, 6]  # label column is still index 6
        assert (labels_from_vbo == 5).all(), \
            f"labels in VBO: unique={np.unique(labels_from_vbo)}"
        print("PASS: re_upload_labels updates GPU buffer")
    finally:
        glfw.terminate()


def test_render_with_labels():
    window, ctx = setup_context()
    try:
        cloud = make_grid_cloud(20)
        program = create_point_cloud_program(ctx)
        gpu = upload_cloud(cloud, ctx, program)

        # Setup label registry with distinct colors
        registry = LabelRegistry()
        id_red = registry.add_label("Red", color=(1.0, 0.0, 0.0, 1.0))
        id_blue = registry.add_label("Blue", color=(0.0, 0.0, 1.0, 1.0))
        # Assign labels
        n = cloud.point_count
        cloud.labels[0:n // 2] = id_red
        cloud.labels[n // 2:] = id_blue
        re_upload_labels(gpu)

        tex = create_label_color_texture(ctx, registry)

        # Render with label_blend=0 (vertex colors = white)
        img_vertex = render_to_fbo(ctx, program, gpu, tex, 256, 256, label_blend=0.0)
        # Render with label_blend=1 (label colors = red/blue)
        img_label = render_to_fbo(ctx, program, gpu, tex, 256, 256, label_blend=1.0)

        # Vertex-color render should have some white pixels
        white_pixels_vertex = ((img_vertex[:, :, 0] > 200) &
                               (img_vertex[:, :, 1] > 200) &
                               (img_vertex[:, :, 2] > 200)).sum()
        assert white_pixels_vertex > 100, \
            f"expected many white pixels in vertex-color render, got {white_pixels_vertex}"

        # Label render should have some red and some blue pixels, no white
        red_pixels = ((img_label[:, :, 0] > 200) &
                      (img_label[:, :, 1] < 50) &
                      (img_label[:, :, 2] < 50)).sum()
        blue_pixels = ((img_label[:, :, 0] < 50) &
                       (img_label[:, :, 1] < 50) &
                       (img_label[:, :, 2] > 200)).sum()
        assert red_pixels > 50, f"expected red pixels, got {red_pixels}"
        assert blue_pixels > 50, f"expected blue pixels, got {blue_pixels}"

        print(f"PASS: Label rendering "
              f"(vertex: {white_pixels_vertex} white px, "
              f"label: {red_pixels} red + {blue_pixels} blue px)")
    finally:
        glfw.terminate()


def test_hidden_labels_discard():
    """Hidden labels (alpha=0) should cause points to be discarded."""
    window, ctx = setup_context()
    try:
        cloud = make_grid_cloud(20)
        program = create_point_cloud_program(ctx)
        gpu = upload_cloud(cloud, ctx, program)

        registry = LabelRegistry()
        id_red = registry.add_label("Red", color=(1.0, 0.0, 0.0, 1.0))
        cloud.labels[:] = id_red
        re_upload_labels(gpu)

        tex = create_label_color_texture(ctx, registry)
        img_visible = render_to_fbo(ctx, program, gpu, tex, 256, 256, label_blend=1.0)
        red_visible = ((img_visible[:, :, 0] > 200) &
                       (img_visible[:, :, 1] < 50) &
                       (img_visible[:, :, 2] < 50)).sum()
        assert red_visible > 100, f"expected red when visible, got {red_visible}"

        # Hide the label
        registry.set_visible(id_red, False)
        update_label_color_texture(tex, registry)
        img_hidden = render_to_fbo(ctx, program, gpu, tex, 256, 256, label_blend=1.0)
        red_hidden = ((img_hidden[:, :, 0] > 200) &
                      (img_hidden[:, :, 1] < 50) &
                      (img_hidden[:, :, 2] < 50)).sum()
        assert red_hidden == 0, f"expected no red when hidden, got {red_hidden}"

        print(f"PASS: Hidden label discard (visible={red_visible}, hidden={red_hidden})")
    finally:
        glfw.terminate()


if __name__ == '__main__':
    test_vbo_upload_with_labels()
    test_re_upload_labels()
    test_render_with_labels()
    test_hidden_labels_discard()
    print("\n=== Phase 3 ALL TESTS PASSED ===")
