"""Spin render — 360-degree rotation frame sequence export."""

import os
import math
import moderngl
import numpy as np

from src.rendering.framebuffer import ExportFramebuffer
from src.rendering.point_cloud_renderer import draw_cloud
from src.utils.math_utils import orbit_camera, perspective


def export_spin(app, output_dir: str, frames: int = 360,
                width: int = 1920, height: int = 1080):
    """Render a 360-degree spin as a PNG sequence.

    The camera orbits around the current target at a fixed elevation.
    """
    os.makedirs(output_dir, exist_ok=True)

    fbo = ExportFramebuffer(app.ctx, width, height)
    aspect = width / height

    # Camera parameters from current view
    target = app.camera.target.copy()
    distance = app.camera.distance
    elevation = app.camera.elevation
    fov = app.camera.fov
    near = app.camera.near
    far = app.camera.far

    proj = perspective(fov, aspect, near, far)

    for i in range(frames):
        azimuth = 2.0 * math.pi * i / frames
        eye, view = orbit_camera(target, distance, azimuth, elevation)

        fbo.use()
        fbo.clear(*app.bg_color)

        app.ctx.enable_only(
            moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
        )
        app.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        app.ctx.viewport = (0, 0, width, height)

        # Render the selected cloud
        if 0 <= app.selected_index < len(app.entries):
            entry = app.entries[app.selected_index]
            gpu = entry.full_gpu or entry.preview_gpu
            if gpu:
                draw_cloud(gpu, app.program, entry.model_transform, view, proj,
                           point_size=app.point_size, sharpness=app.point_sharpness,
                           depth_falloff=getattr(app, 'depth_falloff', 1.0),
                           brightness=app.brightness, contrast=app.contrast,
                           saturation=app.saturation,
                           r_gain=app.r_gain, g_gain=app.g_gain, b_gain=app.b_gain,
                           label_blend=app.label_blend,
                           label_texture=app.label_texture,
                           falloff_texture=app.falloff_texture,
                           clip_min=app.clip_min, clip_max=app.clip_max)

        # Overlays
        mvp = (proj @ view).astype(np.float32)
        if app.show_grid:
            app.overlays.draw_grid_infinite(view, proj, eye)
        if app.show_bbox:
            app.overlays.draw_bbox(mvp)

        img = fbo.read_image()
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        img.save(frame_path)

        if (i + 1) % 30 == 0 or i == frames - 1:
            print(f"Spin: {i + 1}/{frames} frames")

    fbo.release()
    app.ctx.screen.use()
    print(f"Spin export complete: {frames} frames -> {output_dir}")
