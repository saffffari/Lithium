"""Still image export from the current view."""

import os
from src.rendering.framebuffer import ExportFramebuffer


def export_still(app, output_path: str, width: int = 1920, height: int = 1080):
    """Render the current view to a PNG file."""
    import moderngl
    import numpy as np
    from src.rendering.point_cloud_renderer import draw_cloud

    fbo = ExportFramebuffer(app.ctx, width, height)
    try:
        fbo.use()
        fbo.clear(*app.bg_color)

        app.ctx.enable_only(
            moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
        )
        app.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        app.ctx.viewport = (0, 0, width, height)

        # Use current camera but with export aspect ratio.
        # Restore in finally so exceptions don't leave stale aspect.
        old_aspect = app.camera.aspect
        app.camera.aspect = width / height
        view = app.camera.get_view_matrix()
        proj = app.camera.get_projection_matrix()

        # Render clouds
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

        # Render overlays
        mvp = (proj @ view).astype(np.float32)
        if app.show_grid:
            app.overlays.draw_grid_infinite(
                view, proj, app.camera.get_eye_position())
        if app.show_bbox:
            app.overlays.draw_bbox(mvp)

        # Save
        img = fbo.read_image()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)
        print(f"Exported: {output_path} ({width}x{height})")
    finally:
        app.camera.aspect = old_aspect
        fbo.release()
        app.ctx.screen.use()
