"""Headless CLI batch renderer for Lithium.

Renders stills or spin animations from point cloud files without opening the GUI.
Uses a hidden GLFW window to provide an OpenGL context for offscreen rendering.
"""

import argparse
import math
import os
import sys
import time

import glfw
import moderngl
import numpy as np

from src.core.camera import Camera, PRESETS
from src.data.loader import load_point_cloud, scan_directory
from src.rendering.shaders import create_point_cloud_program
from src.rendering.point_cloud_renderer import upload_cloud, draw_cloud
from src.rendering.overlay_renderer import OverlayRenderer
from src.rendering.framebuffer import ExportFramebuffer
from src.rendering.label_texture import create_label_color_texture
from src.rendering.falloff_texture import create_falloff_texture
from src.core.envelope import Envelope
from src.data.labels import LabelRegistry
from src.utils.math_utils import orbit_camera, perspective


class HeadlessRenderer:
    """Offscreen point cloud renderer using a hidden GLFW window."""

    def __init__(self):
        self.ctx: moderngl.Context = None
        self.program = None
        self.overlays: OverlayRenderer = None
        self._window = None

    def init(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.VISIBLE, False)

        self._window = glfw.create_window(1, 1, "Lithium Headless", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create hidden GLFW window")

        glfw.make_context_current(self._window)
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.program = create_point_cloud_program(self.ctx)
        self.overlays = OverlayRenderer(self.ctx)
        # Default label and falloff textures (required by shader)
        self.label_texture = create_label_color_texture(self.ctx, LabelRegistry())
        self.falloff_texture = create_falloff_texture(self.ctx, Envelope.hard_circle())

    def shutdown(self):
        if self.overlays:
            self.overlays.release()
        glfw.terminate()

    def render_still(self, file_path: str, output_path: str,
                     width: int = 1920, height: int = 1080,
                     camera_preset: str = 'iso',
                     point_size: float = 3.0, sharpness: float = 0.0,
                     bg_color: tuple = (0.0, 0.0, 0.0),
                     show_grid: bool = False, show_bbox: bool = False):
        """Render a single point cloud to a PNG file."""
        cloud = load_point_cloud(file_path)
        gpu = upload_cloud(cloud, self.ctx, self.program)

        camera = Camera()
        camera.fit_to_bounds(cloud.bounds_min, cloud.bounds_max, animate=False)
        camera.set_preset(camera_preset)
        camera.snap_to_goal()  # headless renderer has no update loop
        camera.aspect = width / height

        view = camera.get_view_matrix()
        proj = camera.get_projection_matrix()
        model = np.eye(4, dtype=np.float32)

        fbo = ExportFramebuffer(self.ctx, width, height)
        fbo.use()
        fbo.clear(*bg_color)

        self.ctx.enable_only(
            moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
        )
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.viewport = (0, 0, width, height)

        draw_cloud(gpu, self.program, model, view, proj,
                   point_size=point_size, sharpness=sharpness,
                   label_texture=self.label_texture,
                   falloff_texture=self.falloff_texture)

        mvp = (proj @ view).astype(np.float32)
        if show_grid or show_bbox:
            self.overlays.build_bounding_box(cloud.bounds_min, cloud.bounds_max)
            self.overlays.build_grid(cloud.bounds_min, cloud.bounds_max)
        if show_grid:
            self.overlays.draw_grid_infinite(
                view, proj, camera.get_eye_position())
        if show_bbox:
            self.overlays.draw_bbox(mvp)

        img = fbo.read_image()
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        img.save(output_path)

        fbo.release()
        gpu.release()

    def render_spin(self, file_path: str, output_dir: str,
                    frames: int = 360,
                    width: int = 1920, height: int = 1080,
                    camera_preset: str = 'iso',
                    point_size: float = 3.0, sharpness: float = 0.0,
                    bg_color: tuple = (0.0, 0.0, 0.0),
                    show_grid: bool = False, show_bbox: bool = False):
        """Render a 360-degree spin as a PNG sequence."""
        cloud = load_point_cloud(file_path)
        gpu = upload_cloud(cloud, self.ctx, self.program)

        camera = Camera()
        camera.fit_to_bounds(cloud.bounds_min, cloud.bounds_max, animate=False)
        camera.set_preset(camera_preset)
        camera.snap_to_goal()  # headless renderer has no update loop
        camera.aspect = width / height

        target = camera.target.copy()
        distance = camera.distance
        elevation = camera.elevation
        fov = camera.fov
        near = camera.near
        far = camera.far
        model = np.eye(4, dtype=np.float32)

        proj = perspective(fov, width / height, near, far)

        if show_grid or show_bbox:
            self.overlays.build_bounding_box(cloud.bounds_min, cloud.bounds_max)
            self.overlays.build_grid(cloud.bounds_min, cloud.bounds_max)

        os.makedirs(output_dir, exist_ok=True)
        fbo = ExportFramebuffer(self.ctx, width, height)

        for i in range(frames):
            azimuth = 2.0 * math.pi * i / frames
            eye, view = orbit_camera(target, distance, azimuth, elevation)

            fbo.use()
            fbo.clear(*bg_color)

            self.ctx.enable_only(
                moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
            )
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.viewport = (0, 0, width, height)

            draw_cloud(gpu, self.program, model, view, proj,
                       point_size=point_size, sharpness=sharpness,
                       label_texture=self.label_texture)

            mvp = (proj @ view).astype(np.float32)
            if show_grid:
                self.overlays.draw_grid_infinite(view, proj, eye)
            if show_bbox:
                self.overlays.draw_bbox(mvp)

            img = fbo.read_image()
            img.save(os.path.join(output_dir, f"frame_{i:04d}.png"))

            if (i + 1) % 30 == 0 or i == frames - 1:
                print(f"  Spin: {i + 1}/{frames} frames")

        fbo.release()
        gpu.release()


def _parse_bg_color(value: str) -> tuple:
    """Parse background color from 'R,G,B' floats or '#RRGGBB' hex."""
    value = value.strip()
    if value.startswith('#'):
        value = value.lstrip('#')
        r = int(value[0:2], 16) / 255.0
        g = int(value[2:4], 16) / 255.0
        b = int(value[4:6], 16) / 255.0
        return (r, g, b)
    parts = [float(x.strip()) for x in value.split(',')]
    if len(parts) != 3:
        raise ValueError("Background color must be R,G,B or #RRGGBB")
    return tuple(parts)


def _collect_files(path: str) -> list[str]:
    """Collect point cloud files from a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    elif os.path.isdir(path):
        files = scan_directory(path)
        if not files:
            print(f"No point cloud files found in {path}")
            sys.exit(1)
        return files
    else:
        print(f"Path not found: {path}")
        sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='lithium',
        description='Lithium — Point Cloud Visualizer and Batch Renderer',
    )
    sub = parser.add_subparsers(dest='command')

    # Shared options
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('input', help='Point cloud file or directory')
    common.add_argument('-o', '--output', default='./output',
                        help='Output directory (default: ./output)')
    common.add_argument('--width', type=int, default=1920,
                        help='Render width in pixels (default: 1920)')
    common.add_argument('--height', type=int, default=1080,
                        help='Render height in pixels (default: 1080)')
    common.add_argument('--camera', default='iso',
                        choices=list(PRESETS.keys()),
                        help='Camera preset (default: iso)')
    common.add_argument('--point-size', type=float, default=3.0,
                        help='Point size (default: 3.0)')
    common.add_argument('--sharpness', type=float, default=0.0,
                        help='Point sharpness 0=hard 1=soft (default: 0.0)')
    common.add_argument('--bg', default='0,0,0',
                        help='Background color as R,G,B or #RRGGBB (default: 0,0,0)')
    common.add_argument('--grid', action='store_true',
                        help='Show grid floor')
    common.add_argument('--bbox', action='store_true',
                        help='Show bounding box')

    # render subcommand
    sub.add_parser('render', parents=[common],
                   help='Batch render stills from point clouds')

    # spin subcommand
    spin_parser = sub.add_parser('spin', parents=[common],
                                 help='Batch render 360-degree spin animations')
    spin_parser.add_argument('--frames', type=int, default=360,
                             help='Number of frames per spin (default: 360)')

    return parser


def cli_main(argv: list[str] | None = None):
    """Entry point for CLI batch rendering."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    bg_color = _parse_bg_color(args.bg)
    files = _collect_files(args.input)
    os.makedirs(args.output, exist_ok=True)

    renderer = HeadlessRenderer()
    renderer.init()

    t0 = time.time()
    print(f"Lithium CLI — {args.command} | {len(files)} file(s) | {args.width}x{args.height} | camera={args.camera}")

    opts = dict(
        width=args.width, height=args.height,
        camera_preset=args.camera,
        point_size=args.point_size, sharpness=args.sharpness,
        bg_color=bg_color,
        show_grid=args.grid, show_bbox=args.bbox,
    )

    if args.command == 'render':
        for i, f in enumerate(files):
            name = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(args.output, f"{name}.png")
            print(f"[{i + 1}/{len(files)}] {os.path.basename(f)} -> {out_path}")
            renderer.render_still(f, out_path, **opts)

    elif args.command == 'spin':
        for i, f in enumerate(files):
            name = os.path.splitext(os.path.basename(f))[0]
            out_dir = os.path.join(args.output, name)
            print(f"[{i + 1}/{len(files)}] {os.path.basename(f)} -> {out_dir}/")
            renderer.render_spin(f, out_dir, frames=args.frames, **opts)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    renderer.shutdown()


if __name__ == '__main__':
    cli_main()
