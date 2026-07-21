#!/usr/bin/env python
"""wgpu (WebGPU→Vulkan) massive point cloud demo — EDL renderer spike.

The render2 proof-of-concept for 3Photon 1.1's clean-slate renderer:

  pass 1  point-list → MRT: rgba8 color + r32float linear eye depth
          (+ depth24plus for z-test), one draw per ≤12M-pt chunk
  pass 2  fullscreen EDL (eye-dome lighting): per-pixel log-depth
          gradient against 8 neighbors → soft ambient occlusion that
          makes bare-earth structure readable at 1px/point
  pass 3  blit to swapchain (final stays offscreen so screenshots are
          a plain texture readback)

Input:  packed .npy from convert_laz.py (16 B/point).
Drive:  LMB orbit · RMB/MMB pan · wheel dolly · [ / ] EDL strength
        P screenshot · Esc quit. Idles into a slow cinematic orbit.

Usage:  demo.py <prefix>  [--screenshot-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import wgpu
from rendercanvas.auto import RenderCanvas, loop

CHUNK_PTS = 12_000_000          # ≤192 MB per vertex buffer

SHADER_POINTS = """
struct U {
    mvp: mat4x4<f32>,
    eye: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) eyedist: f32,
};

@vertex
fn vs_main(@location(0) p: vec3<f32>, @location(1) c: vec4<f32>) -> VSOut {
    var out: VSOut;
    out.pos = u.mvp * vec4<f32>(p, 1.0);
    out.col = c;
    out.eyedist = length(p - u.eye.xyz);
    return out;
}

struct FSOut {
    @location(0) col: vec4<f32>,
    @location(1) lin_d: f32,
};

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;
    out.col = vec4<f32>(in.col.rgb, 1.0);
    out.lin_d = in.eyedist;
    return out;
}
"""

SHADER_EDL = """
@group(0) @binding(0) var t_col: texture_2d<f32>;
@group(0) @binding(1) var t_dep: texture_2d<f32>;
struct P { strength: f32, radius: f32, _p0: f32, _p1: f32 };
@group(0) @binding(2) var<uniform> prm: P;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VSOut;
    out.pos = vec4<f32>(xy[i], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let ip = vec2<i32>(in.pos.xy);
    let dims = vec2<i32>(textureDimensions(t_dep));
    let zc = textureLoad(t_dep, ip, 0).r;
    var col = textureLoad(t_col, ip, 0).rgb;
    if (zc <= 0.0) {
        // background: subtle vertical wash so silhouettes read
        let t = f32(ip.y) / f32(dims.y);
        return vec4<f32>(mix(vec3<f32>(0.020, 0.024, 0.032),
                             vec3<f32>(0.008, 0.009, 0.012), t), 1.0);
    }
    let lc = log2(max(zc, 0.0001));
    var resp: f32 = 0.0;
    let r = i32(prm.radius);
    var offs = array<vec2<i32>, 8>(
        vec2<i32>(-1, 0), vec2<i32>(1, 0), vec2<i32>(0, -1), vec2<i32>(0, 1),
        vec2<i32>(-1, -1), vec2<i32>(1, 1), vec2<i32>(-1, 1), vec2<i32>(1, -1));
    for (var k = 0; k < 8; k = k + 1) {
        let q = clamp(ip + offs[k] * r, vec2<i32>(0, 0), dims - 1);
        let zn = textureLoad(t_dep, q, 0).r;
        if (zn > 0.0) {
            resp = resp + max(0.0, lc - log2(max(zn, 0.0001)));
        } else {
            resp = resp + 0.18;   // silhouette against empty sky
        }
    }
    let shade = exp(-prm.strength * resp);
    col = col * (0.12 + 0.88 * shade);
    // mild contrast S-curve that keeps the palette instead of the old
    // reinhard-lift that washed everything to cream
    col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)),
              vec3<f32>(0.88)) * 1.04;
    return vec4<f32>(col, 1.0);
}
"""

SHADER_BLIT = """
@group(0) @binding(0) var t_src: texture_2d<f32>;

struct VSOut { @builtin(position) pos: vec4<f32> };

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VSOut {
    var xy = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    var out: VSOut;
    out.pos = vec4<f32>(xy[i], 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return textureLoad(t_src, vec2<i32>(in.pos.xy), 0);
}
"""


def perspective(fov_y, aspect, near, far):
    f = 1.0 / math.tan(fov_y / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = far / (near - far)
    m[2, 3] = far * near / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up):
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = up2
    m[2, :3] = -fwd
    m[:3, 3] = -m[:3, :3] @ eye
    return m


class Orbit:
    def __init__(self, bounds_min, bounds_max):
        self.target = (np.asarray(bounds_min) + np.asarray(bounds_max)) * 0.5
        span = float(np.linalg.norm(np.asarray(bounds_max)
                                    - np.asarray(bounds_min)))
        self.dist = span * 0.38
        self.yaw = math.radians(35.0)
        self.pitch = math.radians(26.0)
        self.span = span
        self.last_input = 0.0

    def eye(self):
        cp = math.cos(self.pitch)
        return self.target + self.dist * np.array([
            cp * math.cos(self.yaw), cp * math.sin(self.yaw),
            math.sin(self.pitch)], dtype=np.float32)

    def matrices(self, aspect):
        eye = self.eye()
        view = look_at(eye, self.target,
                       np.array([0, 0, 1], dtype=np.float32))
        near = max(self.dist * 0.002, 0.5)
        far = self.dist * 6 + self.span * 2
        proj = perspective(math.radians(55), aspect, near, far)
        return (proj @ view).T.copy(), eye   # column-major for WGSL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--screenshot-dir", default=".")
    args = ap.parse_args()

    meta = json.loads(Path(args.prefix + ".json").read_text())
    pts = np.load(args.prefix + ".bin", mmap_mode="r")
    n_total = len(pts)
    print(f"{n_total:,} points | block "
          f"{np.subtract(meta['bounds_max'], meta['bounds_min']).round(0)} m")

    canvas = RenderCanvas(
        size=(1920, 1080),
        title=f"3Photon render2 spike — {n_total/1e6:.0f}M pts")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    print("adapter:", adapter.info["device"], "|", adapter.info["backend_type"])
    device = adapter.request_device_sync(required_limits=adapter.limits)
    ctx = canvas.get_context("wgpu")
    fmt = ctx.get_preferred_format(adapter)
    ctx.configure(device=device, format=fmt)

    # ---- vertex buffers (chunked) -----------------------------------
    t0 = time.perf_counter()
    raw = pts.view(np.uint8).reshape(n_total, 16)
    chunks = []
    for start in range(0, n_total, CHUNK_PTS):
        end = min(start + CHUNK_PTS, n_total)
        buf = device.create_buffer_with_data(
            data=np.ascontiguousarray(raw[start:end]),
            usage=wgpu.BufferUsage.VERTEX)
        chunks.append((buf, end - start))
    print(f"uploaded {len(chunks)} chunks "
          f"({n_total * 16 / 1e9:.2f} GB) in "
          f"{time.perf_counter() - t0:.1f}s")

    ubo = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM
                               | wgpu.BufferUsage.COPY_DST)
    edl_ubo = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM
                                   | wgpu.BufferUsage.COPY_DST)

    sm_pts = device.create_shader_module(code=SHADER_POINTS)
    sm_edl = device.create_shader_module(code=SHADER_EDL)
    sm_blit = device.create_shader_module(code=SHADER_BLIT)

    pipe_pts = device.create_render_pipeline(
        layout="auto",
        vertex={
            "module": sm_pts, "entry_point": "vs_main",
            "buffers": [{
                "array_stride": 16,
                "attributes": [
                    {"format": wgpu.VertexFormat.float32x3,
                     "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.unorm8x4,
                     "offset": 12, "shader_location": 1},
                ]}],
        },
        primitive={"topology": wgpu.PrimitiveTopology.point_list},
        depth_stencil={
            "format": wgpu.TextureFormat.depth24plus,
            "depth_write_enabled": True,
            "depth_compare": wgpu.CompareFunction.less,
        },
        fragment={"module": sm_pts, "entry_point": "fs_main", "targets": [
            {"format": wgpu.TextureFormat.rgba8unorm},
            {"format": wgpu.TextureFormat.r32float},
        ]},
    )
    pipe_edl = device.create_render_pipeline(
        layout="auto",
        vertex={"module": sm_edl, "entry_point": "vs_main"},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        fragment={"module": sm_edl, "entry_point": "fs_main",
                  "targets": [{"format": wgpu.TextureFormat.rgba8unorm}]},
    )
    pipe_blit = device.create_render_pipeline(
        layout="auto",
        vertex={"module": sm_blit, "entry_point": "vs_main"},
        primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        fragment={"module": sm_blit, "entry_point": "fs_main",
                  "targets": [{"format": fmt}]},
    )

    state = {"size": (0, 0), "tex": None, "edl": [2.6, 1.0],
             "shot": False, "frames": 0, "t_fps": time.perf_counter(),
             "fps": 0.0, "total_frames": 0}
    cam = Orbit(meta["bounds_min"], meta["bounds_max"])

    def ensure_targets(w, h):
        if state["size"] == (w, h):
            return
        state["size"] = (w, h)
        mk = lambda f, extra=0: device.create_texture(
            size=(w, h, 1), format=f,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | extra)
        t_col = mk(wgpu.TextureFormat.rgba8unorm,
                   wgpu.TextureUsage.TEXTURE_BINDING)
        t_dep = mk(wgpu.TextureFormat.r32float,
                   wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_SRC)
        t_z = mk(wgpu.TextureFormat.depth24plus)
        t_fin = mk(wgpu.TextureFormat.rgba8unorm,
                   wgpu.TextureUsage.TEXTURE_BINDING
                   | wgpu.TextureUsage.COPY_SRC)
        bg_edl = device.create_bind_group(
            layout=pipe_edl.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": t_col.create_view()},
                {"binding": 1, "resource": t_dep.create_view()},
                {"binding": 2, "resource": {"buffer": edl_ubo,
                                            "offset": 0, "size": 16}},
            ])
        bg_blit = device.create_bind_group(
            layout=pipe_blit.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": t_fin.create_view()}])
        bg_pts = device.create_bind_group(
            layout=pipe_pts.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {"buffer": ubo,
                                                 "offset": 0, "size": 80}}])
        state["tex"] = (t_col, t_dep, t_z, t_fin, bg_edl, bg_blit, bg_pts)

    def pick_world_point(lx: float, ly: float):
        """Depth-readback pick: logical click coords → world position.

        Reads a 64-px strip of the linear-depth target around the click
        (64*4 B satisfies the 256-B bytes_per_row alignment), then
        unprojects eye + ray * depth. Returns None over background.
        """
        w, h = state["size"]
        if state["tex"] is None or w == 0:
            return None
        lw, lh = canvas.get_logical_size()
        px = int(lx * w / max(lw, 1))
        py = int(ly * h / max(lh, 1))
        if not (0 <= px < w and 0 <= py < h):
            return None
        x0 = min(max(px - 32, 0), max(w - 64, 0))
        strip_w = min(64, w)
        t_dep = state["tex"][1]
        data = device.queue.read_texture(
            {"texture": t_dep, "mip_level": 0, "origin": (x0, py, 0)},
            {"offset": 0, "bytes_per_row": 256, "rows_per_image": 1},
            (strip_w, 1, 1))
        depths = np.frombuffer(bytes(data), dtype=np.float32)[:strip_w]
        d = float(depths[min(px - x0, strip_w - 1)])
        if d <= 0.0:
            # background — try the nearest hit in the strip so a click
            # just off a ridge still lands
            hits = depths[depths > 0.0]
            if hits.size == 0:
                return None
            d = float(hits.min())
        eye = cam.eye()
        fwd = cam.target - eye
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 0, 1.0])
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        tanf = math.tan(math.radians(55) / 2)
        ndc_x = 2.0 * (px + 0.5) / w - 1.0
        ndc_y = 1.0 - 2.0 * (py + 0.5) / h
        ray = fwd + tanf * (ndc_x * (w / h) * right + ndc_y * up)
        ray = ray / np.linalg.norm(ray)
        return eye + ray * d

    def save_screenshot():
        w, h = state["size"]
        t_fin = state["tex"][3]
        data = device.queue.read_texture(
            {"texture": t_fin, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1))
        from PIL import Image
        img = Image.frombytes("RGBA", (w, h), bytes(data))
        out = Path(args.screenshot_dir) / f"palisades_{int(time.time())}.png"
        img.convert("RGB").save(out)
        print(f"screenshot -> {out}")

    def frame():
        w, h = canvas.get_physical_size()
        if w == 0 or h == 0:
            canvas.request_draw(frame)
            return
        ensure_targets(w, h)
        t_col, t_dep, t_z, t_fin, bg_edl, bg_blit, bg_pts = state["tex"]

        # idle cinematic orbit
        now = time.perf_counter()
        if now - cam.last_input > 3.0:
            cam.yaw += 0.0022

        mvp, eye = cam.matrices(w / h)
        ub = np.zeros(20, dtype=np.float32)
        ub[:16] = mvp.reshape(-1)
        ub[16:19] = eye
        device.queue.write_buffer(ubo, 0, ub.tobytes())
        device.queue.write_buffer(edl_ubo, 0, np.array(
            state["edl"] + [0.0, 0.0], dtype=np.float32).tobytes())

        enc = device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {"view": t_col.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store,
                 "clear_value": (0, 0, 0, 0)},
                {"view": t_dep.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store,
                 "clear_value": (0, 0, 0, 0)},
            ],
            depth_stencil_attachment={
                "view": t_z.create_view(),
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            })
        rp.set_pipeline(pipe_pts)
        rp.set_bind_group(0, bg_pts)
        for buf, count in chunks:
            rp.set_vertex_buffer(0, buf)
            rp.draw(count)
        rp.end()

        rp2 = enc.begin_render_pass(color_attachments=[
            {"view": t_fin.create_view(), "load_op": wgpu.LoadOp.clear,
             "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)}])
        rp2.set_pipeline(pipe_edl)
        rp2.set_bind_group(0, bg_edl)
        rp2.draw(3)
        rp2.end()

        tex = ctx.get_current_texture()
        rp3 = enc.begin_render_pass(color_attachments=[
            {"view": tex.create_view(), "load_op": wgpu.LoadOp.clear,
             "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 1)}])
        rp3.set_pipeline(pipe_blit)
        rp3.set_bind_group(0, bg_blit)
        rp3.draw(3)
        rp3.end()
        device.queue.submit([enc.finish()])

        state["total_frames"] += 1
        # Two automatic captures early in the idle orbit so headless
        # driving of the demo still produces evidence.
        if state["total_frames"] in (240, 900):
            state["shot"] = True
        if state["shot"]:
            state["shot"] = False
            save_screenshot()

        state["frames"] += 1
        if now - state["t_fps"] >= 1.0:
            state["fps"] = state["frames"] / (now - state["t_fps"])
            state["frames"] = 0
            state["t_fps"] = now
            canvas.set_title(
                f"3Photon render2 spike — {n_total/1e6:.0f}M pts — "
                f"{state['fps']:.0f} fps — EDL {state['edl'][0]:.1f}")
            print(f"fps {state['fps']:6.1f}   edl {state['edl'][0]:.2f}")
        canvas.request_draw(frame)

    drag = {"btn": 0, "x": 0.0, "y": 0.0}

    def on_event(ev):
        et = ev["event_type"]
        cam.last_input = time.perf_counter()
        if et == "double_click":
            hit = pick_world_point(ev["x"], ev["y"])
            if hit is not None:
                # Re-pivot: orbit target jumps to the clicked surface
                # point; ease the dolly in a touch so the pivot change
                # reads spatially.
                cam.target = np.asarray(hit, dtype=np.float32)
                cam.dist = max(cam.dist * 0.72, cam.span * 0.01)
        elif et == "pointer_down":
            drag["btn"] = ev["button"]
            drag["x"], drag["y"] = ev["x"], ev["y"]
        elif et == "pointer_up":
            drag["btn"] = 0
        elif et == "pointer_move" and drag["btn"]:
            dx, dy = ev["x"] - drag["x"], ev["y"] - drag["y"]
            drag["x"], drag["y"] = ev["x"], ev["y"]
            if drag["btn"] == 1:
                cam.yaw -= dx * 0.005
                cam.pitch = min(max(cam.pitch + dy * 0.005, -1.45), 1.55)
            else:
                eye = cam.eye()
                fwd = cam.target - eye
                fwd /= np.linalg.norm(fwd)
                right = np.cross(fwd, [0, 0, 1])
                right /= np.linalg.norm(right)
                up = np.cross(right, fwd)
                k = cam.dist * 0.0012
                cam.target += (-right * dx + up * dy) * k
        elif et == "wheel":
            cam.dist *= math.exp(ev["dy"] * 0.001)
            cam.dist = min(max(cam.dist, cam.span * 0.01), cam.span * 4)
        elif et == "key_down":
            k = ev.get("key", "")
            if k == "Escape":
                canvas.close()
            elif k == "p":
                state["shot"] = True
            elif k == "[":
                state["edl"][0] = max(0.0, state["edl"][0] - 0.2)
            elif k == "]":
                state["edl"][0] += 0.2

    canvas.add_event_handler(
        on_event, "pointer_down", "pointer_up", "pointer_move",
        "wheel", "key_down", "double_click")
    canvas.request_draw(frame)
    loop.run()


if __name__ == "__main__":
    main()
