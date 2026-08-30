#!/usr/bin/env python
"""Streaming out-of-core LOD renderer over a .3pc c-store.

The full-collection companion to demo.py: instead of uploading one
resident blob, every frame

  1. frustum-culls the tile grid (AABB vs clip planes),
  2. picks a level per tile by projected point spacing (screen-space
     error ≤ ~1.8 px), degraded farthest-first under the point budget,
  3. draws whatever is resident (a coarser resident level stands in
     while the wanted one streams), and
  4. asks the IO thread for missing buffers; uploads are capped per
     frame, LRU-evicted past the VRAM budget.

Same EDL post + orbit + double-click pivot as demo.py.

Usage: demo_lod.py <store.3pc> [--budget-mpts 150] [--vram-gb 14]
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import wgpu
from rendercanvas.auto import RenderCanvas, loop

from demo import (SHADER_EDL, SHADER_BLIT, perspective, look_at, Orbit)

L0_SPACING = 0.30          # m — approximate native pulse spacing
PX_ERROR = 1.8             # target projected spacing, px
UPLOAD_BYTES_PER_FRAME = 256 * 1024 * 1024

SHADER_POINTS_Q = """
struct U {
    mvp: mat4x4<f32>,
    eye: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
struct Tile {
    bmin: vec4<f32>,
    span: vec4<f32>,
};
@group(1) @binding(0) var<uniform> tile: Tile;

struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) eyedist: f32,
};

@vertex
fn vs_main(@location(0) q: vec4<u32>, @location(1) c: vec4<f32>) -> VSOut {
    let p = tile.bmin.xyz + (vec3<f32>(vec3<u32>(q.xyz)) / 65535.0)
        * tile.span.xyz;
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


class Tile:
    __slots__ = ("name", "bmin", "bmax", "center", "radius", "levels",
                 "resident", "pending", "ubo_bg", "last_used")

    def __init__(self, entry, center_off):
        self.name = entry["name"]
        self.bmin = np.asarray(entry["bounds_min"]) - center_off
        self.bmax = np.asarray(entry["bounds_max"]) - center_off
        self.center = (self.bmin + self.bmax) * 0.5
        self.radius = float(np.linalg.norm(self.bmax - self.bmin)) * 0.5
        self.levels = {int(k): v for k, v in entry["levels"].items()}
        self.resident: dict[int, tuple] = {}   # level -> (buf, count, bytes)
        self.pending: set[int] = set()
        self.ubo_bg = None
        self.last_used = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--budget-mpts", type=float, default=150.0)
    ap.add_argument("--vram-gb", type=float, default=14.0)
    ap.add_argument("--screenshot-dir", default=".")
    args = ap.parse_args()

    root = Path(args.store)
    index = json.loads((root / "index.json").read_text())
    gmin = np.asarray(index["bounds_min"])
    gmax = np.asarray(index["bounds_max"])
    center = (gmin + gmax) * 0.5
    tiles = [Tile(e, center) for e in index["tiles"]]
    n_levels = index["n_levels"]
    total_l0 = sum(t.levels.get(0, 0) for t in tiles)
    print(f"{len(tiles)} tiles, L0 {total_l0/1e9:.2f}B pts, "
          f"{n_levels} levels")

    canvas = RenderCanvas(size=(1920, 1080), title="Lithium c-store demo")
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync(required_limits=adapter.limits)
    ctx = canvas.get_context("wgpu")
    fmt = ctx.get_preferred_format(adapter)
    ctx.configure(device=device, format=fmt)
    print("adapter:", adapter.info["device"], "|", adapter.info["backend_type"])

    sm_pts = device.create_shader_module(code=SHADER_POINTS_Q)
    sm_edl = device.create_shader_module(code=SHADER_EDL)
    sm_blit = device.create_shader_module(code=SHADER_BLIT)

    pipe_pts = device.create_render_pipeline(
        layout="auto",
        vertex={
            "module": sm_pts, "entry_point": "vs_main",
            "buffers": [{
                "array_stride": 12,
                "attributes": [
                    {"format": wgpu.VertexFormat.uint16x4,
                     "offset": 0, "shader_location": 0},
                    {"format": wgpu.VertexFormat.unorm8x4,
                     "offset": 8, "shader_location": 1},
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

    ubo = device.create_buffer(size=80, usage=wgpu.BufferUsage.UNIFORM
                               | wgpu.BufferUsage.COPY_DST)
    edl_ubo = device.create_buffer(size=16, usage=wgpu.BufferUsage.UNIFORM
                                   | wgpu.BufferUsage.COPY_DST)

    tile_bgl = pipe_pts.get_bind_group_layout(1)
    for t in tiles:
        tb = np.zeros(8, dtype=np.float32)
        tb[0:3] = t.bmin
        tb[4:7] = t.bmax - t.bmin
        tub = device.create_buffer_with_data(
            data=tb.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        t.ubo_bg = device.create_bind_group(
            layout=tile_bgl,
            entries=[{"binding": 0,
                      "resource": {"buffer": tub, "offset": 0, "size": 32}}])

    # ---- IO thread ---------------------------------------------------
    req_q: queue.Queue = queue.Queue()
    done_q: queue.Queue = queue.Queue()

    def io_worker():
        while True:
            tile, lvl = req_q.get()
            if tile is None:
                return
            path = root / f"L{lvl}" / f"{tile.name}.bin"
            try:
                data = np.fromfile(path, dtype=np.uint8)
                done_q.put((tile, lvl, data))
            except OSError as e:
                print(f"io: {path.name}: {e}")
                done_q.put((tile, lvl, None))

    threading.Thread(target=io_worker, daemon=True).start()
    threading.Thread(target=io_worker, daemon=True).start()

    cam = Orbit(gmin - center, gmax - center)
    state = {"size": (0, 0), "tex": None, "edl": [2.6, 1.0],
             "frames": 0, "t_fps": time.perf_counter(), "fps": 0.0,
             "frame_no": 0, "vram": 0, "drawn": 0, "shot": False,
             "total_frames": 0}
    budget_pts = int(args.budget_mpts * 1e6)
    vram_budget = int(args.vram_gb * 1e9)

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

    def frustum_planes(mvp):
        m = mvp.T  # row-major math view
        rows = [m[3] + m[0], m[3] - m[0], m[3] + m[1],
                m[3] - m[1], m[3] + m[2], m[3] - m[2]]
        return [r / (np.linalg.norm(r[:3]) + 1e-12) for r in rows]

    def select(planes, eye, h):
        """(tile, level) pairs to draw this frame + stream requests."""
        tanf = math.tan(math.radians(55) / 2)
        vis = []
        for t in tiles:
            c4 = np.append(t.center, 1.0)
            if any(float(p @ c4) < -t.radius for p in planes):
                continue
            dist = max(float(np.linalg.norm(t.center - eye)) - t.radius,
                       1.0)
            ppm = h / (2.0 * dist * tanf)     # px per meter at tile
            # Coarsest level whose point spacing still projects under
            # the error budget; nothing qualifies up close → L0.
            want = 0
            for lvl in range(n_levels - 1, 0, -1):
                if (0.5 * (2 ** (lvl - 1))) * ppm <= PX_ERROR:
                    want = lvl
                    break
            want = min(want, max(t.levels))
            vis.append((dist, t, want))
        vis.sort(key=lambda v: v[0])

        chosen = []
        total = 0
        for dist, t, want in vis:
            lvl = want
            while total + t.levels.get(lvl, 0) > budget_pts \
                    and lvl < max(t.levels):
                lvl += 1                       # degrade under budget
            total += t.levels.get(lvl, 0)
            chosen.append((t, lvl))
        return chosen

    def upload_pending():
        budget = UPLOAD_BYTES_PER_FRAME
        while budget > 0:
            try:
                tile, lvl, data = done_q.get_nowait()
            except queue.Empty:
                return
            tile.pending.discard(lvl)
            if data is None:
                continue
            buf = device.create_buffer_with_data(
                data=data, usage=wgpu.BufferUsage.VERTEX)
            tile.resident[lvl] = (buf, len(data) // 12, len(data))
            state["vram"] += len(data)
            budget -= len(data)

    def evict():
        if state["vram"] <= vram_budget:
            return
        entries = []
        for t in tiles:
            for lvl, (buf, cnt, nbytes) in t.resident.items():
                entries.append((t.last_used, t, lvl, nbytes))
        entries.sort()
        for last, t, lvl, nbytes in entries:
            if state["vram"] <= vram_budget * 0.9:
                break
            if last >= state["frame_no"] - 1:
                break                          # visible now — stop
            buf = t.resident.pop(lvl)[0]
            buf.destroy()
            state["vram"] -= nbytes

    def save_screenshot():
        w, h = state["size"]
        t_fin = state["tex"][3]
        data = device.queue.read_texture(
            {"texture": t_fin, "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
            (w, h, 1))
        from PIL import Image
        img = Image.frombytes("RGBA", (w, h), bytes(data))
        out = Path(args.screenshot_dir) / f"cstore_{int(time.time())}.png"
        img.convert("RGB").save(out)
        print(f"screenshot -> {out}", flush=True)

    def pick_world_point(lx, ly):
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

    def frame():
        w, h = canvas.get_physical_size()
        if w == 0 or h == 0:
            canvas.request_draw(frame)
            return
        ensure_targets(w, h)
        t_col, t_dep, t_z, t_fin, bg_edl, bg_blit, bg_pts = state["tex"]
        state["frame_no"] += 1

        now = time.perf_counter()
        if now - cam.last_input > 3.0:
            cam.yaw += 0.0016

        mvp, eye = cam.matrices(w / h)
        ub = np.zeros(20, dtype=np.float32)
        ub[:16] = mvp.reshape(-1)
        ub[16:19] = eye
        device.queue.write_buffer(ubo, 0, ub.tobytes())
        device.queue.write_buffer(edl_ubo, 0, np.array(
            state["edl"] + [0.0, 0.0], dtype=np.float32).tobytes())

        upload_pending()
        chosen = select(frustum_planes(mvp @ np.eye(4, dtype=np.float32)),
                        eye, h)

        enc = device.create_command_encoder()
        rp = enc.begin_render_pass(
            color_attachments=[
                {"view": t_col.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)},
                {"view": t_dep.create_view(), "load_op": wgpu.LoadOp.clear,
                 "store_op": wgpu.StoreOp.store, "clear_value": (0, 0, 0, 0)},
            ],
            depth_stencil_attachment={
                "view": t_z.create_view(),
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.store,
                "depth_clear_value": 1.0,
            })
        rp.set_pipeline(pipe_pts)
        rp.set_bind_group(0, bg_pts)
        drawn = 0
        for t, want in chosen:
            t.last_used = state["frame_no"]
            use = None
            if want in t.resident:
                use = want
            else:
                if want not in t.pending:
                    t.pending.add(want)
                    req_q.put((t, want))
                coarser = [l for l in t.resident if l > want]
                finer = [l for l in t.resident if l < want]
                if coarser:
                    use = min(coarser)
                elif finer:
                    use = min(finer)
            if use is None:
                continue
            buf, cnt, _b = t.resident[use]
            rp.set_bind_group(1, t.ubo_bg)
            rp.set_vertex_buffer(0, buf)
            rp.draw(cnt)
            drawn += cnt
        rp.end()
        state["drawn"] = drawn

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

        evict()

        state["total_frames"] += 1
        if state["total_frames"] in (300, 1200):
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
                f"Lithium c-store — {state['drawn']/1e6:.0f}M drawn / "
                f"{total_l0/1e9:.1f}B total — {state['fps']:.0f} fps — "
                f"VRAM {state['vram']/1e9:.1f}G")
            print(f"fps {state['fps']:6.1f}  drawn {state['drawn']/1e6:7.1f}M"
                  f"  vram {state['vram']/1e9:5.2f}G  q {req_q.qsize()}",
                  flush=True)
        canvas.request_draw(frame)

    drag = {"btn": 0, "x": 0.0, "y": 0.0}

    def on_event(ev):
        et = ev["event_type"]
        cam.last_input = time.perf_counter()
        if et == "double_click":
            hit = pick_world_point(ev["x"], ev["y"])
            if hit is not None:
                cam.target = np.asarray(hit, dtype=np.float32)
                cam.dist = max(cam.dist * 0.72, cam.span * 0.001)
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
            cam.dist = min(max(cam.dist, cam.span * 0.001), cam.span * 4)
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
