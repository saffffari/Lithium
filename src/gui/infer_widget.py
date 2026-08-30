"""Light Table INFER button — the cloud-constellation loader.

A full-width button at the bottom of the LIGHT_TABLE sidebar that runs
the active project's model on the current cloud. The button IS the
progress display: it renders the actual point cloud (downsampled to a
few hundred points) as a slowly-rotating constellation and animates it
through the inference lifecycle:

  idle      dim grey constellation, slow spin, INFER label. Hover
            brightens. Disabled (flat, no spin) without a model/cloud.
  running   the spin speeds up, an orbiting comet circles the cloud,
            and a scan sweep passes through the points — points behind
            the sweep ignite in the accent color. A thin baseline bar
            eases toward ~92% (the subprocess reports no granular
            progress; honesty lives in the elapsed-seconds counter).
  complete  a radial ignition wave snaps every point to its PREDICTED
            class color (sampled from the applied labels through the
            live registry) with an expanding pulse ring; the colored
            constellation then breathes for a few seconds.
  failed    the constellation collapses to a red ember and recovers.

Pure ImGui draw-list rendering; numpy for the per-frame rotation of
<=650 points. Idle cost is one matmul + one draw loop; nothing is
computed when the widget is off-screen (ImGui clips, we early-out on
zero visible height).

The widget deliberately owns no inference plumbing: panels.py passes
in ``start_infer(app, index)`` (the shared batch-runner entry point,
which freezes the label namespace at submit) and the runner object to
observe. This module never imports panels — no import cycle.
"""

from __future__ import annotations

import math
import time

import imgui
import numpy as np

from src.gui.theme import (
    OP1_ORANGE, OP1_BLUE, OP1_GREEN, OP1_RED, OP1_GRAY, OP1_DIM, col32,
)
from src.gui.scale import s
from src.gui.op1_widgets import draw_styled_rect, op1_section

# Accent for the scan sweep — teal reads "process" against the warm
# label palette without colliding with the status colors.
_SWEEP = (0.30, 0.82, 0.75, 1.0)
_EMBER = (0.95, 0.25, 0.18, 1.0)

_MAX_PTS = 650

# constellation cache: file_key -> {pts (N,3) unit-box centered,
# sample_idx (N,), point_count}
_cache: dict[str, dict] = {}


def _constellation_for(entry) -> dict | None:
    """Sampled, normalized points for the entry's loaded cloud."""
    fk = getattr(entry, "file_key", None)
    gpu = entry.full_gpu or entry.preview_gpu
    cloud = gpu.cloud_data if gpu is not None else None
    if cloud is None or cloud.positions is None or cloud.point_count == 0:
        return None
    key = fk or entry.file_path
    hit = _cache.get(key)
    if hit is not None and hit["point_count"] == cloud.point_count:
        return hit

    n = cloud.point_count
    stride = max(1, n // _MAX_PTS)
    sample_idx = np.arange(0, n, stride, dtype=np.int64)[:_MAX_PTS]
    pts = np.asarray(cloud.positions[sample_idx], dtype=np.float32).copy()
    center = (pts.min(axis=0) + pts.max(axis=0)) * 0.5
    pts -= center
    span = float(np.abs(pts).max()) or 1.0
    pts /= span  # unit box, centered
    entry_c = {"pts": pts, "sample_idx": sample_idx, "point_count": n}
    if len(_cache) > 16:
        _cache.clear()  # tiny LRU stand-in; rebuild is cheap
    _cache[key] = entry_c
    return entry_c


def _sampled_label_colors(entry, sample_idx, registry) -> np.ndarray | None:
    """(N,3) float colors for the sampled points via the live registry."""
    gpu = entry.full_gpu or entry.preview_gpu
    cloud = gpu.cloud_data if gpu is not None else None
    if cloud is None or cloud.labels is None:
        return None
    if sample_idx[-1] >= len(cloud.labels):
        return None
    lbl = np.asarray(cloud.labels)[sample_idx]
    out = np.empty((len(lbl), 3), dtype=np.float32)
    color_lut: dict[int, tuple] = {}
    for i, lid in enumerate(lbl):
        lid = int(lid)
        c = color_lut.get(lid)
        if c is None:
            info = registry.get(lid) if registry is not None else None
            c = tuple(info.color[:3]) if (info is not None and lid != 0) \
                else (0.45, 0.45, 0.45)
            color_lut[lid] = c
        out[i] = c
    return out


def _anim(app) -> dict:
    st = getattr(app, "_lt_infer_anim", None)
    if st is None:
        st = {"state": "idle", "t0": 0.0, "file_key": None,
              "colors": None, "runner": None}
        app._lt_infer_anim = st
    return st


def draw_light_table_infer(app, *, start_infer, model_label: str | None,
                           can_run: bool, registry) -> None:
    """Draw the INFER section. See module docstring for the states.

    ``start_infer(app, index) -> bool`` launches the shared batch
    runner on one cloud; ``model_label`` names the checkpoint that
    would run (None = no model); ``can_run`` gates the whole affair;
    ``registry`` is app.label_registry (for ignition colors).
    """
    th = imgui.get_text_line_height()
    if not op1_section("INFER", _SWEEP,
                       fill_color=(_SWEEP[0] * 0.08, _SWEEP[1] * 0.08,
                                   _SWEEP[2] * 0.08, 1.0)):
        return

    dl = imgui.get_window_draw_list()
    wx, wy = imgui.get_cursor_screen_pos()
    w = imgui.get_content_region_available_width()
    h = th * 5.5
    if w <= 0:
        return

    st = _anim(app)
    now = time.perf_counter()
    entry = (app.entries[app.selected_index]
             if 0 <= app.selected_index < len(app.entries) else None)
    fk = getattr(entry, "file_key", None) if entry is not None else None

    # Selecting a different cloud resets any finished animation state.
    if st["file_key"] != fk and st["state"] in ("complete", "failed"):
        st["state"] = "idle"
        st["colors"] = None

    # ---- observe the runner we started -------------------------------
    runner = st.get("runner")
    if st["state"] == "running":
        if runner is None or (not runner.running):
            if runner is not None and runner.completed > 0:
                st["state"] = "complete"
                st["t0"] = now
                st["colors"] = None  # sampled lazily after drain applies
            elif runner is not None and runner.cancelled:
                st["state"] = "idle"
            else:
                st["state"] = "failed"
                st["t0"] = now
            st["runner"] = None

    # Ignition colors become available once the main-thread drain has
    # applied the predicted labels to the in-memory cloud; sample on
    # first need, retry a few frames if the drain hasn't landed yet.
    con = _constellation_for(entry) if entry is not None else None
    if (st["state"] == "complete" and st["colors"] is None
            and con is not None):
        st["colors"] = _sampled_label_colors(entry, con["sample_idx"],
                                             registry)

    # ---- background panel -------------------------------------------
    draw_styled_rect(dl, wx, wy, w, h, _SWEEP, rounding=s(3),
                     thickness=1.2)

    hovered = imgui.is_mouse_hovering_rect(wx, wy, wx + w, wy + h)
    clicked = hovered and imgui.is_mouse_clicked(0)

    ready = can_run and entry is not None and con is not None
    # A button should look like one: faint fill, brighter on hover.
    dl.add_rect_filled(wx, wy, wx + w, wy + h,
                       col32((_SWEEP[0], _SWEEP[1], _SWEEP[2], 0.12 if (hovered and ready) else 0.05)), s(3))

    # ---- constellation ----------------------------------------------
    # Constellation on the right, big label on the left — they used to
    # share the centre and overprint each other.
    cx = wx + w * 0.80
    cy = wy + h * 0.45
    scale = h * 0.36

    if con is not None:
        pts = con["pts"]
        state = st["state"]
        spin = {"idle": 0.30, "running": 1.05,
                "complete": 0.22, "failed": 0.30}[state]
        ang = now * spin
        ca, sa = math.cos(ang), math.sin(ang)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        rx = x * ca + z * sa          # screen-x
        rz = -x * sa + z * ca         # depth
        # Slight isometric tip so flat scans still read as 3D.
        ry = y * 0.85 - rz * 0.28

        if state == "failed":
            # Collapse toward the center then recover over 1.6 s.
            k = min(1.0, (now - st["t0"]) / 1.6)
            shrink = 1.0 - 0.75 * math.sin(k * math.pi)
            rx, ry = rx * shrink, ry * shrink

        px = cx + rx * scale
        py = cy - ry * scale
        depth = (rz + 1.2) / 2.4  # 0 far .. 1 near

        base_alpha = 0.30 if state == "idle" and not (hovered and ready) \
            else 0.45
        n = len(px)

        # Scan sweep (running): loops left→right every 1.7 s with a
        # soft trailing gradient behind the edge.
        if state == "running":
            phase = ((now - st["t0"]) % 1.7) / 1.7
            sweep_x = wx + w * (phase * 1.3 - 0.15)
            behind = np.clip((sweep_x - px) / (w * 0.35), 0.0, 1.0)
        else:
            behind = None

        ignite_k = None
        if state == "complete":
            # Radial ignition wave: points flash white then settle to
            # class color, staggered from constellation center outward.
            r_norm = np.sqrt(rx * rx + ry * ry)
            ignite_k = np.clip(((now - st["t0"]) * 1.4 - r_norm) / 0.35,
                               0.0, 1.0)

        colors = st.get("colors") if state == "complete" else None
        sweep_u32_cache = {}
        for i in range(n):
            d = float(depth[i])
            r = s(0.9 + 1.5 * d)
            a = base_alpha * (0.35 + 0.65 * d)
            if state == "idle" or state == "running":
                c = (0.62, 0.62, 0.62)
                if behind is not None:
                    b = float(behind[i])
                    if b > 0.0:
                        c = (0.62 + (_SWEEP[0] - 0.62) * b,
                             0.62 + (_SWEEP[1] - 0.62) * b,
                             0.62 + (_SWEEP[2] - 0.62) * b)
                        a = a + 0.40 * b * d
            elif state == "complete":
                k = float(ignite_k[i]) if ignite_k is not None else 1.0
                if colors is not None:
                    tc = colors[i]
                else:
                    tc = (0.62, 0.62, 0.62)
                # white flash at the wavefront (k near the middle)
                flash = math.sin(min(k, 1.0) * math.pi) * 0.85
                c = (tc[0] + (1.0 - tc[0]) * flash,
                     tc[1] + (1.0 - tc[1]) * flash,
                     tc[2] + (1.0 - tc[2]) * flash)
                breathe = 0.9 + 0.1 * math.sin((now - st["t0"]) * 2.1)
                a = (0.30 + 0.55 * d) * (0.4 + 0.6 * k) * breathe
            else:  # failed
                k = min(1.0, (now - st["t0"]) / 1.6)
                hot = math.sin(k * math.pi)
                c = (0.62 + (_EMBER[0] - 0.62) * hot,
                     0.62 + (_EMBER[1] - 0.62) * hot,
                     0.62 + (_EMBER[2] - 0.62) * hot)
            key = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255),
                   int(a * 255))
            u32 = sweep_u32_cache.get(key)
            if u32 is None:
                u32 = col32((c[0], c[1], c[2], a))
                sweep_u32_cache[key] = u32
            dl.add_rect_filled(px[i] - r, py[i] - r, px[i] + r, py[i] + r,
                               u32)

        # Orbiting comet + trail while running.
        if state == "running":
            orbit_rx = scale * 1.12
            orbit_ry = scale * 0.42
            for tail in range(7):
                ta = now * 2.6 - tail * 0.09
                ox = cx + math.cos(ta) * orbit_rx
                oy = cy - math.sin(ta) * orbit_ry
                fade = 1.0 - tail / 7.0
                dl.add_circle_filled(
                    ox, oy, s(2.4) * (0.5 + 0.5 * fade),
                    col32((_SWEEP[0], _SWEEP[1], _SWEEP[2],
                           0.85 * fade)), 8)

        # Expanding pulse ring on completion (first 0.9 s).
        if state == "complete":
            k = (now - st["t0"]) / 0.9
            if k < 1.0:
                ring_r = scale * (0.2 + 1.15 * k)
                dl.add_circle(cx, cy, ring_r,
                              col32((_SWEEP[0], _SWEEP[1], _SWEEP[2],
                                     0.7 * (1.0 - k))), 48, s(2.0))
    else:
        msg = "no cloud loaded" if entry is None else "cloud loading..."
        tw = imgui.calc_text_size(msg)[0]
        dl.add_text(cx - tw * 0.5, cy - th * 0.5, col32(OP1_DIM), msg)

    # ---- baseline progress bar + status strip -----------------------
    bar_y = wy + h - th * 1.55
    state = st["state"]
    if state == "running":
        elapsed = now - st["t0"]
        frac = 0.92 * (1.0 - math.exp(-elapsed / 8.0))
        dl.add_rect_filled(wx + s(6), bar_y, wx + s(6) + (w - s(12)) * frac,
                           bar_y + s(2), col32(_SWEEP))
        status = f"INFERRING  {elapsed:4.1f}s   (click to cancel)"
        status_col = _SWEEP
    elif state == "complete":
        dl.add_rect_filled(wx + s(6), bar_y, wx + w - s(6), bar_y + s(2),
                           col32(OP1_GREEN))
        npts = con["point_count"] if con else 0
        status = f"COMPLETE - {npts:,} pts labeled  (Ctrl+Z undoes)"
        status_col = OP1_GREEN
    elif state == "failed":
        status = "FAILED - see LOG on the TRAIN tab"
        status_col = OP1_RED
    elif not can_run:
        status = "no trained model - TRAIN one first"
        status_col = OP1_DIM
    elif entry is None:
        status = "select a cloud"
        status_col = OP1_DIM
    else:
        status = "INFER"
        status_col = _SWEEP if (hovered and ready) else OP1_GRAY

    # ---- the big label: say what the button does -----------------------
    from src.gui.theme import OP1_WHITE
    if state == "running":
        big, big_col = "INFERRING", _SWEEP
    elif state == "complete":
        big, big_col = "LABELED", OP1_GREEN
    elif state == "failed":
        big, big_col = "FAILED", OP1_RED
    elif not can_run:
        big, big_col = "NO MODEL", OP1_DIM
    elif entry is None:
        big, big_col = "SELECT A CLOUD", OP1_DIM
    else:
        big, big_col = "RUN INFERENCE", (_SWEEP if hovered else OP1_WHITE)
    _gui = getattr(app, "gui", None)
    _fd = getattr(_gui, "font_display", None)
    if _fd is not None:
        imgui.push_font(_fd)
        imgui.set_window_font_scale(0.5)
    else:
        imgui.set_window_font_scale(1.7)
    _bw, _bh = imgui.calc_text_size(big)
    dl.add_text(wx + s(14), cy - _bh * 0.5 - th * 0.15, col32(big_col), big)
    imgui.set_window_font_scale(1.0)
    if _fd is not None:
        imgui.pop_font()

    dl.add_text(wx + s(8), wy + h - th * 1.25, col32(status_col), status)
    if model_label:
        mw = imgui.calc_text_size(model_label)[0]
        dl.add_text(wx + w - mw - s(8), wy + h - th * 1.25,
                    col32(OP1_DIM), model_label)

    imgui.dummy(w, h + s(4))

    # ---- interaction -------------------------------------------------
    if not clicked:
        return
    if state == "running":
        r = st.get("runner") or getattr(app, "contact_sheets_infer_runner",
                                        None)
        if r is not None and (now - st["t0"]) > 0.5:
            r.cancel()
        return
    if not ready or state == "complete" and (now - st["t0"]) < 1.0:
        return
    if start_infer(app, app.selected_index):
        st["state"] = "running"
        st["t0"] = now
        st["file_key"] = fk
        st["colors"] = None
        st["runner"] = getattr(app, "contact_sheets_infer_runner", None)
