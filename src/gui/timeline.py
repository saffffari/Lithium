"""Timeline scrubber widget for 4D point cloud sequences.

Draws a horizontal bar at the bottom of the viewport with frame markers,
a current-frame indicator, and click-to-seek interaction.
"""

import imgui
import numpy as np

from src.gui.theme import *


def draw_timeline(app) -> bool:
    """Draw the 4D sequence timeline. Returns True if the user seeked."""
    if app.sequence is None or app.sequence.frame_count < 2:
        return False

    seq = app.sequence
    n = seq.frame_count

    # Place at bottom of the viewport (right of both sidebars when in Light Table)
    sw = app._left_chrome_width()
    margin = 12
    tl_x = sw + margin
    tl_w = app.width - sw - margin * 2
    tl_h = 48
    tl_y = app.height - tl_h - margin

    dl = imgui.get_foreground_draw_list()

    # Background
    dl.add_rect_filled(tl_x, tl_y, tl_x + tl_w, tl_y + tl_h, col32((0.04, 0.04, 0.04, 0.95)))
    dl.add_rect(tl_x, tl_y, tl_x + tl_w, tl_y + tl_h, col32(FAINT), 0.0, 0, 1.0)

    # Frame markers
    track_y = tl_y + tl_h - 18
    track_h = 14
    for i in range(n):
        fx = tl_x + (i / max(n - 1, 1)) * tl_w
        # Color: grey=unlabeled, green=has labels
        color = GREEN if seq.has_labels(i) else FAINT
        dl.add_rect_filled(fx - 1, track_y, fx + 1, track_y + track_h, col32(color))

    # Current frame indicator
    cx = tl_x + (seq.current_index / max(n - 1, 1)) * tl_w
    dl.add_rect_filled(cx - 3, tl_y + 4, cx + 3, tl_y + tl_h - 4, col32(ORANGE))

    # Frame text
    text = f"Frame {seq.current_index + 1}/{n}  |  {seq.frame_name()}"
    dl.add_text(tl_x + 8, tl_y + 4, col32(WHITE), text[:80])

    # Handle click-to-seek
    mx, my = imgui.get_mouse_pos()
    if (tl_x <= mx <= tl_x + tl_w and tl_y <= my <= tl_y + tl_h
            and imgui.is_mouse_clicked(0) and not imgui.get_io().want_capture_mouse):
        rel = (mx - tl_x) / max(tl_w, 1)
        new_index = int(round(rel * (n - 1)))
        if new_index != seq.current_index:
            return _do_seek(app, new_index)

    return False


def _do_seek(app, index: int) -> bool:
    """Seek the sequence to a new frame and update the GPU cloud."""
    if app.sequence is None:
        return False
    if not (0 <= app.selected_index < len(app.entries)):
        return False

    entry = app.entries[app.selected_index]

    # LS-1: persist the outgoing frame's labels through the normal
    # per-entry path before it can be LRU-evicted. entry.file_key is
    # kept pointing at the *current* frame (set below on every seek),
    # so the length guard in _persist_cloud_labels lines up.
    old_gpu = entry.full_gpu or entry.preview_gpu
    if old_gpu is not None and old_gpu.cloud_data is not None:
        try:
            app._persist_cloud_labels(entry, old_gpu.cloud_data)
        except Exception as e:
            print(f"[timeline] persist before seek failed: {e}")

    try:
        new_cloud = app.sequence.seek(index)
    except Exception as e:
        print(f"Seek failed: {e}")
        return False
    # ``seek`` clamps out-of-range indices (arrow-key stepping past the
    # ends passes current±1 unclamped) — use the clamped value for the
    # name/key lookups below so they can't IndexError.
    index = app.sequence.current_index

    # Swap the GPU cloud for the selected entry
    from src.rendering.point_cloud_renderer import upload_cloud
    # Release the old GPU
    if entry.full_gpu is not None:
        entry.full_gpu.release()
        entry.full_gpu = None
    if entry.preview_gpu is not None:
        entry.preview_gpu.release()
        entry.preview_gpu = None

    gpu = upload_cloud(new_cloud, app.ctx, app.program)
    entry.full_gpu = gpu
    entry.preview_gpu = gpu
    entry.point_count = new_cloud.point_count
    entry.bounds_min = new_cloud.bounds_min.copy()
    entry.bounds_max = new_cloud.bounds_max.copy()
    entry.name = app.sequence.frame_name(index)
    # LS-1: keep the session entry's file_key pinned to the current
    # frame so every per-stroke persist lands in the right catalog slot.
    entry.file_key = app.sequence.frame_key(index)
    app.selection_buffer.resize(new_cloud.point_count)
    app.selection_buffer.clear()
    # The new cloud is a different Python object — the label-count cache
    # keyed by id(cloud) will auto-miss on the new cloud, but purge old
    # entries to avoid unbounded growth on long scrubs.
    if hasattr(app, 'label_count_cache'):
        app.label_count_cache.clear()
    app._overlays_dirty = True
    return True
