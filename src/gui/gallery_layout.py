"""Gallery grid layout computation.

The Contact Sheets grid lives in the viewport area to the right of the
sidebar. Each cell is a fixed square of ``cell_size`` pixels (controlled
by Ctrl+wheel) padded by CELL_MARGIN. The number of columns is whatever
fits the gallery width; rows are however many it takes to hold every
entry. When the total content height exceeds the visible area, the user
scrolls vertically — cells *never* shrink to fit more entries on screen.
"""

import math
import numpy as np

from src.utils.math_utils import perspective, look_at


# Pixels of padding between and around cells (in screen space)
CELL_MARGIN = 12

# Hard limits on the user-controlled cell size, in framebuffer pixels.
# 80 is roughly the smallest size where the filename label is still
# legible; 1600 is "one cell fills a 5K viewport" — at that point the
# user is effectively previewing a single cloud full-screen and should
# probably enter Light Table, but we don't stop them.
MIN_CELL_SIZE = 80
MAX_CELL_SIZE = 1600


def compute_grid(cloud_count: int, area_w: int, area_h: int, cell_size: int):
    """Compute grid dimensions for a fixed-cell-size scrollable gallery.

    Returns ``(cols, rows, cell_w, cell_h, content_h)``:
      cols / rows  → grid shape (rows is a *content* row count, not
                     visible row count — many rows may be off-screen)
      cell_w/h     → individual cell pixel size (always equal to
                     cell_size; kept as separate fields for the legacy
                     callers that already destructure four values)
      content_h    → total scrollable content height in pixels,
                     including the trailing margin row
    """
    cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, int(cell_size)))
    if cloud_count <= 0:
        return 1, 0, cell_size, cell_size, 0

    # Use the user-selected cell_size as a *minimum* to decide column
    # count, then stretch each cell to consume any remaining horizontal
    # space so the gallery fills the full viewport width with no dead
    # band on the right edge.
    step = cell_size + CELL_MARGIN
    cols = max(1, (max(area_w, 1) - CELL_MARGIN) // step)
    # Distribute leftover pixels back into cell width.
    fit_w = max(MIN_CELL_SIZE,
                (max(area_w, 1) - CELL_MARGIN * (cols + 1)) // cols)
    fit_w = min(MAX_CELL_SIZE, fit_w)
    rows = math.ceil(cloud_count / cols)
    content_h = rows * (fit_w + CELL_MARGIN) + CELL_MARGIN
    return int(cols), int(rows), int(fit_w), int(fit_w), int(content_h)


def cell_rect(index: int, cols: int,
              cell_w: int, cell_h: int,
              area_x: int, area_y: int,
              scroll_y: int = 0) -> tuple[int, int, int, int]:
    """Return the screen-space rect for a cell, post-scroll.

    ``area_x`` / ``area_y`` are the top-left of the gallery viewport.
    ``scroll_y`` is the current scroll offset (positive = scrolled
    down). The returned rect may sit fully outside the gallery area —
    callers are expected to skip cells they don't need to draw.
    """
    col = index % cols
    row = index // cols
    x = area_x + CELL_MARGIN + col * (cell_w + CELL_MARGIN)
    y = area_y + CELL_MARGIN + row * (cell_h + CELL_MARGIN) - scroll_y
    return int(x), int(y), int(cell_w), int(cell_h)


def visible_row_range(scroll_y: int, area_h: int,
                      cell_h: int, total_rows: int) -> tuple[int, int]:
    """Return ``(first_row, last_row_exclusive)`` overlapping the viewport.

    Used to skip iteration / rendering of off-screen cells. Both
    indices are clamped to ``[0, total_rows]``. The result is the
    half-open range of rows whose pixel rect intersects the viewport.
    """
    if total_rows <= 0:
        return 0, 0
    step = cell_h + CELL_MARGIN
    if step <= 0:
        return 0, total_rows
    first = max(0, min(total_rows, (scroll_y - CELL_MARGIN) // step))
    last = max(first, min(total_rows, (scroll_y + area_h + step - 1) // step + 1))
    return int(first), int(last)


def cell_viewport_local(index: int, cols: int,
                        cell_w: int, cell_h: int,
                        area_h: int,
                        scroll_y: int = 0):
    """OpenGL viewport (local to a render target) for a cell.

    ``area_h`` is the *visible* area height — same as the cache RT's
    height. The returned viewport is in bottom-left origin coordinates
    relative to the cache RT, NOT the window. Cells whose visible
    portion lies outside ``[0, area_h]`` should be skipped by the
    caller; this function still returns valid (possibly clipped) ints.
    """
    col = index % cols
    row = index // cols
    local_x = CELL_MARGIN + col * (cell_w + CELL_MARGIN)
    local_y_top = CELL_MARGIN + row * (cell_h + CELL_MARGIN) - scroll_y
    gl_x = local_x
    gl_y = area_h - local_y_top - cell_h
    return int(gl_x), int(gl_y), int(cell_w), int(cell_h)


def fit_view_matrix(bounds_min: np.ndarray, bounds_max: np.ndarray,
                    aspect: float, padding: float = 0.8,
                    azimuth: float = math.pi / 4,
                    elevation: float = math.pi / 6,
                    zoom: float = 1.0):
    """Compute view + projection matrices to fit a cloud in a gallery cell.

    azimuth / elevation / zoom let every cell share a common orbit so
    the user can inspect all previews from the same angle before loading.
    """
    center = (bounds_min + bounds_max) / 2.0
    extent = bounds_max - bounds_min
    max_dim = max(np.linalg.norm(extent), 1e-6)

    distance = (max_dim * 1.2 / padding) / max(zoom, 0.05)

    eye = np.array([
        center[0] + distance * math.cos(elevation) * math.sin(azimuth),
        center[1] + distance * math.sin(elevation),
        center[2] + distance * math.cos(elevation) * math.cos(azimuth),
    ], dtype=np.float32)

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    view = look_at(eye, center, up)

    fov = math.radians(45.0)
    near = distance * 0.01
    far = distance * 10.0
    proj = perspective(fov, aspect, near, far)

    return view, proj


def cell_index_from_mouse(mx: float, my: float, cols: int,
                          cell_w: int, cell_h: int, cloud_count: int,
                          area_x: int = 0, area_y: int = 0,
                          scroll_y: int = 0) -> int:
    """Convert mouse position (screen coords, top-left origin) to cell index.

    Accounts for scroll offset. Returns -1 if the cursor is outside the
    gallery area or in an inter-cell gap.
    """
    local_x = mx - area_x - CELL_MARGIN
    local_y = my - area_y - CELL_MARGIN + scroll_y
    cell_step_x = cell_w + CELL_MARGIN
    cell_step_y = cell_h + CELL_MARGIN
    if local_x < 0 or local_y < 0:
        return -1
    if cell_step_x <= 0 or cell_step_y <= 0:
        return -1
    col = int(local_x) // cell_step_x
    row = int(local_y) // cell_step_y
    if col < 0 or col >= cols:
        return -1
    # Reject the gap between cells (otherwise clicks in the margin
    # would land on whichever cell is to their left/above).
    if int(local_x) % cell_step_x >= cell_w:
        return -1
    if int(local_y) % cell_step_y >= cell_h:
        return -1
    idx = row * cols + col
    if idx < 0 or idx >= cloud_count:
        return -1
    return idx
