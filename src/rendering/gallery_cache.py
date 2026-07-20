"""Per-entry texture cache for the Contact Sheets gallery.

Each gallery entry owns a small RGBA16F square texture (sized to the
current ``cell_size``) into which we render its preview point cloud
*once* and re-use forever — until the entry's orbit camera changes,
the global shader knobs change, or the cell size changes. With this
cache:

- Scrolling is free: visible cells just blit their cached texture
  into the gallery render target. No point-cloud draws.
- Off-screen entries are never rendered, never re-rendered, and the
  cache is bounded by an LRU so a 5000-entry library doesn't OOM the
  GPU. Off-screen-but-recent entries stay warm; truly cold entries
  get evicted and lazily re-rendered when scrolled back into view.
- Adding more entries does not invalidate or shrink anything that's
  already cached. Past contacts retain their texture forever (subject
  to the LRU cap).
- The cache RT (the gallery render target the visible cells are blitted
  into) only needs to be the size of the *visible* gallery viewport,
  not the full content height. This drops VRAM usage for huge libraries
  and removes the "resize wipes everything" pathology.

Invalidation strategy
---------------------
- Cell size changes  → all per-entry textures dropped (they're the
  wrong size). Cells lazily re-render at the new size as they scroll
  into view.
- Global shader knob changes (point size, sharpness, falloff,
  brightness, contrast, saturation, RGB gains, label blend) → bump
  ``global_version``; per-entry caches whose stored ``global_version``
  doesn't match are considered dirty.
- Per-entry change (orbit az/el/zoom, new preview GPU, bounds update)
  → bumps that entry's per-cell version tuple.
- Entry removed from library → its cache slot is dropped on next prune.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import moderngl

from src.rendering.render_target import RenderTarget


# Maximum number of cells re-rendered per frame. Bounds frame time
# while a fresh project or post-resize state populates. Bumped from
# 32 → 24 because the per-entry pipeline does a full RT bind per cell
# (more state churn) but we're rendering far fewer cells per frame
# overall thanks to off-screen culling.
MAX_DIRTY_PER_FRAME = 24

# Hard cap on the number of cached entry textures we keep around.
# 768 × ~220px square × RGBA16F ≈ 290 MB worst case (220 * 220 * 8 *
# 768 bytes). We aim for a budget that comfortably accommodates a
# scrolled-through gallery without thrashing on a typical workstation
# GPU. Eviction is LRU.
MAX_CACHE_ENTRIES = 768


def _entry_key(entry: Any) -> Any:
    """Stable identity key for a CloudEntry. file_key survives reload,
    id() is session-stable for everything else."""
    fk = getattr(entry, 'file_key', None)
    if fk:
        return fk
    return id(entry)


class _CachedCell:
    """One entry's per-cell render target.

    Owns its own framebuffer + color texture (depth is shared from the
    cache because we only ever render one cell at a time and never need
    to keep depth between renders). Cell size is fixed at construction
    — when ``cell_size`` changes, the whole _CachedCell is released and
    a new one is allocated lazily.
    """

    __slots__ = ("size", "color", "fbo", "version", "global_version")

    def __init__(self, ctx: moderngl.Context, size: int,
                 shared_depth: moderngl.Texture):
        self.size = int(size)
        self.color = ctx.texture((self.size, self.size), 4, dtype='f2')
        self.color.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.color.repeat_x = False
        self.color.repeat_y = False
        self.fbo = ctx.framebuffer(
            color_attachments=[self.color],
            depth_attachment=shared_depth,
        )
        # Per-entry version tuple (orbit, transform, bounds, gpu identity)
        self.version: tuple = ()
        # Snapshot of the global params at the moment of last render
        self.global_version: int = -1

    def release(self) -> None:
        try:
            if self.fbo is not None:
                self.fbo.release()
        except Exception:
            pass
        try:
            if self.color is not None:
                self.color.release()
        except Exception:
            pass
        self.fbo = None
        self.color = None


class GalleryCache:
    """Per-entry texture cache + visible-area scratch RT.

    Two distinct GL resources live here:

    1. ``self._cells``: ``OrderedDict[entry_key, _CachedCell]`` — the
       persistent per-entry preview textures, one per gallery cloud.
       LRU-evicted when over ``MAX_CACHE_ENTRIES``.
    2. ``self.rt``: a single RGBA16F render target sized to the
       *visible* gallery viewport. Each frame we blit visible entries'
       cached textures into this RT, then post-tonemap composites it
       on top of ``scene_rt``. This RT is allocated lazily and never
       preserves contents across frames — it's pure scratch space.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        # Visible-area scratch render target (re-used every frame)
        self.rt: RenderTarget | None = None
        self.area_w = 0
        self.area_h = 0
        # Per-entry caches: insertion order = LRU order, oldest first
        self._cells: "OrderedDict[Any, _CachedCell]" = OrderedDict()
        # Global shader-param identity. Bumped whenever a uniform shared
        # by every cell changes; per-cell snapshots compared against this.
        self._global_version: int = 0
        self._cell_size: int = 0
        # Shared depth buffer for the cell render target. We only render
        # one cell at a time and never read depth back, so a single
        # texture sized to the largest expected cell is reused for all
        # cells regardless of their stored size (depth is just scratch).
        self._shared_depth: moderngl.Texture | None = None
        self._shared_depth_size: int = 0
        # Background color used for cell clears; must match App.bg_color.
        self._bg: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Lifecycle

    def ensure_target(self, area_w: int, area_h: int,
                      bg: tuple[float, float, float]) -> None:
        """Allocate / resize the visible-area scratch RT.

        The RT does NOT cache contents — it's cleared and refilled
        every frame from per-entry caches. So resizing it is cheap and
        does not invalidate the per-entry textures (which is the whole
        point of the rewrite).
        """
        area_w = max(1, int(area_w))
        area_h = max(1, int(area_h))
        if self.rt is None:
            self.rt = RenderTarget(
                self.ctx, area_w, area_h,
                components=4, color_dtype='f2', with_depth=False,
            )
        elif (self.rt.width, self.rt.height) != (area_w, area_h):
            self.rt.resize(area_w, area_h)
        self.area_w = area_w
        self.area_h = area_h
        self._bg = bg

    def set_cell_size(self, cell_size: int) -> None:
        """Switch the cached cell size, dropping any cells that don't match.

        Called whenever the user Ctrl+wheels the gallery zoom. Cells
        rendered at the old size are released; new ones are allocated
        lazily as their entries scroll into view.
        """
        cell_size = int(cell_size)
        if cell_size == self._cell_size:
            return
        self._cell_size = cell_size
        for cell in self._cells.values():
            cell.release()
        self._cells.clear()
        # Drop the shared depth too — it'll be recreated at the new size
        # on next render.
        if self._shared_depth is not None:
            try:
                self._shared_depth.release()
            except Exception:
                pass
            self._shared_depth = None
            self._shared_depth_size = 0

    def release(self) -> None:
        if self.rt is not None:
            self.rt.release()
            self.rt = None
        for cell in self._cells.values():
            cell.release()
        self._cells.clear()
        if self._shared_depth is not None:
            try:
                self._shared_depth.release()
            except Exception:
                pass
            self._shared_depth = None

    # ------------------------------------------------------------------
    # Invalidation

    def bump_global(self, global_key: tuple) -> None:
        """Mark every cached cell as needing re-render against new globals.

        We don't actually clear anything — we just bump a version
        counter. Cells whose stored ``global_version`` doesn't match are
        considered dirty by ``is_dirty`` and will be re-rendered the
        next time they appear on screen. This is much cheaper than the
        old "wipe the entire cache" path because cells the user never
        scrolls to never get re-rendered.
        """
        if global_key != getattr(self, '_global_key', None):
            self._global_key = global_key
            self._global_version += 1

    def invalidate_entry(self, entry: Any) -> None:
        cell = self._cells.pop(_entry_key(entry), None)
        if cell is not None:
            cell.release()

    def invalidate_all(self) -> None:
        for cell in self._cells.values():
            cell.release()
        self._cells.clear()

    # ------------------------------------------------------------------
    # Per-entry render slot management

    def _ensure_shared_depth(self, size: int) -> moderngl.Texture:
        """Create / grow the shared scratch depth texture as needed."""
        if self._shared_depth is None or self._shared_depth_size < size:
            if self._shared_depth is not None:
                try:
                    self._shared_depth.release()
                except Exception:
                    pass
            self._shared_depth = self.ctx.depth_texture((size, size))
            self._shared_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._shared_depth.repeat_x = False
            self._shared_depth.repeat_y = False
            try:
                self._shared_depth.compare_func = ''
            except Exception:
                pass
            self._shared_depth_size = size
        return self._shared_depth

    def get_or_alloc(self, entry: Any) -> _CachedCell:
        """Return the cache slot for an entry, allocating if missing.

        Touches LRU order so the entry is now "most recently used".
        Triggers eviction if we've exceeded ``MAX_CACHE_ENTRIES``.
        """
        key = _entry_key(entry)
        cell = self._cells.get(key)
        if cell is not None:
            self._cells.move_to_end(key)
            return cell

        depth = self._ensure_shared_depth(self._cell_size)
        cell = _CachedCell(self.ctx, self._cell_size, depth)
        self._cells[key] = cell
        self._evict_if_needed()
        return cell

    def _evict_if_needed(self) -> None:
        while len(self._cells) > MAX_CACHE_ENTRIES:
            _, victim = self._cells.popitem(last=False)
            victim.release()

    def is_dirty(self, entry: Any, version: tuple) -> bool:
        cell = self._cells.get(_entry_key(entry))
        if cell is None:
            return True
        if cell.global_version != self._global_version:
            return True
        return cell.version != version

    def mark_rendered(self, entry: Any, version: tuple) -> None:
        cell = self._cells.get(_entry_key(entry))
        if cell is None:
            return
        cell.version = version
        cell.global_version = self._global_version

    def cached_texture(self, entry: Any) -> moderngl.Texture | None:
        cell = self._cells.get(_entry_key(entry))
        return cell.color if cell is not None else None

    def cached_fbo(self, entry: Any) -> moderngl.Framebuffer | None:
        cell = self._cells.get(_entry_key(entry))
        return cell.fbo if cell is not None else None

    def prune(self, live_keys: set) -> None:
        """Drop cache slots for entries no longer in the gallery."""
        if not self._cells:
            return
        stale = [k for k in self._cells if k not in live_keys]
        for k in stale:
            cell = self._cells.pop(k, None)
            if cell is not None:
                cell.release()

    # ------------------------------------------------------------------
    # Visible RT helpers

    def begin_frame_visible(self) -> None:
        """Bind the visible-area scratch RT and clear it to bg."""
        if self.rt is None:
            return
        self.rt.use()
        self.rt.clear(self._bg[0], self._bg[1], self._bg[2], 1.0)
