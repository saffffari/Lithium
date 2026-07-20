"""3Photon — Point Cloud Visualizer and Renderer.

Entry point: GLFW window, ModernGL context, ImGui overlay, main render loop.
Supports gallery view (all clouds) and individual view (single cloud, full resolution).
"""

import sys
import os
import time
import math
import glfw
import moderngl
import numpy as np

# Force UTF-8 on stdout/stderr so Unicode characters in print statements
# (e.g. the navigation arrow ``→`` in _navigate_cloud) don't crash the
# GLFW key callback when stdout is redirected to a file or to a console
# whose codepage is cp1252. Without this, a single arrow-key press in a
# subprocess-launched session raises UnicodeEncodeError inside the
# callback, which poisons ImGui frame state and locks the app in an
# assertion loop. errors='replace' makes any future Unicode print fail
# loudly-but-locally instead of taking the whole app down.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# Ensure src is importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.camera import Camera
from src.core.input_state import ClickDetector
from src.core.undo import UndoStack, apply_label
from src.core.selection import SelectionBuffer
from src.core.tools.pick_tool import (
    pick_point, pick_nearest_point, pick_visible_disk,
    PICK_RADIUS_PX, BRUSH_SNAP_RADIUS_PX, CURSOR_SNAP_RADIUS_PX,
)
from src.core.tools.box_tool import box_select
from src.core.tools.lasso_tool import lasso_select
from src.core.tools.brush_tool import brush_select, screen_to_world
from src.core.tools.curve_tool import curve_select
from src.core.tools.measure_tool import MeasureState
from src.core.measure_registry import MeasureRegistry
from src.data.loader import load_point_cloud, scan_directory, scan_directory_recursive
from src.data.library_catalog import LibraryCatalog, SMART_ALL, SMART_RECENT
from src.data.labels import LabelRegistry, locked_mask, LabelCountCache
from src.data.resampler import voxel_downsample
from src.data.sequence import PointCloudSequence, detect_sequence
from src.data.cloud_store import (
    save_cloud_data, load_cloud_data, save_cloud_labels, load_cloud_labels,
    save_preview_labels, load_preview_labels,
    is_source_stale, check_catalog_integrity, format_integrity_summary,
)
from src.data.catalog_lock import acquire_lock, release_lock
from src.utils.file_hash import compute_file_key
from src.core.stroke_recorder import LabelStrokeRecorder
from src.rendering.shaders import create_point_cloud_program, reload_all as reload_all_shaders
from src.rendering.point_cloud_renderer import (
    upload_cloud, draw_cloud, re_upload_labels, update_selection, GPUCloud,
    set_shared_point_uniforms,
)
from src.rendering.overlay_renderer import OverlayRenderer
from src.rendering.gizmo_renderer import GizmoRenderer, unproject_mouse, AXIS_NONE
from src.rendering.cursor3d import Cursor3D
from src.rendering.label_texture import create_label_color_texture, update_label_color_texture
from src.rendering.falloff_texture import create_falloff_texture, update_falloff_texture
from src.rendering.render_target import RenderTarget
from src.rendering.postprocess import FullscreenPass
from src.rendering.gallery_cache import GalleryCache, MAX_DIRTY_PER_FRAME

# Cap on how many previously-cached previews we load from disk and upload to
# the GPU per frame. A view switch into a 1000-entry project used to do all
# N loads + uploads synchronously on the main thread — now they're spread
# across frames, prioritised to whichever cells are currently visible. 8 ×
# ~3-5 ms/upload comfortably fits a 60 fps frame, and visible cells (≤ 24
# in a typical layout) populate within 3 frames of scrolling into view.
MAX_LAZY_PREVIEW_UPLOADS_PER_FRAME = 8
from src.core.envelope import Envelope
from src.gui.imgui_layer import ImGuiLayer
from src.gui.panels import draw_settings_panel, draw_main_menu_bar
from src.gui.measure_panel import draw_measure_panel
from src.gui.timeline import draw_timeline
from src.gui.shortcuts_panel import draw_shortcuts_overlay
from src.gui.radial_menu import draw_radial_menu
from src.gui.label_panel import draw_color_picker_popup
from src.gui import gallery_layout


# View modes live in src.core.modes — re-exported here so existing
# `from src.main import MODE_*` imports keep working while callers
# migrate. New code should import from src.core.modes directly.
from src.core.modes import (  # noqa: E402,F401
    MODE_CONTACT_SHEETS,
    MODE_LIGHT_TABLE,
    MODE_AUTOMATION,
    MODE_GALLERY,
    MODE_INDIVIDUAL,
)

# Hold time (seconds) before an in-place RMB press opens the radial
# preset menu. Shorter than a deliberate orbit hesitation, longer than a
# regular press — users who start moving before this elapses just orbit.
_RADIAL_MENU_DELAY = 0.42


def _fmt_distance(d: float, unit: str = "u") -> str:
    """Format a raw world-unit distance with its label.

    No unit conversion is performed — `d` is in whatever units the point
    cloud positions use, and `unit` names that unit (e.g. 'mm', 'm', 'u').
    """
    if d >= 10_000:
        return f"{d:.1f} {unit}"
    if d >= 100:
        return f"{d:.2f} {unit}"
    if d >= 1:
        return f"{d:.3f} {unit}"
    return f"{d:.4f} {unit}"


def _cloud_unit_label(entry) -> str:
    """Unit label for the given CloudEntry's positions."""
    unit = getattr(entry, 'source_unit', 'raw')
    if unit and unit != 'raw':
        return unit
    # Fallback heuristic for clouds loaded before source_unit was persisted.
    path = getattr(entry, 'file_path', '')
    if path.lower().endswith(('.las', '.laz')):
        return "m"
    return "u"


class CloudEntry:
    """Represents a loaded cloud with both preview and full-res GPU data."""
    def __init__(self, file_path: str, file_key: str | None = None):
        import math as _math
        self.file_path = file_path
        self.name = os.path.basename(file_path)
        # Link back to the persistent library entry this cloud came from
        # (None for single in-memory loads that predate the library).
        self.file_key: str | None = file_key
        self.preview_gpu: GPUCloud | None = None
        self.full_gpu: GPUCloud | None = None
        self.bounds_min: np.ndarray | None = None
        self.bounds_max: np.ndarray | None = None
        self.point_count: int = 0
        self.model_transform = np.eye(4, dtype=np.float32)
        # Coordinate metadata — preserved from the source loader so
        # multi-cloud alignment and measurement units work correctly.
        self.local_origin = np.zeros(3, dtype=np.float64)  # offset subtracted at load (e.g. LAS centroid)
        self.source_unit: str = 'raw'                       # 'mm', 'm', 'um', 'vx', 'raw'
        # Per-entry gallery preview camera state so the Contact Sheets
        # orbit only rotates the currently selected cell.
        self.orbit_az: float = _math.pi / 4
        self.orbit_el: float = _math.pi / 6
        self.orbit_zoom: float = 1.0
        # Triangulated mesh — Poisson surface reconstruction of the
        # point cloud, used in HOLOGRAM and as an optional LIGHT TABLE
        # display mode. Built lazily off-thread by
        # ``LibraryCatalog.queue_mesh_build`` and uploaded to the GPU
        # when the display mode first asks for it. ``mesh_dirty=True``
        # forces a rebuild on the next render — set by the label edit
        # path when the on-disk mesh's vertex_labels are now stale.
        from src.rendering.mesh_renderer import GPUMesh as _GPUMesh
        self.mesh_gpu: _GPUMesh | None = None
        self.mesh_dirty: bool = False
        # Geometric primitives derived from this cloud's labeled subsets
        # (planes through endplate points, centroids of body labels,
        # etc). Re-derived after any label edit; persisted alongside the
        # cloud in primitives/<file_key>.json so HOLOGRAM measurements
        # survive restarts. Empty list = "not derived yet" or "no
        # labels of the configured primitive classes."

    def release(self):
        if self.preview_gpu:
            self.preview_gpu.release()
        if self.full_gpu:
            self.full_gpu.release()
        if self.mesh_gpu:
            self.mesh_gpu.release()


class App:
    """Main application state."""

    def __init__(self):
        self.camera = Camera()
        self.gallery_camera = Camera()  # separate camera for gallery
        self.entries: list[CloudEntry] = []
        # Scene coordinate system
        self.scene_unit: str = 'mm'  # display/measurement unit: 'mm', 'm', 'um', 'vx', 'raw'
        self.scene_origin = np.zeros(3, dtype=np.float64)  # global reference origin
        self.program = None
        self.ctx: moderngl.Context = None
        self.window = None
        self.gui: ImGuiLayer = None
        self.overlays: OverlayRenderer = None
        self.gizmo: GizmoRenderer = None
        self.cursor3d: Cursor3D = None
        # Persistent library catalog — tracks every cloud ever imported.
        # Initialised lazily so tests / CLI paths that never touch the GUI
        # don't spin up the preview worker thread.
        self.catalog: LibraryCatalog | None = None
        # Active "view" in the Contact Sheet tab:
        #   None              → current session only (entries list directly)
        #   ("folder", path)  → all library entries under a folder subtree
        #   ("project", id) → entries in a named / smart project
        self.active_view: tuple[str, str] | None = None
        self.width = 1280
        self.height = 720
        # Viewport clear color — locked to 95% black so the 3D canvas
        # sits flush with the sidebar panel wells. scene_rt is HDR f16;
        # ACES + gamma 2.2 brighten low values, so we feed LINEAR
        # radiance that round-trips to the desired sRGB output:
        #   linear 0.003 → ~0.05 sRGB (window + viewport + tile fill)
        self.bg_color = (0.003, 0.003, 0.003)
        self.cell_bg_color = (0.003, 0.003, 0.003)
        self.point_size = 3.0
        self.point_sharpness = 0.0
        self.depth_falloff = 0.0  # 0 = no brightness falloff, >0 darkens distant points
        # Display mode for the LIGHT TABLE / HOLOGRAM viewport: one of
        # 'points', 'mesh', 'both'. HOLOGRAM mode auto-switches to
        # 'mesh' during render; LIGHT TABLE respects this setting.
        # Meshes are Poisson-reconstructed lazily — see
        # LibraryCatalog.queue_mesh_build and _ensure_mesh_gpu.
        self.display_mode: str = "points"

        # HDR offscreen → tonemap compositing pipeline. scene_rt and
        # pp_tonemap are created in init_gl once the ModernGL context
        # exists. The only per-frame knob is exposure, which feeds the
        # ACES tonemap in post_tonemap.frag.
        self.exposure = 1.0
        self.scene_rt: RenderTarget | None = None
        self.pp_tonemap: FullscreenPass | None = None
        # Depth of field — auto-focus to camera.distance (orbit pivot)
        # so whatever you're looking at stays sharp. Strength = 0
        # disables it entirely in the point vert shader.
        self.dof_enabled = False
        self.dof_strength = 4.0
        self.brightness = 0.0
        self.contrast = 1.0
        self.saturation = 1.0
        self.r_gain = 1.0
        self.g_gain = 1.0
        self.b_gain = 1.0
        self.frame_count = 0
        self._fps_frame_count = 0
        self._total_point_count = 0
        # Per-frame sidebar-width memo (see _left_chrome_width).
        self._left_chrome_frame: int = -1
        self._left_chrome_cached: int = 0
        # Sidebar's ACTUAL right edge in framebuffer pixels, measured from
        # the imgui window each frame (panels.draw_sidebar). imgui lays out
        # in logical/display units while the GL gallery viewport is in
        # framebuffer pixels; at fractional DPI those differ, so the grid
        # must start here, not at the logical sidebar_width(). 0 = unset.
        self._sidebar_right_fb: int = 0
        self.fps = 0.0
        self.last_fps_time = 0.0
        self.gui_visible = True
        self.show_bbox = False
        self.show_grid = False
        self._overlays_dirty = True

        # View mode
        self.mode = MODE_INDIVIDUAL
        self.selected_index = 0
        self.hover_index = -1

        # Gizmo drag state
        self._gizmo_drag_origin = np.zeros(3, dtype=np.float32)
        self._gizmo_drag_dir = np.zeros(3, dtype=np.float32)

        # Mouse click / drag state machine (single vs double vs drag).
        self._click_detector = ClickDetector()
        self._lmb_held = False
        self._rmb_held = False
        self._mmb_held = False
        self._mmb_last_y = 0.0           # last Y for MMB drag DOF adjust

        # Dolly-under-cursor pivot cache. Within a single scroll burst we
        # reuse the world point picked on the first tick so the camera
        # doesn't re-pick every tick (which can drift onto adjacent points
        # and add jitter). Invalidated if the cursor moves meaningfully
        # or the time gap between ticks exceeds a threshold.
        self._dolly_pivot_cache: tuple[np.ndarray, float, float, float] | None = None

        # Radial preset menu — activated by holding RMB in place for
        # _RADIAL_MENU_DELAY seconds; releasing commits the hovered slice.
        self._rmb_press_time: float | None = None
        self._radial_menu_active = False
        self._radial_menu_center: tuple[float, float] | None = None
        self._radial_menu_selected = -1

        # Legacy compat — gpu_clouds list for panels
        self.gpu_clouds: list[GPUCloud] = []

        # Automation CLI
        self.cli = None

        # Annotation state. Labels are strictly project-scoped now —
        # an empty registry at startup (just "Unlabeled" at id 0) means
        # the user cannot add labels until they open or create a
        # project. _sync_project_state swaps the registry in/out when
        # active_view changes.
        self.label_registry = LabelRegistry()
        self.undo_stack = UndoStack()
        self.active_label_id: int = 0
        self.label_blend: float = 1.0  # 0 = vertex colors, 1 = label colors.
        # Default ON because the primary workflow is "label and view labels":
        # on a freshly-imported labelled cloud, thumbnails should immediately
        # show the labels in colour. The L key still toggles to 0.0 when the
        # user wants to inspect raw point colours; that toggle now persists
        # to prefs so the choice survives restart.
        self.label_texture = None  # created in init_gl
        self.label_count_cache = LabelCountCache()
        # Direct-paint stroke recorder. One instance owns the in-progress
        # brush/box/lasso/curve/pick stroke and flushes a single
        # LabelCommand to the undo stack on mouse release.
        self.stroke_recorder = LabelStrokeRecorder()

        # Periodic in-stroke autosave. Between strokes the catalog labels
        # file is already current (every release writes through). This
        # timer only matters mid-brush-drag — long careful strokes get a
        # snapshot every minute so a crash loses at most one minute of
        # in-progress work.
        self._last_autosave_time: float = 0.0
        self._autosave_interval: float = 15.0

        # Periodic auto-snapshot to backups/<timestamp>/. Belt-and-
        # braces against the catalog being overwritten with bad state
        # (mass-paint mistake, undo-buffer loss, etc.) — gives the user
        # rollback points without having to remember to hit Save Points.
        # Snapshots only fire when there has been at least one label
        # mutation since the last snapshot, so an idle session doesn't
        # generate redundant copies. Pruning keeps the most recent
        # ``_snapshot_keep`` directories so backup storage stays bounded.
        self._last_snapshot_time: float = time.perf_counter()
        self._snapshot_interval: float = 600.0   # 10 minutes
        self._snapshot_keep: int = 12             # ~2 hours of history
        self._snapshot_dirty: bool = False

        # Multi-select state for the Contact Sheets cloud list. Plain click
        # populates this with a single index; Ctrl+click toggles; Shift+click
        # extends a range from the anchor. ``selected_index`` is still the
        # primary single-select used by Light Table; this set is a transient
        # superset used by batch actions (e.g. RUN INFERENCE on N clouds).
        # Cleared on tab change so the user doesn't accidentally batch over a
        # stale selection from a previous session.
        self.contact_sheets_selected: set[int] = set()
        self.contact_sheets_selected_anchor: int | None = None

        # Background batch-inference runner (Contact Sheets RUN INFERENCE).
        # Holds a thread + status object the UI can poll without touching
        # subprocess internals. None when no batch is in flight.
        self.contact_sheets_infer_runner = None
        # Queue of (entry, new_labels) tuples produced by the worker thread,
        # drained on the main thread each frame so GPU writes happen on the
        # GL-owning thread, not in the worker. ``deque`` is used because its
        # ``append`` and ``popleft`` are individually atomic under the GIL —
        # safer than list.append + list.pop(0) which combines two operations
        # in the producer / consumer dance.
        from collections import deque as _deque
        self._pending_label_applies = _deque()

        # Sticky status banner for save / load errors. Packaged .exe
        # has no stdout — every ``print()`` on failure is invisible to
        # the user. ``set_status_banner`` writes here; ``draw_status_banner``
        # in src/gui/status_banner.py renders it at the top of the viewport.
        # None when no banner is active. Phase 9/10 fixes convert each
        # silent ``print(...)`` site to a banner write.
        self._status_banner = None

        # Register the built-in main-thread handlers right after the
        # event-queue dicts go up. CP-7's pose_applied refresh is the
        # first one; future ports (Phase 8) join the same line below.
        self._register_default_event_handlers_pending = True

        # Generic subprocess→main-thread event queue (Pattern A).
        # Every subprocess reader thread (PolyPose, ImagingRunner,
        # ContactSheetsInferRunner, PT-v3, ...) is supposed to call
        # ``app.post_event(kind, payload)`` instead of mutating App
        # state directly. The main loop drains all queues each frame in
        # ``drain_all_events`` and dispatches to a registered handler
        # for each kind. Centralising the pattern keeps the reader
        # threads ignorant of App internals and gives a single chokepoint
        # for thread-safety regressions.
        #
        # Handlers run on the main GL thread; safe to touch app.*, GPU,
        # ImGui state. Phase 8 ports each existing reader's ``_on_event``
        # closure to ``post_event``; this commit only lays the rails.
        self._pending_events: dict[str, _deque] = {}
        self._event_handlers: dict[str, "callable"] = {}

        # Point falloff is now driven by a single softness knob 0..1.
        # 0 = hard circle, 1 = linear gradient. The Envelope object is
        # rebuilt from this scalar each time it changes (see panels.py).
        self.point_softness: float = 0.0
        self.point_falloff = Envelope.hard_circle()
        self.falloff_texture = None  # created in init_gl
        # Brush distance falloff — 0..1 scalar that controls the *depth*
        # slab around the snapped surface point. 0 = brush is a full
        # sphere of brush_radius (paints front-to-back through the
        # volume); 1 = thin slab right at the hovered surface (only
        # paints what's visually closest to the camera). Implemented as
        # an along-view-axis filter applied on top of the sphere
        # selection, so the visible rim still matches the in-plane reach.
        self.brush_distance_falloff: float = 0.0

        # Clip box (world-space AABB, pre-model-transform).
        # The shader reads clip_min / clip_max as uniforms; the user-facing
        # state is clip_fraction_lo + clip_fraction_hi per axis, synced via
        # _sync_clip_box().
        self.clip_min = np.array([-1e9, -1e9, -1e9], dtype=np.float32)
        self.clip_max = np.array([1e9, 1e9, 1e9], dtype=np.float32)
        self.clip_fraction_lo = np.zeros(3, dtype=np.float32)  # 0..1, clips from min side
        self.clip_fraction_hi = np.zeros(3, dtype=np.float32)  # 0..1, clips from max side
        self.clip_enabled: bool = True                      # master on/off for the clipping effect
        self.show_clip_planes: bool = True                  # viewport plane preview
        self.clip_axis_enabled = [True, True, True]        # per-axis clip on/off

        # Multi-viewport state
        self.viewport_count: int = 1                       # 1-4 viewport panels
        # Secondary cameras for split views (index 0 is always self.camera).
        # Presets: top (look down -Z), front (look along -Y), right (look along +X).
        # Each secondary camera gets its target + distance synced to the
        # primary every frame so all views show the same region.
        self._secondary_cameras = [Camera() for _ in range(3)]
        for cam, preset in zip(self._secondary_cameras,
                                ('top', 'front', 'right')):
            cam.set_preset(preset)
            cam.snap_to_goal()
            cam.projection = 'orthographic'

        # Selection state
        self.active_tool: str | None = None  # None, 'pick', 'box', 'lasso', 'polygon', 'brush', 'curve', 'measure_line', 'measure_angle', 'measure_landmark'
        self._measure: MeasureState | None = None  # active measurement session

        # Measurement registry — committed (completed) measurements.
        self.measure_registry: MeasureRegistry = MeasureRegistry()
        # Right sidebar (measurement panel) — closed until first commit.
        self.measure_panel_open: bool = False
        # Selection state for the measure panel tree (set of item ids).
        self.measure_selection: set = set()
        self._measure_last_sel_id: str | None = None
        # Anchor drag — set when user LMB-drags a committed anchor.
        self._measure_drag_item_id: str | None = None
        self._measure_drag_anchor_idx: int = -1
        # The selection buffer is kept around so legacy code paths that
        # passed `selection_buffer.mask` to the point cloud VBO still
        # have a valid empty mask to hand off. Under the direct-paint
        # workflow nothing writes to it — labels go straight onto
        # cloud_data.labels via apply_label / LabelStrokeRecorder.
        self.selection_buffer: SelectionBuffer = SelectionBuffer(0)
        # Box / lasso drag state
        self._drag_start: tuple[float, float] | None = None
        self._drag_current: tuple[float, float] | None = None
        self._lasso_path: list[tuple[float, float]] = []
        # True if Ctrl was held at mouse-press: the current stroke is
        # an eraser (writes label 0) instead of the active label.
        self._drag_ctrl: bool = False
        # True between Shift+LMB press and release while a selection tool
        # is engaged. Ensures the release path runs even if the user lets
        # go of Shift mid-drag (so partial strokes still commit cleanly).
        self._tool_engaged: bool = False
        # Cached (zbuf, tolerance) tuple built once at brush stroke PRESS
        # so the per-tick hidden-point filter doesn't recompute the cloud
        # depth grid 60 times a second. Cleared on stroke RELEASE.
        self._brush_visibility_zbuf = None
        # Brush state
        self.brush_radius: float = 1.0  # world units, auto-scaled to cloud on select
        self._brush_painting: bool = False
        # Curve state
        self.curve_threshold_px: float = 15.0
        # Depth limiting (infinity = no limit)
        self.selection_max_depth: float = float('inf')

        # 4D sequence
        self.sequence: PointCloudSequence | None = None

        # Async full-resolution loading
        from concurrent.futures import ThreadPoolExecutor
        self._full_res_executor = ThreadPoolExecutor(max_workers=1)
        self._full_res_pending = None  # (index, Future, model_xform) or None

        # Training sidecar
        self.training_runner = None

        # Shortcuts overlay
        self.show_shortcuts = False

        # Gallery double-click tracking
        self._gallery_last_click_index: int = -1
        self._gallery_last_click_time: float = 0.0

        # Gallery multi-selection (indices into self.entries). Always
        # contains the currently-active selected_index when non-empty.
        # Shift-click extends a range from the last anchor, Ctrl-click
        # toggles individual cells.
        self._gallery_multi_sel: set[int] = set()
        self._gallery_sel_anchor: int = -1  # last plain-clicked index for shift-range

        # Live filter string for the Contact Sheets search bar. Matches
        # the cloud's display name, case-insensitive, substring.
        self.gallery_search: str = ""

        # Gallery right-click context menu
        self._gallery_ctx_target: int = -1      # currently-targeted cloud for the popup
        self._gallery_ctx_pending_target: int = -1  # set on RMB press; committed to menu on release if no drag
        self._gallery_ctx_open_pending: bool = False  # set when an RMB click fires; panels.py calls open_popup
        self._gallery_rename_target: int = -1   # target of the rename modal
        self._gallery_rename_buf: str = ""      # rename text input buffer

        # Orbit drag tracking for Contact Sheets previews. The actual
        # azimuth/elevation/zoom live on each CloudEntry so orbiting
        # only rotates the selected cell.
        self._gallery_rmb_orbit_last: tuple[float, float] | None = None
        self._gallery_rmb_dragged: bool = False

        # Contact Sheets fixed-cell scrollable grid state.
        # gallery_cell_size is the user-controlled icon size in pixels
        # (Ctrl+wheel to resize). Loaded from ~/.3photon/prefs.json on
        # startup so the user's preferred size sticks across sessions.
        # _gallery_scroll_y is the vertical scroll offset in pixels;
        # clamped to [0, content_h - area_h] every frame.
        # _gallery_grid_cache holds (cols, rows, content_h) memoised
        # by (count, area_w, cell_size) so the per-frame layout calc
        # never goes through compute_grid more than once per change.
        from src.utils.prefs import load_prefs
        prefs = load_prefs()
        self.gallery_cell_size: int = int(prefs.get("gallery_cell_size", 220))
        # Labels panel height (draggable resize handle in sidebar)
        self._labels_list_h: float = float(prefs.get("labels_list_h", 0.0))
        # Pointcept training env paths (persisted across sessions)
        self._train_python_exe: str = prefs.get("train_python_exe", "")
        self._train_pointcept_dir: str = prefs.get("train_pointcept_dir", "")
        # Model registry state (lazy-initialized when train tab opens)
        self._train_model_registry = None
        self._train_selected_model_idx: int = 0  # 0 = new model
        self._train_new_model_name: str = ""
        self._train_active_model_id: str = ""  # model_id of running train
        self._train_model_finalized: bool = False
        # Inference model selector — model_id of the user's pick for
        # RUN INFERENCE on Contact Sheets. Empty string or "auto" means
        # "highest best_miou among models with a checkpoint on disk".
        # Persists across sessions so the choice survives a restart.
        self._inference_selected_model_id: str = str(
            prefs.get("inference_selected_model_id", ""))
        # Library-row right-click context menu state. Holds the project_id
        # whose menu is currently open (empty string = no menu). Used by
        # the project rows to render Delete on a deliberate click instead
        # of the old behaviour where a stray RMB nuked the project.
        self._project_ctx_target: str = ""
        # Restore Light Table view settings from prefs
        # User mode: 'researcher' (default, all 6 tabs) or 'clinician'
        # (Imaging / Hologram / Overwatch only). Persisted across sessions.
        self.user_mode: str = str(prefs.get("user_mode", "researcher"))
        if self.user_mode not in ("researcher", "clinician"):
            self.user_mode = "researcher"
        self.point_size = float(prefs.get("point_size", self.point_size))
        self.depth_falloff = float(prefs.get("depth_falloff", self.depth_falloff))
        self.brightness = float(prefs.get("brightness", self.brightness))
        self.contrast = float(prefs.get("contrast", self.contrast))
        self.saturation = float(prefs.get("saturation", self.saturation))
        self.r_gain = float(prefs.get("r_gain", self.r_gain))
        self.g_gain = float(prefs.get("g_gain", self.g_gain))
        self.b_gain = float(prefs.get("b_gain", self.b_gain))
        self.label_blend = float(prefs.get("label_blend", self.label_blend))
        self.exposure = float(prefs.get("exposure", self.exposure))
        self._gallery_scroll_y: float = 0.0
        self._gallery_scroll_target_y: float = 0.0  # smoothed wheel scroll
        self._gallery_grid_cache: tuple = ()
        self._gallery_grid_cache_key: tuple = ()
        # Scrollbar drag state — set when the user grabs the gutter handle.
        self._gallery_scrollbar_dragging: bool = False
        self._gallery_scrollbar_drag_offset: float = 0.0
        # Scrollbar geometry — populated by the overlay draw each frame
        # so the mouse handler has up-to-date hit rects without
        # recomputing them. Default is "no scrollbar yet".
        self._gallery_scrollbar_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._gallery_scrollbar_handle: tuple[int, int, int, int] = (0, 0, 0, 0)

        # Color picker modal state — label id being edited, or None
        self._editing_label_id: int | None = None
        self._editing_color: tuple | None = None
        self._editing_name_buf: str = ""

        # Screenshot / spin render resolution (editable from the EXPORT panel)
        self.export_width: int = 1920
        self.export_height: int = 1080

        # GLFW standard cursors (created in init_window)
        self._cursor_default = None
        self._cursor_crosshair = None
        self._cursor_hand = None
        self._cursor_resize_ew = None

        # Sidebar drag state — set by _handle_sidebar_resize in panels.py.
        # 'left' is the only sidebar now; 'right' is dead but the field
        # type stays str|None so the existing handler still slots in.
        self._sidebar_drag_active: str | None = None  # None | 'left'
        self._sidebar_drag_start_x: float = 0.0
        self._sidebar_drag_start_width: int = 0

        # Inline expansion state for the EXPORT block
        self._export_expanded: bool = False
        # Viewport HUD FOV popup state

    def init_window(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 4)

        # HiDPI: render the framebuffer at the monitor's *native* pixel
        # resolution instead of letting the compositor upscale a
        # logical-sized buffer (which looks blurry on fractional-scale
        # Wayland/KDE, e.g. 1.6x). With this on, get_framebuffer_size
        # returns physical pixels; window-logical cursor coords are then
        # converted to framebuffer space by _window_to_fb in the
        # selection handlers. Hints exist on GLFW 3.4+ (guarded).
        if hasattr(glfw, "SCALE_FRAMEBUFFER"):
            glfw.window_hint(glfw.SCALE_FRAMEBUFFER, True)
        if hasattr(glfw, "SCALE_TO_MONITOR"):
            glfw.window_hint(glfw.SCALE_TO_MONITOR, True)
        if hasattr(glfw, "WAYLAND_APP_ID"):
            try:
                glfw.window_hint_string(glfw.WAYLAND_APP_ID, "3photon")
            except Exception:
                pass

        # Set Windows AppUserModelID BEFORE creating the window so the
        # taskbar groups this process under its own icon, not python.exe.
        if sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    '3Photon.PointCloud.Viewer'
                )
            except Exception:
                pass

        # Get monitor work area for maximized launch
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        glfw.window_hint(glfw.MAXIMIZED, True)

        self.window = glfw.create_window(mode.size.width, mode.size.height,
                                         "3 P H O T O N", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        # Read actual framebuffer size after maximize
        self.width, self.height = glfw.get_framebuffer_size(self.window)

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Set window icon
        self._set_window_icon()

        # Force dark title bar on Windows via DWM
        self._apply_dark_title_bar()

        glfw.set_framebuffer_size_callback(self.window, self._on_resize)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_char_callback(self.window, self._on_char)
        glfw.set_drop_callback(self.window, self._on_drop)
        glfw.set_window_focus_callback(self.window, self._on_focus)

        # Standard cursors for selection tools (see _update_tool_cursor)
        try:
            self._cursor_default = glfw.create_standard_cursor(glfw.ARROW_CURSOR)
            self._cursor_crosshair = glfw.create_standard_cursor(glfw.CROSSHAIR_CURSOR)
            self._cursor_hand = glfw.create_standard_cursor(glfw.HAND_CURSOR)
            self._cursor_resize_ew = glfw.create_standard_cursor(glfw.HRESIZE_CURSOR)
        except Exception:
            self._cursor_default = None
            self._cursor_crosshair = None
            self._cursor_hand = None
            self._cursor_resize_ew = None

    def init_gl(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.program = create_point_cloud_program(self.ctx)
        self.overlays = OverlayRenderer(self.ctx)
        self.gizmo = GizmoRenderer(self.ctx)
        self.cursor3d = Cursor3D(self.ctx)
        # Mesh display pipeline (HOLOGRAM + optional LIGHT TABLE toggle).
        # Lazy import so a missing rendering dependency (e.g. shader file
        # not on disk for a half-checked-out repo) doesn't kill the app
        # at startup — the user can still work with points.
        try:
            from src.rendering.mesh_renderer import MeshRenderer
            self.mesh_renderer = MeshRenderer(self.ctx)
        except Exception as e:
            print(f"[mesh] renderer init failed; mesh display disabled: {e}")
            self.mesh_renderer = None
        self.label_texture = create_label_color_texture(self.ctx, self.label_registry)
        self.falloff_texture = create_falloff_texture(self.ctx, self.point_falloff)

        # HDR offscreen scene target + final tonemap pass. Everything the
        # scene pass draws lands here in RGBA16F linear space; the tonemap
        # pass ACES-maps it to the backbuffer.
        self.scene_rt = RenderTarget(self.ctx, self.width, self.height)
        self.pp_tonemap = FullscreenPass(self.ctx, 'pp_tonemap', 'post_tonemap.frag')
        # Gallery cache: persistent RGBA16F target for Contact Sheets.
        # See gallery_cache.py for rationale. pp_gallery_blit samples it
        # into scene_rt each frame at the gallery area viewport.
        self.gallery_cache = GalleryCache(self.ctx)
        self.pp_gallery_blit = FullscreenPass(self.ctx, 'pp_gallery_blit', 'gallery_blit.frag')

        # === SSAO pipeline ============================================
        # Two single-channel R16F render targets (raw AO + blurred AO),
        # plus two fullscreen passes (sample + bilateral blur). The
        # tonemap pass reads the blurred AO at texture unit 1 and
        # multiplies the HDR scene before applying ACES — contact
        # darkening reads through the curve naturally rather than
        # crushing already-tonemapped pixels.
        #
        # Depth-derived normals (dFdx/dFdy in the SSAO shader) mean no
        # G-buffer is needed; the scene's existing depth texture is
        # the only input. Cost on a 4090 at 5K is well under a
        # millisecond, so SSAO runs unconditionally — the only toggle
        # is ``ssao_enabled`` which swaps the AO texture for a 1×1
        # white so the tonemap multiply becomes a no-op.
        from src.rendering.ssao import generate_kernel, generate_noise
        self.ssao_rt = RenderTarget(
            self.ctx, self.width, self.height,
            components=1, color_dtype='f2', with_depth=False,
        )
        self.ssao_blur_rt = RenderTarget(
            self.ctx, self.width, self.height,
            components=1, color_dtype='f2', with_depth=False,
        )
        self.pp_ssao = FullscreenPass(self.ctx, 'pp_ssao', 'ssao.frag')
        self.pp_ssao_blur = FullscreenPass(
            self.ctx, 'pp_ssao_blur', 'ssao_blur.frag')

        # Upload the hemisphere kernel as uniform vec3[KERNEL_SIZE].
        # Generated once with a fixed seed so AO is deterministic
        # across runs (helpful when comparing screenshots).
        kernel = generate_kernel(32)
        try:
            self.pp_ssao.prog['u_kernel'].write(kernel.tobytes())
        except KeyError:
            pass

        # 4×4 noise texture for per-pixel kernel rotation. GL_REPEAT
        # tiles it across the screen so adjacent pixels rotate the
        # kernel differently; the blur pass cleans up the resulting
        # noise pattern.
        noise = generate_noise(4)
        self.ssao_noise_tex = self.ctx.texture(
            (noise.shape[1], noise.shape[0]),
            components=3, dtype='f4', data=noise.tobytes(),
        )
        self.ssao_noise_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.ssao_noise_tex.repeat_x = True
        self.ssao_noise_tex.repeat_y = True

        # 1×1 white texture used to disable AO without changing the
        # tonemap shader — bound to u_ao when ssao_enabled is False.
        white = np.ones((1, 1, 1), dtype=np.float32)
        self.ssao_disabled_tex = self.ctx.texture(
            (1, 1), components=1, dtype='f4', data=white.tobytes(),
        )
        self.ssao_disabled_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # Sampler unit bindings — the tonemap shader reads HDR from
        # unit 0 and AO from unit 1; the SSAO shader reads depth from
        # unit 0 and noise from unit 1; the blur shader reads raw AO
        # from unit 0 and depth from unit 1. Set once after program
        # creation so subsequent ``.render(textures={n: tex})`` calls
        # land in the right sampler.
        for prog, bindings in (
            (self.pp_tonemap.prog,      {'u_hdr': 0, 'u_ao': 1}),
            (self.pp_ssao.prog,         {'u_depth': 0, 'u_noise': 1}),
            (self.pp_ssao_blur.prog,    {'u_ssao': 0, 'u_depth': 1}),
            # u_src happens to work without an explicit binding (OpenGL
            # defaults sampler2D uniforms to texture unit 0 at link
            # time), but ``refresh_from_registry`` rebuilds the program
            # without re-applying that default — so the first frame
            # after a shader hot-reload samples from whatever was bound
            # to unit 0 by the prior frame. Explicit init removes the
            # hazard. See UNI-001 in REPORT.md.
            (self.pp_gallery_blit.prog, {'u_src': 0}),
        ):
            for name, unit in bindings.items():
                try:
                    prog[name].value = unit
                except KeyError:
                    continue

        # SSAO tuning. Bumped from the initial conservative defaults
        # after a visual pass — strength 0.8 + power 1.9 reads as
        # clearly-defined contact darkening in endplate/pedicle
        # crevices without crossing into the "rendered" look. Tune
        # by editing this dict at runtime; F6 toggles for A/B.
        self.ssao_params = {
            'radius':   8.0,    # mm in view-space units
            'bias':     0.025,  # suppress self-occlusion on flat faces
            'power':    1.9,    # exponent applied to AO factor
            'strength': 0.8,    # 0 = no AO, 1 = full
        }
        self.ssao_enabled = True

        # === Selection-outline pipeline =============================
        # When a vertebra is selected in HOLOGRAM we want a 2-pixel
        # orange silhouette stroke around its outer boundary — NOT a
        # body tint. We render selected meshes into a single-channel
        # mask buffer that shares scene_rt's depth (so occluded parts
        # don't outline through other bones), then a fullscreen edge-
        # detection pass reads the mask and alpha-blends an orange
        # stroke onto the tonemapped backbuffer.
        self.selection_mask_tex = self.ctx.texture(
            (self.width, self.height), components=1, dtype='f2',
        )
        self.selection_mask_tex.filter = (
            moderngl.NEAREST, moderngl.NEAREST)
        self.selection_mask_tex.repeat_x = False
        self.selection_mask_tex.repeat_y = False
        # Share scene_rt.depth so depth-test against the already-drawn
        # scene excludes occluded fragments from the mask.
        self.selection_mask_fbo = self.ctx.framebuffer(
            color_attachments=[self.selection_mask_tex],
            depth_attachment=self.scene_rt.depth,
        )
        self.pp_outline = FullscreenPass(
            self.ctx, 'pp_outline', 'pp_outline.frag')
        try:
            self.pp_outline.prog['u_mask'].value = 0
        except KeyError:
            pass

    def init_gui(self):
        self.gui = ImGuiLayer(self.window)

    def _gallery_filter_ready(self) -> list:
        """Return the entries that should appear in the Contact Sheets
        grid: has bounds (so the layout can size a cell for it), and
        passes the search filter (if any).

        Note: ``preview_gpu`` is *not* required. Cells whose GPU buffer
        hasn't been uploaded yet appear as blank tiles in the layout and
        get populated lazily by ``_render_gallery`` as they scroll into
        view (see ``_ensure_preview_gpu``). This keeps view-switch O(N)
        for metadata only — the heavy disk-decompress + GPU-upload work
        is bounded per frame instead of blocking the main thread on
        every entry up front.

        The returned list is (entry_index, entry) tuples so callers can
        map cell-index back to the global self.entries index.
        """
        needle = self.gallery_search.strip().lower() if self.gallery_search else ""
        ready = []
        for i, e in enumerate(self.entries):
            if e.bounds_min is None or e.bounds_max is None:
                continue
            if needle and needle not in e.name.lower():
                continue
            ready.append((i, e))
        return ready

    def _ensure_preview_gpu(self, entry) -> bool:
        """Lazy-load + upload the preview GPU buffer for ``entry``.

        Returns True if a load+upload actually happened on this call (so
        the caller can decrement its per-frame budget). Returns False if
        the entry already has a preview_gpu, or there's nothing on disk
        to load (no cached .npz, no catalog), or the load failed.

        Called from ``_render_gallery`` for visible cells whose buffers
        haven't been uploaded yet — this is what replaces the old
        synchronous "upload everything in set_active_view" path.
        """
        if entry.preview_gpu is not None:
            return False
        if self.catalog is None:
            return False
        file_key = getattr(entry, 'file_key', None)
        if not file_key:
            return False
        lib_entry = self.catalog.entries.get(file_key)
        if lib_entry is None:
            return False
        preview = self.catalog.get_preview(lib_entry)
        if preview is None:
            return False
        preview.model_transform = entry.model_transform
        # Restore preview-resolution labels if persisted alongside the cloud.
        stored_prev = load_preview_labels(
            file_key, expect_count=preview.point_count)
        if stored_prev is None:
            # No preview-res labels saved, but the cloud may carry
            # full-resolution manual labels in the catalog store. Transfer
            # them onto the (downsampled) preview points by nearest
            # neighbour so the thumbnail shows its labels in Contact Sheets
            # without the cloud ever being opened, and cache the result.
            stored_prev = self._derive_preview_labels_from_full(
                file_key, preview)
            if stored_prev is not None:
                try:
                    save_preview_labels(file_key, stored_prev)
                except Exception:
                    pass
        if stored_prev is not None and preview.labels is not None:
            preview.labels[:] = stored_prev
        gpu = upload_cloud(preview, self.ctx, self.program)
        entry.preview_gpu = gpu
        if entry.bounds_min is None:
            entry.bounds_min = preview.bounds_min.copy()
            entry.bounds_max = preview.bounds_max.copy()
        if not entry.point_count:
            entry.point_count = preview.point_count
        self.gpu_clouds.append(gpu)
        return True

    def _derive_preview_labels_from_full(self, file_key, preview):
        """Nearest-neighbour transfer of full-resolution catalog labels
        onto the preview's downsampled points.

        The gallery preview is a voxel-downsample that keeps no source
        indices (LibraryCatalog._build_preview saves positions+colors
        only), so a cloud labelled only at full resolution has no
        preview-res labels and renders flat-unlabelled in Contact Sheets.
        This maps each preview point to its nearest full-res point and
        copies that label across.

        Returns an int32 array of length ``preview.point_count``, or None
        when there are no usable full labels / coordinates to map from.
        """
        full_labels = load_cloud_labels(file_key)
        if full_labels is None:
            return None
        full_labels = np.asarray(full_labels, dtype=np.int32).reshape(-1)
        if not full_labels.any():
            return None  # all-unlabelled — nothing to show
        loaded = load_cloud_data(file_key)
        if loaded is None:
            return None
        full_cloud, _meta = loaded
        full_xyz = np.asarray(full_cloud.positions, dtype=np.float32)
        prev_xyz = np.asarray(preview.positions, dtype=np.float32)
        if full_xyz.shape[0] != full_labels.shape[0] or prev_xyz.size == 0:
            return None
        try:
            from scipy.spatial import cKDTree
            _, idx = cKDTree(full_xyz).query(prev_xyz, k=1)
        except Exception:
            return None
        return full_labels[idx].astype(np.int32)

    def _ensure_mesh_gpu(self, entry) -> bool:
        """Lazy-load + upload the mesh GPU buffer for ``entry``.

        Path:
        - If ``entry.mesh_gpu`` is already valid and not dirty → return False
          (nothing to do).
        - If labels changed (``mesh_dirty=True``) → delete the cached
          mesh file so the catalog's queue_mesh_build doesn't no-op on
          the stale cache, queue a rebuild, return False this frame.
        - If a cached mesh exists on disk → load + upload.
        - Otherwise → queue a background Poisson build via the catalog
          and return False; the next render call after the build
          completes will pick it up.

        Returns True when an upload actually happened this call. Safe to
        call every frame; the cache-hit path is one dict lookup.
        """
        if self.mesh_renderer is None:
            return False
        if entry is None:
            return False
        if entry.mesh_gpu is not None and not entry.mesh_dirty:
            return False
        file_key = getattr(entry, "file_key", None)
        if not file_key:
            return False
        from src.data.cloud_store import (
            load_cloud_mesh, has_cloud_mesh, cloud_mesh_path,
        )
        from src.rendering.mesh_renderer import upload_mesh

        # Dirty path: the on-disk cache is stale because labels (or
        # other inputs to Poisson) changed since it was written. Without
        # this branch ``_ensure_mesh_gpu`` would happily re-upload the
        # stale file and clear the dirty flag, so subsequent label-aware
        # mesh features (label tinting, label-masked picking) would
        # silently render the old geometry. queue_mesh_build no-ops if
        # ``has_cloud_mesh`` is True, so we have to drop the cache file
        # before queueing.
        if entry.mesh_dirty and has_cloud_mesh(file_key):
            try:
                cloud_mesh_path(file_key).unlink()
            except OSError as e:
                print(f"[mesh] cache invalidate failed for "
                      f"{file_key}: {e}")
            # Fall through to the has_cloud_mesh==False branch below
            # which queues the rebuild.

        if not has_cloud_mesh(file_key):
            # Queue a build; the result lands in cloud_mesh_path next
            # session-tick of poll_pending_meshes (called from the main
            # loop). User sees a brief "no mesh yet" frame and then the
            # surface appears once the build finishes.
            if self.catalog is not None:
                lib_entry = self.catalog.entries.get(file_key)
                if lib_entry is not None:
                    self.catalog.queue_mesh_build(lib_entry)
            return False

        mesh_data = load_cloud_mesh(file_key)
        if mesh_data is None:
            return False
        try:
            gpu_mesh = upload_mesh(
                mesh_data, self.ctx, self.mesh_renderer.program,
                mask_program=self.mesh_renderer.mask_program)
        except Exception as e:
            print(f"[mesh] upload failed for {file_key}: {e}")
            return False
        if entry.mesh_gpu is not None:
            entry.mesh_gpu.release()
        entry.mesh_gpu = gpu_mesh
        entry.mesh_dirty = False
        return True

    def _gallery_filter_ready_cached(self) -> list:
        """Per-frame memoised version of _gallery_filter_ready.

        The gallery filter is called twice per frame (render + overlay)
        and O(N) over the entire library. Cache by frame_count so both
        callers share the result while still refreshing every frame.
        """
        if getattr(self, '_gallery_ready_frame', -1) != self.frame_count:
            self._gallery_ready_cache = self._gallery_filter_ready()
            self._gallery_ready_frame = self.frame_count
        return self._gallery_ready_cache

    def _find_entry_by_key(self, file_key: str | None,
                            file_path: str | None = None):
        """Return the first CloudEntry matching file_key or file_path.

        Session-level dedup helper for import paths so re-importing the
        same cloud updates the existing row instead of stacking copies.
        """
        abs_path = os.path.abspath(file_path) if file_path else None
        for e in self.entries:
            if file_key is not None and e.file_key == file_key:
                return e
            if abs_path is not None and os.path.abspath(e.file_path) == abs_path:
                return e
        return None

    def _autoadd_to_active_project(self, file_key: str | None):
        """If the user has a real (non-smart) project active, append
        the newly-imported file_key to it. Smart views (All / Recent /
        Missing) are skipped because they're derived, not user-owned.
        """
        if file_key is None or self.catalog is None:
            return
        view = self.active_view
        if view is None or view[0] != "project":
            return
        cid = view[1]
        if cid.startswith("smart:"):
            return
        if cid not in self.catalog.projects:
            return
        self.catalog.add_to_project(cid, [file_key])

    def _load_cloud_via_catalog(self, file_key: str | None,
                                 source_path: str | None):
        """Load a cloud, preferring the catalog data file over re-parsing the source.

        Returns ``(cloud, source_was_used)``: a populated PointCloudData
        and a flag indicating whether we had to fall back to the source
        loader (True) or got it straight from the catalog (False). The
        catalog data path is the source-of-truth once a cloud has been
        imported, so we only re-parse the original PLY/LAS/NIfTI on
        first import or when no catalog entry exists yet.
        """
        # 1) Catalog-first: if we have a file_key and a saved data file,
        #    skip the source loader entirely.
        if file_key:
            cached = load_cloud_data(file_key)
            if cached is not None:
                cloud, meta = cached
                if source_path and is_source_stale(meta, source_path):
                    print(
                        f"Source file '{source_path}' has been modified "
                        f"since the catalog last cached this cloud — keeping "
                        f"the catalog version with your labels. Re-import "
                        f"manually if you want to refresh the points."
                    )
                # Apply any persisted labels on top.
                stored_labels = load_cloud_labels(file_key)
                if stored_labels is not None and len(stored_labels) == cloud.point_count:
                    cloud.labels[:] = stored_labels
                elif stored_labels is not None:
                    print(
                        f"Catalog labels for {file_key} have a length "
                        f"mismatch ({len(stored_labels)} vs {cloud.point_count}); "
                        f"discarding."
                    )
                return cloud, False

        # 2) Fall back to the source loader.
        if not source_path:
            raise IOError("No source path and no catalog entry to load from")
        cloud = load_point_cloud(source_path)

        # 2b) Check for companion _labels.npy (e.g. VerSe converter output).
        #     Apply raw label ids directly — registry entries will be created
        #     on the main thread when the GPU upload happens.
        has_labels = (cloud.labels is not None
                      and cloud.point_count > 0
                      and int(cloud.labels.max()) > 0)
        if not has_labels and source_path:
            from src.data.labels_io import find_companion_label_array
            comp = find_companion_label_array(source_path)
            if comp is not None:
                try:
                    raw = np.load(comp).astype(np.int32)
                    if raw.ndim == 1 and len(raw) == cloud.point_count:
                        cloud.labels[:] = raw
                        has_labels = True
                        print(f"Applied companion labels: "
                              f"{os.path.basename(comp)}")
                except Exception as e:
                    print(f"Companion label load failed: {e}")

        # 3) Persist into the catalog so the next open is catalog-fast.
        if file_key:
            save_cloud_data(file_key, cloud, source_path=source_path)
            if has_labels:
                # CP-4 + LS-7 (post-audit): use the per-key lock and
                # surface a failure through the status banner so the
                # user notices a first-import label-persist error
                # before the next session loses the .npy companion.
                err = save_cloud_labels(
                    file_key, cloud.labels, catalog=self.catalog)
                if err:
                    try:
                        self.set_status_banner(
                            f"Label save failed during import: {err}",
                            level="error",
                            source=f"cloud_store.save_cloud_labels:{file_key}",
                        )
                    except AttributeError:
                        pass

        return cloud, True

    def _save_view_prefs(self) -> None:
        """Persist Light Table view settings so they survive app restart."""
        from src.utils.prefs import update_prefs
        update_prefs({
            'point_size': float(self.point_size),
            'depth_falloff': float(self.depth_falloff),
            'brightness': float(self.brightness),
            'contrast': float(self.contrast),
            'saturation': float(self.saturation),
            'r_gain': float(self.r_gain),
            'g_gain': float(self.g_gain),
            'b_gain': float(self.b_gain),
            'label_blend': float(self.label_blend),
            'exposure': float(self.exposure),
        })

    def _persist_cloud_labels(self, entry, cloud) -> None:
        """Write the cloud's current labels to the catalog labels store.

        Hooked from ``_after_label_mutation`` so every brush stroke,
        box/lasso/curve commit, pick, undo, and redo is persisted
        immediately. No-op if the entry has no file_key.

        Guards against the user painting on a preview cloud while the
        full-resolution cloud isn't loaded yet — that would write
        preview-length labels to the catalog's full-resolution labels
        file, where the next session's load (which checks against the
        full point count) silently discards them. Skip with a print so
        the loss is visible.
        """
        file_key = getattr(entry, 'file_key', None)
        if not file_key or cloud is None or cloud.labels is None:
            return
        expected = int(getattr(entry, 'point_count', 0) or 0)
        if expected and cloud.labels.shape[0] != expected:
            print(f"[persist] skip {entry.name}: labels length "
                  f"{cloud.labels.shape[0]} != entry full-res {expected}; "
                  f"open in Light Table first to load the full cloud, then "
                  f"re-paint.")
            return
        # LS-7: route save failures through the sticky status banner so
        # the user sees them — packaged .exe has no stdout.
        # CP-4: pass catalog so the save acquires the per-file_key lock,
        # serialising any concurrent background write to the same key.
        err = save_cloud_labels(file_key, cloud.labels, catalog=self.catalog)
        if err:
            try:
                self.set_status_banner(
                    f"Label save failed for {entry.name}: {err}",
                    level="error",
                    source=f"cloud_store.save_cloud_labels:{file_key}",
                )
            except AttributeError:
                pass

    def load_file(self, path: str):
        """Load a single file at full resolution and register it in the library."""
        # Compute the file_key first so we can check the catalog before
        # touching the source file at all.
        self._ensure_catalog()
        prospective_key = None
        if self.catalog is not None and os.path.isfile(path):
            try:
                prospective_key = compute_file_key(path)
            except Exception:
                prospective_key = None

        try:
            cloud, used_source = self._load_cloud_via_catalog(prospective_key, path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return

        # Register with the persistent library so the cloud is browsable
        # in future sessions without re-importing.
        lib_entry = None
        if self.catalog is not None and os.path.isfile(path):
            lib_entry = self.catalog.register_file(path)

        file_key = lib_entry.file_key if lib_entry else prospective_key
        self._autoadd_to_active_project(file_key)

        # Dedup: if the same cloud is already in the session, refresh
        # its GPU data in place rather than appending a second copy.
        existing = self._find_entry_by_key(file_key, path)
        if existing is not None:
            cloud.model_transform = existing.model_transform
            gpu = upload_cloud(cloud, self.ctx, self.program)
            # Release the old GPU buffer unconditionally — the prior
            # ``is not preview_gpu`` guard meant the typical single-file
            # case (full_gpu == preview_gpu, same object) leaked ~32 MB
            # per re-import. ``release()`` is idempotent on the
            # underlying GL resource even if preview_gpu still aliases it.
            if existing.full_gpu is not None:
                existing.full_gpu.release()
            existing.full_gpu = gpu
            existing.preview_gpu = gpu
            existing.bounds_min = cloud.bounds_min.copy()
            existing.bounds_max = cloud.bounds_max.copy()
            existing.point_count = cloud.point_count
            self.gpu_clouds.append(gpu)
            try:
                self.selected_index = self.entries.index(existing)
            except ValueError:
                pass
            self._overlays_dirty = True
            if getattr(self, 'cursor3d', None) is not None:
                self.cursor3d.clear()
            print(f"Refreshed {existing.name} ({cloud.point_count:,} pts)")
            return

        entry = CloudEntry(path, file_key=file_key)
        entry.bounds_min = cloud.bounds_min.copy()
        entry.bounds_max = cloud.bounds_max.copy()
        entry.point_count = cloud.point_count
        # Coordinate metadata from the loader
        meta = cloud.metadata if cloud.metadata else {}
        wo = meta.get('world_offset')
        if wo is not None:
            entry.local_origin = np.array(wo, dtype=np.float64)
        entry.source_unit = meta.get('source_unit', 'raw')
        cloud.model_transform = entry.model_transform

        gpu = upload_cloud(cloud, self.ctx, self.program)
        entry.full_gpu = gpu
        entry.preview_gpu = gpu  # same for single-file load
        self.entries.append(entry)
        self.gpu_clouds.append(gpu)
        self._overlays_dirty = True
        if getattr(self, 'cursor3d', None) is not None:
            self.cursor3d.clear()

        # Cache metrics + a downsampled preview into the library so the
        # contact sheet stays fast next launch.
        if lib_entry is not None and self.catalog is not None:
            self.catalog.update_metrics(
                lib_entry.file_key,
                cloud.point_count,
                cloud.bounds_min,
                cloud.bounds_max,
            )
            if not os.path.exists(lib_entry.preview_path(self.catalog.dir)):
                self.catalog.queue_preview_build(lib_entry)

        # Initialize clip box to the first loaded cloud's bounds
        if len(self.entries) == 1:
            self.selected_index = 0
            self.reset_clip()
            self.cursor3d.clear()
        print(f"Loaded {cloud.point_count:,} points from {entry.name}")

    def _ensure_catalog(self):
        """Create the library catalog on first use."""
        if self.catalog is None:
            self.catalog = LibraryCatalog()
            # LS-5: surface any startup corruption the catalog
            # detected through the sticky status banner. Without
            # this the user sees an empty library with no
            # explanation when index.json couldn't be parsed.
            for err in getattr(self.catalog, "_init_errors", ()):
                try:
                    self.set_status_banner(
                        err, level="warn", source="library_catalog.init",
                    )
                except AttributeError:
                    pass

    # -- Status banner ----------------------------------------------------

    def set_status_banner(self, message: str, *,
                          level: str = "error",
                          source: str = "") -> None:
        """Show a sticky status bar at the top of the viewport.

        ``level`` is 'error' / 'warn' / 'info' (color tint only).
        ``source`` is an opaque identifier used for auto-coalescing —
        repeated calls with the same ``source`` increment a counter
        instead of stacking a second bar, so an error loop (e.g.
        autosave failing every 15 s) doesn't drown the UI.

        Safe to call from the main thread only. Subprocess readers
        should funnel through ``post_event`` and let the registered
        handler call this on dispatch.
        """
        from src.gui.status_banner import ErrorBanner
        current = self._status_banner
        if (current is not None
                and source
                and current.source == source
                and current.level == level):
            current.suppressed_count += 1
            current.message = message
            return
        self._status_banner = ErrorBanner(
            message=message, level=level, source=source, suppressed_count=0,
        )

    def clear_status_banner(self) -> None:
        """Dismiss the current banner. Used by tests and by code that
        knows the underlying condition has been resolved."""
        self._status_banner = None

    # -- Pattern A subprocess→main-thread event queue --------------------


    def _on_training_event(self, payload) -> None:
        """Main-thread handler for ``training_event`` (ST-5).

        PT-v3 runner reader thread posts parsed-line payloads here;
        we forward the raw line into ``app.cli`` (the CLI panel
        iterates ``cli.output`` every frame) and apply any model-
        registry updates. The reader thread previously did both
        directly off-thread.
        """
        try:
            raw_line = payload["raw_line"]
            project_id = payload["project_id"]
            model_id = payload["model_id"]
            updates = payload["updates"]
        except (TypeError, KeyError):
            return
        if self.cli and raw_line:
            self.cli.log(raw_line, "info")
        if (updates and project_id and model_id
                and getattr(self, "_train_model_registry", None) is not None):
            try:
                self._train_model_registry.update_model(
                    project_id, model_id, **updates)
            except Exception as e:
                print(f"[training] registry update failed: {e}")


    def _on_infer_log(self, payload) -> None:
        """Main-thread handler for ``infer_log`` (TS-001 post-audit).

        The standalone inference subprocess streams stdout through a
        reader thread; we ship each line here instead of letting the
        reader poke ``app.cli`` directly (it appends-and-rebinds a list
        the main thread iterates every frame).
        """
        if not self.cli:
            return
        try:
            line = payload["line"]
            level = payload["level"]
        except (TypeError, KeyError):
            return
        if line:
            self.cli.log(str(line), str(level))

    def _on_infer_finished(self, payload) -> None:
        """Main-thread handler for ``infer_finished`` (TS-001 post-audit).

        Posts the success / failure footer line after the subprocess
        reader observes ``proc.wait()`` returning. Payload is a dict so
        ``pred_dir`` can travel through the queue without depending on
        any closure on the reader thread side.
        """
        if not self.cli:
            return
        try:
            rc = int(payload["returncode"])
            pred_dir = str(payload["pred_dir"])
        except (TypeError, KeyError, ValueError):
            return
        if rc == 0:
            self.cli.log(
                f"Inference complete. Predictions in: {pred_dir}", "success")
        else:
            self.cli.log(f"Inference failed (exit {rc}).", "error")


    def _ensure_default_event_handlers(self) -> None:
        """Register built-in event handlers on first frame. Called from
        ``drain_all_events`` so we don't need to thread the import
        through __init__ ordering."""
        if not getattr(self, "_register_default_event_handlers_pending", False):
            return
        self._register_default_event_handlers_pending = False
        self.register_event_handler(
            "training_event",
            lambda app, payload: app._on_training_event(payload),
        )
        # TS-001 (post-audit): the standalone inference reader was the
        # last unported subprocess→main-thread mutation site.
        self.register_event_handler(
            "infer_log",
            lambda app, payload: app._on_infer_log(payload),
        )
        self.register_event_handler(
            "infer_finished",
            lambda app, payload: app._on_infer_finished(payload),
        )

    def register_event_handler(self, kind: str, handler) -> None:
        """Wire a main-thread handler for ``kind`` events.

        ``handler`` is called with ``(self, payload)`` for every event of
        this kind drained from the queue. Registration is idempotent —
        re-registering overrides the prior handler. Must be called from
        the main thread before any subprocess starts posting.
        """
        self._event_handlers[kind] = handler

    def post_event(self, kind: str, payload) -> None:
        """Enqueue a subprocess→main-thread event. Safe from any thread.

        Uses one ``collections.deque`` per kind — ``append`` is atomic
        under the GIL, so the reader thread can call this without
        locks. The main loop drains the queue in ``drain_all_events``.
        """
        q = self._pending_events.get(kind)
        if q is None:
            from collections import deque as _deque
            # Concurrent first-post-of-kind across threads is benign:
            # whichever ``setdefault`` lands first wins, and the other
            # rebinds to the same deque object via the dict lookup
            # before its ``append`` runs.
            q = self._pending_events.setdefault(kind, _deque())
        q.append(payload)

    def drain_all_events(self, max_per_kind: int = 16) -> int:
        """Drain queued events on the main thread and dispatch handlers.

        Called once per frame from ``_render_frame``'s polling block.
        ``max_per_kind`` caps how many events of any one kind run per
        frame so a burst from a subprocess can't stall the frame loop;
        unhandled events stay on the queue for the next frame.

        Returns the total number of events dispatched this call.
        """
        # Stubs / tests may not have the lazy-init hook — guard so the
        # queue contract stays testable in isolation.
        try:
            self._ensure_default_event_handlers()
        except AttributeError:
            pass
        dispatched = 0
        for kind, q in list(self._pending_events.items()):
            if not q:
                continue
            handler = self._event_handlers.get(kind)
            for _ in range(max_per_kind):
                if not q:
                    break
                try:
                    payload = q.popleft()
                except IndexError:
                    break
                if handler is None:
                    # No handler registered yet — drop the event so the
                    # queue doesn't grow unbounded. Loud print so an
                    # un-wired event kind shows up in dev runs.
                    print(f"[events] dropped {kind!r}: no handler")
                    continue
                try:
                    handler(self, payload)
                    dispatched += 1
                except Exception as e:
                    print(f"[events] {kind!r} handler raised: {e}")
        return dispatched

    def load_directory(self, directory: str):
        """Import all point clouds from a directory into the library.

        If the directory contains a numbered sequence (e.g. frame_001.ply,
        frame_002.ply), it is loaded as a 4D PointCloudSequence instead.
        Every file is registered with the persistent library catalog so
        it remains browsable in future sessions; cached previews are
        reused when available.
        """
        # Check if this looks like a 4D sequence
        seq_files = detect_sequence(directory)
        if seq_files and len(seq_files) >= 2:
            self._load_as_sequence(seq_files)
            return

        files = scan_directory(directory)
        if not files:
            print(f"No point cloud files found in {directory}")
            return

        self._ensure_catalog()

        for f in files:
            lib_entry = self.catalog.register_file(f)
            if lib_entry is None:
                continue

            self._autoadd_to_active_project(lib_entry.file_key)

            # Dedup against what's already in the session: if an entry
            # with the same file_key (or path) is present, refresh its
            # metadata rather than appending a duplicate row.
            existing = self._find_entry_by_key(lib_entry.file_key, f)
            if existing is not None:
                if lib_entry.bounds_min is not None:
                    existing.bounds_min = lib_entry.bounds_min.copy()
                    existing.bounds_max = lib_entry.bounds_max.copy()
                existing.point_count = (
                    lib_entry.point_count or existing.point_count)
                existing.file_key = lib_entry.file_key
                print(f"Skipped duplicate: {existing.name}")
                continue

            entry = CloudEntry(f, file_key=lib_entry.file_key)
            if lib_entry.bounds_min is not None:
                entry.bounds_min = lib_entry.bounds_min.copy()
                entry.bounds_max = lib_entry.bounds_max.copy()
            entry.point_count = lib_entry.point_count
            self.entries.append(entry)

            # Cached previews load lazily in _render_gallery; queue a build
            # only for entries with nothing on disk yet.
            preview_path = lib_entry.preview_path(self.catalog.dir)
            if os.path.exists(preview_path):
                print(f"Cached: {entry.name}")
            else:
                self.catalog.queue_preview_build(lib_entry)
                print(f"Queued: {entry.name}")

        # Mark the freshly-imported batch as the active view so folder
        # switching has a reference point. Exception: if the user is
        # currently inside a user-owned project, stay there so the
        # imports land in that project (they've already been
        # auto-added via _autoadd_to_active_project).
        av = self.active_view
        in_user_project = (
            av is not None and av[0] == "project"
            and not str(av[1]).startswith("smart:")
        )
        if not in_user_project:
            self.active_view = ("folder", os.path.abspath(directory))

        if len(self.entries) > 1:
            self.mode = MODE_CONTACT_SHEETS
        self._fit_camera(animate=False)
        self.reset_clip()
        self.cursor3d.clear()
        self._overlays_dirty = True
        print(f"Loaded {len(files)} files from {directory}")

    def load_directory_recursive(self, root: str):
        """Walk a tree and register every supported file as an independent entry.

        Used for hierarchical datasets where the top level isn't the
        data directory (e.g. VerSe: ``verse_points/01_training/sub-XXX/NNN.ply``,
        141 subjects × ~10 files each, 1400 files total). Unlike
        ``load_directory``, this bypasses the 4D sequence detector —
        files in the same leaf directory are treated as independent
        samples, not frames of a time series.

        Register-only: reads bounds + point count from file headers for
        the catalog, queues preview builds on a background thread. The
        main loop remains responsive even while 1000+ files are being
        indexed because the heavy lifting happens off-thread.
        """
        files = scan_directory_recursive(root)
        if not files:
            print(f"No point cloud files found under {root}")
            return

        self._ensure_catalog()
        print(f"Registering {len(files)} files from {root}...")

        loaded_entries: list[CloudEntry] = []
        for f in files:
            lib_entry = self.catalog.register_file(f)
            if lib_entry is None:
                continue

            self._autoadd_to_active_project(lib_entry.file_key)

            # Dedup against the session list like load_directory does.
            existing = self._find_entry_by_key(lib_entry.file_key, f)
            if existing is not None:
                if lib_entry.bounds_min is not None:
                    existing.bounds_min = lib_entry.bounds_min.copy()
                    existing.bounds_max = lib_entry.bounds_max.copy()
                existing.point_count = (
                    lib_entry.point_count or existing.point_count)
                existing.file_key = lib_entry.file_key
                continue

            entry = CloudEntry(f, file_key=lib_entry.file_key)
            if lib_entry.bounds_min is not None:
                entry.bounds_min = lib_entry.bounds_min.copy()
                entry.bounds_max = lib_entry.bounds_max.copy()
            entry.point_count = lib_entry.point_count
            self.entries.append(entry)
            loaded_entries.append(entry)

            # Cached previews load lazily in _render_gallery; queue a build
            # only for entries with nothing on disk yet.
            preview_path = lib_entry.preview_path(self.catalog.dir)
            if not os.path.exists(preview_path):
                self.catalog.queue_preview_build(lib_entry)

        av = self.active_view
        in_user_project = (
            av is not None and av[0] == "project"
            and not str(av[1]).startswith("smart:")
        )
        if not in_user_project:
            self.active_view = ("folder", os.path.abspath(root))

        if len(self.entries) > 1:
            self.mode = MODE_CONTACT_SHEETS
        self._fit_camera(animate=False)
        self.reset_clip()
        self.cursor3d.clear()
        self._overlays_dirty = True
        print(
            f"Registered {len(loaded_entries)} new entries "
            f"(previews building in background)"
        )

    def set_active_view(self, view: tuple[str, str] | None):
        """Switch the Contact Sheet view to a library folder / project.

        Drops every current CloudEntry (releasing its GPU buffers) and
        rebuilds from the library's entries for the chosen view.

        Preview load + GPU upload are deferred to ``_render_gallery``
        (via ``_ensure_preview_gpu``) so this stays O(N) for metadata
        only — a 1000-entry view used to do 1000 zlib decompresses + GPU
        allocs synchronously here, blocking the main thread for seconds.
        Now the view appears immediately and visible cells populate over
        the next few frames; off-screen cells never load until scrolled
        into view.

        Entries with no cached preview .npz on disk are queued for
        background build via ``queue_preview_build`` as before.
        """
        self._ensure_catalog()
        if self.catalog is None:
            return

        if view is None:
            target_entries = []
        elif view[0] == "folder":
            target_entries = self.catalog.entries_in_folder(view[1], recursive=True)
        elif view[0] == "project":
            target_entries = self.catalog.entries_in_project(view[1])
        else:
            target_entries = []

        # Release GPU resources for the outgoing session
        for e in self.entries:
            e.release()
        self.entries = []
        self.gpu_clouds = []
        self.selected_index = 0
        self.sequence = None

        for lib_entry in target_entries:
            entry = CloudEntry(lib_entry.file_path, file_key=lib_entry.file_key)
            if lib_entry.bounds_min is not None:
                entry.bounds_min = lib_entry.bounds_min.copy()
                entry.bounds_max = lib_entry.bounds_max.copy()
            entry.point_count = lib_entry.point_count
            self.entries.append(entry)

            # Only queue a build if there's nothing cached yet — cached
            # previews load lazily in _render_gallery as cells scroll in.
            preview_path = lib_entry.preview_path(self.catalog.dir)
            if not os.path.exists(preview_path) and lib_entry.exists():
                self.catalog.queue_preview_build(lib_entry)

            self.catalog.touch(lib_entry.file_key)

        self.active_view = view
        self.mode = MODE_CONTACT_SHEETS
        self._overlays_dirty = True
        self.reset_clip()
        self.cursor3d.clear()
        # SC-10: mirror the _select_entry_by_index pattern — clear
        # in-flight tool state on view-switch so a half-drawn lasso /
        # polygon doesn't land its next click on a freshly-loaded
        # cloud. The selection buffer is already cloud-scoped so it
        # gets cleared per-entry on first paint, but the lasso path
        # is App-scoped and survives until explicitly emptied.
        self._lasso_path = []
        self.undo_stack.clear()

        # --- Apply or restore project-level ontology ---
        self._sync_project_state(view)

        # Switching views = reset gallery scroll. The new view's first
        # entry should be in the top-left, not somewhere off-screen
        # because we were scrolled deep in the previous view.
        self._gallery_scroll_y = 0.0
        self._gallery_scroll_target_y = 0.0
        self._gallery_grid_cache_key = ()
        if self.entries:
            self._fit_camera(animate=False)
        print(
            f"Library view: {view} -> {len(self.entries)} entries"
        )

    def _seek_sequence(self, index: int):
        """Seek the 4D sequence to a new frame, delegating GPU swap to timeline module."""
        if self.sequence is None:
            return
        from src.gui.timeline import _do_seek
        _do_seek(self, index)

    def _select_entry_by_index(self, new_entry_idx: int,
                               *, announce: bool = False) -> bool:
        """Switch the active cloud to ``entries[new_entry_idx]``.

        Single source of truth for "make this cloud the active one" —
        used by both arrow-key navigation and click handlers in the
        sidebar / gallery. Without funneling both paths through here,
        clicking a row only updated ``selected_index`` while the
        camera, full-res load, clip box, undo stack and stale-label
        persistence all stayed pinned to the previous cloud, which
        manifested as "clicking a cloud shows nothing" (the new cloud
        renders, just off-screen from where the camera was looking).

        Operations performed when switching:

        - Persist any pending labels for the cloud being left so the
          user doesn't lose unsaved paint when changing focus.
        - Clear the selection buffer + undo stack — both are scoped
          to the previous cloud's point count and would corrupt
          if applied to a different-shape cloud.
        - Reset the clip box to the new cloud's bounds.
        - Trigger full-res load if the new cloud hasn't loaded yet.
        - Slide the orbit pivot to the new cloud's centre while
          preserving distance / azimuth / elevation, so the zoom
          carries over across rapid stepping.

        Returns True if the switch actually happened (False on no-op
        or out-of-range index).
        """
        if not (0 <= new_entry_idx < len(self.entries)):
            return False
        if new_entry_idx == self.selected_index:
            return False
        new_entry = self.entries[new_entry_idx]

        # Persist labels for the cloud we're leaving.
        if 0 <= self.selected_index < len(self.entries):
            old_entry = self.entries[self.selected_index]
            old_gpu = old_entry.full_gpu or old_entry.preview_gpu
            if old_gpu is not None and old_gpu.cloud_data is not None:
                self._persist_cloud_labels(old_entry, old_gpu.cloud_data)

        # Switch.
        self.selected_index = new_entry_idx
        self.selection_buffer.clear()
        # Undo stack is global but every command's indices are scoped
        # to the cloud it was painted on. Switching clouds invalidates
        # all history — clearing here means the new cloud starts with
        # a fresh undo timeline, no chance of an old stroke landing on
        # a cloud whose point_count differs.
        self.undo_stack.clear()

        # Reset clip box to the new cloud's bounds so the previous
        # cloud's clip doesn't hide most of the new cloud's points.
        self.reset_clip()

        # Sync the selection buffer size to the new cloud.
        self._sync_selection_buffer()

        # Trigger full-res load if needed.
        if new_entry.full_gpu is None:
            self._load_full_resolution(new_entry_idx)

        # Slide the orbit pivot to the new cloud's centre without
        # touching distance/azimuth/elevation, so the user's zoom
        # carries over across arrow-nav and click-nav steps.
        if (new_entry.bounds_min is not None
                and new_entry.bounds_max is not None):
            center = ((new_entry.bounds_min + new_entry.bounds_max) / 2.0
                      ).astype(np.float32)
            self.camera.focus_on(center)

        self._overlays_dirty = True
        self.cursor3d.clear()
        self._auto_scale_brush()
        if announce:
            print(f"→ {new_entry.name}")
        return True

    def _navigate_cloud(self, delta: int):
        """Move to the next/previous cloud in the current view.

        Arrow-key navigation in Light Table: steps through the filtered
        entry list (respects project view + search filter), persists
        labels for the current cloud before switching, loads full-res
        for the new cloud, and slides the orbit pivot to the new
        cloud's centre. Camera distance / azimuth / elevation are
        preserved so arrow-stepping through similarly-sized clouds
        (e.g. per-bone crops) doesn't reset zoom each step. Press
        F/Space to re-fit explicitly.

        Delegates the switch itself to ``_select_entry_by_index`` so
        sidebar clicks (which use the same method directly) and
        arrow-key navigation produce identical state transitions.
        """
        visible = self._gallery_filter_ready_cached()
        if not visible:
            return

        # Find current cloud's position in the visible list.
        current_vis_idx = None
        for vi, (entry_idx, _entry) in enumerate(visible):
            if entry_idx == self.selected_index:
                current_vis_idx = vi
                break

        if current_vis_idx is None:
            # Current cloud not in visible list — jump to first/last.
            new_vis_idx = 0 if delta > 0 else len(visible) - 1
        else:
            new_vis_idx = current_vis_idx + delta

        # Clamp to bounds.
        new_vis_idx = max(0, min(len(visible) - 1, new_vis_idx))
        new_entry_idx, new_entry = visible[new_vis_idx]

        if self._select_entry_by_index(new_entry_idx, announce=False):
            print(f"→ {new_entry.name} ({new_vis_idx + 1}/{len(visible)})")

    def _load_as_sequence(self, frame_paths: list[str]):
        """Load a directory as a time-series sequence."""
        print(f"Detected 4D sequence: {len(frame_paths)} frames")
        self.sequence = PointCloudSequence(frame_paths, cache_size=3)
        # Load first frame into a single CloudEntry
        cloud = self.sequence.get_frame(0)
        entry = CloudEntry(frame_paths[0])
        entry.bounds_min = cloud.bounds_min.copy()
        entry.bounds_max = cloud.bounds_max.copy()
        entry.point_count = cloud.point_count
        cloud.model_transform = entry.model_transform
        gpu = upload_cloud(cloud, self.ctx, self.program)
        entry.full_gpu = gpu
        entry.preview_gpu = gpu
        self.entries.append(entry)
        self.gpu_clouds.append(gpu)
        self.selected_index = 0
        self.mode = MODE_LIGHT_TABLE  # go straight to Light Table for annotation
        self._fit_camera(animate=False)
        self._overlays_dirty = True
        print(f"Sequence loaded: {cloud.point_count:,} points per frame")

    def _poll_catalog(self):
        """Check for newly completed library previews and upload them."""
        if not self.catalog:
            return
        completed = self.catalog.poll_pending()
        for lib_entry in completed:
            # Find matching session CloudEntry by file_key
            for entry in self.entries:
                if entry.file_key != lib_entry.file_key or entry.preview_gpu is not None:
                    continue
                if lib_entry.preview_data is None:
                    break
                lib_entry.preview_data.model_transform = entry.model_transform
                # Apply persisted preview-resolution labels at preview-
                # arrive time too (mirrors the load_directory path).
                stored_prev = load_preview_labels(
                    entry.file_key,
                    expect_count=lib_entry.preview_data.point_count)
                if stored_prev is not None and lib_entry.preview_data.labels is not None:
                    lib_entry.preview_data.labels[:] = stored_prev
                gpu = upload_cloud(lib_entry.preview_data, self.ctx, self.program)
                entry.preview_gpu = gpu
                if lib_entry.bounds_min is not None:
                    entry.bounds_min = lib_entry.bounds_min.copy()
                    entry.bounds_max = lib_entry.bounds_max.copy()
                entry.point_count = lib_entry.point_count
                self.gpu_clouds.append(gpu)
                self._overlays_dirty = True
                print(f"Ready: {entry.name} ({lib_entry.point_count:,} pts)")
                break


    def _poll_mesh_builds(self) -> None:
        """Mark session entries as mesh-dirty when a background mesh
        build completes. The next render loop will pick up the new
        mesh from disk via ``_ensure_mesh_gpu``. Cheap when nothing's
        pending (one lock + empty-list check)."""
        if not self.catalog:
            return
        completed = self.catalog.poll_pending_meshes()
        if not completed:
            return
        key_set = set(completed)
        for entry in self.entries:
            if entry.file_key in key_set:
                # Force the next render to reload from disk. If a stale
                # mesh_gpu is sitting around, _ensure_mesh_gpu releases
                # it before assigning the new one.
                entry.mesh_dirty = True

    def _load_full_resolution(self, index: int):
        """Queue async full-resolution load. Non-blocking.

        Disk I/O runs on a background thread; ``_poll_full_res`` does
        the GPU upload on the main thread when the data is ready.
        """
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        if entry.full_gpu is not None:
            return
        if (self._full_res_pending is not None
                and self._full_res_pending[0] == index):
            return
        file_key = getattr(entry, 'file_key', None)
        file_path = entry.file_path
        model_xform = entry.model_transform.copy()
        future = self._full_res_executor.submit(
            self._load_cloud_via_catalog, file_key, file_path
        )
        self._full_res_pending = (index, future, model_xform, file_key)
        print(f"Loading full res: {entry.name} ...")

    def _poll_full_res(self):
        """Upload completed async full-res load to GPU."""
        if self._full_res_pending is None:
            return
        index, future, model_xform, expected_key = self._full_res_pending
        if not future.done():
            return
        self._full_res_pending = None
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        # Guard against index reuse after view switch — if the entry at
        # this index is now a different cloud, discard the stale load.
        if getattr(entry, 'file_key', None) != expected_key:
            return
        if entry.full_gpu is not None:
            return
        try:
            cloud, _ = future.result()
            cloud.model_transform = model_xform

            # If the user painted labels on the preview cloud while the
            # full-res load was in flight, the preview labels are newer
            # than whatever the catalog had on disk. Merge them: for each
            # point in the full-res cloud that has a matching preview
            # point, prefer the preview's label if it is non-zero.
            preview_gpu = entry.preview_gpu
            if (preview_gpu is not None
                    and preview_gpu.cloud_data is not None
                    and preview_gpu.cloud_data.labels is not None
                    and (preview_gpu.cloud_data.labels != 0).any()):
                preview_labels = preview_gpu.cloud_data.labels
                # If counts match (preview is a subsample, so they usually
                # don't), direct copy. Otherwise just persist what the
                # preview had — the catalog file is the source of truth
                # for the next load and it was already flushed on paint.
                if len(preview_labels) == cloud.point_count:
                    cloud.labels[:] = preview_labels

            gpu = upload_cloud(cloud, self.ctx, self.program)
            entry.full_gpu = gpu
            entry.bounds_min = cloud.bounds_min.copy()
            entry.bounds_max = cloud.bounds_max.copy()
            entry.point_count = cloud.point_count
            # Coordinate metadata from catalog round-trip
            meta = cloud.metadata if cloud.metadata else {}
            wo = meta.get('world_offset')
            if wo is not None:
                entry.local_origin = np.array(wo, dtype=np.float64)
            if meta.get('source_unit'):
                entry.source_unit = meta['source_unit']
            if self.selected_index == index:
                self._sync_selection_buffer()
                self.selection_buffer.clear()
                update_selection(gpu, self.selection_buffer.mask)
                self.camera.fit_to_bounds(entry.bounds_min, entry.bounds_max)
                self._auto_scale_brush()
                self._overlays_dirty = True
            print(f"Full res: {entry.name} ({cloud.point_count:,} pts)")
        except Exception as e:
            print(f"Error loading full res {entry.name}: {e}")

    def _ensure_full_resolution(self, index: int):
        """Synchronous full-res load. Blocks until on GPU.

        For batch/export paths only; interactive paths use
        ``_load_full_resolution`` (async).
        """
        if index < 0 or index >= len(self.entries):
            return
        entry = self.entries[index]
        if entry.full_gpu is not None:
            return
        try:
            cloud, _ = self._load_cloud_via_catalog(
                getattr(entry, 'file_key', None), entry.file_path
            )
            cloud.model_transform = entry.model_transform
            gpu = upload_cloud(cloud, self.ctx, self.program)
            entry.full_gpu = gpu
            entry.bounds_min = cloud.bounds_min.copy()
            entry.bounds_max = cloud.bounds_max.copy()
            entry.point_count = cloud.point_count
            print(f"Full res: {entry.name} ({cloud.point_count:,} pts)")
        except Exception as e:
            print(f"Error loading full res {entry.name}: {e}")

    def reset_clip(self):
        """Clear all clip cuts back to nothing-clipped on every axis."""
        self.clip_fraction_lo[:] = 0.0
        self.clip_fraction_hi[:] = 0.0
        self._sync_clip_box()

    def _sync_clip_box(self):
        """Recompute clip_min / clip_max uniforms from dual clip fractions.

        Each axis has two independent fractions:
          clip_fraction_lo[axis] — pushes clip_min inward from bmin
          clip_fraction_hi[axis] — pushes clip_max inward from bmax

        Both at 0 = full span visible. Both at 0.5 = only the centre
        sliver remains. This allows clipping from both sides at once
        (e.g., isolating a sagittal slab).
        """
        if not self.clip_enabled:
            self.clip_min[:] = -1e9
            self.clip_max[:] = 1e9
            return
        if not (0 <= self.selected_index < len(self.entries)):
            self.clip_min[:] = -1e9
            self.clip_max[:] = 1e9
            return
        entry = self.entries[self.selected_index]
        if entry.bounds_min is None:
            self.clip_min[:] = -1e9
            self.clip_max[:] = 1e9
            return
        bmin = entry.bounds_min.astype(np.float32)
        bmax = entry.bounds_max.astype(np.float32)
        span = bmax - bmin
        pad = np.maximum(np.abs(span) * 0.001, 1e-4)
        axis_enabled = getattr(self, 'clip_axis_enabled', [True, True, True])
        for axis in range(3):
            # Disabled axis = no clipping on that axis, regardless of fractions
            if not axis_enabled[axis]:
                self.clip_min[axis] = -1e9
                self.clip_max[axis] = 1e9
                continue
            cut_lo = float(span[axis]) * float(self.clip_fraction_lo[axis])
            cut_hi = float(span[axis]) * float(self.clip_fraction_hi[axis])
            self.clip_min[axis] = float(bmin[axis] + cut_lo - pad[axis])
            self.clip_max[axis] = float(bmax[axis] - cut_hi + pad[axis])

    def _fit_camera(self, animate: bool = True):
        """Fit camera to the combined bounds of all loaded clouds.

        ``animate=False`` snaps instantly (used on initial load so nothing
        swings in); the default smoothly eases to the new framing.
        """
        bounds = [(e.bounds_min, e.bounds_max) for e in self.entries
                  if e.bounds_min is not None]
        if not bounds:
            return
        all_min = np.min([b[0] for b in bounds], axis=0)
        all_max = np.max([b[1] for b in bounds], axis=0)
        self.camera.fit_to_bounds(all_min, all_max, animate=animate)
        self._overlays_dirty = True

    def _on_cloud_selected(self, index: int, enter_light_table: bool = False):
        """Single entry point for switching the active cloud.

        Handles selection-buffer resize, previous-cloud selection clear,
        clip AABB reset, camera fit, overlay rebuild, and optionally
        full-resolution load + mode switch to Light Table.
        """
        if not (0 <= index < len(self.entries)):
            return
        self.selected_index = index
        if enter_light_table:
            self._load_full_resolution(index)
        # Resize + clear the selection buffer so any prior cloud's mask is gone
        self._sync_selection_buffer()
        self.selection_buffer.clear()
        # Drop the undo stack — every command's indices are sized to
        # the cloud it was recorded against, so a redo applied across
        # a cloud switch would IndexError.
        self.undo_stack.clear()
        gpu = self._current_gpu_cloud()
        if gpu is not None:
            update_selection(gpu, self.selection_buffer.mask)

        entry = self.entries[index]
        # Refresh bounds from the actual GPU positions so the camera
        # frames exactly where the visible points live (preview bounds
        # can lag behind the full-res data after an on-demand load).
        if gpu is not None:
            pos = gpu.cloud_data.positions
            if pos.size:
                entry.bounds_min = pos.min(axis=0)
                entry.bounds_max = pos.max(axis=0)
        if entry.bounds_min is not None and entry.bounds_max is not None:
            self.camera.fit_to_bounds(entry.bounds_min, entry.bounds_max)
        self.cursor3d.clear()
        self.reset_clip()
        self._overlays_dirty = True
        if enter_light_table:
            self.mode = MODE_LIGHT_TABLE
            # Suppress the cursor-place double-click gesture that would
            # otherwise fire on the RELEASE of the same double-click that
            # promoted us from Contact Sheets → Light Table.
            self._click_detector.reset()

    def _rebuild_overlays(self):
        """Rebuild overlay geometry."""
        bounds = [(e.bounds_min, e.bounds_max) for e in self.entries
                  if e.bounds_min is not None]
        if not bounds:
            return
        all_min = np.min([b[0] for b in bounds], axis=0)
        all_max = np.max([b[1] for b in bounds], axis=0)
        self.overlays.build_bounding_box(all_min, all_max)
        self.overlays.build_grid(all_min, all_max)
        self._overlays_dirty = False

    def run(self):
        self.init_window()
        self.init_gl()
        self.init_gui()

        from src.gui.automation import CLIEngine
        self.cli = CLIEngine(self)

        # Initialise the persistent library catalog so the Contact Sheet
        # tab can immediately show every cloud the user has previously
        # imported — even before they drop anything new into the app.
        self._ensure_catalog()

        # Single-instance lock. If another live 3Photon owns the catalog
        # we refuse to launch — two writers would race on label files
        # and silently corrupt each other. Stale locks (PID is dead)
        # are cleared automatically.
        ok, blocking_pid = acquire_lock()
        if not ok:
            print(
                f"Refusing to start: another 3Photon instance is using the "
                f"library catalog (pid {blocking_pid}). Close it first, or "
                f"manually delete ~/.3photon/library/.lock if it's stale."
            )
            return

        # Read-only integrity sweep so the user sees what's in the catalog
        # at startup. Reports orphan files and missing data referenced by
        # the index — never auto-fixes anything.
        try:
            status = check_catalog_integrity(catalog=self.catalog)
            print(format_integrity_summary(status))
        except Exception as e:
            print(f"Catalog integrity check failed: {e}")

        # Load from command line
        if len(sys.argv) > 1:
            path = sys.argv[1]
            if os.path.isdir(path) or os.path.isfile(path):
                self.import_path(path)
                self._fit_camera(animate=False)
            else:
                print(f"Path not found: {path}")

        if not self.entries:
            # No CLI path — start with a blank canvas. The user picks
            # a collection / folder from the sidebar to load clouds.
            self.mode = MODE_CONTACT_SHEETS
            self.active_view = None

        self.last_fps_time = time.time()
        self._last_frame_time = time.perf_counter()
        # Per-frame try/except + finally cleanup: any exception inside the
        # frame body (GL error, NaN math, missing file on a background load)
        # is logged and the loop continues instead of silently taking the
        # app down with no trace. Cleanup runs even on a hard failure so
        # the training subprocess + GL context get released.
        import traceback
        try:
            while not glfw.window_should_close(self.window):
                try:
                    # Poll events at the TOP of the loop so this frame's
                    # render reflects the freshest input. The old order
                    # (render → swap → poll) meant an event arriving
                    # mid-render wasn't visible until the iteration AFTER
                    # the next one — up to two frames of input-to-pixel
                    # latency. Polling first closes that gap to one frame.
                    glfw.poll_events()

                    # Frame-rate-independent delta time, clamped so a hitched frame
                    # never teleports smoothed values.
                    now = time.perf_counter()
                    dt = min(now - self._last_frame_time, 0.1)
                    self._last_frame_time = now

                    self.camera.update(dt)
                    self._tick_radial_menu()
                    self._tick_autosave()
                    self._tick_auto_snapshot()
                    # Apply any labels produced by the background batch-
                    # inference worker. Up to 4 clouds per frame: each
                    # apply is one full re_upload_labels (4 bytes/pt) +
                    # cache invalidate + autosave write — measures well
                    # under 2 ms total at this batch size, so 4 stays
                    # comfortably inside one 60 fps frame budget. Higher
                    # throughput means a 50-cloud batch visibly lands in
                    # ~13 frames instead of 50.
                    from src.gui.panels import drain_pending_label_applies
                    drain_pending_label_applies(self, max_per_frame=4)

                    # Drain the generic Pattern A subprocess→main-thread
                    # queue. No callers wired in this commit (Phase 8
                    # ports each existing reader's _on_event closure to
                    # post_event); the drain runs no-op until then.
                    self.drain_all_events()

                    self._poll_catalog()
                    self._poll_full_res()
                    self._poll_mesh_builds()
                    self._update_tool_cursor()
                    self._render_frame()
                    self._update_fps()
                    glfw.swap_buffers(self.window)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"[frame error] {type(e).__name__}: {e}")
                    traceback.print_exc()
                    # ImGui state recovery: if the frame body crashed
                    # AFTER `gui.new_frame()` ran but BEFORE `gui.render()`,
                    # ImGui's internal frame counter is left dangling and
                    # the next `new_frame()` will assert
                    # ("Forgot to call Render() or EndFrame()..."). Drain
                    # the half-built frame here so the loop can continue
                    # cleanly. Both end_frame() (close the frame) and
                    # render() (build draw data) are tried — render() is
                    # the documented recovery path but end_frame() is the
                    # cheaper fallback if render() also throws.
                    try:
                        import imgui as _imgui
                        # render() implies end_frame() — try it first.
                        # If that also fails, try bare end_frame().
                        try:
                            _imgui.render()
                        except Exception:
                            try:
                                _imgui.end_frame()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Try to pump events so the user can at least close the
                    # window instead of having a dead application.
                    try:
                        glfw.poll_events()
                    except Exception:
                        pass
        finally:
            self._cleanup()

    def _render_frame(self):
        self.gui.new_frame()

        if self._overlays_dirty and self.entries:
            self._rebuild_overlays()

        self.ctx.viewport = (0, 0, self.width, self.height)

        # Top menu bar (File / Edit / View / Help). Drawn first so the
        # sidebar + gallery layout can offset by its height. This is an
        # ImGui accumulation call — no GL state touched until the final
        # self.gui.render() at the end of the frame.
        self._menu_bar_height = draw_main_menu_bar(self)

        # Sticky status banner for save / load failures. Sits just under
        # the menu bar. No-op when no banner is active.
        from src.gui.status_banner import draw_status_banner
        draw_status_banner(self)

        # === HDR scene pass ===
        # Everything the scene draws (cloud, gizmo, grid, bbox, cursor,
        # gallery cells) lands in scene_rt as linear RGBA16F. The
        # tonemap pass below ACES-maps it to the backbuffer.
        self.scene_rt.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        bg = self.bg_color
        self.scene_rt.clear(bg[0], bg[1], bg[2], 1.0, depth=1.0)

        if self.mode == MODE_CONTACT_SHEETS:
            self._render_gallery()
        elif self.mode == MODE_LIGHT_TABLE:
            self._render_individual()
        elif self.mode == MODE_AUTOMATION:
            # Render the selected cloud behind the sidebar/CLI so there's
            # still something to look at — matches the Light Table feel
            self._render_individual()

        # === Selection mask render ===================================
        # No mode has a concept of "selected meshes" after the HOLOGRAM
        # removal, so the outline pass is always inactive.
        outline_active = self._render_selection_mask()

        # === SSAO pass + bilateral blur ===
        # Runs once per frame against the just-completed scene depth
        # buffer. Skipped entirely when ssao_enabled is False — the
        # tonemap below binds a 1×1 white texture instead so the
        # multiply becomes a no-op and the curve is unchanged. Cost
        # on a 4090 is well under a millisecond at full resolution.
        ao_tex = self.ssao_disabled_tex
        if (self.ssao_enabled
                and self.scene_rt is not None
                and self.ssao_rt is not None
                and self.ssao_blur_rt is not None
                and self.pp_ssao is not None
                and self.pp_ssao_blur is not None):
            # The SSAO shader needs the camera's projection and its
            # inverse to reconstruct view-space positions from NDC
            # depth and to project sample points back to UV. We use
            # the primary camera — gallery / split views share the
            # same projection within a frame anyway.
            proj = self.camera.get_projection_matrix()
            try:
                inv_proj = np.linalg.inv(proj.astype(np.float64))
            except np.linalg.LinAlgError:
                inv_proj = None
            if inv_proj is not None:
                # GLSL expects column-major; numpy is row-major. Transpose
                # before tobytes() so the upload matches the shader layout.
                proj_gl = proj.astype(np.float32).T.tobytes()
                inv_proj_gl = inv_proj.astype(np.float32).T.tobytes()

                noise_scale = (
                    self.width / float(self.ssao_noise_tex.width),
                    self.height / float(self.ssao_noise_tex.height),
                )

                # --- SSAO sample pass ---
                self.ssao_rt.use()
                self.ctx.viewport = (0, 0, self.ssao_rt.width,
                                      self.ssao_rt.height)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.BLEND)
                # Clear to 1.0 so any pixel the shader early-outs on
                # (far plane) is fully lit — the texture format is R16F
                # so we set color via clear() RGBA tuple, only .r matters.
                self.ssao_rt.clear(1.0, 1.0, 1.0, 1.0)
                # Upload matrices via the raw program (FullscreenPass
                # uniform dispatch handles scalars and tuples cleanly;
                # raw bytes need the program API).
                try:
                    self.pp_ssao.prog['u_projection'].write(proj_gl)
                    self.pp_ssao.prog['u_inv_projection'].write(inv_proj_gl)
                except KeyError:
                    pass
                self.pp_ssao.render(
                    uniforms={
                        'u_noise_scale': noise_scale,
                        'u_radius':   float(self.ssao_params['radius']),
                        'u_bias':     float(self.ssao_params['bias']),
                        'u_power':    float(self.ssao_params['power']),
                        'u_strength': float(self.ssao_params['strength']),
                    },
                    textures={
                        0: self.scene_rt.depth,
                        1: self.ssao_noise_tex,
                    },
                )

                # --- Bilateral blur ---
                self.ssao_blur_rt.use()
                self.ctx.viewport = (0, 0, self.ssao_blur_rt.width,
                                      self.ssao_blur_rt.height)
                self.ssao_blur_rt.clear(1.0, 1.0, 1.0, 1.0)
                self.pp_ssao_blur.render(
                    uniforms={
                        'u_texel_size': (
                            1.0 / self.ssao_blur_rt.width,
                            1.0 / self.ssao_blur_rt.height,
                        ),
                        'u_depth_threshold': 0.0005,
                    },
                    textures={
                        0: self.ssao_rt.color,
                        1: self.scene_rt.depth,
                    },
                )
                ao_tex = self.ssao_blur_rt.color

        # === Tonemap blit to backbuffer ===
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.pp_tonemap.render(
            uniforms={'u_exposure': float(self.exposure)},
            textures={0: self.scene_rt.color, 1: ao_tex},
        )

        # === Selection outline overlay ===============================
        # Drawn AFTER tonemap so the orange stroke lands in display
        # space (won't be re-graded by ACES) and on top of the scene.
        # Alpha-blended onto the backbuffer; the shader writes alpha=0
        # everywhere except the 2-pixel-wide outer outline of the mask.
        if outline_active and self.pp_outline is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.pp_outline.render(
                uniforms={
                    'u_texel_size': (1.0 / max(self.width, 1),
                                     1.0 / max(self.height, 1)),
                    # OP1_ORANGE in display space — matches the rest
                    # of the app's selection accent.
                    'u_outline_color': (0.976, 0.573, 0.141),
                    'u_outline_alpha': 1.0,
                },
                textures={0: self.selection_mask_tex},
            )

        # Restore default GL state so downstream ImGui (the gallery
        # overlay uses ImGui foreground draws, not raw GL) runs the
        # same as before.
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ImGui overlay
        if self.gui_visible:
            draw_settings_panel(self)

        # Measurement panel — right sidebar, closed by default.
        if self.gui_visible:
            draw_measure_panel(self)

        # HOLOGRAM-only right sidebar: vertebra selector + scene
        # controls. Anchored to the right edge using sidebar_width_right;
        # `_right_chrome_width` already reserves space so the viewport
        # rects don't extend underneath this panel.

        # The old HOLOGRAM inspector overlay used to live on the right
        # edge — but the new right-side panel (vertebra selector +
        # scene controls) now owns that space. Inspector content needs
        # to fold into the new panel or move to the left sidebar's
        # MEASUREMENTS section; suppressing it here keeps the layout
        # readable until that consolidation lands.

        # Gallery cell borders + filename labels (foreground draw list)
        if self.mode == MODE_CONTACT_SHEETS:
            self._draw_gallery_overlay()

        # Draw selection tool rubber-band overlays (foreground draw list)
        self._draw_tool_overlay()

        # Floating horizontal tool toolbar — top-left of viewport, Light Table only
        self._draw_tool_toolbar()

        # Workstation-grade world-axis gnomon in the bottom-left
        self._draw_gnomon_hud()

        # Draw 4D timeline if sequence is loaded (Light Table and Automation
        # both render the current cloud, so the scrubber is useful in both)
        if self.sequence is not None and self.mode in (MODE_LIGHT_TABLE, MODE_AUTOMATION):
            draw_timeline(self)

        # Shortcuts overlay
        if self.show_shortcuts:
            draw_shortcuts_overlay(self)

        # Radial preset menu overlay (RMB long-press)
        if self._radial_menu_active and self._radial_menu_center is not None:
            draw_radial_menu(self._radial_menu_center, self._radial_menu_selected)

        # Color picker modal (only when editing a label)
        if self._editing_label_id is not None:
            draw_color_picker_popup(self)

        self.gui.render()
        self.frame_count += 1
        self._fps_frame_count += 1

    def _draw_tool_toolbar(self):
        """Horizontal tool toolbar — top-left of viewport.

        Shown in LIGHT TABLE (full toolset: paint, select, gizmo,
        measurement) and HOLOGRAM (measurement-only: DST / ANG / LMK).
        The paint / box / lasso / brush / move / rotate tools operate
        on a single cloud's labels and don't make sense across the
        HOLOGRAM multi-cloud scene, so they're filtered out there.
        Every other mode hides the toolbar entirely and clears the
        active tool so stale state doesn't leak across tabs.
        """
        # Mode → allowed tool-name set (or None for "all").
        HOLOGRAM_TOOLS = {"measure_line", "measure_angle",
                          "measure_landmark"}
        if self.mode == MODE_LIGHT_TABLE:
            allowed: set[str] | None = None  # full toolset
        else:
            allowed = set()  # toolbar hidden in this mode

        # Drop any active tool that isn't allowed in the current mode.
        # Without this the selection stays in `active_tool` while the
        # toolbar UI disappears, which (a) hides the 3D cursor because
        # the render path is in a tool-specific drawing state, and (b)
        # leaks stale lasso paths / drag-rectangles / gizmo into other
        # tabs. Runs every frame so any path that flips mode (button
        # click, keyboard shortcut) drops the tool atomically.
        if (self.active_tool is not None
                and allowed is not None
                and self.active_tool not in allowed):
            self.active_tool = None
            self._drag_start = None
            self._drag_current = None
            self._lasso_path = []
            if self.gizmo is not None:
                self.gizmo.visible = False
            self._measure = None

        if not allowed and allowed is not None:
            # allowed is an explicit empty set → no toolbar at all.
            return

        import imgui
        from src.gui.scale import s
        from src.gui.theme import (OP1_RED, OP1_BLUE, OP1_GREEN, OP1_WHITE,
                                   OP1_ORANGE, OP1_DIM, OP1_GRAY, col32)
        from src.gui.op1_widgets import (
            draw_styled_rect, draw_styled_rect_active, BUTTON_ROUNDING,
            _icon_pick, _icon_box, _icon_lasso, _icon_polygon, _icon_brush,
            _icon_curve, _icon_move, _icon_rotate,
            _icon_measure_line, _icon_measure_angle, _icon_measure_landmark,
            _TOOL_TOOLTIPS,
        )

        tools_full = [
            ("PCK", "pick",         OP1_RED,                _icon_pick),
            ("BOX", "box",          OP1_BLUE,               _icon_box),
            ("LSO", "lasso",        OP1_GREEN,              _icon_lasso),
            ("PLY", "polygon",      OP1_GREEN,              _icon_polygon),
            ("BRU", "brush",        OP1_WHITE,              _icon_brush),
            ("CRV", "curve",        (0.7, 0.5, 0.95, 1.0), _icon_curve),
            ("MOV", "move",         OP1_ORANGE,             _icon_move),
            ("ROT", "rotate",       (1.0, 0.75, 0.2, 1.0), _icon_rotate),
            ("DST", "measure_line", (1.0, 0.2, 0.2, 1.0),  _icon_measure_line),
            ("ANG", "measure_angle",(1.0, 0.2, 0.2, 1.0),  _icon_measure_angle),
            ("LMK", "measure_landmark",(1.0, 0.2, 0.2, 1.0), _icon_measure_landmark),
        ]
        if allowed is None:
            tools = tools_full
        else:
            tools = [t for t in tools_full if t[1] in allowed]

        th = imgui.get_text_line_height()
        btn_sz = th * 2.2
        gap    = s(3)
        pad    = s(6)
        n      = len(tools)
        bar_w  = n * btn_sz + (n - 1) * gap + pad * 2
        bar_h  = btn_sz + pad * 2

        sw     = self._left_chrome_width()
        menu_h = getattr(self, '_menu_bar_height', 0.0)

        imgui.set_next_window_position(sw + s(8), menu_h + s(8), imgui.ALWAYS)
        imgui.set_next_window_size(bar_w, bar_h, imgui.ALWAYS)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.04, 0.04, 0.04, 0.88)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING,  (pad, pad))
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING,    (gap, 0.0))
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, s(6))

        flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE |
                 imgui.WINDOW_NO_MOVE      | imgui.WINDOW_NO_SCROLLBAR |
                 imgui.WINDOW_NO_SAVED_SETTINGS)
        imgui.begin("##toolbar", flags=flags)
        dl = imgui.get_window_draw_list()

        for label, name, color, icon_fn in tools:
            wx, wy = imgui.get_cursor_screen_pos()
            is_active = self.active_tool == name

            if is_active:
                draw_styled_rect_active(dl, wx, wy, btn_sz, btn_sz,
                                        color, rounding=BUTTON_ROUNDING)
            else:
                draw_styled_rect(dl, wx, wy, btn_sz, btn_sz, OP1_DIM,
                                 rounding=BUTTON_ROUNDING, thickness=1.0)

            icon_fn(dl,
                    wx + btn_sz * 0.5,
                    wy + btn_sz * 0.38,
                    btn_sz * 0.25,
                    col32(color if is_active else OP1_GRAY))

            lbl_w = imgui.calc_text_size(label)[0]
            dl.add_text(wx + (btn_sz - lbl_w) * 0.5,
                        wy + btn_sz - th * 1.05,
                        col32(color if is_active else OP1_GRAY), label)

            imgui.invisible_button(f"##tool_{name}", btn_sz, btn_sz)
            if imgui.is_item_clicked(0):
                self._toggle_tool(name)
            if imgui.is_item_hovered():
                tip = _TOOL_TOOLTIPS.get(name, "")
                if tip:
                    imgui.set_tooltip(tip)
            imgui.same_line()

        imgui.end()
        imgui.pop_style_var(3)
        imgui.pop_style_color(1)

    def _draw_gnomon_hud(self):
        """Compact world-axis gnomon overlaid on the viewport bottom-left.

        Inspired by the OP-1 wireframe sphere: a faint circle for the sphere
        silhouette, a center dot, and three short colored lines with small
        dot caps for X/Y/Z. No heavy backdrop — just a translucent circle
        tint so the lines are legible over the point cloud.

        Visible in LIGHT TABLE and HOLOGRAM — both render a 3D scene that
        benefits from anatomical R/S/A orientation. In HOLOGRAM the world
        frame is the reference vertebra's local anatomical frame (since
        the reference renders at identity), so the gnomon's R/S/A labels
        correctly indicate the reference's right / superior / anterior
        directions.
        """
        if not self.gui_visible:
            return
        if self.mode != MODE_LIGHT_TABLE:
            return
        import imgui
        from src.gui.scale import s
        from src.gui.theme import OP1_RED, OP1_GREEN, OP1_BLUE, col32

        dl = imgui.get_foreground_draw_list()
        th = imgui.get_text_line_height()

        # --- Placement — bottom-left of viewport ----------------------
        size = s(72)          # compact; was 150
        pad = s(16)
        sw = self._left_chrome_width()
        cx = sw + pad + size * 0.5
        cy = self.height - pad - size * 0.5

        # --- Sphere silhouette (faint filled disc + ring) -------------
        sphere_r = size * 0.5
        dl.add_circle_filled(cx, cy, sphere_r,
                             imgui.get_color_u32_rgba(0.04, 0.04, 0.04, 0.55),
                             40)
        dl.add_circle(cx, cy, sphere_r,
                      imgui.get_color_u32_rgba(0.28, 0.28, 0.28, 0.70),
                      40, s(0.8))
        # Equator line (horizontal guide)
        dl.add_line(cx - sphere_r, cy, cx + sphere_r, cy,
                    imgui.get_color_u32_rgba(0.22, 0.22, 0.22, 0.50), s(0.7))

        # --- Project world axes ---------------------------------------
        view = self.camera.get_view_matrix()
        R = np.asarray(view[:3, :3], dtype=np.float32)

        arm = size * 0.40          # shorter arms
        stem_thick = s(1.4)        # thinner stems
        dot_r = s(2.5)             # tip dot radius
        label_offset = s(6)

        axes = [
            (np.array([1.0, 0.0, 0.0], dtype=np.float32), 'R', OP1_RED),
            (np.array([0.0, 1.0, 0.0], dtype=np.float32), 'S', OP1_GREEN),
            (np.array([0.0, 0.0, 1.0], dtype=np.float32), 'A', OP1_BLUE),
        ]

        entries = []
        for axis_vec, label, color in axes:
            local = R @ axis_vec
            sx = float(local[0])
            sy = -float(local[1])
            sz = float(local[2])
            entries.append((sz, sx, sy, label, color))

        entries.sort(key=lambda e: -e[0])

        for sz, sx, sy, label, color in entries:
            forward = max(0.0, min(1.0, (-sz + 1.0) * 0.5))
            front_alpha = 0.45 + 0.55 * forward
            color_front = (color[0], color[1], color[2], front_alpha)

            ax = cx + sx * arm
            ay = cy + sy * arm
            dl.add_line(cx, cy, ax, ay, col32(color_front), stem_thick)
            dl.add_circle_filled(ax, ay, dot_r, col32(color_front), 10)

            # Tiny axis label
            mag2 = sx * sx + sy * sy
            if mag2 > 1e-6:
                inv = 1.0 / math.sqrt(mag2)
                lbl_cx = ax + sx * inv * label_offset
                lbl_cy = ay + sy * inv * label_offset
            else:
                lbl_cx = ax
                lbl_cy = ay - label_offset
            lbl_w = imgui.calc_text_size(label)[0]
            dl.add_text(lbl_cx - lbl_w * 0.5, lbl_cy - th * 0.5,
                        col32(color_front), label)

        # --- Center dot -----------------------------------------------
        dl.add_circle_filled(cx, cy, s(2.5),
                             imgui.get_color_u32_rgba(0.85, 0.85, 0.85, 0.90),
                             10)

    def _draw_empty_splash(self):
        """Small grey space-invader + prompt for the empty gallery state.

        Matches the camera-panel alien style — same op1_alien renderer,
        same neutral grey, just a bit larger so it reads as a centered
        focal element without dominating the viewport.
        """
        import imgui
        from src.gui.scale import s
        from src.gui.theme import OP1_GRAY, col32
        from src.gui.op1_widgets import op1_alien

        dl = imgui.get_foreground_draw_list()
        th = imgui.get_text_line_height()

        sw = self._left_chrome_width()
        menu_h = int(getattr(self, '_menu_bar_height', 0))
        area_x = sw
        area_y = menu_h
        area_w = max(self.width - sw, 1)
        area_h = max(self.height - menu_h, 1)

        cx = area_x + area_w * 0.5
        cy = area_y + area_h * 0.5

        # Slightly larger than the camera-panel mini-aliens (th*0.13),
        # but still small and understated. Alien is 11x8 pixels.
        pixel = max(s(2.0), th * 0.22)
        alien_h = pixel * 8

        op1_alien(dl, cx, cy, pixel, col32(OP1_GRAY),
                  angle=0.0, profile='front')

        # Prompt centered below the alien.
        prompt = "drop a point cloud or point 3photon at a directory"
        prompt_w = imgui.calc_text_size(prompt)[0]
        prompt_y = cy + alien_h * 0.5 + s(14)
        dl.add_text(cx - prompt_w * 0.5, prompt_y,
                    col32(OP1_GRAY), prompt)

    def _draw_tool_overlay(self):
        """Draw rubber-band rectangle or lasso path for active drag."""
        import imgui
        from src.gui.scale import s
        dl = imgui.get_foreground_draw_list()

        # Always draw committed measurements (registry) in modes that
        # show the measurement toolbar. HOLOGRAM measurements live in
        # the same registry as LIGHT TABLE ones; anchors are stored in
        # world space, so the on-screen position is correct as long as
        # the user is in the same mode (and reference frame, for
        # HOLOGRAM) the anchors were placed in.
        if self.mode == MODE_LIGHT_TABLE:
            self._draw_measure_registry_overlay(dl, s)

        if (self.mode != MODE_LIGHT_TABLE
                or self.active_tool is None):
            return
        orange = imgui.get_color_u32_rgba(0.95, 0.55, 0.15, 1.0)
        fill = imgui.get_color_u32_rgba(0.95, 0.55, 0.15, 0.15)

        if self.active_tool == 'box' and self._drag_start is not None and self._drag_current is not None:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_current
            dl.add_rect_filled(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), fill)
            dl.add_rect(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1),
                        orange, 0.0, 0, s(2.0))
        elif self.active_tool in ('lasso', 'curve') and len(self._lasso_path) >= 2:
            curve_color = imgui.get_color_u32_rgba(0.6, 0.5, 0.9, 1.0) if self.active_tool == 'curve' else orange
            # Curve line width matches the pixel threshold so the user
            # sees the selection envelope as they draw. DPI-scaled so it
            # reads at the same physical width on a 5K display.
            width = s(self.curve_threshold_px) if self.active_tool == 'curve' else s(2.0)
            for i in range(len(self._lasso_path) - 1):
                x0, y0 = self._lasso_path[i]
                x1, y1 = self._lasso_path[i + 1]
                dl.add_line(x0, y0, x1, y1, curve_color, max(width, s(2.0)))
        elif self.active_tool == 'polygon' and self._lasso_path:
            # Polygon: discrete vertices. Draw committed segments,
            # vertex handles, and a "rubber band" preview line from the
            # last vertex to the current cursor so the user can see
            # where the next click will land. The first vertex gets a
            # slightly larger handle so the user can target it visually
            # if they want to close back onto it (the double-click
            # close gesture works anywhere — the visual is just for
            # spatial intuition).
            mx, my = imgui.get_mouse_pos()
            poly_col = imgui.get_color_u32_rgba(0.20, 0.85, 0.55, 1.0)
            preview_col = imgui.get_color_u32_rgba(0.20, 0.85, 0.55, 0.5)
            line_w = s(2.0)
            # committed segments
            for i in range(len(self._lasso_path) - 1):
                x0, y0 = self._lasso_path[i]
                x1, y1 = self._lasso_path[i + 1]
                dl.add_line(x0, y0, x1, y1, poly_col, line_w)
            # rubber-band preview line from last vertex to cursor
            lx, ly = self._lasso_path[-1]
            dl.add_line(lx, ly, mx, my, preview_col, line_w)
            # close-the-loop hint from cursor back to the first vertex
            # (fainter, so the user reads the rubber band as primary)
            if len(self._lasso_path) >= 3:
                fx, fy = self._lasso_path[0]
                dl.add_line(mx, my, fx, fy, preview_col, line_w * 0.6)
            # vertex handles
            handle_r = s(3.0)
            for i, (vx, vy) in enumerate(self._lasso_path):
                r = handle_r * (1.5 if i == 0 else 1.0)
                dl.add_circle_filled(vx, vy, r, poly_col, 8)
        elif self.active_tool == 'brush':
            # Preview circle for the brush radius at the current cursor
            # position. Radius is in world units; project it to pixels
            # using the camera's vertical field-of-view so the on-screen
            # ring matches the actual selection sphere at the cloud center.
            mx, my = imgui.get_mouse_pos()
            cam_dist = max(self.camera.distance, 1e-4)
            half_h_world = cam_dist * math.tan(self.camera.fov * 0.5)
            pixels_per_unit = self.height * 0.5 / max(half_h_world, 1e-6)
            radius_px = max(2.0, self.brush_radius * pixels_per_unit)
            ring = imgui.get_color_u32_rgba(0.95, 0.55, 0.15, 0.65)
            dl.add_circle(mx, my, radius_px, ring, 48, s(0.8))
            # Crosshair at the center so the exact point is clear.
            dl.add_line(mx - s(4), my, mx + s(4), my, ring, s(1.0))
            dl.add_line(mx, my - s(4), mx, my + s(4), ring, s(1.0))
        elif self.active_tool in ('measure_line', 'measure_angle', 'measure_landmark') and self._measure is not None:
            self._draw_measure_overlay(dl, s)

    def _draw_measure_overlay(self, dl, s):
        """Draw measurement lines, anchor dots, and value labels using ImGui draw list.

        All geometry is projected from world space → screen space so it
        sits on top of everything in the viewport.  Called only from
        _draw_tool_overlay when a measure tool is active.
        """
        import imgui

        measure = self._measure

        # --- unit label for this cloud ---
        unit = self.scene_unit

        # --- world → screen helper ---
        # Projection targets the primary viewport rect (not the full
        # framebuffer) so overlay dots land on the cloud's actual
        # on-screen position with the sidebar / menu bar offsets
        # accounted for. Batched matmul via project_points_batch
        # (see math_utils) when many anchors are in play.
        from src.utils.math_utils import project_points_batch
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp  = (proj @ view).astype(np.float32)
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()

        def w2s(pos):
            """Single-point projection into primary-viewport screen px,
            or None if behind / off-screen. The bulk path below may call
            project_points_batch directly; this helper keeps the existing
            single-shot call sites readable.
            """
            results = project_points_batch(mvp, [pos], vp_w, vp_h)
            r = results[0]
            if r is None:
                return None
            return (vp_x + r[0], vp_y + r[1])

        # --- colours ---
        r, g, b, a = measure.color
        col_line    = imgui.get_color_u32_rgba(r, g, b, a)
        col_preview = imgui.get_color_u32_rgba(r, g, b, 0.50)
        col_dot     = imgui.get_color_u32_rgba(r, g, b, 1.0)
        col_text    = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        col_bg      = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.72)

        dot_r  = s(5.0)
        line_w = s(2.2)

        # --- project anchors in a single batched matmul ---
        # Anchors are cloud-local — resolve to world via the current
        # scene state before projecting. project_points_batch can't
        # consume the resolver directly because it expects a flat
        # list of world points, so we resolve here and pass world.
        # Resolved-to-None anchors stay None in the screen list.
        resolve = self._build_measure_resolver()
        world_anchors = [resolve(a) for a in measure.anchors]
        _raw = project_points_batch(
            mvp,
            [p for p in world_anchors if p is not None],
            vp_w, vp_h,
        )
        _raw_iter = iter(_raw)
        screen_anchors = []
        for w in world_anchors:
            if w is None:
                screen_anchors.append(None)
                continue
            r = next(_raw_iter, None)
            screen_anchors.append(
                None if r is None else (vp_x + r[0], vp_y + r[1])
            )

        # --- magnetic snap spring ---
        # Advance the animated hover indicator one step toward the true snap
        # target.  The spring constant increases quadratically as the indicator
        # closes in, giving a magnetic "accelerating pull" feel.
        hover_sp = w2s(measure.hover_pos) if measure.hover_pos is not None else None
        cursor_sp = imgui.get_mouse_pos()   # raw cursor in screen pixels
        anim_sp = measure.tick_snap_anim(hover_sp, cursor_sp)

        # --- pop animations (single expanding ring on anchor click) ---
        now = time.perf_counter()
        _POP_DURATION   = 0.22
        _POP_MAX_RADIUS = s(14.0)
        _POP_LINE_W     = s(1.8)
        cutoff = now - _POP_DURATION
        measure.pop_events = [(pos, t) for pos, t in measure.pop_events if t > cutoff]
        for pop_pos, pop_t in measure.pop_events:
            pop_sp = w2s(pop_pos)
            if pop_sp is None:
                continue
            age      = now - pop_t
            progress = age / _POP_DURATION
            ease     = 1.0 - (1.0 - progress) ** 2   # ease-out expansion
            radius   = _POP_MAX_RADIUS * ease
            alpha    = (1.0 - progress) ** 1.4
            pop_col  = imgui.get_color_u32_rgba(r, g, b, alpha)
            dl.add_circle(pop_sp[0], pop_sp[1], max(radius, 1.0), pop_col, 32, _POP_LINE_W)

        # --- draw lines ---
        # Build draw-point list: projected anchors + animated hover endpoint
        draw_pts = list(screen_anchors)
        is_preview_last = False
        if not measure.complete and anim_sp is not None:
            draw_pts.append(anim_sp)
            is_preview_last = True

        for i in range(len(draw_pts) - 1):
            p0, p1 = draw_pts[i], draw_pts[i + 1]
            if p0 is None or p1 is None:
                continue
            use_preview = is_preview_last and i == len(draw_pts) - 2
            dl.add_line(p0[0], p0[1], p1[0], p1[1],
                        col_preview if use_preview else col_line, line_w)

        # --- helpers ---

        def _pop_scale(anchor_world):
            """Ease-out dot scale for the 180ms after an anchor is placed."""
            _DUR = 0.18
            for pop_pos, pop_t in measure.pop_events:
                if np.linalg.norm(pop_pos - anchor_world) < 1e-4:
                    age = now - pop_t
                    if age < _DUR:
                        t_n = age / _DUR
                        return 1.0 + (1.0 - t_n) ** 2 * 1.4
                    break
            return 1.0

        def _draw_dot(sp, world_pos):
            """Filled dot with pop scaling."""
            this_r = dot_r * _pop_scale(world_pos)
            dl.add_circle_filled(sp[0], sp[1], this_r + s(1.5),
                                 imgui.get_color_u32_rgba(0, 0, 0, 0.50), 16)
            dl.add_circle_filled(sp[0], sp[1], this_r, col_dot, 16)

        def _draw_arrowhead(tip, base, size):
            """Solid filled triangle arrowhead at tip, pointing away from base."""
            dx = tip[0] - base[0];  dy = tip[1] - base[1]
            ln = math.sqrt(dx*dx + dy*dy)
            if ln < 0.001:
                return
            ux, uy = dx/ln, dy/ln          # unit toward tip
            px, py = -uy, ux               # perpendicular
            bx = tip[0] - ux * size
            by = tip[1] - uy * size
            hw = size * 0.52
            dl.add_triangle_filled(
                tip[0], tip[1],
                bx + px*hw, by + py*hw,
                bx - px*hw, by - py*hw,
                col_line,
            )

        def _draw_arc(vertex, arm_a, arm_c, radius, col, steps=28):
            """Circular arc at vertex sweeping the interior angle between arm_a and arm_c."""
            ang_a = math.atan2(arm_a[1] - vertex[1], arm_a[0] - vertex[0])
            ang_c = math.atan2(arm_c[1] - vertex[1], arm_c[0] - vertex[0])
            diff  = ang_c - ang_a
            # Normalise to [-π, π] so we always sweep the shorter (interior) arc
            while diff >  math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            for i in range(steps):
                t0 = ang_a + diff * i       / steps
                t1 = ang_a + diff * (i + 1) / steps
                p0 = (vertex[0] + radius * math.cos(t0), vertex[1] + radius * math.sin(t0))
                p1 = (vertex[0] + radius * math.cos(t1), vertex[1] + radius * math.sin(t1))
                dl.add_line(p0[0], p0[1], p1[0], p1[1], col, s(1.3))

        def _bisector_sp(vertex, arm_a, arm_c, offset):
            """Screen point along the angle bisector from vertex, offset pixels out."""
            da = (arm_a[0]-vertex[0], arm_a[1]-vertex[1])
            dc = (arm_c[0]-vertex[0], arm_c[1]-vertex[1])
            la = math.sqrt(da[0]**2+da[1]**2)
            lc = math.sqrt(dc[0]**2+dc[1]**2)
            if la < 0.001 or lc < 0.001:
                return vertex
            da = (da[0]/la, da[1]/la)
            dc = (dc[0]/lc, dc[1]/lc)
            bx = da[0]+dc[0];  by = da[1]+dc[1]
            lb = math.sqrt(bx*bx+by*by)
            if lb < 0.001:           # antiparallel arms → use perpendicular
                bx, by, lb = -da[1], da[0], 1.0
            return (vertex[0]+bx/lb*offset, vertex[1]+by/lb*offset)

        def draw_label(sp, text):
            if sp is None:
                return
            tw, th = imgui.calc_text_size(text)
            pad = s(4.0)
            tx = sp[0] - tw * 0.5
            ty = sp[1] - th - s(12.0)
            dl.add_rect_filled(tx - pad, ty - pad * 0.6,
                               tx + tw + pad, ty + th + pad * 0.6,
                               col_bg, s(3.0))
            dl.add_text(tx, ty, col_text, text)

        # --- anchor dots / arrowheads ---
        # Angle tool: vertex (idx 1) stays a dot; arm tips (idx 0, 2) are arrows.
        # Line tool: all dots.
        is_angle = measure.mode == 'measure_angle'
        vertex_sp = screen_anchors[1] if (is_angle and len(screen_anchors) >= 2) else None

        for idx, sp in enumerate(screen_anchors):
            if sp is None:
                continue
            world = measure.anchors[idx]
            is_arm_tip = is_angle and idx in (0, 2)
            if is_arm_tip:
                # Arrowhead pointing away from the vertex
                base = vertex_sp if vertex_sp is not None else sp
                _draw_arrowhead(sp, base, s(10.0))
            else:
                _draw_dot(sp, world)

        # --- angle arc (drawn between committed arms or with hover arm) ---
        if is_angle and vertex_sp is not None:
            arc_col   = imgui.get_color_u32_rgba(r, g, b, 0.65)
            arc_r     = s(18.0)
            if measure.complete and len(screen_anchors) >= 3:
                a_sp = screen_anchors[0];  c_sp = screen_anchors[2]
                if a_sp and c_sp:
                    _draw_arc(vertex_sp, a_sp, c_sp, arc_r, arc_col)
            elif len(measure.anchors) == 2 and anim_sp is not None and screen_anchors[0] is not None:
                _draw_arc(vertex_sp, screen_anchors[0], anim_sp, arc_r,
                          imgui.get_color_u32_rgba(r, g, b, 0.35))

        # --- animated hover indicator ---
        if anim_sp is not None and not measure.complete:
            from src.core.tools.measure_tool import SNAP_PULL_RADIUS_PX
            lock = 0.0
            if hover_sp is not None:
                dx = anim_sp[0] - hover_sp[0];  dy = anim_sp[1] - hover_sp[1]
                lock = max(0.0, 1.0 - math.sqrt(dx*dx+dy*dy) / SNAP_PULL_RADIUS_PX)
            ch_r  = s(5.0 + lock * 4.0)
            col_h = imgui.get_color_u32_rgba(r, g, b, 0.55 + lock * 0.45)
            dl.add_circle_filled(anim_sp[0], anim_sp[1], dot_r * 0.85, col_h, 12)
            dl.add_line(anim_sp[0]-ch_r, anim_sp[1], anim_sp[0]+ch_r, anim_sp[1], col_h, s(1.4))
            dl.add_line(anim_sp[0], anim_sp[1]-ch_r, anim_sp[0], anim_sp[1]+ch_r, col_h, s(1.4))

        # --- measurement value ---
        # Active-session value helpers now take the resolver since
        # MeasureState anchors are cloud-local.
        if measure.mode == 'measure_line':
            if len(measure.anchors) == 1 and anim_sp is not None and screen_anchors[0] is not None:
                d = measure.get_distance_to_hover(resolve)
                if d is not None:
                    mid = ((screen_anchors[0][0]+anim_sp[0])*0.5,
                           (screen_anchors[0][1]+anim_sp[1])*0.5)
                    draw_label(mid, _fmt_distance(d, unit))
            elif measure.complete and len(screen_anchors) >= 2:
                d  = measure.get_distance(resolve)
                p0, p1 = screen_anchors[0], screen_anchors[1]
                if d is not None and p0 and p1:
                    draw_label(((p0[0]+p1[0])*0.5, (p0[1]+p1[1])*0.5), _fmt_distance(d, unit))

        elif is_angle and vertex_sp is not None:
            if len(measure.anchors) == 2 and anim_sp is not None and screen_anchors[0] is not None:
                ang = measure.get_angle_to_hover_deg(resolve)
                if ang is not None:
                    lsp = _bisector_sp(vertex_sp, screen_anchors[0], anim_sp, s(42.0))
                    draw_label(lsp, f"{ang:.1f}\u00b0")
            elif measure.complete and len(screen_anchors) >= 3:
                ang = measure.get_angle_deg(resolve)
                a_sp = screen_anchors[0];  c_sp = screen_anchors[2]
                if ang is not None and a_sp and c_sp:
                    lsp = _bisector_sp(vertex_sp, a_sp, c_sp, s(42.0))
                    draw_label(lsp, f"{ang:.1f}\u00b0")

    def _draw_measure_registry_overlay(self, dl, s_fn):
        """Draw all committed measurements from the registry (static, no spring).

        Uses the same screen-projection and drawing helpers as the active-
        session overlay but with committed anchor positions.  Selected items
        are drawn at full opacity; others at 75%.

        Called in LIGHT TABLE and HOLOGRAM so measurements persist while
        the user uses other tools or orbits. Anchors are cloud-local
        (see ``src/core/measure_registry.py::Anchor``); we resolve them
        through the current scene state each frame so HOLOGRAM
        reference-vertebra switches re-anchor the measurements onto
        their bones automatically.
        """
        import imgui

        if not self.measure_registry.items:
            return
        if self.mode != MODE_LIGHT_TABLE:
            return

        from src.utils.math_utils import project_points_batch
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp  = (proj @ view).astype(np.float32)
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()

        unit = self.scene_unit
        resolve = self._build_measure_resolver()

        sel = getattr(self, 'measure_selection', set())
        drag_id  = getattr(self, '_measure_drag_item_id', None)
        drag_idx = getattr(self, '_measure_drag_anchor_idx', -1)

        def _draw_arrowhead(tip, base, size, col):
            dx = tip[0] - base[0];  dy = tip[1] - base[1]
            ln = math.sqrt(dx * dx + dy * dy)
            if ln < 0.001:
                return
            ux, uy = dx / ln, dy / ln
            px, py = -uy, ux
            bx = tip[0] - ux * size;  by = tip[1] - uy * size
            hw = size * 0.52
            dl.add_triangle_filled(tip[0], tip[1],
                                   bx + px * hw, by + py * hw,
                                   bx - px * hw, by - py * hw, col)

        def _draw_arc(vertex, arm_a, arm_c, radius, col, steps=24):
            ang_a = math.atan2(arm_a[1] - vertex[1], arm_a[0] - vertex[0])
            ang_c = math.atan2(arm_c[1] - vertex[1], arm_c[0] - vertex[0])
            diff  = ang_c - ang_a
            while diff >  math.pi: diff -= 2 * math.pi
            while diff < -math.pi: diff += 2 * math.pi
            for i in range(steps):
                t0 = ang_a + diff * i       / steps
                t1 = ang_a + diff * (i + 1) / steps
                p0 = (vertex[0] + radius * math.cos(t0), vertex[1] + radius * math.sin(t0))
                p1 = (vertex[0] + radius * math.cos(t1), vertex[1] + radius * math.sin(t1))
                dl.add_line(p0[0], p0[1], p1[0], p1[1], col, s_fn(1.2))

        def _bisector_sp(vertex, arm_a, arm_c, offset):
            da = (arm_a[0] - vertex[0], arm_a[1] - vertex[1])
            dc = (arm_c[0] - vertex[0], arm_c[1] - vertex[1])
            la = math.sqrt(da[0] ** 2 + da[1] ** 2)
            lc = math.sqrt(dc[0] ** 2 + dc[1] ** 2)
            if la < 0.001 or lc < 0.001:
                return vertex
            da = (da[0] / la, da[1] / la)
            dc = (dc[0] / lc, dc[1] / lc)
            bx = da[0] + dc[0];  by = da[1] + dc[1]
            lb = math.sqrt(bx * bx + by * by)
            if lb < 0.001:
                bx, by, lb = -da[1], da[0], 1.0
            return (vertex[0] + bx / lb * offset, vertex[1] + by / lb * offset)

        def draw_label(sp, text, col_bg, col_text):
            if sp is None:
                return
            tw, th = imgui.calc_text_size(text)
            pad = s_fn(4.0)
            tx = sp[0] - tw * 0.5
            ty = sp[1] - th - s_fn(12.0)
            dl.add_rect_filled(tx - pad, ty - pad * 0.6,
                               tx + tw + pad, ty + th + pad * 0.6,
                               col_bg, s_fn(3.0))
            dl.add_text(tx, ty, col_text, text)

        for item in self.measure_registry.items:
            is_sel   = item.id in sel
            is_drag  = (item.id == drag_id)
            alpha    = 1.0 if (is_sel or is_drag) else 0.60

            r, g, b, _ = item.color
            col_line = imgui.get_color_u32_rgba(r, g, b, alpha)
            col_dot  = imgui.get_color_u32_rgba(r, g, b, alpha)
            col_bg   = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.70 * alpha)
            col_text = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, alpha)
            col_arc  = imgui.get_color_u32_rgba(r, g, b, 0.60 * alpha)

            dot_r  = s_fn(5.0)
            line_w = s_fn(2.0)

            # Resolve cloud-local anchors to world via the current
            # scene-state resolver. Anchors whose cloud is no longer
            # in scene resolve to None and become None in ``sps`` —
            # the existing dot/line drawing already skips None entries.
            def _project(world_pt):
                if world_pt is None:
                    return None
                p = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0],
                             dtype=np.float32)
                clip = mvp @ p
                if abs(clip[3]) < 1e-8:
                    return None
                ndc = clip[:3] / clip[3]
                if ndc[2] < -1.0 or ndc[2] > 1.0:
                    return None
                sx = vp_x + (ndc[0] + 1.0) * 0.5 * vp_w
                sy = vp_y + (1.0 - ndc[1]) * 0.5 * vp_h
                return (sx, sy)
            sps = [_project(resolve(a)) for a in item.anchors]

            # Lines
            for i in range(len(sps) - 1):
                if sps[i] and sps[i + 1]:
                    dl.add_line(sps[i][0], sps[i][1],
                                sps[i + 1][0], sps[i + 1][1], col_line, line_w)

            is_angle    = (item.mode == 'measure_angle')
            is_landmark = (item.mode == 'measure_landmark')
            vertex_sp   = sps[1] if (is_angle and len(sps) >= 2) else None

            # Landmark: dot + crosshair + name label
            if is_landmark and len(sps) >= 1 and sps[0]:
                sp = sps[0]
                if is_drag and 0 == drag_idx:
                    dl.add_circle(sp[0], sp[1], dot_r + s_fn(3.5),
                                  imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.5),
                                  16, s_fn(1.5))
                # Outer ring
                dl.add_circle(sp[0], sp[1], dot_r + s_fn(3.0), col_dot, 16, s_fn(1.3))
                # Filled center
                dl.add_circle_filled(sp[0], sp[1], dot_r, col_dot, 12)
                # Crosshair
                ch = s_fn(9.0)
                g_ch = s_fn(4.0)
                dl.add_line(sp[0] - ch, sp[1], sp[0] - g_ch, sp[1], col_dot, s_fn(1.2))
                dl.add_line(sp[0] + g_ch, sp[1], sp[0] + ch, sp[1], col_dot, s_fn(1.2))
                dl.add_line(sp[0], sp[1] - ch, sp[0], sp[1] - g_ch, col_dot, s_fn(1.2))
                dl.add_line(sp[0], sp[1] + g_ch, sp[0], sp[1] + ch, col_dot, s_fn(1.2))
                # Name label
                draw_label(sp, item.name, col_bg, col_text)
                continue   # skip the generic drawing below

            # Dots / arrowheads
            for idx, sp in enumerate(sps):
                if sp is None:
                    continue
                is_arm = is_angle and idx in (0, 2)
                if is_arm:
                    base = vertex_sp if vertex_sp else sp
                    _draw_arrowhead(sp, base, s_fn(9.0), col_dot)
                else:
                    # Highlight drag anchor with a larger ring
                    if is_drag and idx == drag_idx:
                        dl.add_circle(sp[0], sp[1], dot_r + s_fn(3.5),
                                      imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.5),
                                      16, s_fn(1.5))
                    dl.add_circle_filled(sp[0], sp[1], dot_r + s_fn(1.5),
                                         imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.45), 16)
                    dl.add_circle_filled(sp[0], sp[1], dot_r, col_dot, 16)

            # Arc (angle)
            if is_angle and vertex_sp and len(sps) >= 3 and sps[0] and sps[2]:
                _draw_arc(vertex_sp, sps[0], sps[2], s_fn(18.0), col_arc)

            # Value label \u2014 both modes need the resolver since
            # ``get_value`` walks the anchors and looks each up.
            if item.mode == 'measure_line' and len(sps) >= 2 and sps[0] and sps[1]:
                v = item.get_value(resolve)
                if v is not None:
                    mid = ((sps[0][0] + sps[1][0]) * 0.5,
                           (sps[0][1] + sps[1][1]) * 0.5)
                    draw_label(mid, _fmt_distance(v, unit), col_bg, col_text)
            elif is_angle and vertex_sp and len(sps) >= 3 and sps[0] and sps[2]:
                v = item.get_value(resolve)
                if v is not None:
                    lsp = _bisector_sp(vertex_sp, sps[0], sps[2], s_fn(42.0))
                    draw_label(lsp, f"{v:.1f}\u00b0", col_bg, col_text)

    def _viewport_rects(self) -> list[tuple[int, int, int, int]]:
        """Compute per-viewport OpenGL rects (x, y, w, h) for the current
        viewport_count. OpenGL convention: origin at bottom-left.

        Rects include a small gap between adjacent viewports so the dark
        outer background shows through as a separator line.

        Layout (viewport 0 is always the primary/interactive camera):
          1: full visible area
          2: primary right half, secondary (top) left half
          3: primary right full-height, two stacked quads on the left
             (top view = top-left, front view = bottom-left)
          4: 2x2 grid — primary top-right, top top-left,
             front bottom-left, right bottom-right
        """
        from src.gui.scale import s as _s
        vc = max(1, min(4, int(getattr(self, 'viewport_count', 1))))
        sw = self._left_chrome_width()
        rw = self._right_chrome_width()
        menu_h = int(getattr(self, '_menu_bar_height', 0))
        ax = int(sw)
        ay = 0
        aw = max(1, self.width - ax - int(rw))
        ah = max(1, self.height - menu_h)

        if vc == 1:
            return [(ax, ay, aw, ah)]

        # Gap between adjacent viewports (dark separator line)
        gap = max(2, int(_s(4)))
        g = gap // 2  # each viewport shrinks by half the gap on a shared edge

        hw = aw // 2
        hh = ah // 2

        if vc == 2:
            return [
                # primary: right half, left edge pulled right by g
                (ax + hw + g, ay, aw - hw - g, ah),
                # top: left half, right edge pulled left by g
                (ax, ay, hw - g, ah),
            ]
        if vc == 3:
            return [
                # primary: right full-height, left edge pulled right
                (ax + hw + g, ay, aw - hw - g, ah),
                # top: top-left quad — right edge + bottom edge pulled in
                (ax, ay + hh + g, hw - g, ah - hh - g),
                # front: bottom-left quad — right edge + top edge pulled in
                (ax, ay, hw - g, hh - g),
            ]
        # vc == 4: 2x2 grid — every viewport shrinks on two shared edges
        return [
            # primary: top-right
            (ax + hw + g, ay + hh + g, aw - hw - g, ah - hh - g),
            # top: top-left
            (ax, ay + hh + g, hw - g, ah - hh - g),
            # front: bottom-left
            (ax, ay, hw - g, hh - g),
            # right: bottom-right
            (ax + hw + g, ay, aw - hw - g, hh - g),
        ]

    def _primary_viewport_screen_rect(self) -> tuple[int, int, int, int]:
        """Primary viewport rect in ImGui/draw-list screen coords
        (top-left origin). Used by any code that projects world points
        to screen-pixel positions — measurement overlays, hit tests,
        crosshair placement — so projections land on the cloud's
        actual on-screen position instead of the full framebuffer
        centre. For single-viewport mode this is the visible area
        minus sidebar + menu bar; for multi-viewport it's whichever
        rect holds the primary camera.
        """
        rects = self._viewport_rects()
        vx, vy, vw, vh = rects[0] if rects else (0, 0, self.width, self.height)
        # OpenGL viewport is bottom-left origin; ImGui is top-left.
        vy_top = self.height - (vy + vh)
        return (vx, vy_top, vw, vh)

    def _sync_secondary_cameras(self):
        """Copy primary target + distance to secondary cameras so all
        viewports frame the same region. Each secondary keeps its own
        azimuth/elevation from its preset.
        """
        primary = self.camera
        for cam in self._secondary_cameras:
            cam.target = primary.target.copy()
            cam._target_goal = primary._target_goal.copy()
            cam.distance = primary.distance
            cam._distance_goal = primary._distance_goal
            cam.fov = primary.fov
            cam.near = primary.near
            cam.far = primary.far

    def _render_individual(self):
        """Render the selected cloud into 1..4 viewports. Viewport 0 is
        always the primary (interactive) camera; viewports 1..3 are
        fixed-preset spectator views sharing the primary's target +
        distance (set each frame via _sync_secondary_cameras)."""
        self.ctx.enable_only(
            moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
        )
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        rects = self._viewport_rects()
        vc = len(rects)
        if vc > 1:
            self._sync_secondary_cameras()

        # Camera per viewport: primary first, then top/front/right
        viewport_cams: list = [self.camera] + list(self._secondary_cameras)
        viewport_cams = viewport_cams[:vc]

        # Cache the clip-plane axis-enabled gate once per frame
        _ax_en = getattr(self, 'clip_axis_enabled', [True, True, True])
        _any_clip = any(
            _ax_en[a] and (self.clip_fraction_lo[a] > 1e-4
                           or self.clip_fraction_hi[a] > 1e-4)
            for a in range(3)
        )
        _show_clip = (self.clip_enabled and self.show_clip_planes
                      and _any_clip)
        if _show_clip:
            self.overlays.build_clip_planes(self)

        # Pre-resolve the entry once — it's the same across viewports
        entry = None
        gpu = None
        if 0 <= self.selected_index < len(self.entries):
            entry = self.entries[self.selected_index]
            gpu = entry.full_gpu or entry.preview_gpu

        dof_str = float(self.dof_strength) if self.dof_enabled else 0.0

        # Consistent viewport tone across single- and multi-viewport
        # modes. The visible 3D area is always painted with the sidebar-
        # matching grey (`pad_bg` — scene_rt 0.008 → sRGB ~0.058 after
        # ACES + gamma). In multi-viewport mode, each viewport rect
        # then clears to the deeper bg_color so point clouds sit on a
        # darker field and the padding color shows through in the gaps.
        pad_bg = (0.008, 0.008, 0.008)
        bg = self.bg_color
        vp_bg = (float(bg[0]), float(bg[1]), float(bg[2]))

        sw = self._left_chrome_width()
        menu_h = int(getattr(self, '_menu_bar_height', 0))
        pad_x = int(sw)
        pad_y = 0
        pad_w = max(1, self.width - pad_x)
        pad_h = max(1, self.height - menu_h)
        self.ctx.scissor = (pad_x, pad_y, pad_w, pad_h)
        self.ctx.clear(pad_bg[0], pad_bg[1], pad_bg[2], 1.0)

        # Single-panel mode: keep the whole visible area at pad_bg (no
        # per-viewport recolour below). Multi-panel: each rect will
        # override with the darker viewport bg.
        if vc == 1:
            vp_bg = None

        # Shared point-cloud uniforms — brightness / contrast / saturation /
        # RGB gain / depth falloff / label blend / textures / DOF / clip box
        # are identical across every viewport (secondaries sync distance +
        # fov from primary). Write them once here instead of ~13 uniform
        # writes per draw_cloud call × vc viewports. Per-viewport draw_cloud
        # calls pass skip_shared=True and only touch model / view / proj /
        # point_size.
        if gpu is not None:
            set_shared_point_uniforms(
                self.program,
                brightness=self.brightness, contrast=self.contrast,
                saturation=self.saturation,
                r_gain=self.r_gain, g_gain=self.g_gain, b_gain=self.b_gain,
                depth_falloff=self.depth_falloff,
                label_blend=self.label_blend,
                label_texture=self.label_texture,
                falloff_texture=self.falloff_texture,
                dof_strength=dof_str,
                # Focal plane anchored to the 3D cursor's world position,
                # not the camera's orbit distance. Pan or orbit, focus
                # stays on the user's chosen anchor.
                dof_focus_dist=self._cursor_focal_distance(),
                clip_min=self.clip_min, clip_max=self.clip_max,
            )

        # Mesh display path. HOLOGRAM used to force 'mesh' (clinician-
        # facing surface view), but the Poisson mesh occludes the
        # primitive overlays (plane disks, axis arrows) drawn inside the
        # vertebra. We now respect ``display_mode`` everywhere so the
        # measurements read clearly against the see-through point cloud.
        # Switch to mesh/both via the display-mode toggle if you want
        # the surface look back.
        effective_mode = self.display_mode
        want_mesh = effective_mode in ("mesh", "both")
        if want_mesh and entry is not None and self.mesh_renderer is not None:
            self._ensure_mesh_gpu(entry)
            from src.rendering.mesh_renderer import set_shared_mesh_uniforms
            set_shared_mesh_uniforms(
                self.mesh_renderer,
                brightness=self.brightness, contrast=self.contrast,
                saturation=self.saturation,
                r_gain=self.r_gain, g_gain=self.g_gain, b_gain=self.b_gain,
                label_blend=self.label_blend,
                label_texture=self.label_texture,
                clip_min=self.clip_min, clip_max=self.clip_max,
            )

        for i, (rect, cam) in enumerate(zip(rects, viewport_cams)):
            vx, vy, vw, vh = rect
            if vw < 4 or vh < 4:
                continue

            self.ctx.viewport = (vx, vy, vw, vh)
            self.ctx.scissor = (vx, vy, vw, vh)
            # Always clear depth. In multi-viewport also clear color
            # with the darker viewport bg so each panel is visually
            # distinct from the padding around it.
            if vp_bg is not None:
                self.ctx.clear(vp_bg[0], vp_bg[1], vp_bg[2], 1.0, depth=1.0)
            else:
                self.ctx.clear(depth=1.0)

            cam.aspect = vw / max(vh, 1)
            view = cam.get_view_matrix()
            proj = cam.get_projection_matrix()
            mvp = (proj @ view).astype(np.float32)
            cam_pos = cam.get_eye_position()

            # Grid first (behind the cloud). Infinite adaptive — fine
            # lines close in, coarser far out, fades to the horizon.
            if self.show_grid:
                self.overlays.draw_grid_infinite(view, proj, cam_pos)

            # Draw the selected cloud. Shared uniforms (color grading,
            # textures, clip box, DOF) were written once before the loop
            # via set_shared_point_uniforms — this call only needs the
            # per-viewport model / view / projection / point_size.
            # Mesh + points cooperate cleanly: in 'both' mode the mesh
            # draws first (depth-tested as opaque triangles), then
            # points draw on top with their existing alpha-blend path.
            # Polygon offset on the mesh nudges it back so points sit
            # cleanly on the surface without z-fighting.
            mesh_drawn = False
            if (effective_mode in ("mesh", "both")
                    and entry is not None
                    and entry.mesh_gpu is not None
                    and self.mesh_renderer is not None):
                from src.rendering.mesh_renderer import draw_mesh
                # No HOLOGRAM selection concept after the SpineLab
                # removal; LIGHT TABLE has no per-cloud selection, so
                # mesh selection-highlight is always off here.
                is_sel = False
                draw_mesh(
                    entry.mesh_gpu, self.mesh_renderer,
                    entry.model_transform, view, proj,
                    selected=is_sel,
                )
                mesh_drawn = True

            if gpu is not None and effective_mode != "mesh":
                draw_cloud(
                    gpu, self.program, entry.model_transform, view, proj,
                    point_size=self.point_size,
                    skip_shared=True,
                )

            # Gizmo: render in every viewport so the pivot is visible
            # from all angles. Build once per frame (primary cam), draw
            # with each viewport's MVP so screen-space size is consistent.
            if entry is not None and self.gizmo.visible \
                    and entry.bounds_min is not None:
                center = (entry.bounds_min + entry.bounds_max) / 2.0
                t = entry.model_transform
                center_h = np.array([*center, 1.0], dtype=np.float32)
                center_transformed = (t @ center_h)[:3]
                self.gizmo.dimmed = self.active_tool is not None
                self.gizmo.build(center_transformed, cam.distance, cam.fov)
                self.gizmo.draw(view, proj)

            # Remaining overlays
            if self.show_bbox:
                self.overlays.draw_bbox(mvp)
            if _show_clip:
                self.overlays.draw_clip_planes(mvp, show_fill=True)

            # Selection depth-limit plane: primary viewport only (the
            # active selection tool operates against the primary camera).
            if (i == 0 and self.active_tool in ('box', 'lasso', 'curve')
                    and np.isfinite(self.selection_max_depth)):
                self._draw_depth_limit_plane(view, mvp)

            # 3D cursor: draw in every viewport so it reads as a shared
            # anchor point across panels. Pass DOF state so the cursor
            # blooms with the same circle-of-confusion as the points
            # when FOCUS is turned up.
            self.cursor3d.draw(
                view, proj, cam.distance, cam.fov,
                dof_strength=float(self.dof_strength) if self.dof_enabled else 0.0,
                dof_focus_dist=self._cursor_focal_distance(camera=cam),
                point_size=float(self.point_size),
            )


        # Restore full viewport + disable scissor for subsequent passes
        self.ctx.scissor = None
        self.ctx.viewport = (0, 0, self.width, self.height)


    # ------------------------------------------------------------------
    # Selection-outline mask render
    # ------------------------------------------------------------------

    def _render_selection_mask(self) -> bool:
        """No mode renders a selection-outline mask after the SpineLab
        (HOLOGRAM) removal. LIGHT TABLE's active cloud isn't a selection
        in the Blender sense — it's the single thing being viewed — so
        there is never a set to outline. Stub returns False so the
        caller skips the outline pass."""
        return False


    def _gallery_scroll_by(self, y_offset: float) -> None:
        """Plain wheel scroll: nudge the gallery vertically.

        Snap-to-target style — we don't smooth on a timer, we just
        clamp on the next frame in ``_clamp_gallery_scroll``.
        """
        # Step is one cell + margin per wheel notch — feels close to
        # Lightroom and Finder. Negative y_offset = scroll down (the
        # GLFW convention is wheel-up = positive y).
        from src.gui import gallery_layout as _gl
        step = float(self.gallery_cell_size + _gl.CELL_MARGIN)
        self._gallery_scroll_y -= float(y_offset) * step
        self._gallery_scroll_target_y = self._gallery_scroll_y
        # Final clamp happens in _render_gallery once content_h is known.

    def _gallery_zoom_cells(self, y_offset: float, window) -> None:
        """Ctrl+wheel: resize gallery cells, persist preference.

        Cursor-anchored: the entry the user is hovering over stays
        visually fixed across the resize. Bumps the cache cell size
        (drops old cached textures, lazily re-renders at new size as
        cells scroll into view) and writes the new size to prefs.json.
        """
        from src.gui import gallery_layout as _gl
        old_size = self.gallery_cell_size
        # Geometric step so each notch feels like the same proportional
        # change at any size; smaller-than-1.15 step because users tend
        # to spam the wheel and we want it to feel responsive but not
        # jumpy.
        factor = 1.10 if y_offset > 0 else (1.0 / 1.10)
        new_size = int(round(old_size * factor))
        # Always move at least one pixel so a slow-scroll doesn't get
        # rounded back to the same value.
        if y_offset > 0 and new_size <= old_size:
            new_size = old_size + 1
        if y_offset < 0 and new_size >= old_size:
            new_size = old_size - 1
        new_size = max(_gl.MIN_CELL_SIZE, min(_gl.MAX_CELL_SIZE, new_size))
        if new_size == old_size:
            return

        # Cursor-anchor: figure out which cell is under the cursor in
        # the OLD layout, then after resize set scroll_y so the same
        # cell sits at the same screen position. Bail to a simple ratio
        # rescale if there's no cursor / no entries.
        sw = self._left_chrome_width()
        menu_h = int(getattr(self, '_menu_bar_height', 0))
        area_w = max(self.width - sw, 1)
        ready = self._gallery_filter_ready_cached()

        try:
            mx, my = glfw.get_cursor_pos(window)
        except Exception:
            mx, my = 0.0, 0.0

        if ready and mx >= sw and my >= menu_h:
            # Old grid: which row is currently under the cursor?
            cols_old, _rows_old, _cell_old, _content_old = self._gallery_grid(
                len(ready), area_w
            )
            local_y_old = my - menu_h - _gl.CELL_MARGIN + self._gallery_scroll_y
            step_old = old_size + _gl.CELL_MARGIN
            row_old = max(0, int(local_y_old // step_old))
            row_offset_in_cell = local_y_old - row_old * step_old
        else:
            row_old = -1
            row_offset_in_cell = 0.0

        self.gallery_cell_size = new_size
        # Force grid cache invalidation so the next render uses new size
        self._gallery_grid_cache_key = ()
        self._gallery_elide_key = None

        # Persist immediately. Cheap (small JSON), and the user almost
        # certainly wants the new size to stick — no other place would
        # save it.
        try:
            from src.utils.prefs import update_prefs
            update_prefs({"gallery_cell_size": int(new_size)})
        except Exception as e:
            print(f"prefs save failed: {e}")

        if row_old >= 0:
            # Position scroll so the same row sits at the same y.
            step_new = new_size + _gl.CELL_MARGIN
            new_local_y = row_old * step_new + row_offset_in_cell * (step_new / step_old)
            self._gallery_scroll_y = max(
                0.0, new_local_y - (my - menu_h - _gl.CELL_MARGIN)
            )
            self._gallery_scroll_target_y = self._gallery_scroll_y

    def _handle_gallery_scrollbar_press(self, mx: float, my: float,
                                         area_h: int) -> bool:
        """Test mouse press against the scrollbar; start a drag if it hits.

        Returns True if the press was consumed by the scrollbar (caller
        should NOT also try to select a cell). The scrollbar geometry
        was stashed by the previous frame's overlay draw.
        """
        rx0, ry0, rx1, ry1 = self._gallery_scrollbar_rect
        if rx1 <= rx0 or ry1 <= ry0:
            return False
        if not (rx0 <= mx <= rx1 and ry0 <= my <= ry1):
            return False

        hx0, hy0, hx1, hy1 = self._gallery_scrollbar_handle
        if hy0 <= my <= hy1:
            # Grab the handle: remember the offset so the cursor stays
            # locked to the same point on the handle as the user drags.
            self._gallery_scrollbar_dragging = True
            self._gallery_scrollbar_drag_offset = my - hy0
        else:
            # Page jump: center the handle on the click position then
            # start a drag from there. Feels right for "click in the
            # gutter to scroll past one page".
            handle_h = max(1, hy1 - hy0)
            self._gallery_scrollbar_dragging = True
            self._gallery_scrollbar_drag_offset = handle_h * 0.5
            self._scrollbar_drag_to(my, area_h)
        return True

    def _scrollbar_drag_to(self, mouse_y: float, area_h: int) -> None:
        """Map a mouse Y while dragging the scrollbar to a new scroll_y."""
        rx0, ry0, rx1, ry1 = self._gallery_scrollbar_rect
        track_h = max(1, ry1 - ry0)
        hx0, hy0, hx1, hy1 = self._gallery_scrollbar_handle
        handle_h = max(1, hy1 - hy0)
        # New handle top in track-local coords
        local = mouse_y - ry0 - self._gallery_scrollbar_drag_offset
        local = max(0, min(track_h - handle_h, local))
        # Map handle position back to scroll_y
        max_handle = max(1, track_h - handle_h)
        frac = local / max_handle
        # We need content_h to convert frac → scroll_y. Recompute from
        # the current grid layout (cheap, memoised).
        sw = self._left_chrome_width()
        area_w = max(self.width - sw, 1)
        ready = self._gallery_filter_ready_cached()
        cols, _rows, _cell, content_h = self._gallery_grid(len(ready), area_w)
        max_scroll = max(0, content_h - area_h)
        self._gallery_scroll_y = frac * max_scroll
        self._gallery_scroll_target_y = self._gallery_scroll_y

    def _gallery_grid(self, ready_count: int, area_w: int) -> tuple:
        """Return ``(cols, rows, cell_size, content_h)`` for the current state.

        Memoised by ``(count, area_w, cell_size)`` so the layout calc
        runs once per change instead of once per frame × call site.
        Used by both render and overlay paths plus the click hit-test.
        """
        key = (int(ready_count), int(area_w), int(self.gallery_cell_size))
        if key == self._gallery_grid_cache_key:
            return self._gallery_grid_cache
        cols, rows, cell_w, _cell_h, content_h = gallery_layout.compute_grid(
            ready_count, area_w, 0, self.gallery_cell_size
        )
        result = (cols, rows, cell_w, content_h)
        self._gallery_grid_cache_key = key
        self._gallery_grid_cache = result
        return result

    def _gallery_max_scroll(self, content_h: int, area_h: int) -> float:
        return float(max(0, content_h - area_h))

    def _clamp_gallery_scroll(self, content_h: int, area_h: int) -> None:
        """Clamp scroll_y to valid range. Called every frame after layout."""
        max_y = self._gallery_max_scroll(content_h, area_h)
        if self._gallery_scroll_y > max_y:
            self._gallery_scroll_y = max_y
        if self._gallery_scroll_y < 0:
            self._gallery_scroll_y = 0.0
        # Sync target so wheel input doesn't keep accumulating beyond
        # bounds (would cause "snap-back" when reversed)
        if self._gallery_scroll_target_y > max_y:
            self._gallery_scroll_target_y = max_y
        if self._gallery_scroll_target_y < 0:
            self._gallery_scroll_target_y = 0.0

    def _render_gallery(self):
        """Render the Contact Sheets grid via the per-entry texture cache.

        Strategy
        --------
        1. Compute layout from a fixed user-controlled cell size, NOT
           ``ceil(sqrt(N))``. More entries → more rows, never smaller
           cells.
        2. Cull to the visible row range (post-scroll). Off-screen
           cells are never iterated.
        3. For each *visible* dirty cell, render the point cloud into
           that entry's persistent ``_CachedCell`` framebuffer. Budgeted
           by ``MAX_DIRTY_PER_FRAME`` so a fresh project populates
           gradually instead of stalling.
        4. Blit each visible cell's cached texture into the gallery
           scratch RT at the right cell viewport. Clean cells cost just
           one fullscreen-tri textured quad each.
        5. Composite the scratch RT into ``scene_rt`` once.

        Adding entries to the library appends to the bottom of the
        scrollable grid; existing tiles keep their cached textures.
        Resizing the window resizes only the scratch RT (cheap, no
        per-cell invalidation).
        """
        ready_pairs = self._gallery_filter_ready_cached()
        ready = [e for _, e in ready_pairs]
        if not ready:
            return

        sw = self._left_chrome_width()
        menu_h = int(getattr(self, '_menu_bar_height', 0))
        area_x = sw
        area_y = menu_h
        area_w = max(self.width - sw, 1)
        area_h = max(self.height - menu_h, 1)

        cols, rows, cell_size, content_h = self._gallery_grid(len(ready), area_w)
        if cols <= 0 or cell_size <= 0:
            return
        self._clamp_gallery_scroll(content_h, area_h)
        scroll_y = int(round(self._gallery_scroll_y))
        aspect = 1.0  # cells are square now

        cache = self.gallery_cache
        bg = tuple(self.bg_color)
        cache.ensure_target(area_w, area_h, bg)
        cache.set_cell_size(cell_size)

        global_key = (
            float(self.point_size), float(self.point_sharpness),
            float(self.depth_falloff),
            float(self.brightness), float(self.contrast), float(self.saturation),
            float(self.r_gain), float(self.g_gain), float(self.b_gain),
            float(self.label_blend),
        )
        cache.bump_global(global_key)
        cache.prune({getattr(e, 'file_key', None) or id(e) for e in ready})

        # Visible-row culling: we only ever touch entries whose row is
        # currently on screen (or partially on screen). This is the
        # primary perf win for large libraries.
        first_row, last_row = gallery_layout.visible_row_range(
            scroll_y, area_h, cell_size, rows
        )
        first_idx = first_row * cols
        last_idx = min(len(ready), last_row * cols)

        # --- Pass 0: lazy preview load + GPU upload for visible cells ----
        # Replaces the old "synchronously upload every entry in
        # set_active_view" path. Only visible-row cells whose preview_gpu
        # is still None get loaded from the cached .npz and pushed to the
        # GPU here, capped at MAX_LAZY_PREVIEW_UPLOADS_PER_FRAME so a fresh
        # 1000-entry view spreads work across frames instead of blocking
        # the main thread for seconds.
        upload_budget = MAX_LAZY_PREVIEW_UPLOADS_PER_FRAME
        for i in range(first_idx, last_idx):
            if upload_budget <= 0:
                break
            entry = ready[i]
            if entry.preview_gpu is not None:
                continue
            if self._ensure_preview_gpu(entry):
                upload_budget -= 1

        # --- Pass 1: re-render dirty visible cells -----------------------
        # Each cell renders into its OWN persistent _CachedCell.fbo
        # (not a shared atlas), so swapping cells is one FBO bind per
        # render — measured cheaper than the old cache-RT-with-viewports
        # approach for the visible cell counts we hit (typically <30).
        self.ctx.enable_only(
            moderngl.DEPTH_TEST | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND
        )
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # Shared uniforms — colour grading, DOF, clip box, label/falloff
        # textures — are identical for every visible gallery cell. Write
        # them to the program once; the per-cell draw_cloud calls pass
        # skip_shared=True so each re-rendered cell only touches the
        # model/view/projection/point_size triple.
        shared_uniforms_set = False

        dirty_budget = MAX_DIRTY_PER_FRAME
        for i in range(first_idx, last_idx):
            entry = ready[i]
            if entry.bounds_min is None or entry.bounds_max is None:
                continue
            gpu = entry.preview_gpu
            if not gpu:
                continue

            version = (
                float(entry.orbit_az), float(entry.orbit_el),
                float(entry.orbit_zoom),
                id(gpu), id(entry.model_transform),
                id(entry.bounds_min), id(entry.bounds_max),
                int(getattr(gpu.cloud_data, 'labels_version', 0)),
            )
            if not cache.is_dirty(entry, version):
                continue
            if dirty_budget <= 0:
                break
            dirty_budget -= 1

            if not shared_uniforms_set:
                set_shared_point_uniforms(
                    self.program,
                    brightness=self.brightness, contrast=self.contrast,
                    saturation=self.saturation,
                    r_gain=self.r_gain, g_gain=self.g_gain, b_gain=self.b_gain,
                    depth_falloff=self.depth_falloff,
                    label_blend=self.label_blend,
                    label_texture=self.label_texture,
                    falloff_texture=self.falloff_texture,
                    dof_strength=0.0, dof_focus_dist=1.0,
                    clip_min=None, clip_max=None,
                )
                shared_uniforms_set = True

            cell = cache.get_or_alloc(entry)
            cell.fbo.use()
            cbg = self.cell_bg_color
            cell.fbo.clear(cbg[0], cbg[1], cbg[2], 1.0, depth=1.0)
            self.ctx.viewport = (0, 0, cell_size, cell_size)

            view, proj = gallery_layout.fit_view_matrix(
                entry.bounds_min, entry.bounds_max, aspect,
                azimuth=entry.orbit_az,
                elevation=entry.orbit_el,
                zoom=entry.orbit_zoom,
            )
            draw_cloud(
                gpu, self.program, entry.model_transform, view, proj,
                point_size=max(self.point_size * 0.7, 1.0),
                skip_shared=True,
            )
            cache.mark_rendered(entry, version)

        # --- Pass 2: blit visible cached cells into the scratch RT ------
        # The scratch RT is gallery-area sized; each visible entry's
        # cached texture is blitted into its post-scroll cell rect via
        # a viewport-restricted fullscreen pass. Cells outside the
        # visible band (already culled above) contribute nothing.
        cache.begin_frame_visible()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        for i in range(first_idx, last_idx):
            entry = ready[i]
            tex = cache.cached_texture(entry)
            if tex is None:
                continue
            gl_x, gl_y, w, h = gallery_layout.cell_viewport_local(
                i, cols, cell_size, cell_size, area_h, scroll_y
            )
            # Reject cells whose visible rect is fully clipped (defensive;
            # culling above should already have skipped them in normal
            # cases, but a fractional scroll can leave a sliver outside).
            if w <= 0 or h <= 0:
                continue
            if gl_x + w <= 0 or gl_y + h <= 0:
                continue
            if gl_x >= area_w or gl_y >= area_h:
                continue
            self.ctx.viewport = (gl_x, gl_y, w, h)
            self.pp_gallery_blit.render(textures={0: tex})

        # --- Pass 3: composite the scratch RT into scene_rt -------------
        self.scene_rt.use()
        scene_gl_y = self.height - area_y - area_h
        self.ctx.viewport = (area_x, scene_gl_y, area_w, area_h)
        self.pp_gallery_blit.render(textures={0: cache.rt.color})

        # Restore default state for downstream passes.
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.ctx.viewport = (0, 0, self.width, self.height)

    def _draw_gallery_overlay(self):
        """Draw cell borders, filename labels, scrollbar, and empty-state hint.

        Iterates only the visible row range (matches the render-time
        culling) so a 5000-cell library still costs the same per-frame
        as a 30-cell visible window.
        """
        import imgui
        from src.gui.scale import s

        sw = self._left_chrome_width()
        menu_h = int(getattr(self, '_menu_bar_height', 0))
        area_x = sw
        area_y = menu_h
        area_w = max(self.width - sw, 1)
        area_h = max(self.height - menu_h, 1)

        ready_pairs = self._gallery_filter_ready_cached()

        # Empty state — small grey alien centered in the viewport as a
        # subtle placeholder. Import controls live in the sidebar.
        if not ready_pairs:
            self._draw_empty_splash()
            return

        cols, rows, cell_w, content_h = self._gallery_grid(len(ready_pairs), area_w)
        cell_h = cell_w
        if cols <= 0:
            return
        scroll_y = int(round(self._gallery_scroll_y))
        first_row, last_row = gallery_layout.visible_row_range(
            scroll_y, area_h, cell_h, rows
        )
        first_idx = first_row * cols
        last_idx = min(len(ready_pairs), last_row * cols)

        dl = imgui.get_foreground_draw_list()
        th = imgui.get_text_line_height()

        # Elided-name cache keyed by cell_w so the overlay doesn't pay
        # O(visible_cells) text measurement every frame at the same size.
        elide_cache = getattr(self, '_gallery_elide_cache', None)
        elide_key = (cell_w,)
        if elide_cache is None or getattr(self, '_gallery_elide_key', None) != elide_key:
            elide_cache = {}
            self._gallery_elide_cache = elide_cache
            self._gallery_elide_key = elide_key

        # No outlines at all — selection state lives entirely in the
        # bottom label strip color. Strips are FULLY opaque so the
        # point cloud above never bleeds through and makes the tile
        # look translucent.
        name_bg_idle = imgui.get_color_u32_rgba(0.05, 0.05, 0.05, 1.0)
        name_bg_selected = imgui.get_color_u32_rgba(
            0.976, 0.573, 0.141, 1.0)    # OP1_ORANGE
        name_bg_multi = imgui.get_color_u32_rgba(
            0.99, 0.70, 0.20, 1.0)       # AMBER
        name_fg = imgui.get_color_u32_rgba(0.98, 1.0, 1.0, 1.0)
        name_fg_on_accent = imgui.get_color_u32_rgba(0.05, 0.05, 0.05, 1.0)
        idx_fg = imgui.get_color_u32_rgba(0.48, 0.55, 1.0, 1.0)

        # Clip to the gallery viewport so labels of partially-visible
        # cells get cut off at the top/bottom edge instead of bleeding
        # over the menu bar / sidebar.
        dl.push_clip_rect(area_x, area_y, area_x + area_w, area_y + area_h, True)

        for cell_i in range(first_idx, last_idx):
            entry_i, entry = ready_pairs[cell_i]
            x, y, w, h = gallery_layout.cell_rect(
                cell_i, cols, cell_w, cell_h, area_x, area_y, scroll_y
            )
            is_selected = (entry_i == self.selected_index)
            is_multi = (entry_i in self._gallery_multi_sel)

            # Bottom label strip — tinted for selection state in place
            # of a frame around the tile.
            if is_selected:
                strip_bg = name_bg_selected
                strip_fg = name_fg_on_accent
            elif is_multi:
                strip_bg = name_bg_multi
                strip_fg = name_fg_on_accent
            else:
                strip_bg = name_bg_idle
                strip_fg = name_fg
            label_h = th + s(6)
            dl.add_rect_filled(x, y + h - label_h, x + w, y + h, strip_bg)
            idx_str = f"{entry_i:02d}"
            idx_w = imgui.calc_text_size(idx_str)[0]
            idx_col = strip_fg if (is_selected or is_multi) else idx_fg
            dl.add_text(x + s(6), y + h - label_h + s(3), idx_col, idx_str)
            max_name_px = w - idx_w - s(18)
            if max_name_px > 10:
                cache_k = entry.name
                name = elide_cache.get(cache_k)
                if name is None:
                    name = entry.name
                    full_w = imgui.calc_text_size(name)[0]
                    if full_w > max_name_px and len(name) > 1:
                        keep = max(1, int(len(name) * (max_name_px / full_w)) - 1)
                        while keep > 1 and imgui.calc_text_size(name[:keep] + "…")[0] > max_name_px:
                            keep -= 1
                        name = name[:keep] + "…"
                    elide_cache[cache_k] = name
                dl.add_text(x + idx_w + s(12), y + h - label_h + s(3), strip_fg, name)

        dl.pop_clip_rect()

        # Vertical scrollbar — drawn last so it sits on top of cells.
        # Only when content is taller than the viewport.
        if content_h > area_h:
            self._draw_gallery_scrollbar(
                dl, area_x, area_y, area_w, area_h, content_h
            )

    def _draw_gallery_empty_state(self, area_x: int, area_y: int,
                                   area_w: int, area_h: int) -> None:
        """Centered hint for an empty Contact Sheets viewport."""
        import imgui
        from src.gui.scale import s
        dl = imgui.get_foreground_draw_list()
        msg_main = "Pick a project or folder in the sidebar"
        msg_sub = "to load its previews"
        c_main = imgui.get_color_u32_rgba(0.65, 0.68, 0.74, 1.0)
        c_sub = imgui.get_color_u32_rgba(0.45, 0.48, 0.54, 1.0)
        w_main = imgui.calc_text_size(msg_main)[0]
        w_sub = imgui.calc_text_size(msg_sub)[0]
        cx = area_x + area_w * 0.5
        cy = area_y + area_h * 0.5
        dl.add_text(cx - w_main * 0.5, cy - s(10), c_main, msg_main)
        dl.add_text(cx - w_sub * 0.5, cy + s(10), c_sub, msg_sub)

    def _draw_gallery_scrollbar(self, dl, area_x: int, area_y: int,
                                 area_w: int, area_h: int,
                                 content_h: int) -> None:
        """Draw a slim vertical scrollbar in the gallery's right gutter.

        Stores the geometry on the app so the mouse handler can hit-test
        and drag it. Same neutral grey as the rest of the chrome — no
        accent color, no hover halo, just a slim track + handle.
        """
        import imgui
        from src.gui.scale import s
        gutter_w = s(8)
        track_x = area_x + area_w - gutter_w - s(2)
        track_y = area_y + s(2)
        track_h = max(1, area_h - s(4))
        c_track = imgui.get_color_u32_rgba(0.10, 0.10, 0.12, 0.55)
        c_handle = imgui.get_color_u32_rgba(0.45, 0.48, 0.55, 0.85)
        dl.add_rect_filled(track_x, track_y,
                           track_x + gutter_w, track_y + track_h, c_track)

        if content_h <= 0:
            return
        ratio = min(1.0, area_h / content_h)
        handle_h = max(s(24), int(track_h * ratio))
        max_scroll = max(1, content_h - area_h)
        frac = self._gallery_scroll_y / max_scroll
        handle_y = track_y + int((track_h - handle_h) * frac)
        dl.add_rect_filled(track_x, handle_y,
                           track_x + gutter_w, handle_y + handle_h, c_handle)

        # Stash geometry for the mouse handler. Tracked separately from
        # the cell layout so we don't have to recompute it on every click.
        self._gallery_scrollbar_rect = (
            int(track_x), int(track_y),
            int(track_x + gutter_w), int(track_y + track_h),
        )
        self._gallery_scrollbar_handle = (
            int(track_x), int(handle_y),
            int(track_x + gutter_w), int(handle_y + handle_h),
        )

    def _apply_dark_title_bar(self):
        """Use Windows DWM API to set title bar to dark mode."""
        try:
            import ctypes
            hwnd = glfw.get_win32_window(self.window)
            DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE,
                ctypes.byref(ctypes.c_int(1)), 4
            )
            # Also set caption color to near-black (COLORREF: 0x00BBGGRR)
            DWMWA_CAPTION_COLOR = 35
            dark_color = ctypes.c_int(0x00101010)  # RGB(16,16,16)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR,
                ctypes.byref(dark_color), 4
            )
            # Set text color to light grey
            DWMWA_TEXT_COLOR = 36
            text_color = ctypes.c_int(0x00C0C0C0)  # RGB(192,192,192)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_TEXT_COLOR,
                ctypes.byref(text_color), 4
            )
        except Exception:
            pass  # Non-Windows or older Windows — skip silently

    def _set_window_icon(self):
        """Set the window icon from the assets directory.

        Provides multiple sizes (16, 32, 48, 256) so Windows picks the
        right resolution for the title bar (small) and taskbar (large).
        Also sets the Windows AppUserModelID so the taskbar groups this
        process under its own icon instead of inheriting python.exe's.
        """
        try:
            from PIL import Image
            icon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'assets', '3photon.png'
            )
            if getattr(sys, 'frozen', False):
                icon_path = os.path.join(sys._MEIPASS, 'assets', '3photon.png')
            if os.path.exists(icon_path):
                img = Image.open(icon_path).convert('RGBA')
                # Supply multiple sizes so the OS picks the best fit for
                # each context (title bar ≈16-32px, taskbar ≈48px, Alt-Tab ≈256px).
                sizes = [16, 32, 48, 256]
                icon_images = []
                for sz in sizes:
                    resized = img.resize((sz, sz), Image.LANCZOS)
                    icon_images.append((sz, sz, resized.tobytes()))
                glfw.set_window_icon(self.window, len(icon_images), icon_images)
        except Exception:
            pass

    def _update_fps(self):
        now = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._total_point_count = sum(e.point_count for e in self.entries)
            pts = self._total_point_count
            mode_names = {MODE_CONTACT_SHEETS: "Contact Sheets", MODE_LIGHT_TABLE: "Light Table", MODE_AUTOMATION: "Train"}
            mode_str = mode_names.get(self.mode, "View")
            pending = self.catalog.pending_count if self.catalog else 0
            pending_str = f" | {pending} loading" if pending > 0 else ""
            glfw.set_window_title(
                self.window,
                f"3 P H O T O N  —  {mode_str}  |  {self.fps:.0f} fps  |  {pts:,} pts{pending_str}"
            )
            self._fps_frame_count = 0
            self.last_fps_time = now

    def _cleanup(self):
        # Save view settings before shutdown
        try:
            self._save_view_prefs()
        except Exception:
            pass

        # Flush labels for clouds that have ACTUAL labels (non-zero) before
        # tearing anything down.
        #
        # LS-11: when a cloud is only loaded at preview resolution
        # (no full_gpu yet — Contact Sheets paint case), the
        # full-resolution save_cloud_labels would refuse the write
        # via the length-mismatch guard, silently dropping the
        # preview's labels on the floor at shutdown. Route those
        # entries through save_preview_labels so the preview-paint
        # work survives a restart.
        for entry in self.entries:
            try:
                file_key = getattr(entry, "file_key", None)
                if not file_key:
                    continue
                if entry.full_gpu is not None and entry.full_gpu.cloud_data is not None:
                    labels = entry.full_gpu.cloud_data.labels
                    if labels is not None and (labels != 0).any():
                        self._persist_cloud_labels(entry, entry.full_gpu.cloud_data)
                    # Also flush the preview labels if it's a distinct buffer.
                    if (entry.preview_gpu is not None
                            and entry.preview_gpu is not entry.full_gpu
                            and entry.preview_gpu.cloud_data is not None):
                        prev = entry.preview_gpu.cloud_data.labels
                        if prev is not None and (prev != 0).any():
                            save_preview_labels(file_key, prev)
                elif (entry.preview_gpu is not None
                      and entry.preview_gpu.cloud_data is not None):
                    # Preview-only — write through save_preview_labels.
                    prev = entry.preview_gpu.cloud_data.labels
                    if prev is not None and (prev != 0).any():
                        save_preview_labels(file_key, prev)
            except Exception:
                pass

        # Restore cursor in case brush tool left it hidden
        try:
            if self.window:
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        except Exception:
            pass

        # Stop the training subprocess first — we own that handle and GLFW
        # teardown doesn't cascade to it. Without this, closing 3Photon
        # mid-training orphans the subprocess and it keeps running in the
        # background holding file locks and GPU memory.
        if getattr(self, 'training_runner', None) is not None:
            try:
                self.training_runner.stop()
            except Exception as e:
                print(f"Training runner stop failed: {e}")
        self.gui.shutdown()
        self.overlays.release()
        self.gizmo.release()
        self.cursor3d.release()
        for entry in self.entries:
            entry.release()
        # Release GPU objects that were not covered by per-entry cleanup
        for obj_name in ('scene_rt', 'pp_tonemap',
                         'pp_gallery_blit', 'gallery_cache',
                         'ssao_rt', 'ssao_blur_rt',
                         'pp_ssao', 'pp_ssao_blur',
                         'ssao_noise_tex', 'ssao_disabled_tex',
                         'selection_mask_fbo', 'selection_mask_tex',
                         'pp_outline',
                         'label_texture', 'falloff_texture'):
            obj = getattr(self, obj_name, None)
            if obj is not None:
                try:
                    obj.release()
                except Exception:
                    pass
        if self.catalog:
            self.catalog.shutdown()
        self._full_res_executor.shutdown(wait=False, cancel_futures=True)
        # Release the catalog single-instance lock so the next launch
        # works immediately. Best-effort; never blocks shutdown.
        try:
            release_lock()
        except Exception as e:
            print(f"Catalog lock release failed: {e}")
        glfw.terminate()

    def _do_undo(self):
        """Undo the last label command on the selected cloud."""
        if not (0 <= self.selected_index < len(self.entries)):
            return
        entry = self.entries[self.selected_index]
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None:
            return
        cloud = gpu.cloud_data
        # Defensive try/except — a malformed undo command would otherwise
        # raise mid-imgui-frame and poison the frame state, which locks
        # the whole app in an assertion loop until the user kills it.
        # The undo stack itself now validates indices before applying,
        # but this catches anything that slips past.
        try:
            cmd = self.undo_stack.undo(cloud)
        except Exception as e:
            print(f"[undo] failed: {type(e).__name__}: {e}")
            return
        if cmd is not None:
            # Route through the shared post-mutation helper so the GPU
            # upload, count cache, sequence flag, AND catalog labels file
            # all get refreshed in one place. This is the same path the
            # paint tools take, so undo state is persisted automatically.
            self._after_label_mutation(gpu, cloud)
            print(f"Undo: {cmd.description}")

    def _do_redo(self):
        """Redo the next label command on the selected cloud."""
        if not (0 <= self.selected_index < len(self.entries)):
            return
        entry = self.entries[self.selected_index]
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None:
            return
        cloud = gpu.cloud_data
        try:
            cmd = self.undo_stack.redo(cloud)
        except Exception as e:
            print(f"[redo] failed: {type(e).__name__}: {e}")
            return
        if cmd is not None:
            self._after_label_mutation(gpu, cloud)
            print(f"Redo: {cmd.description}")

    def _current_gpu_cloud(self):
        """Return the GPUCloud currently being viewed (full or preview)."""
        if not (0 <= self.selected_index < len(self.entries)):
            return None
        entry = self.entries[self.selected_index]
        return entry.full_gpu or entry.preview_gpu

    def _sync_selection_buffer(self):
        """Ensure the selection buffer matches the current cloud size."""
        gpu = self._current_gpu_cloud()
        if gpu is not None:
            self.selection_buffer.resize(gpu.point_count)

    def _auto_scale_brush(self):
        """Set brush radius to ~3% of the selected cloud's extent."""
        if not (0 <= self.selected_index < len(self.entries)):
            return
        entry = self.entries[self.selected_index]
        if entry.bounds_min is not None and entry.bounds_max is not None:
            extent = float(np.linalg.norm(entry.bounds_max - entry.bounds_min))
            self.brush_radius = max(0.01, extent * 0.03)

    def _update_tool_cursor(self):
        """Set the OS mouse cursor based on the active selection tool.

        Safe to call every frame. The tool cursor (crosshair / hidden
        brush) only applies when the pointer is **over the 3D viewport
        in LIGHT_TABLE mode** — over the sidebar or any other tab the
        cursor reverts to the default arrow. Without this, switching
        tabs with brush active leaves the cursor hidden, and the
        crosshair bleeds into sidebar interactions.

        No-op while a sidebar drag is in progress — the resize handle
        owns the cursor during its drag.
        """
        if self.window is None or self._sidebar_drag_active is not None:
            return

        # Tool cursor only takes effect in LIGHT_TABLE and only when the
        # pointer isn't over an ImGui window (sidebar, popup, modal).
        gui = getattr(self, 'gui', None)
        over_viewport = (
            self.mode == MODE_LIGHT_TABLE
            and not (gui is not None and gui.wants_mouse())
        )

        if over_viewport and self.active_tool == 'brush':
            if glfw.get_input_mode(self.window, glfw.CURSOR) != glfw.CURSOR_HIDDEN:
                glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_HIDDEN)
            return

        # Anything else: ensure the OS cursor is visible.
        if glfw.get_input_mode(self.window, glfw.CURSOR) == glfw.CURSOR_HIDDEN:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)

        cursor = self._cursor_default
        if over_viewport and self.active_tool in ('pick', 'box', 'lasso', 'curve'):
            cursor = self._cursor_crosshair
        if cursor is not None:
            glfw.set_cursor(self.window, cursor)

    def _left_chrome_width(self) -> int:
        """Total left-edge chrome width in framebuffer pixels — just the
        main sidebar. The earlier "second-left popout column" was
        retired when HOLOGRAM moved its vertebra controls back into
        the main left sidebar and the data panels moved to the right
        sidebar (matching the Blender-style outliner+settings layout
        the user landed on).
        """
        if not self.gui_visible:
            return 0
        if self._left_chrome_frame == self.frame_count:
            return self._left_chrome_cached
        from src.gui.scale import sidebar_width_left
        # Prefer the sidebar's MEASURED framebuffer right edge (set by
        # panels.draw_sidebar) over the computed logical width — the two
        # diverge at fractional DPI (imgui logical units vs the GL gallery's
        # framebuffer viewport), which let the grid slide under the sidebar.
        # Falls back to the computed width on the first frame / when unset.
        measured = getattr(self, '_sidebar_right_fb', 0)
        self._left_chrome_cached = measured if measured > 0 else sidebar_width_left()
        self._left_chrome_frame = self.frame_count
        return self._left_chrome_cached


    def _right_chrome_width(self) -> int:
        """Right-edge chrome width in framebuffer pixels.

        HOLOGRAM reserves a right sidebar for the data panels
        (graphs on top, tables on bottom — the Blender-style layout
        the user moved to). Other tabs return 0 so their viewport
        spans full width on the right side.
        """
        return 0

    def _toggle_tool(self, name: str | None):
        """Toggle a tool on/off and wire dependent state.

        Tools are mutually exclusive (single active tool model). Picking
        MOVE / ROTATE shows the gizmo and binds its mode; picking any
        other tool hides the gizmo.
        """
        prev_tool = self.active_tool
        self.active_tool = None if self.active_tool == name else name
        # Switching tools while a polygon is in progress would otherwise
        # leave the partial path drawn on top of the next tool's UI and
        # commit it the next time a polygon click lands. Always clear
        # the path on a tool change.
        if prev_tool == 'polygon' or self.active_tool == 'polygon':
            self._lasso_path = []
        if self.active_tool == 'move':
            self.gizmo.visible = True
            self.gizmo.mode = 'translate'
            self.gizmo.dimmed = False
        elif self.active_tool == 'rotate':
            self.gizmo.visible = True
            self.gizmo.mode = 'rotate'
            self.gizmo.dimmed = False
        else:
            self.gizmo.visible = False
            self.gizmo.dimmed = False

        # Measure tools: create fresh session; any other tool clears it.
        if self.active_tool in ('measure_line', 'measure_angle', 'measure_landmark'):
            self._measure = MeasureState(self.active_tool)
        else:
            self._measure = None

        self._update_tool_cursor()
        print(f"Active tool: {self.active_tool}")

    def _commit_active_measurement(self) -> None:
        """Commit the completed active measurement to the registry.

        Called immediately after the last anchor is placed.  The active
        MeasureState is reset so the tool is ready for the next session.
        Auto-opens the measurement panel on the very first commit.
        """
        m = self._measure
        if m is None or not m.complete:
            return
        item = self.measure_registry.add_item(m.mode, m.anchors, m.color)
        first_ever = (len(self.measure_registry.items) == 1)
        if first_ever:
            self.measure_panel_open = True
        # Reset active session so the next click starts fresh.
        m.reset()

    def _measure_hit_test_anchors(self, mx: float, my: float,
                                   radius_px: float = 12.0
                                   ) -> tuple[str, int] | None:
        """Hit-test all committed anchors in screen space.

        Returns (item_id, anchor_idx) for the closest anchor within
        radius_px, or None if none are close enough. Anchors are
        cloud-local; we resolve them through the current scene state
        before screen projection so the hit-test follows bones through
        reference-vertebra switches.
        """
        if not self.measure_registry.items:
            return None
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp  = (proj @ view).astype(np.float32)
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()

        resolve = self._build_measure_resolver()

        best_dist  = radius_px
        best_result = None

        for item in self.measure_registry.items:
            for idx, anchor in enumerate(item.anchors):
                world = resolve(anchor)
                if world is None:
                    continue  # cloud no longer in scene
                p = np.array([world[0], world[1], world[2], 1.0], dtype=np.float32)
                clip = mvp @ p
                if abs(clip[3]) < 1e-8:
                    continue
                ndc = clip[:3] / clip[3]
                if ndc[2] < -1.0 or ndc[2] > 1.0:
                    continue
                sx = vp_x + (ndc[0] + 1.0) * 0.5 * vp_w
                sy = vp_y + (1.0 - ndc[1]) * 0.5 * vp_h
                d  = math.sqrt((sx - mx) ** 2 + (sy - my) ** 2)
                if d < best_dist:
                    best_dist   = d
                    best_result = (item.id, idx)

        return best_result

    def _drop_locked(self, indices: np.ndarray) -> np.ndarray:
        """Remove indices whose current label is locked. Logs once per call if any dropped."""
        gpu = self._current_gpu_cloud()
        if gpu is None or len(indices) == 0:
            return indices
        mask = locked_mask(gpu.cloud_data.labels, self.label_registry)
        if not mask.any():
            return indices
        keep = ~mask[indices]
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"Skipped {dropped:,} locked points")
        return indices[keep]

    def _drop_hidden(self, indices: np.ndarray) -> np.ndarray:
        """Remove indices whose current label is hidden (visible=False)."""
        gpu = self._current_gpu_cloud()
        if gpu is None or len(indices) == 0:
            return indices
        labels = gpu.cloud_data.labels
        if labels is None:
            return indices
        hidden_ids = {
            lid for lid, info in self.label_registry._labels.items()
            if not info.visible
        }
        if not hidden_ids:
            return indices
        pt_labels = labels[indices]
        keep = np.ones(len(indices), dtype=bool)
        for hid in hidden_ids:
            keep &= (pt_labels != hid)
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"Skipped {dropped:,} hidden-layer points")
        return indices[keep]

    # ---- Hidden-point selection (CPU z-buffer occlusion test) -----------
    #
    # The selection tools (box / lasso / curve / brush) used to grab
    # every point whose 2D projection landed inside the cursor region —
    # including points that were occluded by the visible front surface
    # of the cloud. For dense data like vertebrae this meant a single
    # box stroke could pick the front AND the back of the bone, ruining
    # the annotation. The visibility filter below builds a low-res CPU
    # z-buffer from ALL points in the current cloud, then drops any
    # candidate whose view-space depth is greater than the buffer at
    # its pixel (i.e. something else in the cloud is in front of it).

    _ZBUF_DOWNSCALE = 2  # 1 = full resolution, 2 = quarter px count

    def _build_visibility_zbuffer(
        self, gpu, mvp: np.ndarray, view_xform: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """Build a low-res min-depth buffer from every point in the cloud.

        Returns ``(zbuf, tolerance)`` or ``None`` if the cloud is empty
        or the framebuffer collapsed. ``zbuf`` is a 2D float32 array of
        view-space depths (positive = in front of camera, np.inf = no
        point landed there). ``tolerance`` is an absolute depth slack
        derived from the cloud's diagonal so the test scales across
        wildly different unit systems (mm CT scans vs metre lidar).
        """
        if gpu is None or gpu.cloud_data is None:
            return None
        positions = gpu.cloud_data.positions
        n = len(positions)
        if n == 0:
            return None
        scale = max(1, int(self._ZBUF_DOWNSCALE))
        bw = max(1, self.width // scale)
        bh = max(1, self.height // scale)
        zbuf = np.full((bh, bw), np.inf, dtype=np.float32)

        homo = np.empty((n, 4), dtype=np.float32)
        homo[:, :3] = positions
        homo[:, 3] = 1.0
        view_pos = homo @ view_xform.T
        view_z = -view_pos[:, 2]  # positive = in front of camera

        clip = homo @ mvp.T
        wclip = clip[:, 3]
        in_front = view_z > 1e-6
        valid = in_front & (np.abs(wclip) > 1e-8)
        if not valid.any():
            return zbuf, 0.0

        nx = clip[valid, 0] / wclip[valid]
        ny = clip[valid, 1] / wclip[valid]
        # Accept points slightly outside the viewport too — front-most
        # logic still works at the edges. Then clip below.
        sx = ((nx * 0.5 + 0.5) * bw).astype(np.int32)
        sy = ((1.0 - (ny * 0.5 + 0.5)) * bh).astype(np.int32)
        in_view = (sx >= 0) & (sx < bw) & (sy >= 0) & (sy < bh)
        if not in_view.any():
            return zbuf, 0.0

        sx = sx[in_view]
        sy = sy[in_view]
        zs = view_z[valid][in_view]

        # Per-pixel min reduction. np.minimum.at is the canonical
        # ufunc-with-collisions reduction; slower than a vectorized
        # approach but rock-solid for an O(N) write of 5M points.
        np.minimum.at(zbuf, (sy, sx), zs)

        diag = float(np.linalg.norm(gpu.bounds_max - gpu.bounds_min))
        tolerance = max(diag * 0.01, 1e-4)
        return zbuf, tolerance

    def _filter_visible(
        self,
        indices: np.ndarray,
        gpu,
        mvp: np.ndarray,
        view_xform: np.ndarray,
        zbuf_cache: tuple[np.ndarray, float] | None = None,
    ) -> np.ndarray:
        """Drop candidate indices that are occluded by the visible surface.

        Pass ``zbuf_cache`` to reuse a previously-built (zbuf, tolerance)
        — the brush stroke recorder builds it once at PRESS and reuses
        for every per-tick filter so a long drag isn't recomputing the
        whole cloud's depth grid 60 times a second.
        """
        if len(indices) == 0:
            return indices
        cache = zbuf_cache or self._build_visibility_zbuffer(gpu, mvp, view_xform)
        if cache is None:
            return indices
        zbuf, tolerance = cache

        positions = gpu.cloud_data.positions[indices]
        n = len(positions)
        homo = np.empty((n, 4), dtype=np.float32)
        homo[:, :3] = positions
        homo[:, 3] = 1.0
        view_pos = homo @ view_xform.T
        cz = -view_pos[:, 2]

        clip = homo @ mvp.T
        wclip = clip[:, 3]
        bh, bw = zbuf.shape
        # Map each candidate into the same downscaled buffer space.
        valid = (np.abs(wclip) > 1e-8) & (cz > 1e-6)
        sx = np.zeros(n, dtype=np.int32)
        sy = np.zeros(n, dtype=np.int32)
        sx[valid] = ((clip[valid, 0] / wclip[valid] * 0.5 + 0.5) * bw).astype(np.int32)
        sy[valid] = ((1.0 - (clip[valid, 1] / wclip[valid] * 0.5 + 0.5)) * bh).astype(np.int32)
        sx = np.clip(sx, 0, bw - 1)
        sy = np.clip(sy, 0, bh - 1)

        buffer_z = zbuf[sy, sx]
        # A candidate is "visible" if its view-space depth is within
        # `tolerance` of the front-most point at that pixel. Anything
        # noticeably behind the front surface is dropped.
        visible = valid & (cz <= buffer_z + tolerance)
        kept = indices[visible]
        dropped = int(len(indices) - len(kept))
        if dropped > 0:
            print(f"Hidden-point filter dropped {dropped:,} occluded points")
        return kept

    def _install_schema_callback(self) -> None:
        """Deprecated. Labels are project-scoped — the mutation callback
        is installed by ``_sync_project_state`` when a project is
        active, and cleared otherwise. Kept as a no-op so older callers
        that still invoke it after a registry swap don't blow up.
        """
        return

    # -- project-level state (ontology + settings) --------------------------

    def _sync_project_state(self, view: tuple[str, str] | None) -> None:
        """Apply the label registry for the incoming view.

        Labels are strictly project-scoped:
          - Project view + stored ontology -> load that ontology
          - Project view + no ontology yet -> empty registry (project
            inherits nothing; user starts from scratch)
          - Any non-project view (None, folder, smart) -> empty registry
            with no callback, so mutations can't happen anyway
        """
        proj = self._project_for_view(view)
        if proj is not None:
            if proj.ontology_data is not None:
                self.label_registry = LabelRegistry.from_json(proj.ontology_data)
            else:
                self.label_registry = LabelRegistry()
            self.label_registry._on_change_callback = self._on_label_registry_changed
        else:
            # No project active -> empty, read-only registry. Any
            # attempt to add a label is a no-op (the +ADD UI also
            # disables itself in this state).
            self.label_registry = LabelRegistry()
            self.label_registry._on_change_callback = None

        if self.label_texture is not None:
            update_label_color_texture(self.label_texture, self.label_registry)
        self.label_count_cache.clear()

    def _on_label_registry_changed(self, registry: LabelRegistry) -> None:
        """Single mutation hook the registry fires on every edit.

        Wraps two side effects that BOTH need to happen on every label
        change:

        1. Persist the registry back to the active project (so the
           ontology survives a restart).
        2. Re-upload the GPU label-color LUT so the viewport reflects
           the change immediately.

        Without (2), edits via ``set_visible`` (H shortcut, eye icon
        in the label panel) updated the registry but left the GPU's
        texture stale — the sidebar would show the eye flipped while
        the viewport kept rendering the hidden label at full alpha.
        Same trap for color edits if anything ever toggled
        ``info.locked`` outside the existing color-picker code path.
        Routing both side effects through this one method means every
        future ``_on_change``-emitting mutation gets full coverage.
        """
        self._save_project_state(registry)
        if self.label_texture is not None:
            try:
                update_label_color_texture(self.label_texture, registry)
            except Exception as e:
                print(f"[label texture] refresh failed: {e}")

    def _save_project_state(self, registry: LabelRegistry) -> None:
        """Mutation callback: persist registry back to the active project."""
        view = getattr(self, 'active_view', None)
        proj = self._project_for_view(view)
        if proj is None:
            return
        if self.catalog is not None:
            self.catalog.update_project_ontology(proj.id, registry.to_json())

    def _project_for_view(self, view: tuple[str, str] | None):
        """Return the Project for a view tuple, or None."""
        if view is None or view[0] != "project":
            return None
        if self.catalog is None:
            return None
        return self.catalog.projects.get(view[1])

    def _label_name_for(self, label_id: int) -> str:
        """Human-readable label name for a given id, falling back gracefully."""
        if label_id == 0:
            return "Unlabeled"
        info = self.label_registry.get(label_id)
        return info.name if info is not None else str(label_id)

    def _paint_target(self, ctrl: bool) -> int:
        """Resolve the label id that the current stroke will write.

        Ctrl held at press time = erase (write 0). Otherwise write the
        active label. If the active label is locked, return -1 so the
        caller can abort the stroke cleanly.
        """
        if ctrl:
            return 0
        info = self.label_registry.get(self.active_label_id)
        if info is not None and info.locked:
            return -1
        return int(self.active_label_id)

    def zero_label_id_across_catalog(self, label_id: int) -> tuple[int, int]:
        """Set every point with ``label_id`` to 0 across the catalog.

        Called when the user deletes a label class — without this,
        deleting a label just removed the registry entry while the
        numeric ``label_id`` survived on every cloud's labels array on
        disk, creating "ghost-labelled" points that reappeared the
        moment any future label took the same numeric id. That trap
        cost the user 1K painted points and ~10 minutes of confusion,
        so the deletion is now made truly destructive: the registry
        entry goes AND the underlying numeric labels get zeroed.

        Walks both:
        - In-memory cloud_data objects on every CloudEntry (full + preview).
          Zero in place, bump labels_version, re-upload the label
          texture so the gallery + viewport refresh this frame.
        - On-disk ``labels/<fk>.npy`` and ``preview_labels/<fk>.npy``
          for every catalog entry whose cloud isn't loaded right now.
          Read → modify → atomic write via ``save_cloud_labels`` /
          ``save_preview_labels``.

        Returns ``(loaded_clouds_touched, disk_clouds_touched)`` for
        the caller to surface in a CLI log line.
        """
        if label_id == 0:
            return (0, 0)  # zeroing "Unlabeled" is a no-op
        loaded_touched = 0
        disk_touched = 0

        # LS-10: snapshot every catalog entry's labels file BEFORE the
        # destructive zero. The mass-paint deletion is irreversible
        # without this — a wrong label_id wipes work across hundreds
        # of clouds. Backup lands at
        # ``<catalog>/backups/<YYYY-MM-DD_HH-MM-SS>/<fk>.npy``;
        # surface the directory through the status banner so the
        # user has a recovery path.
        catalog_for_snap = getattr(self, "catalog", None)
        if catalog_for_snap is not None:
            try:
                from src.data.cloud_store import snapshot_labels
                file_keys = list(catalog_for_snap.entries.keys())
                if file_keys:
                    backup_dir, copied, skipped = snapshot_labels(file_keys)
                    msg = (f"Pre-delete snapshot of {copied} label files "
                           f"under {backup_dir} (skipped {skipped})")
                    print(f"[delete-label] {msg}")
                    try:
                        self.set_status_banner(
                            msg, level="info",
                            source="zero_label_id_across_catalog",
                        )
                    except AttributeError:
                        pass
            except Exception as e:
                print(f"[delete-label] snapshot failed: {e}")

        # In-memory zero — every CloudEntry, both full + preview GPUs.
        loaded_fkeys: set[str] = set()
        for entry in self.entries:
            if entry.file_key:
                loaded_fkeys.add(entry.file_key)
            for gpu in (entry.full_gpu, entry.preview_gpu):
                if gpu is None or gpu.cloud_data is None:
                    continue
                cloud = gpu.cloud_data
                if cloud.labels is None:
                    continue
                mask = (cloud.labels == int(label_id))
                if not mask.any():
                    continue
                cloud.labels[mask] = 0
                cloud.labels_version += 1
                try:
                    re_upload_labels(gpu, None)
                except Exception as e:
                    print(f"[delete-label] re-upload failed: {e}")
                if hasattr(self, 'label_count_cache'):
                    self.label_count_cache.invalidate_cloud(cloud)
                loaded_touched += 1

        # On-disk zero — every catalog entry whose cloud isn't in
        # self.entries (i.e. not loaded into the working set). We
        # still want their labels NPY rewritten so reopening that
        # cloud doesn't bring the deleted-label points back as
        # ghost-labelled.
        catalog = getattr(self, 'catalog', None)
        if catalog is None:
            return (loaded_touched, 0)
        for fk in list(catalog.entries.keys()):
            if fk in loaded_fkeys:
                # Already zeroed in memory; the next persist call
                # writes the cleaned array to disk. Force a write now
                # so we don't leave stale label data on disk that
                # would resurface on next launch if the in-memory
                # cleanup never gets flushed.
                for entry in self.entries:
                    if entry.file_key == fk:
                        gpu = entry.full_gpu or entry.preview_gpu
                        if gpu is not None and gpu.cloud_data is not None and gpu.cloud_data.labels is not None:
                            # CP-4 + LS-7 (post-audit): per-key lock so
                            # this write doesn't race a background mesh
                            # build / inference; surface failures via
                            # the status banner.
                            err = save_cloud_labels(
                                fk, gpu.cloud_data.labels,
                                catalog=catalog)
                            if err:
                                try:
                                    self.set_status_banner(
                                        f"Label-delete save failed for "
                                        f"{entry.name}: {err}",
                                        level="error",
                                        source=f"cloud_store.save_cloud_labels:{fk}",
                                    )
                                except AttributeError:
                                    pass
                            preview_gpu = entry.preview_gpu
                            if preview_gpu is not None and preview_gpu is not entry.full_gpu and preview_gpu.cloud_data is not None and preview_gpu.cloud_data.labels is not None:
                                from src.data.cloud_store import save_preview_labels
                                save_preview_labels(fk, preview_gpu.cloud_data.labels)
                        break
                continue
            # Cloud isn't loaded — read its labels NPY, zero, write back.
            arr = load_cloud_labels(fk)
            if arr is None or arr.size == 0:
                continue
            if not (arr == int(label_id)).any():
                continue
            arr = arr.copy()
            arr[arr == int(label_id)] = 0
            # CP-4 + LS-7 (post-audit): same per-key lock + banner contract
            # for the disk-only path. The entry isn't loaded so no in-app
            # GPU surface to refresh — the banner is the only user-facing
            # signal a save failed.
            err = save_cloud_labels(fk, arr, catalog=catalog)
            if err:
                try:
                    self.set_status_banner(
                        f"Label-delete disk save failed for {fk}: {err}",
                        level="error",
                        source=f"cloud_store.save_cloud_labels:{fk}",
                    )
                except AttributeError:
                    pass
            # Same treatment for the preview labels file.
            from src.data.cloud_store import (
                preview_labels_path, save_preview_labels,
            )
            prev_path = preview_labels_path(fk)
            if prev_path.exists():
                try:
                    prev = np.load(prev_path)
                    if (prev == int(label_id)).any():
                        prev = prev.copy()
                        prev[prev == int(label_id)] = 0
                        save_preview_labels(fk, prev)
                except (OSError, ValueError) as e:
                    print(f"[delete-label] preview labels rewrite failed "
                          f"for {fk}: {e}")
            disk_touched += 1
        return (loaded_touched, disk_touched)

    def _after_label_mutation(self, gpu, cloud) -> None:
        """Shared post-write bookkeeping: GPU upload, caches, sequence flag,
        and persistent catalog labels (so closing and reopening 3Photon
        keeps the user's annotation state automatically).

        Catalog persistence is suppressed while a brush stroke is in
        progress — writing the full int32 labels array on every tick
        would mean tens of MB of disk traffic per second. Brush strokes
        flush their persist call exactly once at mouse release, after
        ``stroke_recorder.end()``.
        """
        re_upload_labels(gpu, self.selection_buffer.mask)
        cloud.labels_version += 1
        if self.sequence is not None:
            self.sequence._has_labels[self.sequence.current_index] = bool(
                (cloud.labels != 0).any())
        if hasattr(self, 'label_count_cache'):
            self.label_count_cache.invalidate_cloud(cloud)
        # Mirror labels into the preview gpu so the gallery thumbnails
        # stay in sync with strokes painted on the full-resolution cloud.
        # Without this, the user paints in Light Table and then sees an
        # unlabelled thumbnail in Contact Sheets — terrifying false-
        # alarm we want to avoid.
        if 0 <= self.selected_index < len(self.entries):
            entry = self.entries[self.selected_index]
            if gpu is entry.full_gpu and entry.preview_gpu is not None and entry.preview_gpu is not gpu:
                self._propagate_labels_to_preview(entry)
        # Catalog persistence: write the labels file for this cloud's
        # file_key.
        if not self.stroke_recorder.is_active:
            if 0 <= self.selected_index < len(self.entries):
                entry = self.entries[self.selected_index]
                self._persist_cloud_labels(entry, cloud)
        # Mark the session as having unsaved-to-backup work so the next
        # _tick_auto_snapshot fires; idle sessions don't generate
        # redundant copies.
        self._snapshot_dirty = True
        # Labels changed -> derived primitives are stale -> mesh
        # vertex_labels are stale. Flag both for rebuild. The actual
        # primitive re-derive is gated on the stroke ending so a long
        # brush drag doesn't refit 5 planes per frame.
        if not self.stroke_recorder.is_active:
            if 0 <= self.selected_index < len(self.entries):
                entry = self.entries[self.selected_index]
                entry.mesh_dirty = True




    def _propagate_labels_to_preview(self, entry) -> None:
        """Copy labels from the full GPU to the preview GPU via a spatial
        index map. The map is built lazily (first call after a paint on
        full) using ``scipy.spatial.cKDTree`` and cached on the entry,
        so subsequent strokes only pay an O(N_preview) gather + a small
        VBO write.

        Voxel-downsampled previews are byte-exact subsets of the full
        positions, so nearest-neighbour distance is zero — the KD-tree
        is just a fast route to the index.
        """
        full = entry.full_gpu
        prev = entry.preview_gpu
        if full is None or prev is None or full is prev:
            return
        full_cloud = full.cloud_data
        prev_cloud = prev.cloud_data
        if full_cloud is None or prev_cloud is None:
            return
        if full_cloud.labels is None or prev_cloud.labels is None:
            return

        # Cache key: identity of the full cloud_data + length of the
        # preview. Either changing invalidates — covers preview resize
        # AND a full-resolution reload that swaps cloud_data underneath
        # the entry. Without the full-id check, after a re-import the
        # cached mapping would index into the wrong positions array.
        cache_key = (id(full_cloud), prev_cloud.point_count)
        cached = getattr(entry, '_preview_to_full_idx_cache', None)
        mapping = cached[1] if cached is not None and cached[0] == cache_key else None
        if mapping is None:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(np.ascontiguousarray(full_cloud.positions,
                                                   dtype=np.float32))
                # Cap query parallelism at 4 cores. ``workers=-1`` pinned
                # all available threads — on a 16-core workstation the
                # GUI hitched perceptibly mid-paint when a 2M-point tree
                # query ran. Four is enough to keep the build below ~1s
                # for our cloud sizes while leaving the rest of the
                # machine responsive for whatever else the user is doing.
                _, idx = tree.query(prev_cloud.positions, k=1, workers=4)
                mapping = np.asarray(idx, dtype=np.int64)
                entry._preview_to_full_idx_cache = (cache_key, mapping)
            except Exception as e:
                print(f"[label propagate] index build failed for "
                      f"{getattr(entry, 'name', '?')}: {e}")
                return

        prev_cloud.labels[:] = full_cloud.labels[mapping]
        prev_cloud.labels_version += 1
        try:
            re_upload_labels(prev, None)
        except Exception as e:
            print(f"[label propagate] preview re-upload failed for "
                  f"{getattr(entry, 'name', '?')}: {e}")
        if hasattr(self, 'label_count_cache'):
            self.label_count_cache.invalidate_cloud(prev_cloud)
        # Persist preview labels so the Train tab's "labelled" filter
        # and the CLOUDS row swatches see this cloud as labelled even
        # before the user re-opens it in Light Table next session.
        fk = getattr(entry, 'file_key', None)
        if fk:
            try:
                save_preview_labels(fk, prev_cloud.labels)
            except Exception as e:
                print(f"[label propagate] preview labels save failed for "
                      f"{getattr(entry, 'name', '?')}: {e}")

    def _cycle_active_label(self, direction: int = 1) -> None:
        """Move ``active_label_id`` to the next/previous label in the
        project's registry (sorted by id, skipping 0/Unlabeled).

        ``direction`` is +1 or -1. Wraps at the ends so cycling past
        the last label lands on the first, matching the rest of the
        app's "hot-key navigation" muscle memory.

        Called from the LIGHT TABLE G shortcut. No-op when the
        registry has no non-zero labels.
        """
        registry = getattr(self, "label_registry", None)
        if registry is None:
            return
        ids = sorted(
            info.id for info in registry.all_labels() if info.id != 0
        )
        if not ids:
            return
        try:
            i = ids.index(int(self.active_label_id))
        except ValueError:
            # Currently on Unlabeled or a stale id: jump to the first
            # real label rather than rotating from -1, which would
            # land predictably-wrong on the last entry.
            i = -1 if direction > 0 else 0
        new_i = (i + (1 if direction > 0 else -1)) % len(ids)
        new_id = ids[new_i]
        self.active_label_id = new_id
        info = registry.get(new_id)
        if info is not None:
            print(f"Active label: id={new_id} '{info.name}'")

    def _filter_to_active_label_if_erasing(self, gpu, indices: np.ndarray) -> np.ndarray:
        """When Ctrl-erasing, keep only indices whose current label equals
        the active label. Ctrl-click is "subtract from THIS label," not
        "subtract from everything" — otherwise a stray ctrl-stroke wipes
        adjacent labels the user wasn't aiming at.
        """
        if not getattr(self, '_drag_ctrl', False):
            return indices
        if gpu is None or gpu.cloud_data is None or gpu.cloud_data.labels is None:
            return indices
        if len(indices) == 0:
            return indices
        mask = (gpu.cloud_data.labels[indices] == int(self.active_label_id))
        return indices[mask]

    def _paint_indices(self, gpu, indices: np.ndarray) -> int:
        """Atomic paint: drop-locked, apply_label, post-mutation bookkeeping.

        Used by box/lasso/curve/pick where the whole stroke is materialized
        at release time — no incremental state. Returns the number of
        points actually written (0 if none after filtering).
        """
        target = self._paint_target(self._drag_ctrl)
        if target < 0:
            info = self.label_registry.get(self.active_label_id)
            name = info.name if info else str(self.active_label_id)
            print(f"Cannot paint '{name}': label is locked")
            return 0
        indices = self._drop_locked(indices)
        if len(indices) == 0:
            return 0
        indices = self._filter_to_active_label_if_erasing(gpu, indices)
        if len(indices) == 0:
            return 0
        cloud = gpu.cloud_data
        apply_label(cloud, indices, target, self.undo_stack,
                    description=f"Paint {len(indices)} pts as "
                                f"'{self._label_name_for(target)}'")
        self._after_label_mutation(gpu, cloud)
        return len(indices)

    def _window_to_fb(self, mx: float, my: float) -> tuple[float, float]:
        """Convert GLFW *window* (logical) cursor coords to *framebuffer*
        (physical) pixels.

        GLFW reports the cursor in window-logical pixels, but every
        selection handler does its hit-testing in framebuffer space:
        ``self.width``/``self.height``, the viewport rects from
        ``_primary_viewport_screen_rect``, and the snap radii
        (PICK_RADIUS_PX / BRUSH_SNAP_RADIUS_PX) are all framebuffer
        pixels. On a HiDPI / fractional-scale display (e.g. 1.6x on
        Wayland/KDE) the two differ, so feeding raw window coords makes
        the brush/box/lasso/pick land offset from the cursor — shifted
        toward the origin, growing with distance from it. This applies
        the same correction ``_pick_world_position`` already does for the
        3D cursor, so all selection tools agree with the pointer.
        """
        try:
            win_w, win_h = glfw.get_window_size(self.window)
        except Exception:
            return mx, my
        if win_w > 0 and win_h > 0 and (win_w != self.width or win_h != self.height):
            return mx * (self.width / float(win_w)), my * (self.height / float(win_h))
        return mx, my

    def _do_pick(self, mx: float, my: float, shift: bool, ctrl: bool):
        """Pick the nearest point and paint it with the active label.

        Under the direct-paint workflow pick is a one-point stroke. Ctrl
        writes label 0 (erase) instead of the active label.
        """
        gpu = self._current_gpu_cloud()
        if gpu is None:
            return
        mx, my = self._window_to_fb(mx, my)
        entry = self.entries[self.selected_index]
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp = proj @ view @ entry.model_transform
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
        idx = pick_point(gpu.cloud_data.positions, mvp,
                         mx - vp_x, my - vp_y, vp_w, vp_h,
                         radius_px=PICK_RADIUS_PX)
        if idx < 0:
            return
        # _paint_indices reads self._drag_ctrl; set it for this single click.
        self._drag_ctrl = bool(ctrl)
        self._paint_indices(gpu, np.array([idx], dtype=np.int32))

    def _do_brush_at(self, mx: float, my: float):
        """Paint the points under the brush directly onto the active layer.

        Incremental: each tick adds to the in-progress stroke via
        ``self.stroke_recorder``. The recorder captures each point's
        original label on first touch and flushes one ``LabelCommand``
        on release, so a 100-tick stroke becomes a single undo step.
        """
        gpu = self._current_gpu_cloud()
        if gpu is None or not self.stroke_recorder.is_active:
            return
        mx, my = self._window_to_fb(mx, my)
        entry = self.entries[self.selected_index]
        positions = gpu.cloud_data.positions
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        view_xform = view @ entry.model_transform
        mvp = proj @ view_xform
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()

        center = screen_to_world(
            positions, mvp,
            mx - vp_x, my - vp_y, vp_w, vp_h,
            view=view, projection=proj,
            camera_target=self.camera.target,
            camera_distance=self.camera.distance,
            radius_px=BRUSH_SNAP_RADIUS_PX,
        )
        if center is None:
            return
        indices = brush_select(positions, center, self.brush_radius)
        # Depth-slab filter: along-view distance from the snapped center.
        # Falloff = 0 keeps the full sphere (today's behavior); higher
        # values progressively reject points behind / in front of the
        # hovered surface so the brush stops painting through the volume.
        falloff = float(np.clip(self.brush_distance_falloff, 0.0, 1.0))
        if falloff > 1e-3 and len(indices) > 0:
            # World-space camera forward = -row2 of the view matrix's
            # 3x3 rotation block (view space has -Z forward).
            forward = -view[2, :3].astype(np.float32)
            # Min slab keeps a single-voxel-ish slice usable at slider=1
            # instead of selecting literally zero points.
            slab = self.brush_radius * max(1.0 - falloff, 0.02)
            offsets = positions[indices] - center
            depth = offsets @ forward
            keep = np.abs(depth) <= slab
            indices = indices[keep]
        indices = self._drop_locked(indices)
        if len(indices) == 0:
            return
        indices = self._drop_hidden(indices)
        if len(indices) == 0:
            return
        indices = self._filter_to_active_label_if_erasing(gpu, indices)
        if len(indices) == 0:
            return
        added = self.stroke_recorder.add_points(indices)
        if added > 0:
            self._after_label_mutation(gpu, gpu.cloud_data)

    def _finish_box_drag(self):
        """Finalize a box drag: paint every enclosed point as one command."""
        gpu = self._current_gpu_cloud()
        if gpu is None or self._drag_start is None or self._drag_current is None:
            self._drag_start = None
            self._drag_current = None
            return
        entry = self.entries[self.selected_index]
        positions = gpu.cloud_data.positions
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp = proj @ view @ entry.model_transform
        x0, y0 = self._drag_start
        x1, y1 = self._drag_current
        self._drag_start = None
        self._drag_current = None
        # Minimum drag distance to avoid accidental clicks
        if abs(x0 - x1) < 3 and abs(y0 - y1) < 3:
            return
        view_xform = view @ entry.model_transform
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
        # _drag_* are window-logical; hit-test wants framebuffer pixels.
        x0, y0 = self._window_to_fb(x0, y0)
        x1, y1 = self._window_to_fb(x1, y1)
        indices = box_select(positions, mvp,
                             x0 - vp_x, y0 - vp_y,
                             x1 - vp_x, y1 - vp_y,
                             vp_w, vp_h,
                             max_depth=self.selection_max_depth,
                             view_matrix=view_xform)
        indices = self._drop_hidden(indices)
        n = self._paint_indices(gpu, indices)
        if n > 0:
            print(f"Box painted {n:,} points as "
                  f"'{self._label_name_for(self._paint_target(self._drag_ctrl))}'")

    def _finish_lasso_drag(self):
        """Finalize a lasso: paint every point inside the polygon in one command."""
        gpu = self._current_gpu_cloud()
        if gpu is None or len(self._lasso_path) < 3:
            self._lasso_path = []
            return
        entry = self.entries[self.selected_index]
        positions = gpu.cloud_data.positions
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp = proj @ view @ entry.model_transform
        view_xform = view @ entry.model_transform
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
        # _lasso_path is window-logical; hit-test wants framebuffer pixels.
        fb_path = [self._window_to_fb(p[0], p[1]) for p in self._lasso_path]
        lasso_local = [(px - vp_x, py - vp_y) for px, py in fb_path]
        indices = lasso_select(positions, mvp, lasso_local,
                               vp_w, vp_h,
                               max_depth=self.selection_max_depth,
                               view_matrix=view_xform)
        self._lasso_path = []
        indices = self._drop_hidden(indices)
        n = self._paint_indices(gpu, indices)
        if n > 0:
            print(f"Lasso painted {n:,} points as "
                  f"'{self._label_name_for(self._paint_target(self._drag_ctrl))}'")

    def _finish_curve_drag(self):
        """Finalize a curve: paint every point within threshold as one command."""
        gpu = self._current_gpu_cloud()
        if gpu is None or len(self._lasso_path) < 2:
            self._lasso_path = []
            return
        entry = self.entries[self.selected_index]
        positions = gpu.cloud_data.positions
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        mvp = proj @ view @ entry.model_transform
        view_xform = view @ entry.model_transform
        vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
        # _lasso_path is window-logical; hit-test wants framebuffer pixels.
        fb_path = [self._window_to_fb(p[0], p[1]) for p in self._lasso_path]
        curve_local = [(px - vp_x, py - vp_y) for px, py in fb_path]
        indices = curve_select(positions, mvp, curve_local,
                               vp_w, vp_h,
                               threshold_px=self.curve_threshold_px,
                               max_depth=self.selection_max_depth,
                               view_matrix=view_xform)
        self._lasso_path = []
        indices = self._drop_hidden(indices)
        n = self._paint_indices(gpu, indices)
        if n > 0:
            print(f"Curve painted {n:,} points as "
                  f"'{self._label_name_for(self._paint_target(self._drag_ctrl))}'")

    def _draw_depth_limit_plane(self, view: np.ndarray, mvp: np.ndarray):
        """Draw a wireframe quad at the current selection depth cutoff.

        The plane is perpendicular to the camera forward axis and sits at
        view-space distance = self.selection_max_depth.
        """
        # Camera basis in world space from the view matrix rows
        right = view[0, 0:3]
        up = view[1, 0:3]
        forward = -view[2, 0:3]  # view_z points back toward the camera
        eye = self.camera.get_eye_position()
        d = float(self.selection_max_depth)
        center = eye + forward * d
        half = max(d * 0.5, 0.05)
        self.overlays.build_depth_plane(center, right, up, half)
        # Disable depth test so the plane shows through the cloud
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.overlays.draw_depth_plane(mvp)
        self.ctx.enable(moderngl.DEPTH_TEST)

    def _ray_plane_intersect(self, ray_o: np.ndarray, ray_d: np.ndarray,
                              plane_y: float) -> np.ndarray | None:
        """Intersect a ray with the horizontal plane at the given Y level."""
        if abs(ray_d[1]) < 1e-8:
            return None  # Ray parallel to plane
        t = (plane_y - ray_o[1]) / ray_d[1]
        if t < 0:
            # Plane behind camera — use a point at camera distance along the ray instead
            t = self.camera.distance
        return (ray_o + ray_d * t).astype(np.float32)

    def _cursor_focal_distance(self, camera=None) -> float:
        """World-space distance from a camera to the 3D cursor's anchor.

        Used as the DOF focal plane so the cursor *defines* where the
        scene is sharp — pan or orbit, the cursor (and the volume of
        space immediately around it) stays in focus, and depth-of-field
        blur grows symmetrically in front of and behind it.

        Falls back to the camera's orbit distance if the cursor hasn't
        been placed yet (visible=False), so DOF still behaves sensibly
        on first launch before the user picks an anchor.
        """
        cam = camera if camera is not None else self.camera
        if not getattr(self.cursor3d, "visible", False):
            return float(cam.distance)
        cam_pos = cam.get_eye_position()
        cursor_pos = np.asarray(self.cursor3d.position, dtype=np.float32)
        return float(np.linalg.norm(cursor_pos - cam_pos))

    def _pick_world_position(self, mx: float, my: float,
                              radius_px: float = CURSOR_SNAP_RADIUS_PX) -> np.ndarray | None:
        """Find a world-space position under the mouse cursor.

        Snaps to the screen-space-nearest visible point of the currently
        selected cloud. No radius cutoff, no plane-intersection fallback
        — a plane fallback put the orbit cursor visibly offset from the
        clicked point whenever the camera was tilted, which is what
        users were experiencing as "the triangle lands somewhere else".
        Returns None only when there is genuinely no cloud point on
        screen.

        ``radius_px`` is kept in the signature for API compatibility but
        no longer participates in the result — every caller wanted
        "nearest point" semantics, not "nearest within radius".

        Multi-viewport aware: the primary camera's viewport may only be
        part of the framebuffer. Mouse coords arrive in full framebuffer
        space and have to be remapped to viewport-local before
        projection math. If the click is outside the primary viewport,
        returns ``None`` so spectator viewports don't produce stray
        cursor placements.
        """
        del radius_px  # accepted for API compat; no longer used
        # GLFW reports cursor position in *window* (logical) pixels, but
        # every viewport rect, self.width and self.height live in
        # *framebuffer* pixels. On HiDPI displays the two differ by the
        # content-scale factor (e.g. 1.5× / 2× on a 5K monitor with
        # Windows scaling). Without this conversion the picked location
        # gets shifted — and on high scales mirrored across the viewport
        # centre — so cursor placement lands on the opposite side of the
        # cloud or floats off it entirely.
        try:
            win_w, win_h = glfw.get_window_size(self.window)
        except Exception:
            win_w, win_h = self.width, self.height
        if win_w > 0 and win_h > 0 and (win_w != self.width or win_h != self.height):
            mx = mx * (self.width / float(win_w))
            my = my * (self.height / float(win_h))

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()

        # Primary viewport rect (OpenGL convention: origin bottom-left)
        rects = self._viewport_rects()
        vx, vy, vw, vh = rects[0] if rects else (0, 0, self.width, self.height)
        if vw <= 0 or vh <= 0:
            return None

        # Convert OpenGL-bottom-origin viewport Y to ImGui/screen-top-origin
        vy_top = self.height - (vy + vh)

        # Drop clicks that land outside the primary viewport (sidebar,
        # secondary viewports, inter-viewport gap).
        if not (vx <= mx <= vx + vw and vy_top <= my <= vy_top + vh):
            return None

        # Mouse coords relative to the primary viewport's top-left
        mx_local = mx - vx
        my_local = my - vy_top

        # Snap to the screen-space-nearest visible point of the active
        # cloud. No radius cutoff: the user's mental model is "the
        # cursor lands on whatever I'm pointing at on the cloud", and a
        # tight radius produced plane-fallback misses that read as
        # offsets. Mode/selection guards still apply — outside Light
        # Table or with no cloud selected, no point makes sense, so
        # nothing happens.
        if (self.mode == MODE_LIGHT_TABLE
                and 0 <= self.selected_index < len(self.entries)):
            entry = self.entries[self.selected_index]
            gpu = entry.full_gpu or entry.preview_gpu
            if gpu is not None:
                positions = gpu.cloud_data.positions
                # pick_visible_disk replicates the point-cloud shader's
                # size formula so the click registers wherever the
                # *rendered* disk sits — front-facing points win even
                # when DOF/Focus has ballooned them past the centre of
                # back-facing neighbours.
                dof_strength = float(getattr(self, "dof_strength", 0.0)) \
                    if getattr(self, "dof_enabled", False) else 0.0
                dof_focus = self._cursor_focal_distance()
                idx = pick_visible_disk(
                    positions,
                    model=entry.model_transform,
                    view=view, proj=proj,
                    mouse_x=mx_local, mouse_y=my_local,
                    width=vw, height=vh,
                    point_size=float(self.point_size),
                    dof_strength=dof_strength,
                    dof_focus_dist=dof_focus,
                )
                if idx >= 0:
                    local = np.array([*positions[idx], 1.0], dtype=np.float32)
                    world = (entry.model_transform @ local)[:3]
                    return world.astype(np.float32)
        return None


    def _pick_world_position_any(self, mx: float, my: float
                                  ) -> np.ndarray | None:
        """Mode-aware world picker.

        Returns the world-space position under the cursor using the
        correct picker for the active mode:
          - HOLOGRAM → multi-cloud picker (every visible vertebra at
            its reference-local pose).
          - everything else → single-cloud picker against the active
            entry's ``model_transform``.

        Callers that need the hit cloud's entry (e.g. for focus
        framing) should reach for the underlying picker directly; this
        helper drops the entry to keep the common-case API a plain
        ``np.ndarray | None``.
        """
        return self._pick_world_position(mx, my)

    def _build_measure_resolver(self):
        """Return an ``AnchorResolver`` for the current scene state.

        The resolver maps cloud-local Anchors to world positions using
        whichever model-matrix composition matches the active mode:

        - **LIGHT TABLE**: per-entry ``model_transform`` (single-cloud
          view, no reference-anchoring).
        - **HOLOGRAM**: ``reference_local_model(cloud_pose, ref_pose)``
          — the same matrix the renderer uses — so anchors follow
          their bones through reference-vertebra switches.
        - Other modes: identity (the gallery / contact sheets don't
          render the measure registry, so the resolver mostly returns
          None for missing clouds; the few anchors that exist resolve
          to their stored local position).

        Rebuilt each call rather than cached because both modes can
        change between consecutive frames (mode switch, reference
        change, project switch) and any cached resolver would silently
        return stale world positions.
        """
        from src.core.measure_registry import make_resolver
        entries_by_key = {
            e.file_key: e for e in self.entries if e.file_key
        }


    def _pick_anchor(self, mx: float, my: float):
        """Mode-aware picker that returns a full ``PickResult``.

        Unlike ``_pick_world_position_any`` (which strips down to a
        bare world position for cursor / orbit-pivot use), this returns
        the cloud_key + cloud-local position that a measurement anchor
        needs. The cloud-local position is the cloud's mesh-local
        position of the picked point (``positions[idx]`` — already
        cloud-local since point clouds are stored mesh-local and
        composed with the model matrix at render time).

        Used by the measurement-tool placement + drag handlers. The
        returned ``PickResult.as_anchor()`` produces an ``Anchor`` that
        survives reference-vertebra switches.
        """
        from src.core.measure_registry import PickResult
        # LIGHT TABLE (or any future single-cloud mode): pick against
        # the active cloud and use its model_transform for world<->local.
        try:
            win_w, win_h = glfw.get_window_size(self.window)
        except Exception:
            win_w, win_h = self.width, self.height
        if (win_w > 0 and win_h > 0
                and (win_w != self.width or win_h != self.height)):
            mx = mx * (self.width / float(win_w))
            my = my * (self.height / float(win_h))

        rects = self._viewport_rects()
        vx, vy, vw, vh = rects[0] if rects else (
            0, 0, self.width, self.height)
        if vw <= 0 or vh <= 0:
            return None
        vy_top = self.height - (vy + vh)
        if not (vx <= mx <= vx + vw and vy_top <= my <= vy_top + vh):
            return None
        mx_local = mx - vx
        my_local = my - vy_top

        if not (0 <= self.selected_index < len(self.entries)):
            return None
        entry = self.entries[self.selected_index]
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None or gpu.cloud_data is None:
            return None
        positions = gpu.cloud_data.positions
        if positions is None or len(positions) == 0:
            return None

        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix()
        dof_strength = (
            float(getattr(self, "dof_strength", 0.0))
            if getattr(self, "dof_enabled", False) else 0.0
        )
        dof_focus = self._cursor_focal_distance()
        idx = pick_visible_disk(
            positions,
            model=entry.model_transform, view=view, proj=proj,
            mouse_x=mx_local, mouse_y=my_local,
            width=vw, height=vh,
            point_size=float(self.point_size),
            dof_strength=dof_strength,
            dof_focus_dist=dof_focus,
        )
        if idx < 0:
            return None
        local_pos = np.asarray(positions[idx], dtype=np.float32).copy()
        local_h = np.array([*local_pos, 1.0], dtype=np.float32)
        world = (entry.model_transform @ local_h)[:3].astype(np.float32)
        return PickResult(
            world_pos=world, cloud_key=entry.file_key,
            local_pos=local_pos, entry=entry,
        )


    def _tick_autosave(self) -> None:
        """Per-frame autosave check.

        Brush stroke release / box / lasso / curve / pick already write
        through to ``labels/<file_key>.npy`` immediately. This timer
        only matters mid-brush-drag: if the user holds a long careful
        stroke, snapshot the current label state every
        ``self._autosave_interval`` seconds so a crash mid-drag loses
        at most that much in-progress work.

        Out of stroke this is a redundant rewrite of the same bytes
        already on disk — cheap, but bounded by the interval so it's
        not per-frame disk traffic.
        """
        now = time.perf_counter()
        if now - self._last_autosave_time < self._autosave_interval:
            return
        self._last_autosave_time = now
        if not (0 <= self.selected_index < len(self.entries)):
            return
        entry = self.entries[self.selected_index]
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None or gpu.cloud_data is None:
            return
        # LS-9: skip the canonical labels write when a stroke is in
        # progress. Mid-stroke cloud.labels reflects partial brush
        # state; persisting it would freeze a half-painted result
        # on disk that the user can't undo (the stroke isn't on
        # the undo stack until release). Write to a recovery
        # sidecar instead so a crash mid-stroke still has a
        # rollback point.
        try:
            if getattr(self.stroke_recorder, "is_active", False):
                self._persist_recovery_sidecar(entry, gpu.cloud_data)
                return
            self._persist_cloud_labels(entry, gpu.cloud_data)
        except Exception as e:
            print(f"Autosave failed: {e}")

    def _persist_recovery_sidecar(self, entry, cloud) -> None:
        """Write mid-stroke labels to ``labels/<fk>.recovery.npy``.

        LS-9: keeps the canonical ``labels/<fk>.npy`` consistent (only
        rewritten on stroke release) while still giving us a crash
        recovery point during long brush drags. On next launch, App
        scans for .recovery.npy files newer than their canonical
        sibling and offers the user a choice to restore. The
        recovery file is removed on the next clean save.
        """
        file_key = getattr(entry, "file_key", None)
        if not file_key or cloud is None or cloud.labels is None:
            return
        try:
            from src.data import cloud_store
            labels_path = cloud_store.cloud_labels_path(file_key)
            recovery_path = labels_path.parent / f"{file_key}.recovery.npy"
            # Post-audit fix: ``np.save`` unconditionally appends ".npy"
            # when missing. Use the same no-suffix-tmp-base pattern as
            # ``cloud_store.save_cloud_labels`` so the temp file lands
            # at a real intermediate path (not directly on top of the
            # canonical recovery file), then ``os.replace`` is the
            # atomic publish. The previous code did
            # ``tmp.with_suffix("")`` which stripped ``.tmp`` and made
            # np.save write straight to the canonical path — every
            # autosave tick subsequently hit ``FileNotFoundError`` on
            # os.replace because the .tmp file never existed.
            tmp_base = labels_path.parent / f"_tmp_recovery_{file_key}"
            tmp_with_npy = tmp_base.with_suffix(".npy")
            arr = np.ascontiguousarray(cloud.labels, dtype=np.int32)
            np.save(str(tmp_base), arr)
            os.replace(tmp_with_npy, recovery_path)
        except (OSError, ValueError) as e:
            print(f"Recovery sidecar write failed for {file_key}: {e}")

    def _tick_auto_snapshot(self) -> None:
        """Periodic timestamped backup of every loaded cloud's labels.

        Distinct from ``_tick_autosave``: that one rewrites the LIVE
        catalog labels file (overwrite-in-place). This one COPIES the
        live files to ``<catalog>/backups/<timestamp>/`` so the user
        has rollback points if the live state goes bad.

        Fires only when ``_snapshot_dirty`` is set (a label mutation
        happened since the last snapshot) so an idle session doesn't
        produce redundant copies. Old backup directories beyond
        ``_snapshot_keep`` are pruned.
        """
        now = time.perf_counter()
        if now - self._last_snapshot_time < self._snapshot_interval:
            return
        self._last_snapshot_time = now
        if not self._snapshot_dirty:
            return
        file_keys: list[str] = []
        for entry in self.entries:
            gpu = entry.full_gpu or entry.preview_gpu
            cloud = gpu.cloud_data if gpu is not None else None
            fk = getattr(entry, 'file_key', None)
            if cloud is not None and cloud.labels is not None and fk:
                file_keys.append(fk)
        if not file_keys:
            return
        try:
            from src.data.cloud_store import snapshot_labels, prune_old_snapshots
            backup_dir, copied, skipped = snapshot_labels(file_keys)
            self._snapshot_dirty = False
            pruned = prune_old_snapshots(self._snapshot_keep)
            msg = f"[auto-snapshot] {copied} cloud(s) -> {backup_dir.name}"
            if skipped:
                msg += f" ({skipped} skipped)"
            if pruned:
                msg += f" [pruned {pruned} old]"
            print(msg)
        except Exception as e:
            print(f"[auto-snapshot] failed: {type(e).__name__}: {e}")

    def _tick_radial_menu(self) -> None:
        """Per-frame check: open the radial preset menu if RMB has been
        held in place long enough.

        The menu is only eligible in Light Table mode with no active
        selection tool; a confirmed drag (movement past the click radius)
        cancels the pending activation so starting an orbit by flicking
        RMB immediately never triggers the menu.
        """
        if self._radial_menu_active:
            return
        if not self._rmb_held or self._rmb_press_time is None:
            return
        if self.mode != MODE_LIGHT_TABLE or self.active_tool is not None:
            return
        if self._click_detector.any_confirmed_drag():
            # User has started orbiting — cancel the pending menu open.
            self._rmb_press_time = None
            return
        if (time.perf_counter() - self._rmb_press_time) < _RADIAL_MENU_DELAY:
            return

        # Open the menu at the press location.
        mx, my = glfw.get_cursor_pos(self.window)
        self._radial_menu_active = True
        self._radial_menu_center = (mx, my)
        self._radial_menu_selected = -1
        # Cancel any camera orbit drag state so subsequent mouse motion
        # inside the menu doesn't leak into a rotation when the menu
        # closes and camera.on_mouse_move fires again.
        self.camera._dragging = False
        self.camera._orbit_prev_dx = 0.0
        self.camera._orbit_prev_dy = 0.0

    # --- GLFW callbacks ---

    def _on_resize(self, window, width, height):
        self.width = max(width, 1)
        self.height = max(height, 1)
        if self.scene_rt is not None:
            self.scene_rt.resize(self.width, self.height)
        # SSAO targets resize in lockstep with the scene buffer — they
        # sample the same depth and write at the same screen resolution.
        if getattr(self, 'ssao_rt', None) is not None:
            self.ssao_rt.resize(self.width, self.height)
        if getattr(self, 'ssao_blur_rt', None) is not None:
            self.ssao_blur_rt.resize(self.width, self.height)
        # Selection mask shares scene_rt.depth, which was just
        # reallocated above — recreate the texture + FBO so we hold a
        # reference to the NEW depth attachment, not the released one.
        if (getattr(self, 'selection_mask_tex', None) is not None
                and self.scene_rt is not None):
            try:
                self.selection_mask_fbo.release()
            except Exception:
                pass
            try:
                self.selection_mask_tex.release()
            except Exception:
                pass
            self.selection_mask_tex = self.ctx.texture(
                (self.width, self.height), components=1, dtype='f2',
            )
            self.selection_mask_tex.filter = (
                moderngl.NEAREST, moderngl.NEAREST)
            self.selection_mask_tex.repeat_x = False
            self.selection_mask_tex.repeat_y = False
            self.selection_mask_fbo = self.ctx.framebuffer(
                color_attachments=[self.selection_mask_tex],
                depth_attachment=self.scene_rt.depth,
            )
        # Gallery cache: the *visible* scratch RT picks up the new size
        # via ensure_target() on the next _render_gallery call. Per-entry
        # caches are NOT invalidated — cell content is identical at any
        # window size, only the scratch RT and scrollbar geometry change.
        # Clamp scroll_y in case the new viewport is taller than the old.
        if getattr(self, '_gallery_scroll_y', None) is not None:
            self._gallery_scroll_y = max(0.0, float(self._gallery_scroll_y))

    def _on_focus(self, window, focused):
        """Reset all drag states when the window loses focus.

        GLFW swallows button-release events that happen while the window
        is unfocused, which leaves camera/tool drags stuck in their
        pressed state. Clearing everything on focus-loss prevents the
        camera from orbiting unexpectedly when the user Alt-Tabs back.
        """
        if not focused:
            self.camera.cancel_all_drags()
            self._tool_engaged = False
            self._brush_painting = False
            self._rmb_held = False
            self._lmb_held = False
            if hasattr(self, '_gallery_scrollbar_dragging'):
                self._gallery_scrollbar_dragging = False

    def _on_mouse_button(self, window, button, action, mods):
        self.gui.feed_mouse_button(button, action == glfw.PRESS)
        if self.gui.wants_mouse():
            return

        mx, my = glfw.get_cursor_pos(window)

        # State bookkeeping: held flags + click detector. Done up front so
        # every return path below leaves the state consistent.
        click_kind: str | None = None
        if action == glfw.PRESS:
            self._click_detector.on_press(button, mx, my)
            if button == 0:
                self._lmb_held = True
            elif button == 1:
                self._rmb_held = True
                self._rmb_press_time = time.perf_counter()
            elif button == 2:
                self._mmb_held = True
                self._mmb_last_y = my
        elif action == glfw.RELEASE:
            click_kind = self._click_detector.on_release(button, mx, my)
            if button == 0:
                self._lmb_held = False
                # End any in-progress gallery scrollbar drag.
                if self._gallery_scrollbar_dragging:
                    self._gallery_scrollbar_dragging = False
                    return
            elif button == 1:
                self._rmb_held = False
                self._rmb_press_time = None
                # Radial menu commit. If the menu was open, fire the
                # hovered slice's preset (or nothing, for the dead zone)
                # and close the menu. We still forward the release to the
                # camera so its _dragging state is cleaned up, but we
                # return immediately afterwards to skip the normal dispatch.
                if self._radial_menu_active:
                    if self._radial_menu_selected >= 0:
                        from src.gui.radial_menu import RADIAL_PRESETS
                        self.camera.set_preset(
                            RADIAL_PRESETS[self._radial_menu_selected]
                        )
                    self._radial_menu_active = False
                    self._radial_menu_center = None
                    self._radial_menu_selected = -1
                    self.camera.on_mouse_button(button, action, mods)
                    self._click_detector.reset()
                    return
            elif button == 2:
                self._mmb_held = False

        # Selection tool dispatch: Light Table mode.
        #   Shift+LMB = paint with the active label (add)
        #   Ctrl+LMB  = erase (write label 0, subtract)
        # Plain LMB is reserved for camera panning. `_tool_engaged`
        # ensures release always completes on the tool that started the
        # drag — even if the modifier is released mid-drag.
        if self.mode == MODE_LIGHT_TABLE and button == 0:
            shift = bool(mods & glfw.MOD_SHIFT)
            ctrl = bool(mods & glfw.MOD_CONTROL)
            if (action == glfw.PRESS and self.active_tool is not None
                    and (shift or ctrl)):
                self._tool_engaged = True
                # Stamp the erase flag for the entire stroke.
                self._drag_ctrl = ctrl
                # NOTE: _drag_*/_lasso_path stay in window-logical pixels so
                # the ImGui overlay (which draws in logical space) stays
                # correct. Conversion to framebuffer space happens only at
                # hit-test time (_do_*, _finish_*) via _window_to_fb.
                if self.active_tool == 'pick':
                    self._do_pick(mx, my, False, ctrl)
                elif self.active_tool == 'box':
                    self._drag_start = (mx, my)
                    self._drag_current = (mx, my)
                elif self.active_tool in ('lasso', 'curve'):
                    self._lasso_path = [(mx, my)]
                elif self.active_tool == 'polygon':
                    # Polygon: don't reset the path — vertices accumulate
                    # across clicks. Release handler decides whether this
                    # press turns into a vertex (single click) or closes
                    # the polygon (double click). _tool_engaged stays True
                    # so the release branch runs.
                    pass
                elif self.active_tool == 'brush':
                    gpu = self._current_gpu_cloud()
                    target = self._paint_target(ctrl)
                    if gpu is not None and target >= 0:
                        name = self._label_name_for(target)
                        self.stroke_recorder.begin(
                            gpu.cloud_data, target, f"Paint as '{name}'"
                        )
                        self._brush_painting = True
                        # Build a per-stroke visibility z-buffer once, so
                        # the per-tick brush filter is O(C candidates),
                        # not O(N points) every 16 ms during a long drag.
                        try:
                            entry = self.entries[self.selected_index]
                            view = self.camera.get_view_matrix()
                            proj = self.camera.get_projection_matrix()
                            view_xform = view @ entry.model_transform
                            mvp = proj @ view_xform
                            self._brush_visibility_zbuf = (
                                self._build_visibility_zbuffer(gpu, mvp, view_xform)
                            )
                        except Exception:
                            self._brush_visibility_zbuf = None
                        self._do_brush_at(mx, my)
                    elif target < 0:
                        info = self.label_registry.get(self.active_label_id)
                        print(
                            f"Cannot paint '{info.name if info else self.active_label_id}': "
                            f"label is locked"
                        )
                self._click_detector.reset()
                return
            elif (action == glfw.RELEASE
                  and getattr(self, '_tool_engaged', False)):
                self._tool_engaged = False
                if self.active_tool == 'box' and self._drag_start is not None:
                    self._finish_box_drag()
                elif self.active_tool == 'lasso' and self._lasso_path:
                    self._finish_lasso_drag()
                elif self.active_tool == 'curve' and self._lasso_path:
                    self._finish_curve_drag()
                elif self.active_tool == 'polygon':
                    # Polygon UX:
                    #   single Shift/Ctrl-click  → append a vertex
                    #   double Shift/Ctrl-click  → close + commit polygon
                    #                              (needs >= 3 vertices)
                    # The first click of a double-pair already appended
                    # a vertex via the prior 'click' release; the second
                    # release fires here as 'double' and just commits.
                    if click_kind == 'click':
                        self._lasso_path.append((mx, my))
                    elif click_kind == 'double':
                        if len(self._lasso_path) >= 3:
                            self._finish_lasso_drag()
                        else:
                            self._lasso_path = []
                elif self.active_tool == 'brush':
                    self._brush_painting = False
                    # Drop the cached visibility z-buffer regardless of
                    # whether the stroke produced a command.
                    self._brush_visibility_zbuf = None
                    if self.stroke_recorder.is_active:
                        cmd = self.stroke_recorder.end(self.undo_stack)
                        if cmd is not None:
                            print(
                                f"Brush painted {len(cmd.point_indices):,} points as "
                                f"'{self._label_name_for(cmd.new_label)}'"
                            )
                            # Stroke is no longer in progress, so flush
                            # the labels file once for the whole drag.
                            if 0 <= self.selected_index < len(self.entries):
                                entry = self.entries[self.selected_index]
                                gpu = entry.full_gpu or entry.preview_gpu
                                if gpu is not None:
                                    self._persist_cloud_labels(entry, gpu.cloud_data)
                # Selection tools consume the click; don't also dispatch focus.
                self._click_detector.reset()
                return

        if self.mode == MODE_CONTACT_SHEETS and button in (0, 1, 2):
            # Click in gallery to highlight a cloud. Double-click promotes
            # it to Light Table. RMB: open context menu on click, or drag
            # to orbit the selected preview. Wheel scrolls; Ctrl+wheel
            # resizes cells (handled in _on_scroll).
            ready_pairs = self._gallery_filter_ready_cached()
            ready = [e for _, e in ready_pairs]
            sw = self._left_chrome_width()
            menu_h = int(getattr(self, '_menu_bar_height', 0))
            area_w = max(self.width - sw, 1)
            area_h = max(self.height - menu_h, 1)
            cols, rows, cell_w, _content_h = self._gallery_grid(len(ready), area_w)
            cell_h = cell_w
            scroll_y = int(round(self._gallery_scroll_y))

            # Scrollbar takes priority on LMB press — if the user
            # clicked the gutter we route to the scroll handler instead
            # of trying to select a cell behind it.
            if button == 0 and action == glfw.PRESS:
                if self._handle_gallery_scrollbar_press(mx, my, area_h):
                    return

            idx = gallery_layout.cell_index_from_mouse(
                mx, my, cols, cell_w, cell_h, len(ready),
                area_x=sw, area_y=menu_h, scroll_y=scroll_y,
            )
            ready_indices = [i for i, _ in ready_pairs]

            if action == glfw.PRESS:
                if button == 1:
                    # Select the hovered cell so any subsequent orbit drag
                    # rotates it. Stash the target for the release branch,
                    # which opens the context menu if no drag happened.
                    if idx >= 0:
                        target = ready_indices[idx]
                        # If the right-clicked cell isn't part of the
                        # current multi-selection, reset to just that one
                        # so the menu operates on a sensible target set.
                        if target not in self._gallery_multi_sel:
                            self._gallery_multi_sel = {target}
                            self._gallery_sel_anchor = target
                        self._on_cloud_selected(target, enter_light_table=False)
                        self._gallery_ctx_pending_target = target
                    else:
                        self._gallery_ctx_pending_target = -1
                    self._gallery_rmb_orbit_last = (mx, my)
                    self._gallery_rmb_dragged = False
                    return
                elif button == 0 and idx >= 0:
                    target = ready_indices[idx]
                    now = time.time()
                    is_double = (target == self._gallery_last_click_index and
                                 (now - self._gallery_last_click_time) < 0.5)
                    self._gallery_last_click_index = target
                    self._gallery_last_click_time = now

                    shift = bool(mods & glfw.MOD_SHIFT)
                    ctrl = bool(mods & glfw.MOD_CONTROL)

                    if shift and self._gallery_sel_anchor >= 0:
                        # Range-extend from the anchor through target
                        # (using ready_indices order so the visual range
                        # matches the grid the user sees).
                        try:
                            a = ready_indices.index(self._gallery_sel_anchor)
                        except ValueError:
                            a = idx
                        lo, hi = sorted((a, idx))
                        self._gallery_multi_sel = {
                            ready_indices[k] for k in range(lo, hi + 1)
                        }
                    elif ctrl:
                        # Toggle individual membership
                        if target in self._gallery_multi_sel:
                            self._gallery_multi_sel.discard(target)
                        else:
                            self._gallery_multi_sel.add(target)
                        self._gallery_sel_anchor = target
                    else:
                        # Plain click: single selection, reset anchor
                        self._gallery_multi_sel = {target}
                        self._gallery_sel_anchor = target

                    self._on_cloud_selected(target, enter_light_table=is_double)
                    return
            elif action == glfw.RELEASE:
                if button == 1:
                    # If a drag confirmed orbit mode, suppress the menu;
                    # otherwise commit the pending right-click to open it.
                    dragged = self._gallery_rmb_dragged
                    if (not dragged) and self._gallery_ctx_pending_target >= 0:
                        self._gallery_ctx_target = self._gallery_ctx_pending_target
                        self._gallery_ctx_open_pending = True
                    self._gallery_ctx_pending_target = -1
                    self._gallery_rmb_orbit_last = None
                    self._gallery_rmb_dragged = False
                    return


        # Gizmo interaction in Light Table — only when MOVE / ROTATE tool is active
        if (self.mode == MODE_LIGHT_TABLE
                and self.active_tool in ('move', 'rotate')
                and self.gizmo.visible
                and button == 0
                and 0 <= self.selected_index < len(self.entries)):
            if action == glfw.PRESS:
                view = self.camera.get_view_matrix()
                proj = self.camera.get_projection_matrix()
                gmx, gmy = self._window_to_fb(mx, my)
                vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
                ray_o, ray_d = unproject_mouse(
                    gmx - vp_x, gmy - vp_y, vp_w, vp_h, view, proj)
                axis = self.gizmo.hit_test(ray_o, ray_d)
                if axis != AXIS_NONE:
                    entry = self.entries[self.selected_index]
                    self.gizmo.start_drag(axis, entry.model_transform)
                    self._gizmo_drag_origin = ray_o
                    self._gizmo_drag_dir = ray_d
                    self._click_detector.reset()
                    return
            elif action == glfw.RELEASE:
                if self.gizmo.active_axis != AXIS_NONE:
                    self.gizmo.end_drag()
                    self._click_detector.reset()
                    return

        # Anchor drag — LMB press on a committed measurement anchor starts a
        # reposition drag. Must happen BEFORE camera dispatch so the camera
        # doesn't also receive the PRESS and begin panning.
        # Only active when registry has items; skipped during live anchor
        # placement (when the measure tool is actively building a new session
        # with anchors still being placed). Available in HOLOGRAM too so
        # measurements placed in the multi-cloud scene can be tweaked.
        if (self.mode == MODE_LIGHT_TABLE
                and button == 0):
            if action == glfw.PRESS and self._measure_drag_item_id is None:
                hit_result = self._measure_hit_test_anchors(mx, my)
                if hit_result is not None:
                    self._measure_drag_item_id, self._measure_drag_anchor_idx = hit_result
                    self._click_detector.reset()
                    return
            elif action == glfw.RELEASE and self._measure_drag_item_id is not None:
                self._measure_drag_item_id = None
                self._measure_drag_anchor_idx = -1
                self._click_detector.reset()
                return

        self.camera.on_mouse_button(button, action, mods)

        # Cursor-placement gestures — Light Table only, no active tool,
        # clean (non-drag) mouse releases so orbit drags are unaffected.
        # Spacebar is the place+zoom path; this handler covers the
        # silent "drop marker" gesture.
        #   Alt+LMB click → place cursor (no zoom)
        if (self.mode == MODE_LIGHT_TABLE and self.active_tool is None
                and button == 0 and click_kind == 'click'
                and (mods & glfw.MOD_ALT)):
            hit = self._pick_world_position(mx, my)
            if hit is not None:
                self.cursor3d.place(hit)

        # Measure tool anchor placement — plain LMB click snaps to nearest point.
        # No Shift/Ctrl modifier required. Camera still orbits on drag;
        # only clean clicks (no movement) commit an anchor. The
        # cloud-aware picker (``_pick_anchor``) returns a PickResult
        # carrying the cloud_key + cloud-local position, so the
        # measurement follows its bone through any later HOLOGRAM
        # reference-vertebra switch.
        if (self.mode == MODE_LIGHT_TABLE
                and self.active_tool in ('measure_line', 'measure_angle', 'measure_landmark')
                and button == 0 and click_kind == 'click'
                and self._measure is not None):
            pick = self._pick_anchor(mx, my)
            if pick is not None:
                self._measure.add_anchor(pick.as_anchor())
                self._measure.record_pop(pick.world_pos)  # expanding ring burst
                # Auto-commit when the measurement is now complete.
                if self._measure.complete:
                    self._commit_active_measurement()

        # HOLOGRAM 3D-viewport picker. Layered priority:
        #   1. Primitive pick (plane / axis / centroid / frame / line /
        #      box) toggles the inspector's selection set — same
        #      semantic as clicking the matching row in MEASUREMENTS.
        #      Ctrl-click replaces, plain click toggles.
        #   2. If the click missed every primitive, try a bone pick:
        #      toggle the bone in the compared set (same as clicking
        #      the matching button in the VERTEBRAE outliner).
        #      Shift-click sets the bone as the reference frame (also
        #      mirrors the outliner's shift-click behaviour).
        # Drags still orbit the camera; only no-movement clicks pick.
        # Suppressed when a measurement tool is active so the click is
        # treated as an anchor placement (handled above) rather than a
        # primitive/bone select.

    def _on_mouse_move(self, window, x, y):
        # Always feed the click detector so drags register as "moved"
        # even while the cursor is over the GUI sidebar.
        self._click_detector.on_move(x, y)

        # Gallery scrollbar drag — runs before radial / gui guards so the
        # user can drag the handle even if the cursor strays into the
        # sidebar / chrome briefly. Stops on LMB release.
        if self._gallery_scrollbar_dragging:
            menu_h = int(getattr(self, '_menu_bar_height', 0))
            area_h = max(self.height - menu_h, 1)
            self._scrollbar_drag_to(y, area_h)
            self.camera.set_mouse_pos(x, y)
            return

        # Radial menu owns mouse movement while it's open: update the
        # hovered slice and stop here so the camera never orbits in the
        # middle of a menu selection.
        if self._radial_menu_active and self._radial_menu_center is not None:
            from src.gui.radial_menu import slice_from_cursor
            cx, cy = self._radial_menu_center
            self._radial_menu_selected = slice_from_cursor(cx, cy, x, y)
            self.camera.set_mouse_pos(x, y)
            return

        if self.gui.wants_mouse():
            self.camera.set_mouse_pos(x, y)
            return

        # Gallery RMB-drag orbits only the selected preview cell.
        if (self.mode == MODE_CONTACT_SHEETS
                and self._rmb_held
                and self._gallery_rmb_orbit_last is not None
                and 0 <= self.selected_index < len(self.entries)):
            import math as _math
            lx, ly = self._gallery_rmb_orbit_last
            dx = x - lx
            dy = y - ly
            if dx * dx + dy * dy > 4:  # 2 px threshold to start orbit
                self._gallery_rmb_dragged = True
                entry = self.entries[self.selected_index]
                entry.orbit_az += dx * 0.01
                entry.orbit_el -= dy * 0.01
                entry.orbit_el = max(
                    -_math.pi / 2 + 0.05,
                    min(_math.pi / 2 - 0.05, entry.orbit_el),
                )
                self._gallery_rmb_orbit_last = (x, y)
            self.camera.set_mouse_pos(x, y)
            return

        # Committed anchor drag — highest priority in LIGHT TABLE /
        # HOLOGRAM; runs before selection tool and camera so the
        # dragged point follows the cursor. Re-pick the anchor through
        # ``_pick_anchor`` so the new position carries the (possibly
        # different) cloud_key + cloud-local position — supports
        # dragging an anchor off bone A onto bone B and having the
        # anchor follow B from then on.
        if (self.mode == MODE_LIGHT_TABLE
                and self._measure_drag_item_id is not None
                and self._measure_drag_anchor_idx >= 0):
            item = self.measure_registry.get_item(self._measure_drag_item_id)
            if item is not None and self._measure_drag_anchor_idx < len(item.anchors):
                pick = self._pick_anchor(x, y)
                if pick is not None:
                    item.anchors[self._measure_drag_anchor_idx] = pick.as_anchor()
            self.camera.set_mouse_pos(x, y)
            return

        # Measurement-tool hover update — works in LIGHT TABLE and
        # HOLOGRAM. Pulled out of the LIGHT-TABLE-only selection-tool
        # block below so HOLOGRAM's measurement tools get the same
        # real-time rubber-band preview. Doesn't consume the event,
        # so camera orbit remains usable between anchor placements.
        # Hover stays world-space — it's a transient cursor preview,
        # not a committed anchor.
        if (self.mode == MODE_LIGHT_TABLE
                and self.active_tool in (
                    'measure_line', 'measure_angle', 'measure_landmark')
                and self._measure is not None):
            self._measure.hover_pos = self._pick_world_position_any(x, y)

        # Active selection tool drag updates (paint / box / lasso /
        # brush etc. — LIGHT TABLE only, those tools don't exist in
        # HOLOGRAM).
        if self.mode == MODE_LIGHT_TABLE and self.active_tool is not None:
            # _drag_*/_lasso_path stay in window-logical pixels (the ImGui
            # overlay draws in that space); the brush hit-tests immediately
            # so it converts at the call (_do_brush_at -> _window_to_fb).
            if self.active_tool == 'box' and self._drag_start is not None:
                self._drag_current = (x, y)
                self.camera.set_mouse_pos(x, y)
                return
            elif self.active_tool == 'lasso' and self._lasso_path:
                # Subsample: only add if moved > 3 pixels
                lx, ly = self._lasso_path[-1]
                if (x - lx) ** 2 + (y - ly) ** 2 > 9:
                    self._lasso_path.append((x, y))
                self.camera.set_mouse_pos(x, y)
                return
            elif self.active_tool == 'curve' and self._lasso_path:
                lx, ly = self._lasso_path[-1]
                if (x - lx) ** 2 + (y - ly) ** 2 > 9:
                    self._lasso_path.append((x, y))
                self.camera.set_mouse_pos(x, y)
                return
            elif self.active_tool == 'brush' and self._brush_painting:
                self._do_brush_at(x, y)
                self.camera.set_mouse_pos(x, y)
                return

        # Gizmo dragging
        if (self.gizmo and self.gizmo.active_axis != AXIS_NONE and
                0 <= self.selected_index < len(self.entries)):
            view = self.camera.get_view_matrix()
            proj = self.camera.get_projection_matrix()
            gx, gy = self._window_to_fb(x, y)
            vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
            ray_o, ray_d = unproject_mouse(
                gx - vp_x, gy - vp_y, vp_w, vp_h, view, proj)
            delta = ray_o - self._gizmo_drag_origin
            entry = self.entries[self.selected_index]
            entry.model_transform = self.gizmo.update_drag(delta, entry.model_transform)
            self.camera.set_mouse_pos(x, y)
            return

        # Gizmo hover detection
        if (self.mode == MODE_LIGHT_TABLE and self.gizmo and self.gizmo.visible and
                0 <= self.selected_index < len(self.entries)):
            view = self.camera.get_view_matrix()
            proj = self.camera.get_projection_matrix()
            gx, gy = self._window_to_fb(x, y)
            vp_x, vp_y, vp_w, vp_h = self._primary_viewport_screen_rect()
            ray_o, ray_d = unproject_mouse(
                gx - vp_x, gy - vp_y, vp_w, vp_h, view, proj)
            self.gizmo.hovered_axis = self.gizmo.hit_test(ray_o, ray_d)

        # MMB drag adjusts depth-of-field aperture strength. Drag up to
        # increase, down to decrease. Auto-enables DOF on first drag.
        if self._mmb_held and self.mode == MODE_LIGHT_TABLE:
            dy = y - self._mmb_last_y
            self._mmb_last_y = y
            if abs(dy) > 0.5:
                self.dof_strength = max(0.0, min(20.0, self.dof_strength - dy * 0.05))
                self.dof_enabled = self.dof_strength > 0.0
            self.camera.set_mouse_pos(x, y)
            return

        # Suppress camera drag inside the click-ambiguity zone. If a mouse
        # button is held but the click detector hasn't yet confirmed it's a
        # drag (movement still under the click radius), we do NOT feed the
        # move to the camera. Otherwise a 2-3 px wobble during a would-be
        # click would produce a tiny pan AND fire a focus on release —
        # exactly the "feels off" micro-jitter you'd get on LMB tap+release.
        # Keep _last_mouse in sync via set_mouse_pos so the first post-
        # confirmation delta is a single-frame step, not an accumulated jump.
        button_held = self._lmb_held or self._rmb_held or self._mmb_held
        if button_held and not self._click_detector.any_confirmed_drag():
            self.camera.set_mouse_pos(x, y)
            return

        self.camera.on_mouse_move(x, y)

    def _on_scroll(self, window, x_offset, y_offset):
        self.gui.feed_scroll(y_offset)
        if self.gui.wants_mouse():
            return

        ctrl = (glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS or
                glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS)
        shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
                 glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # Contact Sheets wheel handling — three modes:
        #   Ctrl+wheel: resize cells (Lightroom-style icon zoom). Cursor-
        #               anchored so the cell under the cursor stays put.
        #   Shift+wheel: zoom the SELECTED preview cell's orbit camera
        #               (the old "scroll zooms preview" behavior, now
        #               opt-in so it doesn't fight with the new vertical
        #               scroll).
        #   plain wheel: vertical scroll the grid.
        if self.mode == MODE_CONTACT_SHEETS:
            if ctrl:
                self._gallery_zoom_cells(y_offset, window)
                return
            if shift and 0 <= self.selected_index < len(self.entries):
                factor = 1.15 if y_offset > 0 else (1.0 / 1.15)
                entry = self.entries[self.selected_index]
                entry.orbit_zoom = max(0.1, min(10.0, entry.orbit_zoom * factor))
                self.gallery_cache.invalidate_entry(entry)
                return
            self._gallery_scroll_by(y_offset)
            return
        if ctrl and self.mode == MODE_LIGHT_TABLE and self.active_tool is not None:
            if self.active_tool == 'brush':
                factor = 1.15 if y_offset > 0 else (1.0 / 1.15)
                self.brush_radius = max(0.001, self.brush_radius * factor)
                print(f"Brush radius: {self.brush_radius:.3f}")
            elif self.active_tool == 'curve':
                factor = 1.2 if y_offset > 0 else (1.0 / 1.2)
                self.curve_threshold_px = max(1.0, min(100.0, self.curve_threshold_px * factor))
                print(f"Curve threshold: {self.curve_threshold_px:.1f}px")
            elif self.active_tool in ('box', 'lasso'):
                # Adjust depth limit in units proportional to cloud scale
                if np.isinf(self.selection_max_depth):
                    # Initialize from current camera distance
                    self.selection_max_depth = self.camera.distance * 2.0
                factor = 1.15 if y_offset > 0 else (1.0 / 1.15)
                self.selection_max_depth = max(0.01, self.selection_max_depth * factor)
                print(f"Selection depth limit: {self.selection_max_depth:.3f}")
            return

        # MMB held + wheel = point size adjustment. While the middle button
        # is down (the camera pan button in this build), the wheel is
        # repurposed to grow / shrink the point sprite. LMB is deliberately
        # kept free for selection tools. Bounds mirror op1_sphere_size_gauge.
        if self._mmb_held and self.mode == MODE_LIGHT_TABLE:
            factor = 1.0 + y_offset * 0.08
            # Bounds mirror op1_sphere_size_gauge (min 0.005, max 10).
            new_size = max(0.005, min(self.point_size * factor, 10.0))
            self.point_size = new_size
            return

        # Default: dolly-under-cursor scroll zoom. We look for the actual
        # world point below the mouse cursor; if we hit something, the
        # pivot slides toward that point as the distance contracts, so
        # the scroll zooms "into what you're looking at".
        #
        # A cache keeps the pivot stable across a single scroll burst:
        # if the cursor is near its last position and the time between
        # ticks is small, reuse the cached world pivot. This prevents
        # micro-jitter when hovering over regions where _pick_world_position
        # resolves to adjacent points between ticks.
        world_pivot = None
        if self.mode == MODE_LIGHT_TABLE:
            mx, my = glfw.get_cursor_pos(window)
            now = time.perf_counter()
            cache = self._dolly_pivot_cache
            reuse = False
            if cache is not None:
                cached_pos, cached_mx, cached_my, cached_t = cache
                if (now - cached_t) < 0.30 and ((mx - cached_mx) ** 2 + (my - cached_my) ** 2) < 64.0:
                    world_pivot = cached_pos
                    reuse = True
            if not reuse:
                world_pivot = self._pick_world_position(mx, my)
                if world_pivot is not None:
                    self._dolly_pivot_cache = (world_pivot, mx, my, now)
        self.camera.dolly_at(y_offset, world_pivot)

    def _on_key(self, window, key, scancode, action, mods):
        self.gui.feed_key(key, action, mods)
        if self.gui.wants_keyboard():
            return

        if action != glfw.PRESS:
            return

        if key == glfw.KEY_DELETE:
            # Delete selected measurements from the registry (panel must be open).
            sel = getattr(self, 'measure_selection', set())
            if sel and getattr(self, 'measure_panel_open', False):
                self.measure_registry.remove_items(set(sel))
                self.measure_selection = set()
                self._measure_last_sel_id = None
                return

        if key == glfw.KEY_ESCAPE:
            # Priority order:
            # 1. Close the shortcuts overlay if open
            # 2. Drop the active selection tool
            # 3. Clear the current selection
            # 4. Go back to Contact Sheets if a gallery is available
            # ESC no longer closes the window — too easy to hit by accident
            # mid-labeling. Use the window X or Cmd/Alt+F4 to quit.
            if self.show_shortcuts:
                self.show_shortcuts = False
            elif self.active_tool is not None:
                # ESC also drops any polygon-in-progress vertices so the
                # path doesn't haunt the next tool selection.
                self._lasso_path = []
                self.active_tool = None
                self._update_tool_cursor()
                print("Tool cleared")
            elif self.selection_buffer.selected_count > 0:
                self.selection_buffer.clear()
                gpu = self._current_gpu_cloud()
                if gpu is not None:
                    update_selection(gpu, self.selection_buffer.mask)
                print("Selection cleared")
            elif self.mode != MODE_CONTACT_SHEETS and len(self.entries) > 1:
                self.mode = MODE_CONTACT_SHEETS
        elif key == glfw.KEY_F5:
            # Live shader reload — rebuilds every registered program
            # from disk. Core point cloud shader is not (yet) in the
            # registry, so only postprocess / compute shaders live-swap.
            new_objs, failed = reload_all_shaders(self.ctx)
            # Rebind VAOs for any FullscreenPass whose program swapped
            if self.pp_tonemap is not None:
                self.pp_tonemap.refresh_from_registry()
            if getattr(self, 'pp_gallery_blit', None) is not None:
                self.pp_gallery_blit.refresh_from_registry()
            if getattr(self, 'pp_ssao', None) is not None:
                self.pp_ssao.refresh_from_registry()
            if getattr(self, 'pp_ssao_blur', None) is not None:
                self.pp_ssao_blur.refresh_from_registry()
            if getattr(self, 'pp_outline', None) is not None:
                self.pp_outline.refresh_from_registry()
                try:
                    self.pp_outline.prog['u_mask'].value = 0
                except KeyError:
                    pass
            # SSAO program may have been swapped — re-upload kernel
            # and re-set sampler bindings so the new program reads
            # from the right texture units.
            if getattr(self, 'pp_ssao', None) is not None:
                from src.rendering.ssao import generate_kernel
                try:
                    self.pp_ssao.prog['u_kernel'].write(
                        generate_kernel(32).tobytes())
                except KeyError:
                    pass
                for name, unit in (('u_depth', 0), ('u_noise', 1)):
                    try:
                        self.pp_ssao.prog[name].value = unit
                    except KeyError:
                        pass
            if getattr(self, 'pp_ssao_blur', None) is not None:
                for name, unit in (('u_ssao', 0), ('u_depth', 1)):
                    try:
                        self.pp_ssao_blur.prog[name].value = unit
                    except KeyError:
                        pass
            if self.pp_tonemap is not None:
                for name, unit in (('u_hdr', 0), ('u_ao', 1)):
                    try:
                        self.pp_tonemap.prog[name].value = unit
                    except KeyError:
                        pass
            if getattr(self, 'gallery_cache', None) is not None:
                self.gallery_cache.invalidate_all()
            ok = ", ".join(new_objs.keys()) or "nothing"
            fail_s = f" | failed: {', '.join(failed)}" if failed else ""
            print(f"[shader reload] ok: {ok}{fail_s}")
        elif key == glfw.KEY_F6:
            # SSAO on/off — quick A/B for comparing AO contribution to
            # the unlit shading. The flag swaps the AO sampler input
            # in the tonemap between the blurred SSAO buffer and the
            # 1×1 white sentinel, so no shader recompile.
            self.ssao_enabled = not self.ssao_enabled
            print(f"[ssao] {'on' if self.ssao_enabled else 'off'}")
        elif key == glfw.KEY_H and not (mods & glfw.MOD_SHIFT):
            # H is context-dependent. In LIGHT TABLE it toggles
            # visibility of the *active label* so the user can hide
            # what they just painted to see what's underneath — much
            # faster than reaching for the eye icon in the label
            # panel when the hand is on the brush. Outside LIGHT
            # TABLE, plain H still toggles the whole GUI (clean
            # screenshot mode), the long-standing behavior.
            if self.mode == MODE_LIGHT_TABLE:
                aid = int(self.active_label_id)
                if aid != 0:
                    info = self.label_registry.get(aid)
                    if info is not None:
                        self.label_registry.set_visible(aid, not info.visible)
                        state = "visible" if info.visible else "hidden"
                        # `info.visible` is the BOOL on the entry; we
                        # just flipped it via set_visible which mutates
                        # in place, so `state` reflects the new state.
                        print(f"Label '{info.name}' (id={aid}): {state}")
            else:
                self.gui_visible = not self.gui_visible
        elif key == glfw.KEY_H and (mods & glfw.MOD_SHIFT):
            # Shift+H keeps the global GUI-visibility toggle reachable
            # in LIGHT TABLE too, so the user can still take a clean
            # screenshot without leaving the tab.
            self.gui_visible = not self.gui_visible
        elif key == glfw.KEY_G and not (mods & glfw.MOD_SHIFT):
            # G in LIGHT TABLE = cycle through the project's label
            # registry. Hand stays on the brush; no mouse trip to the
            # label panel. In other modes G keeps the legacy
            # LIGHT TABLE ↔ CONTACT SHEETS swap (rarely used now that
            # numbered hotkeys 1–6 jump to any tab directly).
            if self.mode == MODE_LIGHT_TABLE:
                self._cycle_active_label(direction=1)
            elif len(self.entries) > 1:
                self.mode = MODE_CONTACT_SHEETS
            elif self.mode == MODE_CONTACT_SHEETS:
                self.mode = MODE_LIGHT_TABLE
        # Number keys map to tabs in workflow order: 1=SHEETS, 2=LIGHT, 3=TRAIN
        elif key == glfw.KEY_1 and not (mods & glfw.MOD_CONTROL):
            self.mode = MODE_CONTACT_SHEETS
        elif key == glfw.KEY_2 and not (mods & glfw.MOD_CONTROL):
            self.mode = MODE_LIGHT_TABLE
        elif key == glfw.KEY_3 and not (mods & glfw.MOD_CONTROL):
            self.mode = MODE_AUTOMATION
        elif key == glfw.KEY_T:
            self.camera.set_preset('top')
        elif key == glfw.KEY_F:
            # F = focus on the 3D cursor / selected cloud. If the cursor
            # is placed, zoom in tight on it rather than framing all clouds.
            if self.cursor3d.visible:
                self.camera.focus_and_frame(self.cursor3d.position, tighten=0.55)
            elif 0 <= self.selected_index < len(self.entries):
                e = self.entries[self.selected_index]
                if e.bounds_min is not None:
                    self.camera.fit_to_bounds(e.bounds_min, e.bounds_max)
            else:
                self._fit_camera()
        elif key == glfw.KEY_C:
            # C in LIGHT TABLE = un-hide every label (counterpart to
            # H, which toggles the active label's visibility). After
            # H-ing through a few labels to inspect the layers
            # individually, a single C restores the full painted
            # state without having to chase down each one in the
            # label panel. Outside LIGHT TABLE, C keeps its legacy
            # job of clearing the 3D cursor + reframing the camera.
            if self.mode == MODE_LIGHT_TABLE:
                registry = getattr(self, "label_registry", None)
                if registry is not None:
                    changed = 0
                    for info in registry.all_labels():
                        if info.id != 0 and not info.visible:
                            registry.set_visible(info.id, True)
                            changed += 1
                    if changed:
                        print(f"Restored {changed} hidden label(s)")
            else:
                self.cursor3d.clear()
                self._fit_camera()
        elif key == glfw.KEY_R:
            self.camera.set_preset('right')
        elif key == glfw.KEY_I:
            self.camera.set_preset('iso')
        elif key == glfw.KEY_SPACE:
            # Space = place the 3D cursor under the mouse. Pick a point
            # on the visible cloud first; fall back to the ground plane
            # at the current orbit-target height. Also focuses the
            # orbit pivot on the hit so subsequent orbits feel local.
            if self.mode == MODE_LIGHT_TABLE:
                mx, my = glfw.get_cursor_pos(window)
                # _pick_world_position already falls back to a
                # ray-plane intersection when no cloud point is under
                # the cursor, and returns None for clicks outside the
                # primary viewport (sidebar, secondary views, gap).
                hit = self._pick_world_position(mx, my)
                if hit is not None:
                    self.cursor3d.place(hit)
                    target_distance = self.camera.distance
                    if 0 <= self.selected_index < len(self.entries):
                        entry = self.entries[self.selected_index]
                        if (entry.bounds_min is not None
                                and entry.bounds_max is not None):
                            extent = float(np.linalg.norm(
                                entry.bounds_max - entry.bounds_min))
                            target_distance = max(extent * 0.22, 0.05)
                    self.camera.focus_at_distance(hit, target_distance)
        elif key == glfw.KEY_B and (mods & glfw.MOD_SHIFT):
            # Shift+B → toggle bbox overlay (was plain B)
            self.show_bbox = not self.show_bbox
        elif key == glfw.KEY_B and not (mods & glfw.MOD_CONTROL):
            # B in LIGHT TABLE = previous cloud (pairs with V = next).
            # The box-tool shortcut moved to Shift+Alt+B since the
            # hot-path during labelling is navigation, not tool swap.
            # Outside LIGHT TABLE plain B does nothing (no box tool
            # anywhere else anyway).
            if self.mode == MODE_LIGHT_TABLE:
                if self.sequence is not None:
                    self._seek_sequence(self.sequence.current_index - 1)
                else:
                    self._navigate_cloud(-1)
            else:
                self._toggle_tool('box')
        elif key == glfw.KEY_G and (mods & glfw.MOD_SHIFT):
            self.show_grid = not self.show_grid
        elif key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL) and (mods & glfw.MOD_SHIFT):
            # Ctrl+Shift+Z = redo. Must come before the plain Ctrl+Z
            # branch below so the shift modifier isn't swallowed by undo.
            self._do_redo()
        elif key == glfw.KEY_Z and (mods & glfw.MOD_CONTROL):
            self._do_undo()
        elif key == glfw.KEY_ENTER or key == glfw.KEY_KP_ENTER:
            # Legacy two-step commit is gone — paint tools now write
            # labels directly onto the active layer. Leave a hint so
            # users with the old muscle memory know what changed.
            print("Tip: paint directly onto the active layer — "
                  "no explicit apply needed. Ctrl-drag to erase.")
        elif key == glfw.KEY_P and not (mods & glfw.MOD_CONTROL):
            self._toggle_tool('pick')
        elif key == glfw.KEY_O and not (mods & glfw.MOD_CONTROL):
            self._toggle_tool('lasso')
        elif key == glfw.KEY_K and not (mods & glfw.MOD_CONTROL):
            self._toggle_tool('brush')
        elif key == glfw.KEY_U and not (mods & glfw.MOD_CONTROL):
            self._toggle_tool('curve')
        elif key in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
            if self.sequence is not None:
                delta = -1 if key == glfw.KEY_LEFT else 1
                self._seek_sequence(self.sequence.current_index + delta)
            elif self.mode == MODE_LIGHT_TABLE:
                self._navigate_cloud(1 if key == glfw.KEY_RIGHT else -1)
        elif key == glfw.KEY_V and not (mods & (glfw.MOD_CONTROL | glfw.MOD_ALT
                                                | glfw.MOD_SHIFT)):
            # V = next cloud. Same effect as pressing → but reachable
            # with the left hand at typing position; saves the wrist
            # stretch over a long labeling session. Sequence mode
            # advances the timeline; Light Table advances the
            # gallery list.
            if self.sequence is not None:
                self._seek_sequence(self.sequence.current_index + 1)
            elif self.mode == MODE_LIGHT_TABLE:
                self._navigate_cloud(1)
        elif key == glfw.KEY_C and not (mods & (glfw.MOD_CONTROL | glfw.MOD_ALT
                                                | glfw.MOD_SHIFT)):
            # C = previous cloud. Mirrors V on the keyboard so the
            # left hand can navigate the gallery without leaving the
            # home row. Modifier'd C (Ctrl+C copy etc.) stays free.
            if self.sequence is not None:
                self._seek_sequence(self.sequence.current_index - 1)
            elif self.mode == MODE_LIGHT_TABLE:
                self._navigate_cloud(-1)
        elif key == glfw.KEY_SLASH and (mods & glfw.MOD_SHIFT):
            # Shift+/ = ? key
            self.show_shortcuts = not self.show_shortcuts
        elif key == glfw.KEY_L and not (mods & glfw.MOD_CONTROL):
            # Toggle label blend; persist so the user's choice survives
            # restart (otherwise every fresh launch reverts to the default
            # and thumbnails go back to whatever default mode we ship).
            self.label_blend = 1.0 if self.label_blend < 0.5 else 0.0
            print(f"Label blend: {self.label_blend}")
            try:
                from src.utils.prefs import update_prefs
                update_prefs({"label_blend": float(self.label_blend)})
            except Exception as e:
                print(f"[prefs] save label_blend failed: {e}")
        elif key == glfw.KEY_E and (mods & glfw.MOD_CONTROL) and (mods & glfw.MOD_SHIFT):
            from src.gui.panels import _do_spin
            _do_spin(self)
        elif key == glfw.KEY_E and (mods & glfw.MOD_CONTROL) and not (mods & glfw.MOD_SHIFT):
            from src.gui.panels import _do_screenshot
            _do_screenshot(self)
        elif key == glfw.KEY_M and not (mods & glfw.MOD_CONTROL):
            # M = toggle measurement panel (only if there are measurements)
            if self.measure_registry.items or self.measure_panel_open:
                self.measure_panel_open = not self.measure_panel_open

    def _on_char(self, window, char):
        self.gui.feed_char(char)

    def _on_drop(self, window, paths):
        """Handle drag-and-drop of files or directories."""
        for path in paths:
            self.import_path(path)

    def import_path(self, path: str):
        """Import a file or directory into the viewer.

        Directory handling order:
          1. 4D point-cloud sequence (numbered .ply/.las/.laz) -> sequence view
          2. Mixed directory of point cloud files -> gallery load
        """
        if os.path.isdir(path):
            # Hierarchical dataset detection: if the top level has no
            # supported files but descendants do, walk recursively.
            top_level_files = scan_directory(path)
            if not top_level_files:
                nested = scan_directory_recursive(path, max_files=50000)
                if nested:
                    self.load_directory_recursive(path)
                    return
            # Default: existing load_directory behavior (4D sequence or gallery)
            self.load_directory(path)
            return

        if os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext not in ('.ply', '.las', '.laz', '.npz'):
                print(f"Unsupported format: {path}")
                return
            self.load_file(path)
            self._fit_camera()
            # Point cloud files: check for companion _labels.npy
            self._maybe_auto_apply_label_array(path)
            return

        print(f"Path not found: {path}")

    def _maybe_auto_apply_label_array(self, cloud_path: str):
        """After loading a point cloud, check for a companion _labels.npy.

        LS-3: looks up the entry by ``cloud_path`` (canonical absolute
        path) rather than blindly taking ``self.entries[-1]``. The
        last-entry assumption breaks when load_file hit the dedup path
        (refresh in place, no append) or when another thread queued a
        load between load_file and this call.
        """
        from src.data.labels_io import (
            find_companion_label_array, apply_label_array_to_cloud,
        )
        from src.rendering.label_texture import update_label_color_texture
        from src.rendering.point_cloud_renderer import re_upload_labels

        companion = find_companion_label_array(cloud_path)
        if companion is None:
            return
        target_path = os.path.abspath(cloud_path)
        entry = None
        for e in self.entries:
            if os.path.abspath(getattr(e, "file_path", "")) == target_path:
                entry = e
                break
        if entry is None:
            print(f"[companion] no entry for {target_path}; skipping")
            return
        gpu = entry.full_gpu or entry.preview_gpu
        if gpu is None:
            return
        cloud = gpu.cloud_data

        try:
            labeled, created = apply_label_array_to_cloud(
                cloud, companion, self.label_registry,
            )
        except Exception as e:
            print(f"Companion label array apply failed: {e}")
            return

        print(f"Auto-paired labels from {os.path.basename(companion)}: "
              f"{labeled:,} points labeled, {len(created)} new labels created")
        if self.label_texture is not None and created:
            update_label_color_texture(self.label_texture, self.label_registry)
        re_upload_labels(gpu, self.selection_buffer.mask)
        self.label_blend = 1.0
        self.label_count_cache.invalidate_cloud(cloud)
        # Persist to catalog so labels survive restart
        file_key = getattr(entry, 'file_key', None)
        if file_key and cloud.labels is not None:
            # CP-4 + LS-7 (post-audit): per-key lock + status banner on
            # failure. The companion auto-pair runs after load_file
            # returns, which is exactly the window a background
            # mesh-build worker would race a label save on the same key.
            err = save_cloud_labels(
                file_key, cloud.labels, catalog=self.catalog)
            if err:
                try:
                    self.set_status_banner(
                        f"Companion label save failed for {entry.name}: {err}",
                        level="error",
                        source=f"cloud_store.save_cloud_labels:{file_key}",
                    )
                except AttributeError:
                    pass

    # Retained for the main menu bar + older call sites.
    def import_file_dialog(self):
        self._smart_import_dialog(category='all')

    def import_folder_dialog(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askdirectory(title="Import Folder")
            root.destroy()
            if path:
                self.import_path(path)
        except Exception as e:
            print(f"Folder dialog error: {e}")

    def add_points_dialog(self):
        """ADD button: native point-cloud formats (ply/las/laz/npz)."""
        self._smart_import_dialog(category='points')

    def _smart_import_dialog(self, category: str):
        """Unified chooser that autodetects file / files / directory.

        category: 'points'  -> .ply / .las / .laz / .npz
                  'all'     -> everything (used by the menu bar)

        Shows a multi-select file picker with the right filter. If the
        user cancels without picking anything we fall back to a folder
        picker so the same button covers the "pick a folder" case too.
        import_path then dispatches each path (file or directory) on
        its own.
        """
        if category == 'points':
            title = "Add Point Clouds"
            filetypes = [
                ("Point clouds", "*.ply *.las *.laz *.npz"),
                ("PLY files", "*.ply"),
                ("LAS/LAZ files", "*.las *.laz"),
                ("NPZ files", "*.npz"),
                ("All files", "*.*"),
            ]
        else:
            title = "Import"
            filetypes = [
                ("Point clouds", "*.ply *.las *.laz *.npz"),
                ("All files", "*.*"),
            ]

        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            paths = filedialog.askopenfilenames(title=title, filetypes=filetypes)
            if not paths:
                # Fallback: let the user pick a folder from the same button.
                folder = filedialog.askdirectory(title=f"{title} — folder")
                root.destroy()
                if folder:
                    self.import_path(folder)
                return
            root.destroy()
        except Exception as e:
            print(f"Import dialog error: {e}")
            return

        for p in paths:
            self.import_path(p)


def main():
    # Dispatch to CLI if a subcommand is given
    if len(sys.argv) > 1 and sys.argv[1] in ('render', 'spin'):
        from src.cli import cli_main
        cli_main(sys.argv[1:])
        return

    app = App()
    app.run()


if __name__ == '__main__':
    main()
