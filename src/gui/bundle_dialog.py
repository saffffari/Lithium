"""File > Import / Export Project (.li) — modals and the progress overlay.

Three pieces:

* the EXPORT modal: pick a project, tick clouds / labels / models with
  live byte estimates, choose a file, go;
* IMPORT: pick a ``.li`` and it comes in as-is — a new project, or a
  merge into the project it was exported from when that still exists;
* the PROGRESS overlay: a lithium atom that assembles itself from loose
  particles as the job runs (``src.gui.atom_loader``) and settles into a
  stable Li-7 when it finishes.

The work itself runs on a worker thread through ``src.data.bundle``;
progress and completion come back through ``app.post_event`` and are
applied on the main thread, like inference and training already do.
"""

from __future__ import annotations

import os
import threading
import time

import imgui

from src.data import bundle
from src.gui.atom_loader import AtomSim, draw_atom
from src.gui.scale import s
from src.gui.theme import *


# ---------------------------------------------------------------------------
# Module state (immediate mode keeps nothing for us)
# ---------------------------------------------------------------------------

_export_pending = False       # menu asked for the export modal
_export_proj_idx = 0
_export_spec = bundle.ExportSpec(clouds=True, labels=True, models=False)
_export_plan: bundle.ExportPlan | None = None
_export_plan_pid: str | None = None



class _Job:
    """One running (or just finished) export / import."""

    def __init__(self, kind: str, title: str):
        self.kind = kind                  # "export" | "import"
        self.title = title
        self.stage = "starting"
        self.done = 0
        self.total = 0
        self.nbytes = 0
        self.started = time.monotonic()
        self.finished_at: float | None = None
        self.error: str | None = None
        self.cancelled = False
        self.cancel = threading.Event()
        self.result = None                # manifest (export) / ImportResult (import)
        self.out_path = ""
        self.thread: threading.Thread | None = None
        self.atom = AtomSim()             # the progress animation
        self.atom_clock = self.started

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0 if self.finished_at is None else 1.0
        return min(1.0, self.done / self.total)

    @property
    def running(self) -> bool:
        return self.finished_at is None


_job: _Job | None = None


# ---------------------------------------------------------------------------
# Menu entry points
# ---------------------------------------------------------------------------

def request_export(app) -> None:
    global _export_pending, _export_proj_idx, _export_plan, _export_plan_pid
    _export_pending = True
    _export_plan = None
    _export_plan_pid = None
    projects = _projects(app)
    view = getattr(app, "active_view", None)
    if view and view[0] == "project":
        for i, p in enumerate(projects):
            if p.id == view[1]:
                _export_proj_idx = i
                break
    _export_proj_idx = min(_export_proj_idx, max(0, len(projects) - 1))


def request_import(app) -> None:
    """Pick a ``.li`` and import whatever it contains, no questions.

    The bundle's manifest already says what is inside. Target rule:
    if this catalog still has the project the bundle was exported from
    (same ``source_id``), the contents merge into it — that is a
    reader's labels coming home; otherwise a new project is created
    under the bundle's name (de-duplicated). Labels never overwrite
    ones the target already has.
    """
    path = _pick_open_path()
    if path:
        import_bundle_path(app, path)


def import_bundle_path(app, path: str) -> None:
    """Import ``path`` with the same rules as File > Import Project.
    Also used for a ``.li`` dropped on the window or given on the
    command line."""
    try:
        m = bundle.inspect_bundle(path)
    except bundle.BundleError as e:
        app.set_status_banner(f"Import Project: {e}", level="error", source="bundle")
        return
    into = None
    src_id = m["project"].get("source_id")
    if src_id and src_id in app.catalog.projects:
        into = src_id
    _start_import(app, path, None, into, overwrite=False)


def _projects(app):
    cat = getattr(app, "catalog", None)
    if cat is None:
        return []
    return cat.user_projects_recent()


# ---------------------------------------------------------------------------
# Native pickers (tkinter, as the rest of the app does)
# ---------------------------------------------------------------------------

def _pick_open_path() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Import Lithium project",
            filetypes=[("Lithium project bundle", "*.li"), ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception as e:
        print(f"[bundle] open dialog error: {e}")
        return None


def _pick_save_path(default_name: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title="Export Lithium project",
            initialfile=default_name,
            defaultextension=bundle.EXTENSION,
            filetypes=[("Lithium project bundle", "*.li")])
        root.destroy()
        return path or None
    except Exception as e:
        print(f"[bundle] save dialog error: {e}")
        return None


def _safe_filename(name: str) -> str:
    keep = "".join(c if (c.isalnum() or c in " -_.") else "_" for c in name).strip()
    return (keep or "project").replace(" ", "_")


# ---------------------------------------------------------------------------
# Worker launch + event plumbing
# ---------------------------------------------------------------------------

def _ensure_handlers(app) -> None:
    if getattr(app, "_bundle_handlers_registered", False):
        return
    app.register_event_handler("bundle_progress", _on_progress)
    app.register_event_handler("bundle_finished", _on_finished)
    app._bundle_handlers_registered = True


def _on_progress(app, payload) -> None:
    if _job is None:
        return
    _job.stage, _job.done, _job.total, _job.nbytes = payload


def _on_finished(app, payload) -> None:
    global _job
    if _job is None:
        return
    _job.finished_at = time.monotonic()
    _job.error = payload.get("error")
    _job.cancelled = bool(payload.get("cancelled"))
    _job.result = payload.get("result")
    if _job.error:
        app.set_status_banner(f"{_job.title}: {_job.error}", level="error", source="bundle")
        return
    if _job.cancelled:
        return
    if _job.kind == "import" and _job.result is not None:
        try:
            proj = bundle.apply_import(app.catalog, _job.result)
            _job.out_path = proj.name
            app.set_active_view(("project", proj.id))
            app._sync_project_state(("project", proj.id))
        except Exception as e:
            _job.error = f"{type(e).__name__}: {e}"
            app.set_status_banner(f"Import Project: {_job.error}", level="error", source="bundle")


def _start(app, kind: str, title: str, fn) -> None:
    """Run ``fn(progress, cancel)`` on a worker; results come back as events."""
    global _job
    _ensure_handlers(app)
    job = _Job(kind, title)
    _job = job

    def progress(stage, done, total, nbytes):
        app.post_event("bundle_progress", (stage, done, total, nbytes))

    def run():
        try:
            result = fn(progress, job.cancel)
            app.post_event("bundle_finished", {"result": result})
        except bundle.BundleCancelled:
            app.post_event("bundle_finished", {"cancelled": True})
        except Exception as e:  # surfaced in the banner, never swallowed
            app.post_event("bundle_finished", {"error": f"{type(e).__name__}: {e}"})

    job.thread = threading.Thread(target=run, name=f"bundle-{kind}", daemon=True)
    job.thread.start()


def _start_export(app, project_id: str, spec: bundle.ExportSpec, out_path: str) -> None:
    _start(app, "export", "Export Project",
           lambda progress, cancel: bundle.export_bundle(
               app.catalog, project_id, spec, out_path, progress=progress, cancel=cancel))
    _job.out_path = out_path


def _start_import(app, path: str, project_name: str | None, into: str | None,
                  overwrite: bool) -> None:
    _start(app, "import", "Import Project",
           lambda progress, cancel: bundle.import_bundle(
               app.catalog, path, project_name=project_name, into_project_id=into,
               overwrite_labels=overwrite, progress=progress, cancel=cancel))


# ---------------------------------------------------------------------------
# Modals (called at top level of the sidebar draw so nothing clips them)
# ---------------------------------------------------------------------------

def draw_bundle_modals(app) -> None:
    global _export_pending
    if _export_pending:
        imgui.open_popup("Export Project##bundle")
        _export_pending = False
    _draw_export_modal(app)


def _begin_modal(name: str) -> bool:
    imgui.push_style_color(imgui.COLOR_POPUP_BACKGROUND, 0.04, 0.04, 0.04, 0.98)
    imgui.push_style_color(imgui.COLOR_BORDER, *ORANGE)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (s(18), s(14)))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 1.0)
    opened = imgui.begin_popup_modal(
        name, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE)[0]
    if not opened:
        imgui.pop_style_var(2)
        imgui.pop_style_color(2)
    return opened


def _end_modal() -> None:
    imgui.end_popup()
    imgui.pop_style_var(2)
    imgui.pop_style_color(2)


def _section(text: str) -> None:
    imgui.spacing()
    imgui.text_colored(text, *CORAL[:3])
    imgui.spacing()


def _draw_export_modal(app) -> None:
    global _export_proj_idx, _export_plan, _export_plan_pid, _export_spec
    if not _begin_modal("Export Project##bundle"):
        return
    projects = _projects(app)
    if not projects:
        imgui.text_colored("No projects in this catalog.", *DIM[:3])
        if imgui.button("Close", width=s(90)):
            imgui.close_current_popup()
        _end_modal()
        return

    _section("PROJECT")
    imgui.push_item_width(s(360))
    changed, _export_proj_idx = imgui.combo(
        "##bundle_export_proj", _export_proj_idx, [p.name for p in projects])
    imgui.pop_item_width()
    proj = projects[_export_proj_idx]
    if _export_plan is None or _export_plan_pid != proj.id or changed:
        try:
            _export_plan = bundle.plan_export(app.catalog, proj.id)
        except bundle.BundleError as e:
            imgui.text_colored(str(e), *RED[:3])
            _export_plan = None
        _export_plan_pid = proj.id
    plan = _export_plan

    _section("INCLUDE")
    if plan is not None:
        _, _export_spec.clouds = imgui.checkbox(
            f"Point clouds   {len(plan.cloud_keys)} clouds · "
            f"{bundle.human_bytes(plan.cloud_bytes)}##bundle_clouds", _export_spec.clouds)
        imgui.same_line()
        imgui.text_colored("colour always included", *DIM[:3])
        if plan.missing_clouds:
            imgui.text_colored(f"   {plan.missing_clouds} project members have no catalog data "
                               f"and will be skipped", *AMBER[:3])
        _, _export_spec.labels = imgui.checkbox(
            f"Labels   {len(plan.label_keys)} of {len(plan.cloud_keys)} clouds · "
            f"{bundle.human_bytes(plan.label_bytes)}##bundle_labels", _export_spec.labels)
        if plan.models:
            _, _export_spec.models = imgui.checkbox(
                f"Model weights   {len(plan.models)} · "
                f"{bundle.human_bytes(plan.model_bytes)}##bundle_models", _export_spec.models)
            for m in plan.models:
                imgui.text_colored(f"      {m['name']}  ·  {bundle.human_bytes(sum(b for _, _, b in m['files']))}",
                                   *DIM[:3])
        else:
            imgui.text_colored("Model weights   none trained in this project", *DIM[:3])
            _export_spec.models = False

        imgui.spacing()
        total = plan.total_bytes(_export_spec)
        nothing = not (_export_spec.clouds or _export_spec.labels or _export_spec.models)
        if nothing:
            imgui.text_colored("Pick at least one.", *AMBER[:3])
        else:
            imgui.text_colored(f"About {bundle.human_bytes(total)} on disk "
                               f"(clouds shrink once stripped to coord + colour + normal).",
                               *DIM[:3])
        if not _export_spec.labels and plan.label_keys:
            imgui.text_colored("Labels and every label-bearing scalar are left out — "
                               "safe for a blinded reader.", *TEAL[:3])

    imgui.spacing()
    imgui.separator()
    imgui.spacing()
    go = imgui.button("Export...", width=s(110))
    imgui.same_line()
    cancel = imgui.button("Cancel", width=s(90))
    if go and plan is not None and (_export_spec.clouds or _export_spec.labels or _export_spec.models):
        suffix = "".join(c for c, on in (("c", _export_spec.clouds), ("l", _export_spec.labels),
                                         ("m", _export_spec.models)) if on)
        default = f"{_safe_filename(proj.name)}_{suffix}{bundle.EXTENSION}"
        path = _pick_save_path(default)
        if path:
            _start_export(app, proj.id, bundle.ExportSpec(
                _export_spec.clouds, _export_spec.labels, _export_spec.models), path)
            imgui.close_current_popup()
    elif cancel:
        imgui.close_current_popup()
    _end_modal()


# ---------------------------------------------------------------------------
# Progress overlay
# ---------------------------------------------------------------------------

_HOLD_AFTER_DONE = 2.6      # seconds the finished card stays before auto-closing


def draw_bundle_overlay(app) -> None:
    """Draw the running / finished job card. Call once per frame after
    the main panels so it sits on top of everything."""
    global _job
    job = _job
    if job is None:
        return
    now = time.monotonic()
    if not job.running and not job.error and (now - job.finished_at) > _HOLD_AFTER_DONE:
        # let the atom finish forming first (bounded, in case of a stall)
        if job.atom.complete or (now - job.finished_at) > _HOLD_AFTER_DONE + 3.0:
            _job = None
            return

    W, H = app.width, app.height
    cw, ch = s(440), s(300)
    x0, y0 = (W - cw) * 0.5, (H - ch) * 0.5
    dl = imgui.get_foreground_draw_list()
    th = imgui.get_text_line_height()

    # Veil + card, matching the shortcuts overlay.
    dl.add_rect_filled(0, 0, W, H, col32((0.0, 0.0, 0.0, 0.55)))
    dl.add_rect_filled(x0, y0, x0 + cw, y0 + ch, col32((0.04, 0.04, 0.04, 0.97)))
    border = RED if job.error else ORANGE
    dl.add_rect(x0, y0, x0 + cw, y0 + ch, col32(border), 0.0, 0, 2.0)

    title = job.title.upper()
    tw = imgui.calc_text_size(title)[0]
    dl.add_text(x0 + (cw - tw) * 0.5, y0 + th * 0.6, col32(border), title)

    # The atom: loose particles snap into a Li-7 nucleus, then into the
    # three rings of the POINT SIZE icon, as the job progresses. A failed
    # or cancelled job lets them fly apart instead.
    cx, cy = x0 + cw * 0.5, y0 + th * 2.2 + s(70)
    R = s(52)
    frac = job.fraction
    t_run = now - job.started
    ok = job.finished_at is not None and not job.error and not job.cancelled
    sim = job.atom
    if job.error or job.cancelled:
        sim.decay()
    sim.step(now - job.atom_clock, frac, finished=ok)
    job.atom_clock = now
    tint = "error" if job.error else ("cancel" if job.cancelled else "normal")
    glow = (now - job.finished_at) if (ok and sim.complete) else 0.0
    dl.push_clip_rect(x0 + 1, y0 + th * 1.8, x0 + cw - 1, y0 + ch - 1, True)
    draw_atom(dl, sim, cx, cy, R, tint=tint, glow=glow)
    dl.pop_clip_rect()

    # Status lines under the atom.
    ly = cy + R + s(22)
    if job.error:
        lines = [("Failed", RED), (job.error[:70], DIM)]
    elif job.cancelled:
        lines = [("Cancelled — nothing was written", AMBER)]
    elif job.running:
        stage = {"clouds": "writing clouds" if job.kind == "export" else "unpacking clouds",
                 "labels": "labels", "models": "model weights",
                 "manifest": "manifest", "starting": "starting"}.get(job.stage, job.stage)
        el = t_run
        eta = ""
        if job.done > 2 and job.total:
            eta = f"  ·  ~{(el / job.done) * (job.total - job.done):.0f} s left"
        pct = f"{int(frac * 100):d}%  ·  " if job.total else ""
        lines = [(f"{pct}{stage}  {job.done} / {job.total}", WHITE),
                 (f"{bundle.human_bytes(job.nbytes)}{eta}", DIM)]
    else:
        if job.kind == "export":
            lines = [(f"Done  ·  {bundle.human_bytes(job.nbytes)}", WHITE),
                     (os.path.basename(job.out_path), DIM)]
        else:
            r = job.result
            parts = []
            if r is not None:
                parts.append(f"{r.clouds_new} new clouds" + (f", {r.clouds_known} already here"
                                                            if r.clouds_known else ""))
                if r.labels_written or r.labels_skipped:
                    parts.append(f"{r.labels_written} label files"
                                 + (f" ({r.labels_skipped} skipped)" if r.labels_skipped else ""))
                if r.models_registered:
                    parts.append(f"{r.models_registered} model" + ("s" if r.models_registered > 1 else ""))
            lines = [(f"Imported  ·  {job.out_path}", WHITE), ("  ·  ".join(parts), DIM)]
    for text, col in lines:
        if not text:
            continue
        w = imgui.calc_text_size(text)[0]
        dl.add_text(x0 + (cw - w) * 0.5, ly, col32(col), text)
        ly += th * 1.25

    # One button: Cancel while running, Close afterwards. A borderless
    # window hosts it so clicks go through ImGui, not our own hit test.
    bw, bh = s(100), s(26)
    imgui.set_next_window_position(x0 + (cw - bw) * 0.5, y0 + ch - bh - s(14))
    imgui.set_next_window_size(bw + s(2), bh + s(2))
    flags = (imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE |
             imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_BACKGROUND | imgui.WINDOW_NO_SAVED_SETTINGS)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0.0, 0.0))
    imgui.begin("##bundle_progress_btn", flags=flags)
    if job.running:
        if imgui.button("Cancel", width=bw, height=bh):
            job.cancel.set()
    else:
        if imgui.button("Close", width=bw, height=bh) or imgui.is_key_pressed(imgui.get_key_index(imgui.KEY_ENTER)):
            _job = None
    imgui.end()
    imgui.pop_style_var()
