"""Pointcept PTv3 subprocess runner.

Launches Pointcept's ``tools/train.py`` as a subprocess inside the
user-configured training env (conda or WSL2 or any other interpreter)
and parses its stdout into structured progress events for the GUI.

Design
------
- ``PointceptLaunchConfig`` captures *where* to run (python exe,
  pointcept repo path, work dir, config file path, env vars) without
  caring about *what* to train. Any Pointcept-compatible config works.
- ``PointceptRunner`` owns the subprocess lifecycle (spawn, stream,
  cancel, wait) and publishes parsed events onto ``status`` and into
  a ring buffer of raw log lines for the panel.
- ``parse_pointcept_line`` is a pure function so the stdout parser
  can be unit-tested without spawning any subprocesses or torch.

This file intentionally does NOT import torch or pointcept. It's
imported by the viz app (which is torch-free) to *manage* training
that happens in a completely separate interpreter. Keeping it pure
Python means a broken training env can't break the viz app.

Not handled here
----------------
- Installing the training env (see ``env_bootstrap.py`` — future)
- Generating Pointcept configs (see ``config_gen.py``)
- Importing predictions back into the registry (see
  ``src/export/predictions_import.py`` — already built)
"""

import os
import re
import shutil
import sys
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Stdout parser
# ---------------------------------------------------------------------------

# Pointcept's default logger formats training lines roughly as:
#
#   [2024-01-15 14:23:45.123 main INFO] Train: [12/200][  50/150] ...
#     Loss 0.3421 (0.3812) Acc 0.8921 (0.8745) ...
#
# And validation lines as:
#
#   [2024-01-15 14:23:50.456 main INFO] Val result: mIoU/mAcc/allAcc
#   0.6834/0.7123/0.8921
#
# These patterns are deliberately loose — Pointcept's formatter has
# drifted between versions and we want to keep working across minor
# releases. Specifically we don't anchor to the timestamp prefix
# because tqdm output and errors from subprocesses sometimes break
# the prefix pattern.
#
# ``_NUM`` is the float-shaped capture used for every numeric field. The
# fractional part requires at least one digit AFTER the period, so a
# trailing sentence-ending period (Pointcept logs validation lines as
# ``... allAcc 0.7965.``) is left out of the capture instead of being
# slurped in and then crashing ``float()``. The previous loose form
# ``[0-9.eE+-]+`` killed the reader thread on every val-result line.
_NUM = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

_RE_TRAIN_STEP = re.compile(
    r"(?:Train|train)\s*:?\s*\[?\s*(\d+)\s*/\s*(\d+)\s*\]?"
    r"\s*\[?\s*(\d+)\s*/\s*(\d+)\s*\]?"
    # Pointcept formats this field as ``loss: 0.9197`` (lowercase,
    # colon-separated). The old ``Loss\s+`` form REQUIRED whitespace
    # immediately after "Loss" and refused to consume the colon, so
    # every train-step line silently failed to parse — which is why
    # the TRAIN tab's epoch + loss readouts stuck at 0 for the whole
    # run. ``[:\s]+`` lets the connector be a colon, whitespace, or
    # both, matching every Pointcept release we've seen.
    rf".*?Loss[:\s]+({_NUM})"
    rf"(?:\s*\(({_NUM})\))?"
    rf"(?:.*?Acc[:\s]+({_NUM}))?",
    re.IGNORECASE,
)

_RE_VAL_RESULT = re.compile(
    rf"(?:Val|val).*?mIoU(?:/mAcc/allAcc)?\s*"
    rf"({_NUM})\s*/\s*({_NUM})\s*/\s*({_NUM})",
    re.IGNORECASE,
)

_RE_EPOCH_SUMMARY = re.compile(
    rf"Epoch\s+(\d+)\s*(?:/\s*(\d+))?.*?"
    rf"(?:Loss|loss)\s*[:=]?\s*({_NUM})"
    rf"(?:.*?(?:mIoU|miou)\s*[:=]?\s*({_NUM}))?",
)

_RE_CKPT_SAVED = re.compile(
    # Pointcept's actual log line is `Saving checkpoint to: <path>`. The
    # `to:` has a colon AND a trailing space, so the connector class has
    # to accept colon OR whitespace (one or more). The old version used
    # `to\s+` which failed on the colon and captured "to:" itself as the
    # path — corrupting every registry entry to best_checkpoint='to:'.
    r"(?:Save|Saving|Saved)\s+(?:checkpoint|ckpt|model)\s+(?:to[:\s]+)?['\"]?([^'\"\s]+)['\"]?",
    re.IGNORECASE,
)

_RE_BEST_MIOU = re.compile(
    rf"Best\s+(?:val\s+)?mIoU\s*[:=]?\s*({_NUM})",
    re.IGNORECASE,
)

# Legacy format used by our in-house train_script.py — kept so the
# same parser can service both runners without confusing users who
# have a mix of runs in the panel.
_RE_LEGACY = re.compile(
    rf"epoch\s*[:\s]\s*(\d+)\s*/\s*(\d+).*loss\s*[:=]\s*({_NUM})"
    rf"(?:.*miou\s*[:=]\s*({_NUM}))?",
    re.IGNORECASE,
)


def _safe_float(s: str | None) -> float | None:
    """``float(s)`` that returns ``None`` instead of raising.

    The reader thread MUST NOT die on a malformed numeric capture — a
    parser glitch should drop one line, not poison the whole run.
    """
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _safe_int(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


@dataclass
class ParsedLine:
    """Structured view of one stdout line.

    Any field may be ``None`` — the parser fills in only what it
    matched. A completely unrecognized line yields all-None and is
    still appended to the raw log.
    """
    kind: str = "raw"  # 'raw' | 'train_step' | 'val_result' | 'epoch' | 'checkpoint' | 'best'
    epoch: int | None = None
    total_epochs: int | None = None
    step: int | None = None
    total_steps: int | None = None
    loss: float | None = None
    loss_avg: float | None = None
    acc: float | None = None
    miou: float | None = None
    macc: float | None = None
    all_acc: float | None = None
    checkpoint_path: str | None = None
    best_miou: float | None = None


def parse_pointcept_line(line: str) -> ParsedLine:
    """Parse a single stdout line into a ``ParsedLine``.

    Pure function. Order of patterns matters — we check the most
    specific patterns first so a per-step line doesn't accidentally
    match a val-result regex.
    """
    # Per-step training line (most common during a run)
    m = _RE_TRAIN_STEP.search(line)
    if m:
        return ParsedLine(
            kind="train_step",
            epoch=_safe_int(m.group(1)),
            total_epochs=_safe_int(m.group(2)),
            step=_safe_int(m.group(3)),
            total_steps=_safe_int(m.group(4)),
            loss=_safe_float(m.group(5)),
            loss_avg=_safe_float(m.group(6)),
            acc=_safe_float(m.group(7)),
        )

    # End-of-epoch validation result
    m = _RE_VAL_RESULT.search(line)
    if m:
        return ParsedLine(
            kind="val_result",
            miou=_safe_float(m.group(1)),
            macc=_safe_float(m.group(2)),
            all_acc=_safe_float(m.group(3)),
        )

    # Legacy / in-house epoch summary — also matches some Pointcept lines
    m = _RE_LEGACY.search(line)
    if m:
        return ParsedLine(
            kind="epoch",
            epoch=_safe_int(m.group(1)),
            total_epochs=_safe_int(m.group(2)),
            loss=_safe_float(m.group(3)),
            miou=_safe_float(m.group(4)),
        )

    m = _RE_EPOCH_SUMMARY.search(line)
    if m:
        return ParsedLine(
            kind="epoch",
            epoch=_safe_int(m.group(1)),
            total_epochs=_safe_int(m.group(2)),
            loss=_safe_float(m.group(3)),
            miou=_safe_float(m.group(4)),
        )

    m = _RE_CKPT_SAVED.search(line)
    if m:
        return ParsedLine(kind="checkpoint", checkpoint_path=m.group(1))

    m = _RE_BEST_MIOU.search(line)
    if m:
        return ParsedLine(kind="best", best_miou=_safe_float(m.group(1)))

    return ParsedLine(kind="raw")


# ---------------------------------------------------------------------------
# Launch config & status
# ---------------------------------------------------------------------------

@dataclass
class PointceptLaunchConfig:
    """Everything needed to spawn a Pointcept training subprocess.

    ``python_exe`` is a full path to the Python interpreter in the
    training env. Examples:
      - conda on Windows: ``<miniforge>\\envs\\lithium-train\\python.exe``
      - WSL2 Ubuntu:      ``wsl.exe -e /home/<user>/.venv-train/bin/python``
        (we pass the whole command verbatim; the launcher splits on
         spaces so WSL-style invocations work as a single string.)

    ``pointcept_dir`` is the cloned Pointcept repo so we can invoke
    ``tools/train.py`` with the right working directory.

    ``config_file`` is the generated Pointcept config (Python module)
    that describes the dataset, model, optimizer, and schedule.
    ``config_gen.py`` writes it into the run's work_dir.

    ``work_dir`` is where Pointcept will drop its logs and checkpoints.
    Typically ``{project}/training/runs/{run_name}/``.

    ``env`` is merged on top of the current environment for the
    subprocess. Useful for CUDA_VISIBLE_DEVICES or overriding
    PYTHONPATH when the custom dataset class lives outside Pointcept.
    """
    python_exe: str
    pointcept_dir: str
    config_file: str
    work_dir: str
    env: dict[str, str] = field(default_factory=dict)
    # Optional extra CLI args forwarded to train.py verbatim.
    extra_args: list[str] = field(default_factory=list)
    # Optional dir containing Lithium's Pointcept extensions (custom
    # dataset class etc.). When set, the runner bootstraps an import
    # of lithium_dataset before invoking tools/train.py so the
    # LithiumDataset class is in Pointcept's DATASETS registry by
    # the time the trainer builds the dataloader. Required because
    # Pointcept's Config.dump() strips bare ``__import__`` expressions
    # from the generated config (yapf can't serialise them), so any
    # registration done from inside the config file is lost on the
    # second use of the same config path.
    pointcept_ext_dir: str = ""
    # Which train entrypoint inside pointcept_dir. Overridable so we
    # can target their test.py for inference runs without writing a
    # separate runner.
    entrypoint: str = "tools/train.py"


@dataclass
class PointceptStatus:
    running: bool = False
    finished: bool = False
    returncode: int | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    current_step: int = 0
    total_steps: int = 0
    last_loss: float = 0.0
    last_loss_avg: float = 0.0
    last_acc: float = 0.0
    last_miou: float = 0.0
    best_miou: float = 0.0
    last_checkpoint: str = ""
    # Path to ``model_best.pth`` — the checkpoint Pointcept saves
    # alongside ``model_last.pth`` whenever val mIoU improves. Set by
    # probing ``<work_dir>/model/model_best.pth`` rather than parsing
    # logs (Pointcept doesn't print a distinct "saving BEST" line).
    best_checkpoint: str = ""
    log_lines: list[str] = field(default_factory=list)
    error: str = ""


_LOG_RING_CAP = 1000
_LOG_RING_TRIM = 800


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

EventCallback = Callable[[ParsedLine, str], None]


class PointceptRunner:
    """Manages one Pointcept training subprocess.

    Usage::

        runner = PointceptRunner()
        runner.launch(cfg, on_event=my_callback)
        # ...
        if runner.is_running:
            runner.stop()

    The subprocess is daemonized via a reader thread so the viz app's
    main loop never blocks on stdout. All state updates happen under
    the GIL via ``self.status`` — read it from the UI thread without
    locks, but treat each read as a snapshot.
    """

    def __init__(self):
        self.process: subprocess.Popen | None = None
        self.config: PointceptLaunchConfig | None = None
        self.status = PointceptStatus()
        self._thread: threading.Thread | None = None
        self._on_event: EventCallback | None = None

    # -- lifecycle ----------------------------------------------------------

    def launch(self, config: PointceptLaunchConfig,
               on_event: EventCallback | None = None) -> bool:
        """Spawn the training subprocess. Returns True on success.

        If a previous subprocess is still alive this is a no-op and
        returns False so the caller can show "already running" in
        the UI. Use ``stop()`` before re-launching.
        """
        if self.process is not None and self.process.poll() is None:
            return False
        self.config = config
        self._on_event = on_event

        os.makedirs(config.work_dir, exist_ok=True)

        cmd = self._build_command(config)
        merged_env = os.environ.copy()
        # Pointcept has no setup.py — their install pattern is
        # ``export PYTHONPATH=./`` from inside the repo. Prepend the
        # pointcept_dir so the subprocess can ``import pointcept`` no
        # matter where the config file lives. Preserve any existing
        # PYTHONPATH entries so the user's own paths still work.
        pp_parts = [config.pointcept_dir]
        if config.pointcept_ext_dir:
            pp_parts.append(config.pointcept_ext_dir)
        existing_pp = merged_env.get("PYTHONPATH", "")
        if existing_pp:
            pp_parts.append(existing_pp)
        merged_env["PYTHONPATH"] = os.pathsep.join(pp_parts)
        # UTF-8 in the child so tqdm / logging Unicode glyphs don't
        # crash the reader thread on Windows (cp1252 default).
        merged_env["PYTHONIOENCODING"] = "utf-8"
        merged_env.update(config.env)

        self.status = PointceptStatus(running=True)
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=config.pointcept_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,  # line-buffered so progress streams live
                env=merged_env,
            )
        except (OSError, ValueError) as e:
            # Common: python_exe does not exist, pointcept_dir missing,
            # config file path invalid. Surface cleanly to the UI.
            self.status.running = False
            self.status.finished = True
            self.status.error = f"failed to spawn training process: {e}"
            return False

        self._thread = threading.Thread(
            target=self._reader_loop, name="ptv3-reader", daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float = 10.0) -> None:
        """Terminate the subprocess if running. Blocks up to ``timeout``.

        Tries SIGTERM first, escalates to SIGKILL if the process
        doesn't exit. The reader thread will notice the pipe close
        on its own and update status to finished.
        """
        proc = self.process
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        except OSError:
            pass
        self.status.running = False

    @property
    def is_running(self) -> bool:
        return self.status.running and (
            self.process is not None and self.process.poll() is None)

    # -- internals ----------------------------------------------------------

    def _build_command(self, config: PointceptLaunchConfig) -> list[str]:
        """Build the argv for subprocess.Popen.

        Splits ``python_exe`` on whitespace so WSL-style invocations
        like ``wsl.exe -e /home/<user>/.venv-train/bin/python`` work
        without the caller having to pre-split into a list.

        When ``pointcept_ext_dir`` is set, the command is wrapped in a
        ``python -u -c <bootstrap>`` invocation that adds the extension
        dir to ``sys.path``, imports our custom dataset module to fire
        its ``@DATASETS.register_module()`` side effect, and then runs
        the entrypoint via ``runpy.run_path`` with the same argv the
        plain CLI form would have built. This is robust against
        Pointcept's ``Config.dump`` stripping the registration shim
        out of the saved config file.
        """
        python_parts = config.python_exe.split()
        # -u for unbuffered Python so the pipe stays live under
        # Pointcept's mixed tqdm + logging output.
        plain_args = [config.entrypoint,
                      "--config-file", config.config_file,
                      "--options", f"save_path={config.work_dir}",
                      *config.extra_args]
        if config.pointcept_ext_dir:
            ext_dir = config.pointcept_ext_dir.replace("\\", "/")
            entry = config.entrypoint.replace("\\", "/")
            argv_repr = ", ".join(repr(a) for a in plain_args)
            # The monkey-patch on pointcept.engines.defaults.worker_init_fn
            # is what makes ``num_worker > 0`` safe on Windows. Each
            # DataLoader worker spawns a fresh Python interpreter, so the
            # main-process bootstrap import is gone — without this patch
            # workers re-raise "LithiumDataset not in registry". The
            # patched function re-prepends the ext dir and re-imports
            # lithium_dataset before delegating to the original
            # seed-setter, which is what the worker actually needs.
            bootstrap = "\n".join([
                "import sys, runpy",
                f"sys.path.insert(0, r'{ext_dir}')",
                "import lithium_dataset",
                # Replace Pointcept's worker_init_fn with one that re-imports
                # our dataset class on every worker startup. The replacement
                # function lives in a real module file (worker_init.py) so
                # PyTorch can pickle it across the spawn boundary — a function
                # defined inline in this -c bootstrap exists only in the
                # main process's __main__ and cannot be unpickled by workers.
                "import pointcept.engines.defaults as _pcd",
                "from worker_init import photon_worker_init_fn as _3p_wif",
                "_pcd.worker_init_fn = _3p_wif",
                f"sys.argv = [{argv_repr}]",
                f"runpy.run_path(r'{entry}', run_name='__main__')",
            ])
            return [*python_parts, "-u", "-c", bootstrap]
        return [*python_parts, "-u", *plain_args]

    def _reader_loop(self):
        """Drain stdout line-by-line, parse, and publish.

        Runs on its own daemon thread. Resilient to pipe errors —
        a torn-down stdout ends the loop and marks the status
        finished rather than crashing the thread and leaving the UI
        stuck on ``running=True``.
        """
        proc = self.process
        if proc is None or proc.stdout is None:
            self.status.running = False
            self.status.finished = True
            self.status.error = "no stdout pipe on subprocess"
            return
        # Tee stdout to <work_dir>/run.log so any failure is diagnosable
        # after the GUI exits — the in-memory ring buffer alone vanishes
        # with the app. Best-effort: a broken file handle never takes
        # down the reader.
        log_path = os.path.join(self.config.work_dir, "run.log") if self.config else None
        log_fh = None
        if log_path:
            try:
                log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
            except OSError as e:
                self._append_log(f"[runner] could not open run.log: {e}")
                log_fh = None
        try:
            for raw in iter(proc.stdout.readline, ''):
                if not raw:
                    break
                line = raw.rstrip()
                self._append_log(line)
                if log_fh is not None:
                    try:
                        log_fh.write(line + "\n")
                        log_fh.flush()
                    except OSError:
                        log_fh = None
                try:
                    parsed = parse_pointcept_line(line)
                except Exception as e:
                    # A parser bug on a single oddly-formatted line
                    # must not take the reader thread down with it —
                    # the subprocess would then block on its next
                    # stdout write once the pipe filled, hanging the
                    # whole run. Log it and move on.
                    self._append_log(f"[runner] parser error on line: {e}")
                    parsed = ParsedLine(kind="raw")
                self._apply_parsed(parsed)
                cb = self._on_event
                if cb is not None:
                    try:
                        cb(parsed, line)
                    except Exception as e:
                        # A broken UI callback must not take down the
                        # reader thread. Log and continue.
                        self._append_log(f"[runner] event callback error: {e}")
        except OSError as e:
            self.status.error = f"stdout pipe error: {e}"
        finally:
            if log_fh is not None:
                try:
                    log_fh.close()
                except OSError:
                    pass
            try:
                proc.wait()
            except Exception:
                pass
            self.status.returncode = proc.returncode
            self.status.running = False
            self.status.finished = True
            if (self.status.returncode not in (0, None)
                    and not self.status.error):
                self.status.error = (
                    f"training process exited with code {self.status.returncode}")
            elif self.status.returncode == 0:
                # Drop run-named copies of the canonical Pointcept
                # checkpoints next to the work_dir's other artifacts.
                # Pointcept hardcodes ``model/model_best.pth`` /
                # ``model/model_last.pth`` and our internal code (registry
                # auto-pickup, inference resolver) relies on those exact
                # names — so we copy rather than rename, leaving both
                # forms in place. Net effect: when the user browses the
                # training_runs tree, every checkpoint file has the run
                # name baked into its filename, no more pile of
                # identically-named ``model_best.pth`` files to disambiguate
                # by parent directory.
                self._copy_friendly_named_checkpoints()

    def _copy_friendly_named_checkpoints(self) -> None:
        """Write run-named copies of ``model_best.pth`` / ``model_last.pth``
        alongside the originals so the user can identify checkpoints
        without descending into per-run subdirs.

        Copies (not hard links) for cross-drive safety on Windows and so
        a later resumed-training overwrite of the originals doesn't
        clobber the snapshot of *this* run's completion state.
        """
        if self.config is None:
            return
        work_dir = self.config.work_dir
        if not work_dir or not os.path.isdir(work_dir):
            return
        run_name = os.path.basename(os.path.normpath(work_dir))
        model_dir = os.path.join(work_dir, "model")
        pairs = (
            ("model_best.pth", f"{run_name}_best.pth"),
            ("model_last.pth", f"{run_name}_last.pth"),
        )
        for src_name, dst_name in pairs:
            src = os.path.join(model_dir, src_name)
            if not os.path.isfile(src):
                continue
            # Drop the friendly copy at the work_dir root so it's
            # visible at the same level as ``config.py`` / ``train.log``.
            dst = os.path.join(work_dir, dst_name)
            try:
                shutil.copyfile(src, dst)
                self._append_log(f"[checkpoint] {dst_name} written")
            except OSError as e:
                self._append_log(
                    f"[checkpoint] friendly copy failed for {src_name}: {e}")

    def _append_log(self, line: str) -> None:
        log = self.status.log_lines
        log.append(line)
        if len(log) > _LOG_RING_CAP:
            # Trim in chunks so we don't thrash on every append once
            # the buffer is saturated.
            del log[:len(log) - _LOG_RING_TRIM]

    def _apply_parsed(self, parsed: ParsedLine) -> None:
        s = self.status
        if parsed.kind == "train_step":
            if parsed.epoch is not None:
                s.current_epoch = parsed.epoch
            if parsed.total_epochs is not None:
                s.total_epochs = parsed.total_epochs
            if parsed.step is not None:
                s.current_step = parsed.step
            if parsed.total_steps is not None:
                s.total_steps = parsed.total_steps
            if parsed.loss is not None:
                s.last_loss = parsed.loss
            if parsed.loss_avg is not None:
                s.last_loss_avg = parsed.loss_avg
            if parsed.acc is not None:
                s.last_acc = parsed.acc
        elif parsed.kind == "val_result":
            if parsed.miou is not None:
                s.last_miou = parsed.miou
                if parsed.miou > s.best_miou:
                    s.best_miou = parsed.miou
        elif parsed.kind == "epoch":
            if parsed.epoch is not None:
                s.current_epoch = parsed.epoch
            if parsed.total_epochs is not None:
                s.total_epochs = parsed.total_epochs
            if parsed.loss is not None:
                s.last_loss = parsed.loss
            if parsed.miou is not None:
                s.last_miou = parsed.miou
                if parsed.miou > s.best_miou:
                    s.best_miou = parsed.miou
        elif parsed.kind == "checkpoint":
            if parsed.checkpoint_path:
                s.last_checkpoint = parsed.checkpoint_path
        elif parsed.kind == "best":
            if parsed.best_miou is not None and parsed.best_miou > s.best_miou:
                s.best_miou = parsed.best_miou
                # Pointcept writes model_best.pth next to model_last.pth
                # whenever val mIoU improves. Probe for it so the
                # registry can point inference at the right file later.
                # File-existence check keeps us honest if Pointcept's
                # save fails partway through.
                if self.config is not None:
                    bp = os.path.join(
                        self.config.work_dir, "model", "model_best.pth")
                    if os.path.isfile(bp):
                        s.best_checkpoint = bp
