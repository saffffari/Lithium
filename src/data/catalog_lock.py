"""Single-instance lock for the Lithium library catalog.

A lock file at ``~/.lithium/library/.lock`` carries the PID of the
process that owns the catalog. On startup we acquire it; on clean exit
we release it. A second Lithium launch detects the lock, checks whether
the recorded PID is still alive, and either:

- refuses to start (if the other instance is genuinely running), or
- silently overwrites the stale lock and continues (if the previous
  process crashed without releasing it).

The check is psutil-free: ``os.kill(pid, 0)`` on POSIX, and a
``OpenProcess`` syscall on Windows. Both return False on a dead PID
without raising.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from src.data import library_paths


_LOCK_FILENAME = ".lock"


def lock_path() -> Path:
    """Compute the lock file path each time so test monkey-patching of
    ``library_paths.library_dir`` lands in our dispatch."""
    return Path(library_paths.library_dir()) / _LOCK_FILENAME


# ---------------------------------------------------------------------------
# PID liveness
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently running.

    Robust to permission errors (a running process owned by someone else
    still counts as alive). Returns False for negative or zero PIDs.
    """
    if pid <= 0:
        return False

    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
            )
            if not handle:
                return False
            try:
                exit_code = wintypes.DWORD()
                STILL_ACTIVE = 259
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                if not ok:
                    return False
                return exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            # Any failure to query → assume dead so a stuck lock can be
            # cleared. The worst case is two instances racing, which is
            # what the lock is meant to prevent — but a permanently
            # stuck lock is the more annoying failure mode.
            return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Running but owned by another user.
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Acquire / release
# ---------------------------------------------------------------------------

def acquire_lock() -> tuple[bool, int | None]:
    """Try to take ownership of the catalog lock.

    Returns ``(success, blocking_pid)``:

    - ``(True, None)``  — lock acquired (clean state, or stale lock cleared).
    - ``(False, pid)``  — refused; another live Lithium instance owns it.

    Implementation: read the lock file if present, decode the recorded
    PID, check liveness. If alive, refuse. If dead, overwrite. Either
    way the file is left with our own PID + start timestamp on success.
    """
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            blocking_pid = int(payload.get("pid", 0))
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            blocking_pid = 0

        if blocking_pid > 0 and _pid_alive(blocking_pid):
            return False, blocking_pid
        # Stale: previous owner is gone. Fall through and overwrite.

    payload = {
        "pid": int(os.getpid()),
        "started": time.time(),
        "platform": sys.platform,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except OSError as e:
        # CP-6: previously returned (True, None) here ("treat as acquired
        # so the app still launches") which lied to the caller — the
        # lock file never landed and a second instance would race on
        # the same catalog. Return (False, None) so the caller knows
        # the lock genuinely wasn't taken. App launches anyway with
        # ``release_lock`` skipped.
        print(f"Catalog lock acquire failed: {e}")
        return False, None
    return True, None


def release_lock() -> None:
    """Best-effort delete of the lock file. Silent on missing/permission
    errors so cleanup never blocks app shutdown."""
    try:
        path = lock_path()
        if path.exists():
            path.unlink()
    except OSError:
        pass


def lock_owner_pid() -> int | None:
    """Return the PID currently recorded in the lock file (or None)."""
    path = lock_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return int(payload.get("pid", 0)) or None
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None
