"""Library directory resolution — the one place that knows where the
3Photon catalog lives on disk.

Lives in its own module so callers like ``cloud_store`` and
``catalog_lock`` can resolve paths without importing
``library_catalog`` — breaking the historical
``cloud_store ↔ library_catalog`` import cycle flagged as AR-2.

Tests redirect the catalog directory by monkey-patching
``library_paths.library_dir`` (not ``library_catalog.library_dir``,
which is kept as a thin delegating alias for back-compat).

Production redirection is via the ``THREEPHOTON_LIBRARY_DIR`` env
var — set this when running a dev / sandbox build so it can't
accidentally touch the shared installer's catalog at
``~/.3photon/library``. Falls back to the default when unset, so
existing installs behave exactly as before.
"""

from __future__ import annotations

import os


_LIBRARY_DIR_ENV = "THREEPHOTON_LIBRARY_DIR"


def library_dir() -> str:
    """Return the user-wide library directory, creating it on first use.

    Resolution order:

    1. ``THREEPHOTON_LIBRARY_DIR`` env var if set + non-empty. Lets a
       worktree / dev build redirect to its own catalog so it can't
       collide with the user's installed app. The env var is expanded
       through ``os.path.expanduser`` so ``~/.3photon-overnight/library``
       works portably.

    2. ``~/.3photon/library`` — the canonical default for any installer
       build. Kept exactly as before this hook landed so existing
       installs see no behaviour change.

    Either way, the directory tree (``previews/`` subdir) is created
    eagerly so the first cloud_store write doesn't have to mkdir.
    """
    override = os.environ.get(_LIBRARY_DIR_ENV, "").strip()
    if override:
        d = os.path.expanduser(override)
    else:
        home = os.path.expanduser("~")
        d = os.path.join(home, ".3photon", "library")
    os.makedirs(os.path.join(d, "previews"), exist_ok=True)
    return d
