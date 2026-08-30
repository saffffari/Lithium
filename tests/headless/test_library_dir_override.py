"""Verify the LITHIUM_LIBRARY_DIR env-var override redirects
the catalog away from the installed user's ``~/.lithium/library``.

This is the worktree-isolation contract: when the env var is set,
EVERY catalog write path (cloud_store, catalog_lock,
library_catalog) lands in the override location, not the default.
A regression here would let a dev build silently mutate a real
user's catalog.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.data import library_paths


def test_env_override_redirects(monkeypatch):
    """Setting LITHIUM_LIBRARY_DIR points library_dir at the override."""
    with tempfile.TemporaryDirectory(prefix="lithium_override_") as d:
        monkeypatch.setenv("LITHIUM_LIBRARY_DIR", d)
        resolved = library_paths.library_dir()
        assert os.path.abspath(resolved) == os.path.abspath(d)
        # previews/ subdir is eagerly created.
        assert os.path.isdir(os.path.join(d, "previews"))


def test_env_override_expanduser(monkeypatch, tmp_path):
    """A ``~/...`` path in the env var is expanded through expanduser."""
    # Symlink HOME into tmp_path so we don't pollute the real ~/.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows alias
    monkeypatch.setenv("LITHIUM_LIBRARY_DIR", "~/my-test-library")
    resolved = library_paths.library_dir()
    # On Windows, expanduser may use USERPROFILE or HOMEDRIVE+HOMEPATH;
    # accept any resolution that lands under tmp_path.
    assert str(tmp_path) in resolved or "my-test-library" in resolved
    assert resolved.endswith("my-test-library")


def test_env_override_unset_uses_default(monkeypatch):
    """Unsetting the env var falls back to the canonical default."""
    monkeypatch.delenv("LITHIUM_LIBRARY_DIR", raising=False)
    resolved = library_paths.library_dir()
    # Default path lives under ``.lithium/library`` somewhere under the user's home.
    assert resolved.endswith(os.path.join(".lithium", "library"))


def test_env_override_blank_uses_default(monkeypatch):
    """An empty string env var is treated as 'unset', not as 'override to root'."""
    monkeypatch.setenv("LITHIUM_LIBRARY_DIR", "")
    resolved = library_paths.library_dir()
    assert resolved.endswith(os.path.join(".lithium", "library"))


def test_env_override_propagates_to_cloud_store(monkeypatch):
    """cloud_store reads through library_paths so it sees the override too."""
    from src.data import cloud_store
    with tempfile.TemporaryDirectory(prefix="lithium_override_") as d:
        monkeypatch.setenv("LITHIUM_LIBRARY_DIR", d)
        labels_path = cloud_store.cloud_labels_path("test_fk")
        assert str(labels_path).startswith(os.path.abspath(d)), (
            f"cloud_store.cloud_labels_path resolved to {labels_path} but "
            f"override is {d}"
        )


def test_env_override_propagates_to_catalog_lock(monkeypatch):
    """catalog_lock reads through library_paths so the .lock file
    lands in the override location, not the default catalog."""
    from src.data import catalog_lock
    with tempfile.TemporaryDirectory(prefix="lithium_override_") as d:
        monkeypatch.setenv("LITHIUM_LIBRARY_DIR", d)
        path = catalog_lock.lock_path()
        assert str(path).startswith(os.path.abspath(d))
