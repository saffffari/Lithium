"""Pytest fixtures for the tier-1 headless test harness.

Every fixture is scoped tight enough that each test gets a fresh
library directory — no cross-test bleed via the catalog's index.json
or labels/ folder.
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

# Ensure the repo root is on sys.path so ``import src.*`` works when
# pytest is invoked from anywhere inside the repo.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data import library_paths  # noqa: E402

from tests.headless.app_fixture import HeadlessApp  # noqa: E402
from tests.headless import helpers  # noqa: E402


@pytest.fixture
def tmp_library(monkeypatch):
    """Redirect ``library_paths.library_dir()`` to a fresh tempdir.

    Yields the directory path as a str. All code under test that
    resolves the catalog directory via ``library_paths.library_dir()``
    (which is everything post-AR-2: cloud_store, catalog_lock, and
    library_catalog all dispatch through library_paths) lands here
    for the lifetime of the test.
    """
    with tempfile.TemporaryDirectory(prefix="lithium_headless_") as d:
        # Pre-create the previews/labels/data subdirs so save_cloud_labels
        # never has to mkdir on first write.
        os.makedirs(os.path.join(d, "previews"), exist_ok=True)
        os.makedirs(os.path.join(d, "labels"), exist_ok=True)
        os.makedirs(os.path.join(d, "data"), exist_ok=True)
        monkeypatch.setattr(library_paths, "library_dir", lambda: d)
        yield d


@pytest.fixture
def tiny_ply(tmp_path):
    """A 100-point binary PLY in a per-test tempdir.

    Lives in ``tmp_path`` rather than the library tempdir so the
    catalog's source-path matching code paths are exercised against a
    real on-disk file outside the catalog itself.
    """
    path = str(tmp_path / "tiny.ply")
    helpers.make_tiny_ply(path, n=100)
    return path


@pytest.fixture
def headless_app(tmp_library):
    """A fresh HeadlessApp pointed at the per-test library tempdir."""
    app = HeadlessApp()
    try:
        yield app
    finally:
        app.close()


@pytest.fixture(scope="session")
def gl_context():
    """Shared invisible GLFW + ModernGL context for tests that need GL.

    Skips the test entirely if no display/GLFW init is available (CI
    without a GPU, headless Linux without xvfb, etc.). Use sparingly —
    most label-safety tests should be pure data-layer.
    """
    glfw = pytest.importorskip("glfw")
    moderngl = pytest.importorskip("moderngl")

    if not glfw.init():
        pytest.skip("GLFW init failed; no headless GL available")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(1, 1, "Lithium Tier1", None, None)
    if not window:
        glfw.terminate()
        pytest.skip("Could not create hidden GLFW window")

    glfw.make_context_current(window)
    try:
        ctx = moderngl.create_context()
    except Exception as e:
        glfw.terminate()
        pytest.skip(f"ModernGL context creation failed: {e}")

    yield ctx

    try:
        ctx.release()
    except Exception:
        pass
    glfw.terminate()
