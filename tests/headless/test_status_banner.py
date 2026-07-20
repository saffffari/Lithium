"""ErrorBanner + App.set_status_banner contract.

The banner is the only failure-surfacing channel in the packaged .exe
(no stdout). Coalescing must be tight enough that a repeating failure
loop never stacks duplicate bars, but loose enough that distinct
sources keep producing their own messages.
"""

from __future__ import annotations

import pytest

import src.main as main_module
from src.gui.status_banner import ErrorBanner


@pytest.fixture
def app_stub():
    class _Stub:
        set_status_banner = main_module.App.set_status_banner
        clear_status_banner = main_module.App.clear_status_banner

    s = _Stub()
    s._status_banner = None
    return s


def test_initial_is_none(app_stub):
    assert app_stub._status_banner is None


def test_set_creates_banner(app_stub):
    app_stub.set_status_banner("disk full", level="error", source="cloud_store.save")
    b = app_stub._status_banner
    assert isinstance(b, ErrorBanner)
    assert b.message == "disk full"
    assert b.level == "error"
    assert b.source == "cloud_store.save"
    assert b.suppressed_count == 0


def test_same_source_same_level_coalesces(app_stub):
    app_stub.set_status_banner("save 1", level="error", source="A")
    app_stub.set_status_banner("save 2", level="error", source="A")
    app_stub.set_status_banner("save 3", level="error", source="A")
    b = app_stub._status_banner
    # Most-recent message wins; suppressed_count counts the OTHER two.
    assert b.message == "save 3"
    assert b.suppressed_count == 2


def test_different_source_replaces(app_stub):
    app_stub.set_status_banner("disk full", level="error", source="A")
    app_stub.set_status_banner("lock denied", level="error", source="B")
    b = app_stub._status_banner
    assert b.message == "lock denied"
    assert b.source == "B"
    assert b.suppressed_count == 0


def test_different_level_replaces(app_stub):
    """Bumping level (warn → error) should replace, not coalesce."""
    app_stub.set_status_banner("low disk", level="warn", source="A")
    app_stub.set_status_banner("out of disk", level="error", source="A")
    b = app_stub._status_banner
    assert b.message == "out of disk"
    assert b.level == "error"
    assert b.suppressed_count == 0


def test_empty_source_never_coalesces(app_stub):
    """A blank source is a deliberate one-shot — never coalesces with another blank."""
    app_stub.set_status_banner("first", level="error")
    app_stub.set_status_banner("second", level="error")
    b = app_stub._status_banner
    assert b.message == "second"
    assert b.suppressed_count == 0


def test_clear_resets(app_stub):
    app_stub.set_status_banner("x", source="A")
    app_stub.clear_status_banner()
    assert app_stub._status_banner is None
