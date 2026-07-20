"""ST-1..5 Pattern A handler contract tests.

The five subprocess-port commits each registered a main-thread
handler on the Pattern A event queue. The queue mechanics are
already pinned by test_event_queue.py; this file pins the
DECODING + side-effect contract for each handler so a payload-key
rename or a state-mutation regression trips here, not in the field.

We construct minimal stub objects with just the App attributes
each handler reads. ``src.main.App`` methods are bound to the stub
so any drift between this test and the live App method body breaks
the test.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.main as main_module

# ---------------------------------------------------------------------------
# ST-5 — _on_training_event
# ---------------------------------------------------------------------------

def test_st5_training_event_logs_and_updates_registry():
    """Full happy path: raw_line goes to cli.log, updates land in registry."""
    app = SimpleNamespace()
    app._on_training_event = main_module.App._on_training_event.__get__(app)
    app.cli = MagicMock()
    app._train_model_registry = MagicMock()
    app._train_model_registry.update_model = MagicMock()
    app._on_training_event({
        "raw_line": "[runner] epoch 5 miou=0.823",
        "project_id": "proj_X",
        "model_id": "model_42",
        "updates": {"best_miou": 0.823},
    })
    app.cli.log.assert_called_once_with(
        "[runner] epoch 5 miou=0.823", "info",
    )
    app._train_model_registry.update_model.assert_called_once_with(
        "proj_X", "model_42", best_miou=0.823,
    )


def test_st5_training_event_no_updates_still_logs():
    """A raw_line with no model updates still flows to CLI."""
    app = SimpleNamespace()
    app._on_training_event = main_module.App._on_training_event.__get__(app)
    app.cli = MagicMock()
    app._train_model_registry = MagicMock()
    app._on_training_event({
        "raw_line": "[runner] starting epoch 1",
        "project_id": "proj_X",
        "model_id": "model_42",
        "updates": {},
    })
    app.cli.log.assert_called_once()
    app._train_model_registry.update_model.assert_not_called()


def test_st5_training_event_missing_model_id_skips_registry():
    """A training event with model_id=None (early-startup line) logs
    but doesn't trip registry.update_model."""
    app = SimpleNamespace()
    app._on_training_event = main_module.App._on_training_event.__get__(app)
    app.cli = MagicMock()
    app._train_model_registry = MagicMock()
    app._on_training_event({
        "raw_line": "[runner] pre-start banner",
        "project_id": None,
        "model_id": None,
        "updates": {"best_miou": 0.5},
    })
    app.cli.log.assert_called_once()
    app._train_model_registry.update_model.assert_not_called()


def test_st5_training_event_registry_error_logged_not_raised():
    """A registry write that raises must NOT propagate — handler
    catches + logs, then the next event still drains."""
    app = SimpleNamespace()
    app._on_training_event = main_module.App._on_training_event.__get__(app)
    app.cli = MagicMock()
    app._train_model_registry = MagicMock()
    app._train_model_registry.update_model = MagicMock(
        side_effect=OSError("disk full"))
    # No exception leaks.
    app._on_training_event({
        "raw_line": "log",
        "project_id": "proj_X",
        "model_id": "model_42",
        "updates": {"best_miou": 0.9},
    })
    app.cli.log.assert_called_once()
    app._train_model_registry.update_model.assert_called_once()
