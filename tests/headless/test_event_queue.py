"""Pattern A subprocess→main-thread event-queue regression tests.

Pins down the contract that subprocess readers rely on:
- ``post_event`` is thread-safe (multiple producer threads OK).
- ``drain_all_events`` only dispatches on the main thread.
- ``max_per_kind`` caps how many events of one kind drain per call.
- Unhandled kinds drop the payload + log, without growing the queue.
- A raising handler does not block subsequent events.

These tests construct a minimal stub object with just the event-queue
attributes so we don't have to spin up the full App god-object — the
infra is intentionally lightweight to keep this isolation possible.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import pytest

import src.main as main_module


@pytest.fixture
def app_stub():
    """A bare object exposing just the event-queue attributes.

    Reuses the actual ``App`` methods so any drift between this stub
    and the live App will fail the test rather than silently passing.
    """
    class _Stub:
        register_event_handler = main_module.App.register_event_handler
        post_event = main_module.App.post_event
        drain_all_events = main_module.App.drain_all_events

    s = _Stub()
    s._pending_events = {}
    s._event_handlers = {}
    return s


def test_post_and_drain_dispatches_to_registered_handler(app_stub):
    seen = []
    app_stub.register_event_handler("foo", lambda app, p: seen.append(p))
    app_stub.post_event("foo", {"x": 1})
    app_stub.post_event("foo", {"x": 2})
    n = app_stub.drain_all_events()
    assert n == 2
    assert seen == [{"x": 1}, {"x": 2}]


def test_drain_no_handler_drops_event(capsys, app_stub):
    app_stub.post_event("orphan", "payload")
    n = app_stub.drain_all_events()
    assert n == 0
    captured = capsys.readouterr()
    assert "dropped 'orphan'" in captured.out
    # Queue must NOT accumulate dropped kinds — otherwise an un-wired
    # subprocess pumping events would grow memory unbounded.
    assert len(app_stub._pending_events.get("orphan", deque())) == 0


def test_max_per_kind_caps_dispatch(app_stub):
    seen = []
    app_stub.register_event_handler("burst", lambda app, p: seen.append(p))
    for i in range(10):
        app_stub.post_event("burst", i)
    n = app_stub.drain_all_events(max_per_kind=3)
    assert n == 3
    assert seen == [0, 1, 2]
    # The remaining 7 stay queued for the next frame.
    assert len(app_stub._pending_events["burst"]) == 7

    n2 = app_stub.drain_all_events(max_per_kind=100)
    assert n2 == 7
    assert seen == list(range(10))


def test_handler_exception_does_not_block_subsequent_events(capsys, app_stub):
    seen = []

    def boom(app, p):
        if p == "bad":
            raise RuntimeError("kaboom")
        seen.append(p)

    app_stub.register_event_handler("k", boom)
    app_stub.post_event("k", "ok-1")
    app_stub.post_event("k", "bad")
    app_stub.post_event("k", "ok-2")
    n = app_stub.drain_all_events()
    # All three events drain; the one that raised is reported and skipped.
    # The OTHER two run their handler. The dispatched count is 2.
    assert n == 2
    assert seen == ["ok-1", "ok-2"]
    captured = capsys.readouterr()
    assert "'k' handler raised" in captured.out


def test_multiple_producer_threads(app_stub):
    """Concurrent post_event from many threads loses no events."""
    seen = []
    app_stub.register_event_handler("t", lambda app, p: seen.append(p))

    N_THREADS = 8
    PER_THREAD = 200

    barrier = threading.Barrier(N_THREADS)

    def producer(tid):
        barrier.wait()
        for i in range(PER_THREAD):
            app_stub.post_event("t", (tid, i))

    threads = [threading.Thread(target=producer, args=(t,))
               for t in range(N_THREADS)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    # Now drain on the "main" thread.
    n = app_stub.drain_all_events(max_per_kind=N_THREADS * PER_THREAD + 100)
    assert n == N_THREADS * PER_THREAD
    # Every (tid, i) pair must appear exactly once.
    assert sorted(seen) == sorted(
        (t, i) for t in range(N_THREADS) for i in range(PER_THREAD)
    )


def test_multiple_kinds_each_drained_independently(app_stub):
    seen_a = []
    seen_b = []
    app_stub.register_event_handler("a", lambda app, p: seen_a.append(p))
    app_stub.register_event_handler("b", lambda app, p: seen_b.append(p))
    app_stub.post_event("a", 1)
    app_stub.post_event("b", 10)
    app_stub.post_event("a", 2)
    app_stub.post_event("b", 20)
    n = app_stub.drain_all_events()
    assert n == 4
    assert seen_a == [1, 2]
    assert seen_b == [10, 20]


def test_drain_empty_is_noop(app_stub):
    assert app_stub.drain_all_events() == 0
