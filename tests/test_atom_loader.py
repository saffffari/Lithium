"""The lithium-atom progress animation (src/gui/atom_loader.py)."""

from __future__ import annotations

import numpy as np

from src.gui.atom_loader import (
    AtomSim, BEAD, ELECTRON, NEUTRON, PROTON, BEADS_PER_RING, ring_point,
)


def _run(sim, seconds, progress, finished=False, dt=1 / 60):
    for _ in range(int(seconds / dt)):
        sim.step(dt, progress, finished)


def test_composition_is_lithium_7():
    sim = AtomSim()
    assert (sim.kind == PROTON).sum() == 3
    assert (sim.kind == NEUTRON).sum() == 4
    assert (sim.kind == ELECTRON).sum() == 3
    assert (sim.kind == BEAD).sum() == 3 * BEADS_PER_RING


def test_nothing_captured_at_zero_progress():
    sim = AtomSim()
    _run(sim, 2.0, 0.0)
    assert sim.n_locked == 0
    assert (sim.captured_at < 0).all()


def test_locks_grow_with_progress_and_finish_stable():
    sim = AtomSim()
    counts = []
    for p in (0.1, 0.4, 0.7):
        _run(sim, 2.0, p)
        counts.append(sim.n_locked)
    assert counts == sorted(counts) and counts[0] > 0
    _run(sim, 3.0, 1.0, finished=True)
    assert sim.complete
    # locked particles sit on their slots, electrons on their rings
    tgt = sim.slot_targets(sim.t)
    assert np.allclose(sim.pos, tgt)
    e = np.nonzero(sim.kind == ELECTRON)[0]
    assert np.allclose(np.linalg.norm(sim.pos[e], axis=1), 1.0)


def test_finished_job_completes_quickly_from_zero():
    sim = AtomSim()
    _run(sim, 2.0, 0.0, finished=True)
    assert sim.complete


def test_frame_rate_independent():
    for dt in (1 / 20, 1 / 144):
        sim = AtomSim()
        _run(sim, 2.5, 1.0, finished=True, dt=dt)
        assert sim.complete, dt


def test_huge_dt_stays_finite():
    sim = AtomSim()
    sim.step(5.0, 0.5)
    sim.step(1e6, 1.0, finished=True)
    assert np.isfinite(sim.pos).all() and np.isfinite(sim.vel).all()


def test_decay_releases_everything():
    sim = AtomSim()
    _run(sim, 3.0, 1.0, finished=True)
    sim.decay()
    _run(sim, 1.0, 1.0)
    assert sim.n_locked == 0
    assert np.isfinite(sim.pos).all()


def test_ring_point_on_unit_circle():
    for r in range(3):
        for a in np.linspace(0, 6.28, 7):
            assert abs(np.linalg.norm(ring_point(r, a)) - 1.0) < 1e-9
