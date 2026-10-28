"""Lithium-atom progress animation.

Free particles drift as a loose gas. As a job's progress climbs, each one
"switches on" its magnet at a fixed fraction, is pulled toward its slot and
snaps in with a flash: first the seven nucleons of a Li-7 nucleus (three
protons, four neutrons), then the three rings of the POINT SIZE icon bead by
bead, each finished by the electron that rides it. At 100% the result is a
stable lithium atom.

``AtomSim`` is plain Python + numpy so it can be stepped headless in tests;
``draw_atom`` renders one frame onto an ImGui draw list and
``draw_atom_badge`` is the small always-orbiting version used as a
"loading" marker.
"""

from __future__ import annotations

import math
import random

import numpy as np

PROTONS, NEUTRONS, ELECTRONS = 3, 4, 3
BEADS_PER_RING = 30

PROTON, NEUTRON, BEAD, ELECTRON = 0, 1, 2, 3

# Progress fractions at which each group's magnets switch on.
_NUCLEUS_SPAN = (0.02, 0.16)
_RING_START, _RING_SPAN = 0.18, 0.26

_K_SNAP = 70.0          # spring toward the slot once captured
_K_GAS = 5.0            # weak pull toward the drifting gas path
_SNAP_DIST = 0.035      # lock when this close (ring radius = 1)
_MAX_STEP = 1.0 / 120.0

# Three perpendicular great circles, same as the POINT SIZE sphere:
# (u, v) span each ring's plane.
_RINGS = (
    (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),   # XZ (equator)
    (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),   # XY
    (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),   # YZ
)


def ring_point(ring: int, angle: float) -> np.ndarray:
    u, v = _RINGS[ring]
    return math.cos(angle) * u + math.sin(angle) * v


def _nucleus_slots() -> list[tuple[int, np.ndarray]]:
    """Seven packed nucleons: a neutron at the centre and alternating
    protons / neutrons on the octahedron around it (3 p + 4 n)."""
    d = 0.11
    slots = [(NEUTRON, np.zeros(3))]
    axes = [np.array(a, float) for a in
            ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))]
    for i, a in enumerate(axes):
        slots.append((PROTON if i % 2 == 0 else NEUTRON, a * d))
    return slots


class AtomSim:
    """Particle state for one animation. Units: ring radius = 1."""

    ELECTRON_SPEED = 2.2            # rad/s along the ring

    def __init__(self, seed: int = 3):
        rng = random.Random(seed)
        kinds, slots, thresholds, rings, angles = [], [], [], [], []

        nuc = _nucleus_slots()
        lo, hi = _NUCLEUS_SPAN
        for i, (k, p) in enumerate(nuc):
            kinds.append(k); slots.append(p); rings.append(-1); angles.append(0.0)
            thresholds.append(lo + (hi - lo) * i / len(nuc))
        for r in range(3):
            base = _RING_START + _RING_SPAN * r
            start = rng.uniform(0, 2 * math.pi)
            for b in range(BEADS_PER_RING):
                a = start + 2 * math.pi * b / BEADS_PER_RING
                kinds.append(BEAD); slots.append(ring_point(r, a)); rings.append(r); angles.append(a)
                thresholds.append(base + 0.22 * b / BEADS_PER_RING)
            kinds.append(ELECTRON); slots.append(ring_point(r, start)); rings.append(r)
            angles.append(start + r * 2.1)
            thresholds.append(base + 0.24)

        self.kind = np.array(kinds, np.int8)
        self.slot = np.array(slots, float)
        self.threshold = np.array(thresholds, float)
        self.ring = np.array(rings, np.int8)
        self.e_phase = np.array(angles, float)      # electrons: phase offset
        n = len(kinds)

        # Gas: each particle circles a tilted axis at its own radius/speed.
        self._g_r = np.array([rng.uniform(1.15, 1.55) for _ in range(n)])
        self._g_w = np.array([rng.choice((-1, 1)) * rng.uniform(0.25, 0.6) for _ in range(n)])
        self._g_a0 = np.array([rng.uniform(0, 2 * math.pi) for _ in range(n)])
        self._g_h = np.array([rng.uniform(-0.45, 0.45) for _ in range(n)])
        tilt = [rng.uniform(-0.6, 0.6) for _ in range(n)]
        self._g_tilt = np.array(tilt)

        self.t = 0.0
        self.progress = 0.0            # smoothed, what the animation shows
        self.pos = self._gas_targets(0.0)
        self.vel = np.zeros_like(self.pos)
        self.captured_at = np.full(n, -1.0)
        self.locked = np.zeros(n, bool)
        self.flash = np.zeros(n)
        self.decaying = False

    # -- targets -------------------------------------------------------
    def _gas_targets(self, t: float) -> np.ndarray:
        a = self._g_a0 + self._g_w * t
        x = np.cos(a) * self._g_r
        z = np.sin(a) * self._g_r
        y = self._g_h + 0.12 * np.sin(a * 2.0 + self._g_a0)
        # tilt the orbit plane about x
        ct, st = np.cos(self._g_tilt), np.sin(self._g_tilt)
        return np.stack([x, y * ct - z * st, y * st + z * ct], axis=1)

    def slot_targets(self, t: float) -> np.ndarray:
        tg = self.slot.copy()
        e = np.nonzero(self.kind == ELECTRON)[0]
        for i in e:
            tg[i] = ring_point(int(self.ring[i]), self.e_phase[i] + self.ELECTRON_SPEED * t)
        return tg

    def slot_velocities(self, t: float) -> np.ndarray:
        v = np.zeros_like(self.slot)
        for i in np.nonzero(self.kind == ELECTRON)[0]:
            a = self.e_phase[i] + self.ELECTRON_SPEED * t
            v[i] = ring_point(int(self.ring[i]), a + math.pi / 2) * self.ELECTRON_SPEED
        return v

    def electron_angle(self, i: int, t: float | None = None) -> float:
        return self.e_phase[i] + self.ELECTRON_SPEED * (self.t if t is None else t)

    # -- state ---------------------------------------------------------
    @property
    def n_locked(self) -> int:
        return int(self.locked.sum())

    @property
    def complete(self) -> bool:
        return bool(self.locked.all())

    def ring_fill(self, ring: int) -> float:
        m = (self.kind == BEAD) & (self.ring == ring)
        return float(self.locked[m].mean())

    def decay(self) -> None:
        """Release every particle with an outward kick (job failed)."""
        if self.decaying:
            return
        self.decaying = True
        out = self.pos / (np.linalg.norm(self.pos, axis=1, keepdims=True) + 1e-6)
        self.vel += out * 2.2
        self.locked[:] = False
        self.captured_at[:] = -1.0

    # -- integration ---------------------------------------------------
    def step(self, dt: float, target_progress: float, finished: bool = False) -> None:
        dt = max(0.0, min(float(dt), 0.25))
        # Captures spread out even when progress jumps; a finished job
        # sweeps the rest in quickly so the atom completes on screen.
        rate = 1.6 if finished else 0.45
        goal = 1.0 if finished else max(0.0, min(1.0, target_progress))
        if goal > self.progress:
            self.progress = min(goal, self.progress + rate * dt)
        n = max(1, int(math.ceil(dt / _MAX_STEP)))
        h = dt / n
        for _ in range(n):
            self._substep(h)

    def _substep(self, h: float) -> None:
        self.t += h
        t = self.t
        gas = self._gas_targets(t)
        if self.decaying:
            acc = _K_GAS * 0.6 * (gas - self.pos) - 1.2 * self.vel
            self.vel += acc * h
            self.pos += self.vel * h
            self.flash *= math.exp(-h * 6.0)
            return

        newly = (self.captured_at < 0) & (self.threshold <= self.progress)
        self.captured_at[newly] = t
        tgt = self.slot_targets(t)
        tvel = self.slot_velocities(t)

        free = self.captured_at < 0
        live = ~free & ~self.locked
        acc = np.zeros_like(self.pos)
        acc[free] = _K_GAS * (gas[free] - self.pos[free]) - 2.5 * self.vel[free]
        if live.any():
            age = t - self.captured_at[live]
            m = np.clip(age / 0.3, 0.0, 1.0)
            m = (m * m * (3 - 2 * m))[:, None]        # magnet switching on
            k = _K_SNAP
            d = 2.0 * math.sqrt(k) * 0.75
            acc[live] = (m * k * (tgt[live] - self.pos[live])
                         + (1 - m) * _K_GAS * (gas[live] - self.pos[live])
                         - d * (self.vel[live] - m * tvel[live]))
            # electrons ride a circle: feed forward its centripetal pull
            e_live = self.kind[live] == ELECTRON
            acc[live] -= (m * e_live[:, None]) * (self.ELECTRON_SPEED ** 2) * tgt[live]
        self.vel += acc * h
        self.pos += self.vel * h

        if live.any():
            dist = np.linalg.norm(tgt - self.pos, axis=1)
            snap = live & (dist < _SNAP_DIST)
            self.locked |= snap
            self.flash[snap] = 1.0
        self.pos[self.locked] = tgt[self.locked]
        self.vel[self.locked] = 0.0
        self.flash *= math.exp(-h * 5.0)


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------

def _view(azimuth: float, elevation: float) -> np.ndarray:
    ca, sa = math.cos(azimuth), math.sin(azimuth)
    ce, se = math.cos(elevation), math.sin(elevation)
    ry = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    rx = np.array([[1, 0, 0], [0, ce, se], [0, -se, ce]])
    return rx @ ry


def _mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(4))


def _rgba(c, alpha):
    """Colour with alpha, every channel clamped to 0..1 (col32 packs
    bytes and overflows on anything outside)."""
    return (min(1.0, max(0.0, c[0])), min(1.0, max(0.0, c[1])),
            min(1.0, max(0.0, c[2])), min(1.0, max(0.0, alpha)))


_RING_SAMPLES = np.linspace(0, 2 * math.pi, 49)


def draw_atom(dl, sim: AtomSim, cx: float, cy: float, R: float, *,
              tint: str = "normal", glow: float = 0.0) -> None:
    """Draw ``sim`` centred on (cx, cy) with ring radius ``R`` pixels.

    ``tint``: "normal", "error" (reddened) or "cancel" (amber, dimmed).
    ``glow``: 0..1, the "stable atom" pulse shown when a job finishes.
    """
    from src.gui.theme import ORANGE, AMBER, WHITE, DIM, FAINT, RED, col32

    def tinted(c):
        if tint == "error":
            return _mix(c, RED, 0.65)
        if tint == "cancel":
            return _mix(c, (AMBER[0] * 0.6, AMBER[1] * 0.6, AMBER[2] * 0.6, 1.0), 0.5)
        return c

    V = _view(0.62 + sim.t * 0.35, math.radians(25))

    def project(p):
        q = p @ V.T
        return cx + q[..., 0] * R, cy - q[..., 1] * R, q[..., 2]

    # Orbit lines fade in as each ring's beads lock — the finished atom
    # reads exactly like the POINT SIZE icon.
    for r in range(3):
        fill = sim.ring_fill(r) if not sim.decaying else 0.0
        if fill <= 0.0:
            continue
        pts = np.array([ring_point(r, a) for a in _RING_SAMPLES])
        xs, ys, _ = project(pts)
        col = tinted(_mix(FAINT, ORANGE, fill))
        col = (col[0], col[1], col[2], 0.25 + 0.5 * fill * fill)
        c32 = col32(_rgba(col, col[3]))
        for i in range(len(xs) - 1):
            dl.add_line(float(xs[i]), float(ys[i]), float(xs[i + 1]), float(ys[i + 1]),
                        c32, max(1.0, R * 0.022))

    if glow > 0.0:
        u = (glow * 1.6) % 1.0
        halo = tinted((ORANGE[0], ORANGE[1], ORANGE[2], 0.45 * (1 - u)))
        dl.add_circle(cx, cy, R * (0.25 + 0.95 * u), col32(_rgba(halo, halo[3])), 48, max(1.0, R * 0.03))

    xs, ys, zs = project(sim.pos)
    order = np.argsort(zs)
    for i in order:
        k = int(sim.kind[i])
        depth = 0.85 + 0.15 * max(-1.0, min(1.0, float(zs[i])))
        fl = float(sim.flash[i])
        captured = sim.captured_at[i] >= 0 or sim.locked[i]
        if k == PROTON:
            col, rad = ORANGE, 0.085
        elif k == NEUTRON:
            col, rad = WHITE, 0.08
        elif k == ELECTRON:
            col, rad = (1.0, 1.0, 1.0, 1.0), 0.06
        else:
            col = ORANGE if sim.locked[i] else (_mix(DIM, ORANGE, 0.4) if captured else DIM)
            rad = 0.03 if sim.locked[i] else 0.026
        if not captured and k != BEAD:
            col = _mix(DIM, col, 0.35)
        col = _mix(col, AMBER, fl * 0.8)
        col = tinted(col)
        alpha = (0.55 + 0.45 * depth) if captured else 0.5 * depth
        x, y = float(xs[i]), float(ys[i])
        size = R * rad * depth * (1.0 + 0.6 * fl)
        if k == ELECTRON and sim.locked[i]:
            # short trail along the ring
            r = int(sim.ring[i])
            for j in range(1, 6):
                tp = ring_point(r, sim.electron_angle(i) - j * 0.12)
                tx, ty, _ = project(tp)
                dl.add_circle_filled(float(tx), float(ty), size * (1 - j * 0.15),
                                     col32(_rgba(col, alpha * 0.35 * (1 - j / 6))), 10)
            dl.add_circle_filled(x, y, size * 2.2, col32(_rgba(col, 0.18)), 14)
        if fl > 0.05:
            dl.add_circle_filled(x, y, size * 2.4, col32(_rgba(AMBER, 0.3 * fl)), 12)
        dl.add_circle_filled(x, y, max(1.0, size), col32(_rgba(col, alpha)), 12)


_BADGE_RINGS = [np.array([ring_point(r, a) for a in np.linspace(0, 2 * math.pi, 25)])
                for r in range(3)]


def draw_atom_badge(dl, cx: float, cy: float, r: float, t: float, color) -> None:
    """Small always-running atom: three rings, a nucleus and three
    electrons circling. Used where a pulsing "loading" dot used to be."""
    from src.gui.theme import col32
    V = _view(0.62 + t * 0.6, math.radians(25))
    ring_col = col32((color[0], color[1], color[2], 0.55))
    for pts in _BADGE_RINGS:
        q = pts @ V.T
        for i in range(len(q) - 1):
            dl.add_line(cx + q[i, 0] * r, cy - q[i, 1] * r,
                        cx + q[i + 1, 0] * r, cy - q[i + 1, 1] * r, ring_col, 1.0)
    dl.add_circle_filled(cx, cy, max(1.5, r * 0.22), col32(color), 10)
    for ring in range(3):
        p = ring_point(ring, t * 3.0 + ring * 2.1) @ V.T
        dl.add_circle_filled(cx + p[0] * r, cy - p[1] * r, max(1.2, r * 0.16),
                             col32((1.0, 1.0, 1.0, 0.95)), 8)
