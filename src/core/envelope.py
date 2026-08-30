"""Envelope curve for point falloff and selection falloff.

An Envelope is a sorted list of control points (t, value) in [0,1]x[0,1].
Sampling between control points uses linear interpolation. The envelope
can be baked into a fixed-size LUT for GPU upload.

Common use:
    env = Envelope([(0.0, 1.0), (0.5, 0.7), (1.0, 0.0)])
    env.sample(0.25)  # ~0.85
    lut = env.to_lut(256)  # (256,) float32
"""

import numpy as np


class Envelope:
    """Piecewise-linear curve in the unit square.

    Control points are stored as list of (t, value) tuples, both in [0,1].
    The first point must be at t=0, the last at t=1, and points are kept
    sorted by t.
    """

    def __init__(self, points: list[tuple[float, float]] | None = None):
        if points is None:
            points = [(0.0, 1.0), (1.0, 0.0)]
        self._points: list[list[float]] = [list(p) for p in points]
        self._sort_and_clamp()

    def _sort_and_clamp(self):
        for p in self._points:
            p[0] = max(0.0, min(1.0, float(p[0])))
            p[1] = max(0.0, min(1.0, float(p[1])))
        self._points.sort(key=lambda p: p[0])
        # Ensure endpoints exist at t=0 and t=1
        if not self._points or self._points[0][0] > 0.001:
            self._points.insert(0, [0.0, self._points[0][1] if self._points else 1.0])
        if self._points[-1][0] < 0.999:
            self._points.append([1.0, self._points[-1][1]])
        self._points[0][0] = 0.0
        self._points[-1][0] = 1.0

    @property
    def points(self) -> list[tuple[float, float]]:
        """Return a copy of the control points as tuples."""
        return [(p[0], p[1]) for p in self._points]

    def __len__(self) -> int:
        return len(self._points)

    def get(self, index: int) -> tuple[float, float]:
        p = self._points[index]
        return (p[0], p[1])

    def set_point(self, index: int, t: float, value: float,
                  lock_t: bool = False):
        """Move the control point at index. First/last points have locked t.

        lock_t is also respected for internal points if the caller wants
        to only change the value.
        """
        if index < 0 or index >= len(self._points):
            return
        is_endpoint = (index == 0 or index == len(self._points) - 1)
        if is_endpoint or lock_t:
            # Only update the value
            self._points[index][1] = max(0.0, min(1.0, float(value)))
        else:
            # Clamp t to stay between neighbors
            prev_t = self._points[index - 1][0]
            next_t = self._points[index + 1][0]
            new_t = max(prev_t + 0.001, min(next_t - 0.001, float(t)))
            self._points[index][0] = new_t
            self._points[index][1] = max(0.0, min(1.0, float(value)))

    def add_point(self, t: float, value: float) -> int:
        """Insert a new control point. Returns the new index."""
        if len(self._points) >= 16:
            return -1
        new_point = [max(0.0, min(1.0, float(t))),
                     max(0.0, min(1.0, float(value)))]
        # Find insertion index
        insert_idx = 0
        for i, p in enumerate(self._points):
            if p[0] < new_point[0]:
                insert_idx = i + 1
        self._points.insert(insert_idx, new_point)
        self._sort_and_clamp()
        return insert_idx

    def remove_point(self, index: int) -> bool:
        """Remove a control point by index. Cannot remove endpoints."""
        if index <= 0 or index >= len(self._points) - 1:
            return False
        self._points.pop(index)
        return True

    def sample(self, t: float) -> float:
        """Sample the curve at t in [0,1] with linear interpolation."""
        t = max(0.0, min(1.0, t))
        for i in range(len(self._points) - 1):
            t0, v0 = self._points[i]
            t1, v1 = self._points[i + 1]
            if t0 <= t <= t1:
                if t1 - t0 < 1e-8:
                    return v0
                frac = (t - t0) / (t1 - t0)
                return v0 + (v1 - v0) * frac
        return self._points[-1][1]

    def to_lut(self, size: int = 256) -> np.ndarray:
        """Bake the envelope into a (size,) float32 array."""
        ts = np.linspace(0.0, 1.0, size, dtype=np.float32)
        xp = [p[0] for p in self._points]
        fp = [p[1] for p in self._points]
        return np.interp(ts, xp, fp).astype(np.float32)

    def to_json(self) -> list:
        return [[p[0], p[1]] for p in self._points]

    @classmethod
    def from_json(cls, data) -> 'Envelope':
        return cls([(p[0], p[1]) for p in data])

    # Preset factories
    @classmethod
    def flat(cls, value: float = 1.0) -> 'Envelope':
        return cls([(0.0, value), (1.0, value)])

    @classmethod
    def linear_falloff(cls) -> 'Envelope':
        return cls([(0.0, 1.0), (1.0, 0.0)])

    @classmethod
    def hard_circle(cls) -> 'Envelope':
        # Truly flat. The old (0.99, 1.0) -> (1.0, 0.0) drop was an alpha
        # feather from before splats became opaque; with alpha forced to 1
        # it only painted a dark 1-2 px rim on every disc, which reads as a
        # light outline stroke against the neighbouring splat.
        return cls([(0.0, 1.0), (1.0, 1.0)])

    @classmethod
    def soft_bell(cls) -> 'Envelope':
        return cls([(0.0, 1.0), (0.35, 0.9), (0.7, 0.5), (1.0, 0.0)])


def envelope_from_softness(softness: float) -> 'Envelope':
    """Build a point falloff envelope from a single softness knob 0..1.

    softness=0 → hard circle (alpha 1.0 until very edge, then drop)
    softness=1 → linear falloff from 1.0 to 0.0
    """
    s = max(0.0, min(1.0, float(softness)))
    if s <= 1e-6:
        return Envelope.hard_circle()
    knee = 1.0 - s * 0.99
    return Envelope([(0.0, 1.0), (knee, 1.0), (1.0, 0.0)])
