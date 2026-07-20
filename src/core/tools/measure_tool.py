"""Measurement tool state: line distance, angle, and landmark between snapped points.

Usage:
  state = MeasureState('measure_line')       # or 'measure_angle' / 'measure_landmark'
  state.add_anchor(anchor)                   # cloud-local Anchor (from PickResult.as_anchor())
  state.hover_pos = world_pos                # update every mouse move (still world for the
                                              # in-progress cursor)
  d = state.get_distance(resolve)            # resolve = AnchorResolver from measure_registry
  a = state.get_angle_deg(resolve)

Anchors live in cloud-local space — see measure_registry.Anchor — so
measurements stay glued to their bones when the HOLOGRAM reference
vertebra changes. ``hover_pos`` stays world-space because it's just
the in-flight cursor preview; it converts to an Anchor at the moment
add_anchor is called.
"""

import math
import time
import numpy as np

from src.core.measure_registry import Anchor, AnchorResolver

MEASURE_COLOR = (1.0, 0.15, 0.15, 1.0)   # default red

_LINE_ANCHORS     = 2   # distance tool needs 2 points
_ANGLE_ANCHORS    = 3   # angle tool needs 3 points (A=arm, B=vertex, C=arm)
_LANDMARK_ANCHORS = 1   # landmark is a single point

# Screen-space spring constants for snap animation.
# The spring constant increases quadratically as the indicator closes in on
# the snap target, creating the magnetic "accelerating pull" feel.
SNAP_PULL_RADIUS_PX = 52.0   # start pulling within this many screen pixels
SPRING_K_MIN        = 28.0   # spring constant at the edge of the pull zone
SPRING_K_MAX        = 900.0  # spring constant at the snap target
SPRING_DAMPING      = 0.62   # velocity multiplier per frame (lower = more bounce)
SNAP_HARD_PX        = 3.0    # distance at which we hard-snap (stop spring)


class MeasureState:
    """Mutable state for one active measurement session."""

    def __init__(self, mode: str):
        if mode not in ('measure_line', 'measure_angle', 'measure_landmark'):
            raise ValueError(f"Unknown measure mode: {mode!r}")
        self.mode = mode
        # Anchors are stored cloud-local (Anchor objects). Hover stays
        # world-space — it's just the in-flight cursor preview, never
        # committed until add_anchor is called with a real Anchor.
        self.anchors: list[Anchor] = []
        self.hover_pos: np.ndarray | None = None
        self.color: tuple = MEASURE_COLOR

        # --- snap animation state (screen-space spring) ---
        # anim_hover_screen: animated (x, y) in framebuffer pixels — used for
        # drawing.  Distinct from hover_pos (world-space true snap target) so
        # that measurement values are always exact while the visual lags.
        self.anim_hover_screen: tuple[float, float] | None = None
        self.anim_vel_screen:   list[float] = [0.0, 0.0]   # [vx, vy] px/s

        # --- click pop events ---
        # Each entry: (world_pos: np.ndarray, timestamp: float)
        self.pop_events: list[tuple] = []

        # --- timing ---
        self._anim_last_t: float = time.perf_counter()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def max_anchors(self) -> int:
        if self.mode == 'measure_line':
            return _LINE_ANCHORS
        if self.mode == 'measure_angle':
            return _ANGLE_ANCHORS
        return _LANDMARK_ANCHORS

    @property
    def complete(self) -> bool:
        return len(self.anchors) >= self.max_anchors

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def add_anchor(self, anchor: Anchor) -> None:
        """Append an Anchor (cloud-local). If measurement is already
        complete, reset first.

        Callers should obtain the Anchor from a ``PickResult.as_anchor()``
        — the picker provides the cloud_key + cloud-local position by
        inverting the cloud's current model matrix at click time.
        """
        if not isinstance(anchor, Anchor):
            raise TypeError(
                f"add_anchor expects an Anchor; got {type(anchor).__name__}. "
                "Use PickResult.as_anchor() to convert pick results.")
        if self.complete:
            self.anchors = []
            self.anim_hover_screen = None
            self.anim_vel_screen = [0.0, 0.0]
        self.anchors.append(anchor)

    def record_pop(self, world_pos: np.ndarray) -> None:
        """Record a click-pop animation event at world_pos."""
        now = time.perf_counter()
        self.pop_events.append((np.array(world_pos, dtype=np.float32), now))
        # Keep only recent events
        cutoff = now - 0.5
        self.pop_events = [(p, t) for p, t in self.pop_events if t > cutoff]

    def reset(self) -> None:
        self.anchors = []
        self.hover_pos = None
        self.anim_hover_screen = None
        self.anim_vel_screen = [0.0, 0.0]
        self.pop_events = []

    # ------------------------------------------------------------------
    # Spring animation update — call once per frame from the overlay
    # ------------------------------------------------------------------

    def tick_snap_anim(self, hover_sp: tuple[float, float] | None,
                       cursor_sp:  tuple[float, float]) -> tuple[float, float] | None:
        """Advance the screen-space spring toward hover_sp.

        hover_sp   — true snap target in screen pixels (or None)
        cursor_sp  — raw cursor position in screen pixels (fallback when no snap)

        Returns the current animated position, or None if there's nothing to draw.
        """
        now = time.perf_counter()
        dt  = min(now - self._anim_last_t, 0.05)   # cap to avoid big jumps on lag
        self._anim_last_t = now

        if hover_sp is None:
            # No snap target — clear animation
            self.anim_hover_screen = None
            self.anim_vel_screen   = [0.0, 0.0]
            return None

        target_x, target_y = hover_sp

        if self.anim_hover_screen is None:
            # First frame: start at the raw cursor so there's no jarring jump
            self.anim_hover_screen = cursor_sp
            self.anim_vel_screen   = [0.0, 0.0]

        ax, ay   = self.anim_hover_screen
        dx       = target_x - ax
        dy       = target_y - ay
        dist     = math.sqrt(dx * dx + dy * dy)

        if dist > SNAP_PULL_RADIUS_PX * 1.6:
            # Well outside the pull zone — teleport to cursor (no lag in free air)
            self.anim_hover_screen = cursor_sp
            self.anim_vel_screen   = [0.0, 0.0]

        elif dist > SNAP_HARD_PX:
            # Inside the magnetic zone: spring with quadratically increasing k
            t   = max(0.0, 1.0 - dist / SNAP_PULL_RADIUS_PX)
            k   = SPRING_K_MIN + (SPRING_K_MAX - SPRING_K_MIN) * t * t

            vx, vy = self.anim_vel_screen
            vx = vx * SPRING_DAMPING + dx * k * dt
            vy = vy * SPRING_DAMPING + dy * k * dt
            self.anim_vel_screen = [vx, vy]

            new_x = ax + vx * dt
            new_y = ay + vy * dt

            # Clamp: don't let the spring overshoot the target
            # (we want magnetic feel, not harmonic oscillation)
            if (target_x - new_x) * dx < 0:
                new_x = target_x
                self.anim_vel_screen[0] = 0.0
            if (target_y - new_y) * dy < 0:
                new_y = target_y
                self.anim_vel_screen[1] = 0.0

            self.anim_hover_screen = (new_x, new_y)

        else:
            # Hard snap — within SNAP_HARD_PX pixels
            self.anim_hover_screen = hover_sp
            self.anim_vel_screen   = [0.0, 0.0]

        return self.anim_hover_screen

    # ------------------------------------------------------------------
    # Measurement values
    # ------------------------------------------------------------------

    def get_distance(self, resolve: AnchorResolver) -> float | None:
        """Euclidean distance between anchor 0 and anchor 1.

        Returns None if either anchor's cloud can't be resolved
        (e.g. cloud was removed mid-session).
        """
        if len(self.anchors) < 2:
            return None
        a = resolve(self.anchors[0])
        b = resolve(self.anchors[1])
        if a is None or b is None:
            return None
        return float(np.linalg.norm(b - a))

    def get_distance_to_hover(self, resolve: AnchorResolver) -> float | None:
        """Live distance from the last placed anchor to the world-space
        hover position. Hover is world-space because it's the un-committed
        cursor preview; only resolved anchors need scene-state lookup."""
        if not self.anchors or self.hover_pos is None:
            return None
        a = resolve(self.anchors[-1])
        if a is None:
            return None
        return float(np.linalg.norm(self.hover_pos - a))

    def get_angle_deg(self, resolve: AnchorResolver) -> float | None:
        """Angle in degrees at anchor[1] (the vertex) between anchor[0] and anchor[2]."""
        if len(self.anchors) < 3:
            return None
        a = resolve(self.anchors[0])
        b = resolve(self.anchors[1])
        c = resolve(self.anchors[2])
        if a is None or b is None or c is None:
            return None
        return _angle_at_vertex(a, b, c)

    def get_angle_to_hover_deg(self, resolve: AnchorResolver) -> float | None:
        """Live angle at anchor[1] between anchor[0] and the hover position."""
        if len(self.anchors) < 2 or self.hover_pos is None:
            return None
        a = resolve(self.anchors[0])
        b = resolve(self.anchors[1])
        if a is None or b is None:
            return None
        return _angle_at_vertex(a, b, self.hover_pos)

    # ------------------------------------------------------------------
    # Helpers for the overlay renderer
    # ------------------------------------------------------------------

    def preview_points(self, resolve: AnchorResolver) -> list[np.ndarray]:
        """World-space points to draw lines through: resolved anchors
        + hover preview if incomplete. Resolver-failed anchors are
        skipped (the line draws as a partial segment rather than
        spanning to origin)."""
        pts: list[np.ndarray] = []
        for a in self.anchors:
            world = resolve(a)
            if world is not None:
                pts.append(np.asarray(world, dtype=np.float32))
        if not self.complete and self.hover_pos is not None:
            pts.append(np.asarray(self.hover_pos, dtype=np.float32))
        return pts


# ------------------------------------------------------------------
# Internal math
# ------------------------------------------------------------------

def _angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Angle in degrees at point B in the triangle A-B-C."""
    ba = a - b
    bc = c - b
    la = float(np.linalg.norm(ba))
    lc = float(np.linalg.norm(bc))
    if la < 1e-12 or lc < 1e-12:
        return None
    cos_a = float(np.dot(ba, bc)) / (la * lc)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.degrees(math.acos(cos_a))
