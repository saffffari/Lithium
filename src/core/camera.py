"""Orbit camera with smooth pan, zoom, and preset transitions.

Every camera parameter (target, distance, azimuth, elevation) has two copies:

* ``*_goal`` — the value the camera is moving *toward*.
* the public attribute (``target``, ``distance``, ``azimuth``, ``elevation``)
  — the value the renderer actually uses this frame.

Direct drag input (orbit, pan, zoom-drag) updates both in lockstep so the
camera feels physically connected to the mouse. Commanded motion
(scroll-zoom, preset change, fit-to-view, focus-on-point) only touches the
goal, and :py:meth:`Camera.update` eases the live values toward the goals
with frame-rate-independent exponential smoothing. The result is butter-
smooth transitions without introducing lag on manual manipulation.

Additional butter layers on top of the basic ease:

1. **Dolly-under-cursor** (:py:meth:`Camera.dolly_at`) — scroll zooms toward
   the world point under the mouse, not the current pivot, so you zoom into
   what you're looking at.
2. **Distance-aware ease rate** — big normalized deltas ease slightly faster
   so time-to-target stays roughly constant regardless of delta magnitude.
3. **Adaptive low-pass orbit filter** — high-DPI mice produce jittery frame-
   to-frame deltas; a fast-motion-transparent low-pass kills the sub-pixel
   tremble without adding perceptible latency.
4. **Soft pole clamp** — elevation slows smoothly as it approaches the pole
   instead of hitting a hard wall.
"""

import math

import numpy as np
from src.utils.math_utils import orbit_camera, perspective, orthographic


# Camera preset angles (azimuth, elevation)
PRESETS = {
    'front': (0.0, 0.0),
    'back': (np.pi, 0.0),
    'right': (np.pi / 2, 0.0),
    'left': (-np.pi / 2, 0.0),
    'top': (0.0, np.pi / 2 - 0.01),
    'bottom': (0.0, -np.pi / 2 + 0.01),
    'iso': (np.pi / 4, np.pi / 6),
}

# Planar presets lock projection to orthographic — perspective on a
# strict axis-aligned view just distorts the plane without adding
# useful information.
PLANAR_PRESETS = frozenset(
    {'front', 'back', 'right', 'left', 'top', 'bottom'})


# Base smoothing rates (1/seconds). With frame-rate-independent easing, a
# value of R makes the remaining error shrink by (1 - exp(-R * dt)) each
# frame, so R=18 ≈ 95% there after ~170 ms, R=30 ≈ 95% after ~100 ms.
_RATE_ANGLE = 30.0         # orbit ring (fast — angle changes should feel snappy)
_RATE_DISTANCE = 11.0      # dolly/zoom (the big "butter" lever)
_RATE_TARGET = 13.0        # pivot translation (focus / fit / shift+click)

# Elevation pole soft-clamp. The orbit slows smoothly inside the last
# _POLE_SOFT_ZONE radians so you never feel a hard stop.
_POLE_LIMIT = np.pi / 2 - 0.01
_POLE_SOFT_ZONE = 0.35      # ≈ 20° soft zone at each pole

# Drag delta low-pass (used for both orbit and pan). At slow mouse speeds
# we smooth aggressively to kill jitter; at high speeds we pass the delta
# straight through so flicks don't feel delayed.
_ORBIT_JITTER_MIX = 0.62     # weight of the previous delta when mouse is slow
_ORBIT_FAST_THRESHOLD = 11.0  # px — beyond this, no smoothing

# Per-pixel gain for drags. Tuned down substantially so the smoothed
# drags feel precise and controlled rather than flighty.
_ORBIT_GAIN = 0.0022         # radians per pixel of cursor motion
_PAN_GAIN = 0.00095          # world-units per pixel, then scaled by distance

# Velocity-scaled orbit sensitivity. Slow cursor speeds stay at the base
# gain (precise framing); fast flicks boost up to _ORBIT_GAIN_BOOST ×
# so sweeping the view feels broad without sacrificing precision.
_ORBIT_GAIN_BOOST = 1.35      # peak multiplier at high cursor speed
_ORBIT_GAIN_BOOST_SPEED = 18.0  # px/frame at which the boost saturates

# Orbit flick-to-spin inertia. On RMB release, if the last filtered
# delta was faster than _INERTIA_MIN_SPEED, the orbit continues with
# an exponentially-decaying velocity for a fraction of a second.
_INERTIA_MIN_SPEED = 4.0      # px/frame — below this, no inertia kick
_INERTIA_DECAY = 6.5          # 1/s — higher = stops faster (half-life ≈ 0.11 s)
_INERTIA_ASSUMED_FPS = 60.0   # converts px/frame → px/s for rad/s velocity
_INERTIA_LAUNCH_SCALE = 0.55  # damp initial velocity so fast flicks don't overshoot


def _ease(current: float, goal: float, rate: float, dt: float) -> float:
    """Frame-rate-independent exponential easing toward a goal."""
    if dt <= 0.0:
        return current
    alpha = 1.0 - math.exp(-rate * dt)
    return current + (goal - current) * alpha


def _ease_adaptive(current: float, goal: float, base_rate: float, dt: float,
                    scale: float) -> float:
    """Distance-aware exponential easing.

    Normalizes the remaining delta against ``scale`` and ramps the rate
    logarithmically. Tiny deltas ease at the base rate (subtle settle);
    big deltas ease faster (so time-to-target stays bounded) without
    changing the ease *shape*, which keeps the glide cinematic.
    """
    if dt <= 0.0:
        return current
    delta = abs(goal - current)
    if delta < 1e-9:
        return goal
    ref = max(abs(current), abs(goal), scale, 1e-6)
    normalized = delta / ref
    rate_mult = 1.0 + 0.5 * min(math.log1p(normalized * 8.0), 2.5)
    alpha = 1.0 - math.exp(-base_rate * rate_mult * dt)
    return current + (goal - current) * alpha


def _pole_softness(elevation: float) -> float:
    """Fraction (0-1) to scale incoming elevation deltas by.

    Returns 1.0 well away from the pole and falls off smoothly inside
    :data:`_POLE_SOFT_ZONE` of the limit so orbiting near vertical feels
    cushioned rather than blocked.
    """
    slack = _POLE_LIMIT - abs(elevation)
    if slack >= _POLE_SOFT_ZONE:
        return 1.0
    if slack <= 0.0:
        return 0.0
    # Smooth cosine ramp from 0 at the pole to 1 at _POLE_SOFT_ZONE.
    t = slack / _POLE_SOFT_ZONE
    return 0.5 - 0.5 * math.cos(math.pi * t)


class Camera:
    def __init__(self):
        # Live (rendered) state.
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.distance = 5.0
        self.azimuth = np.pi / 4
        self.elevation = np.pi / 6

        # Goal state — what the live state is easing toward.
        self._target_goal = self.target.copy()
        self._distance_goal = self.distance
        self._azimuth_goal = self.azimuth
        self._elevation_goal = self.elevation

        self.fov = np.radians(60.0)
        self.near = 0.01
        self.far = 10000.0
        self.aspect = 16.0 / 9.0
        self.active_preset: str | None = 'iso'  # tracks last-set preset for UI
        self.projection: str = 'perspective'  # 'perspective' | 'orthographic'

        # Interaction state. LMB is intentionally NOT a camera button so
        # it stays free for selection tools.
        self._dragging = False   # right mouse  = orbit
        self._panning = False    # middle mouse = pan
        self._last_mouse = (0.0, 0.0)

        # Adaptive low-pass state for orbit and pan drags. The filter is
        # applied per-gesture so orbit and pan have independent histories.
        self._orbit_prev_dx = 0.0
        self._orbit_prev_dy = 0.0
        self._pan_prev_dx = 0.0
        self._pan_prev_dy = 0.0

        # Orbit flick-to-spin inertia — set on RMB release, decays in update.
        self._orbit_inertia_vel_az = 0.0
        self._orbit_inertia_vel_el = 0.0

    # --- goal/snap helpers --------------------------------------------------

    @staticmethod
    def _smooth_drag_delta(dx: float, dy: float,
                            prev_dx: float, prev_dy: float) -> tuple[float, float]:
        """Adaptive low-pass for drag input deltas.

        At slow cursor speeds (< _ORBIT_FAST_THRESHOLD px/frame) the
        delta is mixed heavily with the previous filtered delta, killing
        sub-pixel tremble. At fast speeds the raw delta passes through
        unchanged so flicks and rapid gestures feel instant. Used by
        both orbit (RMB) and pan (LMB) so the two drag styles feel
        equally silky.
        """
        speed = math.hypot(dx, dy)
        if speed < _ORBIT_FAST_THRESHOLD:
            t = speed / _ORBIT_FAST_THRESHOLD
            mix = _ORBIT_JITTER_MIX * (1.0 - t)
            return ((1.0 - mix) * dx + mix * prev_dx,
                    (1.0 - mix) * dy + mix * prev_dy)
        return dx, dy

    def _snap(self):
        """Collapse goal == live (used for instantaneous state changes)."""
        self._target_goal = self.target.copy()
        self._distance_goal = float(self.distance)
        self._azimuth_goal = float(self.azimuth)
        self._elevation_goal = float(self.elevation)

    def snap_to_goal(self):
        """Jump live state to the goal state instantly (no easing)."""
        self.target = self._target_goal.copy()
        self.distance = float(self._distance_goal)
        self.azimuth = float(self._azimuth_goal)
        self.elevation = float(self._elevation_goal)
        self._update_clip_range()

    def _update_clip_range(self):
        self.near = max(self.distance * 0.001, 1e-4)
        self.far = self.distance * 100.0

    # --- per-frame update ---------------------------------------------------

    def update(self, dt: float):
        """Ease live state toward goal state. Called once per frame."""
        if dt <= 0.0:
            return

        # Orbit inertia — apply first so the angular velocity we add this
        # frame is what the angle ease below then smooths. Only runs when
        # we're NOT actively dragging; a new press zeros the velocity
        # (handled in on_mouse_button).
        if not self._dragging and (abs(self._orbit_inertia_vel_az) > 1e-4
                                    or abs(self._orbit_inertia_vel_el) > 1e-4):
            self.azimuth += self._orbit_inertia_vel_az * dt
            # Soft-clamp the elevation contribution near poles so flicks
            # toward the top don't slam into the wall.
            el_softness = _pole_softness(self.elevation)
            self.elevation += self._orbit_inertia_vel_el * dt * el_softness
            self.elevation = max(-_POLE_LIMIT, min(_POLE_LIMIT, self.elevation))
            self._azimuth_goal = self.azimuth
            self._elevation_goal = self.elevation
            # Exponential decay
            decay = math.exp(-_INERTIA_DECAY * dt)
            self._orbit_inertia_vel_az *= decay
            self._orbit_inertia_vel_el *= decay
            # Snap to zero once below the noise floor so we stop waking up
            # in update loops forever.
            if (abs(self._orbit_inertia_vel_az) < 1e-4
                    and abs(self._orbit_inertia_vel_el) < 1e-4):
                self._orbit_inertia_vel_az = 0.0
                self._orbit_inertia_vel_el = 0.0

        # Target pivot — distance-aware per-component ease so big focus
        # moves glide for roughly the same duration as small ones.
        target_scale = max(self.distance, 1e-3)
        for i in range(3):
            self.target[i] = _ease_adaptive(
                float(self.target[i]),
                float(self._target_goal[i]),
                _RATE_TARGET, dt, target_scale,
            )

        # Distance — distance-aware (relative scale = current distance).
        self.distance = _ease_adaptive(
            self.distance, self._distance_goal,
            _RATE_DISTANCE, dt, max(self.distance, 1e-3),
        )

        # Angles — constant rate is fine; they're already snappy.
        self.azimuth = _ease(self.azimuth, self._azimuth_goal, _RATE_ANGLE, dt)
        self.elevation = _ease(self.elevation, self._elevation_goal, _RATE_ANGLE, dt)
        # Hard safety clamp (the goal is soft-clamped already on input).
        self.elevation = max(-_POLE_LIMIT, min(_POLE_LIMIT, self.elevation))

        self._update_clip_range()

    # --- framing ------------------------------------------------------------

    def fit_to_bounds(self, bounds_min: np.ndarray, bounds_max: np.ndarray,
                      animate: bool = True, padding: float = 1.6):
        """Frame the camera around a bounding box.

        ``padding`` multiplies the bounding-box diagonal to set the
        camera distance. Larger = farther / more headroom around the
        subject; smaller = tighter framing. The previous default of
        2.2 left enough margin to feel "comfortably zoomed out" but
        meant every cloud switch required a manual zoom-in. 1.6
        fills the viewport visibly while still leaving enough margin
        that orbit doesn't clip the cloud.

        With ``animate=False`` the live state snaps to the new framing
        immediately (used on first load so there is no initial zoom-in).
        """
        center = ((bounds_min + bounds_max) / 2.0).astype(np.float32)
        extent = float(np.linalg.norm(bounds_max - bounds_min))
        distance = max(extent * float(padding), 0.1)

        self._target_goal = center.copy()
        self._distance_goal = distance
        if not animate:
            self.target = center.copy()
            self.distance = distance
            self._update_clip_range()

    def focus_on(self, world_pos: np.ndarray):
        """Smoothly move the orbit pivot to ``world_pos`` without changing zoom."""
        self._target_goal = np.asarray(world_pos, dtype=np.float32).copy()

    def focus_at_distance(self, world_pos: np.ndarray, distance: float):
        """Smoothly move the pivot to ``world_pos`` AND ease zoom to the
        given absolute distance. Used by cursor-place to frame a point
        at a scene-relative distance (roughly cloud_extent * 0.22) so
        the focused point is visible with surrounding context.
        """
        self._target_goal = np.asarray(world_pos, dtype=np.float32).copy()
        self._distance_goal = float(max(distance, 0.01))

    def focus_and_frame(self, world_pos: np.ndarray, tighten: float = 0.45):
        """Smoothly focus on a point and zoom in by ``tighten`` (0..1).

        Used for LMB double-click: compound with a plain single-click focus
        so you can fire the click optimistically and the double-click just
        extends it.
        """
        self._target_goal = np.asarray(world_pos, dtype=np.float32).copy()
        tighten = float(max(0.0, min(1.0, tighten)))
        self._distance_goal = max(self._distance_goal * (1.0 - tighten), 0.01)

    def dolly_at(self, offset: float, world_pivot: np.ndarray | None = None):
        """Scroll-zoom with optional dolly-under-cursor.

        ``offset`` is the raw scroll tick (positive = toward you = zoom in).
        If ``world_pivot`` is provided, the pivot goal slides toward that
        world point on zoom-in so the world point under the cursor stays
        approximately under the cursor as the camera closes in.

        On zoom-out we do NOT push the pivot away from the cursor — doing
        so causes the pivot to drift exponentially off the cloud with
        repeated scrolls, which then degrades pan / orbit feel because the
        orbit ring is no longer centered on anything interesting. So
        zoom-out is a pure dolly; zoom-in is dolly-toward-cursor.
        """
        factor = 1.0 - offset * 0.085  # < 1 = zoom in
        factor = max(factor, 0.05)     # safety floor, never invert
        new_distance = max(self._distance_goal * factor, 0.01)

        if world_pivot is not None and factor < 1.0:
            pivot = np.asarray(world_pivot, dtype=np.float32)
            # Slide the target goal toward the cursor point by the same
            # fraction the distance contracts.
            alpha = float(1.0 - factor)
            self._target_goal = (self._target_goal
                                 + (pivot - self._target_goal) * alpha)

        self._distance_goal = new_distance

    # --- matrices -----------------------------------------------------------

    def get_view_matrix(self) -> np.ndarray:
        # Memoise on the orbit state so render paths that call this
        # multiple times per frame (render_individual, gnomon HUD,
        # measure overlays × 2, picking) only actually rebuild the
        # matrix when the camera has moved.
        key = (
            float(self.target[0]), float(self.target[1]), float(self.target[2]),
            float(self.distance), float(self.azimuth), float(self.elevation),
        )
        cache = getattr(self, '_view_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        _, view = orbit_camera(self.target, self.distance, self.azimuth, self.elevation)
        self._view_cache = (key, view)
        return view

    def get_projection_matrix(self) -> np.ndarray:
        key = (
            self.projection, float(self.fov), float(self.aspect),
            float(self.near), float(self.far), float(self.distance),
        )
        cache = getattr(self, '_proj_cache', None)
        if cache is not None and cache[0] == key:
            return cache[1]
        mat = self._compute_projection_matrix()
        self._proj_cache = (key, mat)
        return mat

    def _compute_projection_matrix(self) -> np.ndarray:
        if self.projection == 'orthographic':
            # Half-extent at the camera's target distance — keeps the cloud
            # roughly the same on-screen size as in perspective.
            half_h = self.distance * np.tan(self.fov / 2.0)
            half_w = half_h * self.aspect
            return orthographic(-half_w, half_w, -half_h, half_h,
                                self.near, self.far)
        return perspective(self.fov, self.aspect, self.near, self.far)

    def get_eye_position(self) -> np.ndarray:
        eye, _ = orbit_camera(self.target, self.distance, self.azimuth, self.elevation)
        return eye

    # --- presets ------------------------------------------------------------

    def set_preset(self, name: str):
        """Smoothly rotate to a named viewing preset.

        Planar presets (top/bottom/front/back/left/right) also force the
        projection to orthographic — a perspective view along a pure
        axis just skews the plane and offers no depth cue.
        """
        if name in PRESETS:
            az, el = PRESETS[name]
            # Rotate around the short way (mod 2π).
            az_delta = (az - self._azimuth_goal + np.pi) % (2 * np.pi) - np.pi
            self._azimuth_goal = self._azimuth_goal + az_delta
            self._elevation_goal = el
            self.active_preset = name
            if name in PLANAR_PRESETS:
                self.projection = 'orthographic'

    # --- input --------------------------------------------------------------

    def on_mouse_button(self, button: int, action: int, mods: int):
        """Handle mouse button events.

        LMB (no mods)  = pan
        Shift+LMB      = reserved for selection tools (app-side)
        RMB            = orbit
        MMB            = owned by the app (3D cursor placement)
        """
        pressed = action == 1
        shift = bool(mods & 0x0001)  # GLFW_MOD_SHIFT

        # Any new press cancels pending inertia immediately so a quick
        # grab feels hard-stopped, not like the camera is still drifting.
        if pressed and button in (0, 1):
            self._orbit_inertia_vel_az = 0.0
            self._orbit_inertia_vel_el = 0.0

        # Capture orbit inertia velocity at the moment of RMB release
        # (orbit is now RMB, not LMB).
        if button == 1 and not pressed:
            flick_speed = math.hypot(self._orbit_prev_dx, self._orbit_prev_dy)
            if flick_speed > _INERTIA_MIN_SPEED:
                gain = _ORBIT_GAIN * _INERTIA_ASSUMED_FPS * _INERTIA_LAUNCH_SCALE
                self._orbit_inertia_vel_az = -self._orbit_prev_dx * gain
                self._orbit_inertia_vel_el = self._orbit_prev_dy * gain

        if button == 0:
            # LMB without shift = pan. Shift+LMB is handed off to the
            # selection tool system in main.py.
            if pressed and not shift:
                self._panning = True
            elif not pressed:
                self._panning = False
        elif button == 1:             # RMB = orbit
            # Orbit is always unlocked. If the user grabs RMB while an
            # ortho plane preset is active, we drop the preset on press
            # so the first mouse motion already reads from a free-orbit
            # state (prevents getting stuck in TOP/FRONT/RIGHT views
            # while still letting the preset buttons / alien set them).
            if pressed and self.active_preset is not None:
                self.active_preset = None
            self._dragging = pressed
        # MMB (button 2) is owned by the app (3D cursor placement).

        if not pressed:
            if button == 0:
                self._panning = False
            if button == 1:
                self._dragging = False
            # Reset filter state on release so the next drag starts from
            # zero velocity (no stale bias).
            self._orbit_prev_dx = 0.0
            self._orbit_prev_dy = 0.0
            self._pan_prev_dx = 0.0
            self._pan_prev_dy = 0.0

    def on_mouse_move(self, x: float, y: float):
        """Handle mouse movement. Direct-drag inputs update both live and
        goal state so manual manipulation has zero perceived lag."""
        dx = x - self._last_mouse[0]
        dy = y - self._last_mouse[1]
        self._last_mouse = (x, y)

        if self._dragging:
            # Adaptive low-pass on orbit deltas (shared with pan via the
            # _smooth_drag_delta helper).
            sdx, sdy = self._smooth_drag_delta(
                dx, dy, self._orbit_prev_dx, self._orbit_prev_dy
            )
            self._orbit_prev_dx = sdx
            self._orbit_prev_dy = sdy

            # Velocity-scaled gain: sweeping flicks cover more angle per
            # pixel, slow framing stays at the precise base gain.
            speed = math.hypot(sdx, sdy)
            vel_boost = 1.0 + (_ORBIT_GAIN_BOOST - 1.0) * min(
                speed / _ORBIT_GAIN_BOOST_SPEED, 1.0
            )
            gain = _ORBIT_GAIN * vel_boost

            self.azimuth -= sdx * gain
            # Soft pole clamp — scale incoming elevation delta by how much
            # "slack" remains before the hard limit.
            eff_dy = sdy * _pole_softness(self.elevation)
            self.elevation += eff_dy * gain
            self.elevation = max(-_POLE_LIMIT, min(_POLE_LIMIT, self.elevation))
            self._azimuth_goal = self.azimuth
            self._elevation_goal = self.elevation
            if sdx != 0.0 or sdy != 0.0:
                self.active_preset = None
        elif self._panning:
            # Pan in camera-local XY plane with the same adaptive low-pass
            # used by orbit, so pan feels equally silky rather than raw.
            sdx, sdy = self._smooth_drag_delta(
                dx, dy, self._pan_prev_dx, self._pan_prev_dy
            )
            self._pan_prev_dx = sdx
            self._pan_prev_dy = sdy

            _, view = orbit_camera(self.target, self.distance, self.azimuth, self.elevation)
            right = view[0, 0:3]
            up = view[1, 0:3]
            # Use the distance GOAL (not the live, still-easing distance) so
            # pan sensitivity stays locked while a prior zoom ease is still
            # settling. Otherwise 1 px of cursor produces different amounts
            # of world movement each frame while distance glides.
            pan_speed = self._distance_goal * _PAN_GAIN
            self.target -= right * sdx * pan_speed
            self.target += up * sdy * pan_speed
            self._target_goal = self.target.copy()

    def on_scroll(self, offset: float):
        """Back-compat scroll without dolly-under-cursor.

        Callers that know the world point under the mouse should prefer
        :py:meth:`dolly_at` so the camera zooms toward the cursor.
        """
        self.dolly_at(offset, world_pivot=None)

    def cancel_all_drags(self):
        """Reset all drag states. Called on window focus loss to prevent
        stuck drags when the OS swallows the button-release event."""
        self._dragging = False
        self._panning = False
        self._orbit_prev_dx = 0.0
        self._orbit_prev_dy = 0.0
        self._pan_prev_dx = 0.0
        self._pan_prev_dy = 0.0
        self._orbit_inertia_vel_az = 0.0
        self._orbit_inertia_vel_el = 0.0

    def set_mouse_pos(self, x: float, y: float):
        """Update stored mouse position without triggering movement."""
        self._last_mouse = (x, y)
