"""Mouse click / double-click / drag state machine.

Distinguishes between a true click (press + release with no meaningful
movement and no long hold) and a drag (press + release with movement
above a pixel threshold). Also detects double clicks when two clicks
land on the same button within a short time window and a small spatial
radius.

The detector is deliberately event-stream based — it receives press,
move, and release events from the caller and returns what that release
"counted as" at the moment of release. Single clicks are reported
immediately (they are compound-safe with double clicks in this app:
a single click runs focus, a double click adds a tetrahedron placement
on top).
"""

import time


# Thresholds chosen so a normal mouse-down-release-up without intended
# motion reliably counts as a click, while any deliberate drag does not.
_CLICK_RADIUS_SQ = 16.0      # 4 px movement budget
_CLICK_MAX_HOLD = 0.18       # max press duration to still count as a click
_DOUBLE_TIME = 0.28          # max gap between the two clicks of a double
_DOUBLE_RADIUS_SQ = 64.0     # 8 px position budget for the second click


class ClickDetector:
    """Per-button click / double-click / drag state machine.

    Usage::

        d = ClickDetector()
        d.on_press(button, mx, my)
        d.on_move(mx, my)        # call on every mouse move
        kind = d.on_release(button, mx, my)
        # kind is 'click', 'double', or None (drag or cancelled)
    """

    def __init__(self):
        # Per-button "currently pressed" state.
        self._press_time: dict[int, float] = {}
        self._press_pos: dict[int, tuple[float, float]] = {}
        self._moved: dict[int, bool] = {}
        # Per-button "last click that might pair into a double" state.
        self._last_click: dict[int, tuple[float, float, float]] = {}

    def on_press(self, button: int, x: float, y: float) -> None:
        self._press_time[button] = time.perf_counter()
        self._press_pos[button] = (x, y)
        self._moved[button] = False

    def on_move(self, x: float, y: float) -> None:
        """Mark any currently-held button as 'moved' if the cursor has
        travelled outside its click radius. This is what lets us tell
        a click apart from a drag at release time."""
        for btn, pos in self._press_pos.items():
            if pos is None or self._moved.get(btn, False):
                continue
            dx = x - pos[0]
            dy = y - pos[1]
            if dx * dx + dy * dy > _CLICK_RADIUS_SQ:
                self._moved[btn] = True

    def on_release(self, button: int, x: float, y: float) -> str | None:
        """Consume a release event and classify the gesture.

        Returns:
          * ``'click'`` — short, in-place release (first of a possible pair).
          * ``'double'`` — second click of a double within the time / radius
            window. The stored pair state is cleared so a third click in
            rapid succession starts a new chain as a single.
          * ``None`` — this release was a drag or was otherwise not a click.
        """
        press_pos = self._press_pos.get(button)
        press_time = self._press_time.get(button)
        moved = self._moved.get(button, False)

        # Clear press state for this button before returning.
        self._press_pos[button] = None
        self._press_time[button] = None
        self._moved[button] = False

        if press_pos is None or press_time is None:
            return None
        if moved:
            return None

        now = time.perf_counter()
        if now - press_time > _CLICK_MAX_HOLD:
            # Held too long; not a click (future hook: long-press gestures).
            return None

        last = self._last_click.get(button)
        if last is not None:
            last_t, last_x, last_y = last
            if (now - last_t) < _DOUBLE_TIME:
                dx = x - last_x
                dy = y - last_y
                if dx * dx + dy * dy < _DOUBLE_RADIUS_SQ:
                    # Second click completes a double. Clear pair state so
                    # a third click in rapid succession starts fresh.
                    self._last_click[button] = None
                    return 'double'

        # Single click — remember it so a follow-up may form a double.
        self._last_click[button] = (now, x, y)
        return 'click'

    def reset(self) -> None:
        self._press_time.clear()
        self._press_pos.clear()
        self._moved.clear()
        self._last_click.clear()

    def any_confirmed_drag(self) -> bool:
        """True if any currently-pressed button has crossed the click
        radius. Used by callers to suppress camera drag inside the
        "click ambiguity zone" at the very start of a press, so a
        micro-wobble can't produce both a tiny drag and a click."""
        return any(self._moved.values())
