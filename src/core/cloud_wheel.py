"""Turn high-resolution horizontal wheel events into deliberate cloud steps."""

import math
import time


class CloudWheel:
    def __init__(self):
        self.reset()

    def reset(self):
        self._amount = 0.0
        self._last_event = -math.inf
        self._last_step = -math.inf

    def step(self, offset: float, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        if not math.isfinite(offset) or offset == 0:
            return 0
        if now - self._last_event > 0.35 or self._amount * offset < 0:
            self._amount = 0.0
        self._last_event = now
        self._amount = max(-1.0, min(1.0, self._amount + offset))
        if abs(self._amount) < 1.0 or now - self._last_step < 0.14:
            return 0
        result = 1 if self._amount > 0 else -1
        self._amount = 0.0
        self._last_step = now
        return result
