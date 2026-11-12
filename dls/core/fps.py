"""Thread-safe sliding-window FPS meter (used per camera source and per branch chain)."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional


class FpsMeter:
    def __init__(self, window_s: float = 5.0):
        self._window = window_s
        self._ticks: deque[float] = deque()
        self._lock = threading.Lock()
        self.total = 0
        self.last_tick: Optional[float] = None

    def tick(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._ticks.append(now)
            self.total += 1
            self.last_tick = now
            self._trim(now)

    def rate(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._trim(now)
            if len(self._ticks) < 2:
                return 0.0
            span = max(self._ticks[-1] - self._ticks[0], 1e-6)
            return (len(self._ticks) - 1) / span

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = time.monotonic() if now is None else now
        idle = None if self.last_tick is None else round(now - self.last_tick, 3)
        return {"fps": round(self.rate(now), 2), "frames": self.total, "idle_s": idle}

    def _trim(self, now: float) -> None:
        while self._ticks and now - self._ticks[0] > self._window:
            self._ticks.popleft()
