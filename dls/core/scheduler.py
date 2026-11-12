"""Inference scheduler shared by every camera of one device (pydetect `yolo_motion`).

On a small CPU one 768 inference of the fire model takes ~1 s on all cores, and two at once are
slower in total than one after the other (OpenVINO THROUGHPUT 0.45 vs LATENCY 0.71 inf/s on the
i3-4160). So the device runs ONE inference at a time and the scheduler decides whose turn it is:

  priority  CONFIRM (3)   fire / smoke suspected or just seen: confirm with full frames now
            FULL_DUE (2)  the camera's periodic full frame (fire / smoke scan) is due
            MOSAIC (1)    persons: motion crops or tracked objects
  + aging: a camera that keeps asking without a turn gains one level per `aging_s`, so a busy
    camera cannot starve a quiet one's periodic scan.

acquire() blocks at most `max_wait_s` (a fraction of a frame): the caller runs the motion
detector on every frame and must not stall; a camera that gets no turn simply skips inference on
this frame (its tracker predicts) and asks again on the next one with an older `since`.
Pure Python; thread-safe; `clock` is injectable for tests.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Hashable

CONFIRM, FULL_DUE, MOSAIC = 3, 2, 1
STALE_S = 2.0          # a camera that has not asked for this long starts aging from zero again


class InferenceScheduler:
    def __init__(self, slots: int = 1, aging_s: float = 4.0, max_wait_s: float = 0.1,
                 clock: Callable[[], float] = time.monotonic):
        if slots < 1:
            raise ValueError("slots phải ≥ 1")
        self.slots, self.aging_s, self.max_wait_s = int(slots), float(aging_s), float(max_wait_s)
        self._clock = clock
        self._cond = threading.Condition()
        self._busy = 0
        self._waiting: dict[Hashable, int] = {}          # key → priority, while inside acquire()
        self._since: dict[Hashable, float] = {}          # key → first unanswered request
        self._asked: dict[Hashable, float] = {}          # key → last request
        self.stats: dict[Hashable, dict] = {}

    def acquire(self, key: Hashable, priority: int, max_wait_s: float | None = None) -> bool:
        """True = run the inference now, then call release(key). False = skip this frame."""
        wait = self.max_wait_s if max_wait_s is None else float(max_wait_s)
        with self._cond:
            now = self._clock()
            st = self.stats.setdefault(key, {"granted": 0, "denied": 0, "waited_s": 0.0, "max_age_s": 0.0})
            if now - self._asked.get(key, now) > STALE_S:
                self._since.pop(key, None)
            self._asked[key] = now
            self._since.setdefault(key, now)
            self._waiting[key] = int(priority)
            deadline = now + wait
            try:
                while True:
                    now = self._clock()
                    if self._busy < self.slots and self._best(now) == key:
                        self._busy += 1
                        age = now - self._since.pop(key)
                        st["granted"] += 1
                        st["waited_s"] += age
                        st["max_age_s"] = max(st["max_age_s"], age)
                        return True
                    if now >= deadline:
                        st["denied"] += 1
                        return False
                    self._cond.wait(deadline - now)
            finally:
                self._waiting.pop(key, None)
                self._cond.notify_all()            # the next best waiter may go now

    def release(self, key: Hashable) -> None:
        with self._cond:
            self._busy = max(0, self._busy - 1)
            self._cond.notify_all()

    def forget(self, key: Hashable) -> None:
        with self._cond:
            for d in (self._since, self._asked, self.stats):
                d.pop(key, None)

    def effective(self, key: Hashable, priority: int, now: float) -> float:
        return priority + (now - self._since.get(key, now)) / self.aging_s

    def _best(self, now: float) -> Hashable:
        return max(self._waiting, key=lambda k: (self.effective(k, self._waiting[k], now), -self._since.get(k, now)))


_shared: dict[str, InferenceScheduler] = {}
_shared_lock = threading.Lock()


def shared_scheduler(device: str) -> InferenceScheduler:
    """One scheduler per inference device of the process; slots = concurrent OpenVINO streams
    (dls.core.cpu: DLS_INFER_SLOTS, auto from the CPU budget)."""
    from dls.core.cpu import get_infer_slots
    with _shared_lock:
        if device not in _shared:
            _shared[device] = InferenceScheduler(slots=get_infer_slots())
        return _shared[device]
