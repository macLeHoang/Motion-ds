"""Event gate: area gates, ghost filter and N-of-M confirmation before an event is emitted.

Distilled from dlstreamer-edge-reference/decision/event_decider.py (same defaults):
- area gate: a box under 3e-4 of the frame (FOV-blind speck) or over 0.95 (full-frame glitch)
  never makes an event;
- ghost filter: a track that is old enough (≥ ghost_min_frames and ≥ ghost_min_age_s) and whose
  foot point never moved (spread < ghost_var_eps of the frame) is static clutter (poster, coat on
  a hook). A track that moved once (spread ≥ ghost_move_eps) is exempt for good, so a person who
  stops is kept. The same test runs per foot-point grid cell so it survives tracker id churn; a
  cell touched by a track that moved is a thoroughfare and is exempt. Only `ghost_classes`
  (fire and smoke flicker in place and must never be ghosted);
- confirm: an event key must be seen in ≥ confirm_hits of the last confirm_window frames
  (single-frame false positives never fire).
Time is the caller's clock (buffer PTS) so file playback and live streams behave the same.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, Mapping, Optional

from dls.core.metadata import Detection

PARAM_KEYS = ("min_area", "max_area", "ghost", "ghost_classes", "ghost_min_frames", "ghost_min_age_s",
              "ghost_var_eps", "ghost_move_eps", "ghost_ttl_s", "ghost_cell", "confirm_hits", "confirm_window")
HISTORY = 150
MAX_STATE = 512


@dataclass
class _History:
    t0: float
    last: float
    xs: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    ys: deque = field(default_factory=lambda: deque(maxlen=HISTORY))
    moved: bool = False

    def add(self, x: float, y: float, now: float) -> float:
        self.last = now
        self.xs.append(x)
        self.ys.append(y)
        return max(max(self.xs) - min(self.xs), max(self.ys) - min(self.ys))


class EventGate:
    def __init__(self, min_area: float = 3e-4, max_area: float = 0.95, ghost: bool = True,
                 ghost_classes: Optional[Iterable[str]] = ("person",), ghost_min_frames: int = 30,
                 ghost_min_age_s: float = 4.0, ghost_var_eps: float = 0.02, ghost_move_eps: float = 0.08,
                 ghost_ttl_s: float = 30.0, ghost_cell: float = 0.03, confirm_hits: int = 1,
                 confirm_window: int = 1):
        if not 1 <= confirm_hits <= confirm_window:
            raise ValueError("confirm_hits phải trong [1, confirm_window]")
        self.min_area, self.max_area = min_area, max_area
        self.ghost = ghost
        self.ghost_classes = set(ghost_classes) if ghost_classes else None
        self.ghost_min_frames, self.ghost_min_age_s = ghost_min_frames, ghost_min_age_s
        self.ghost_var_eps, self.ghost_move_eps = ghost_var_eps, ghost_move_eps
        self.ghost_ttl_s, self.ghost_cell = ghost_ttl_s, ghost_cell
        self.confirm_hits, self.confirm_window = confirm_hits, confirm_window
        self._lock = threading.Lock()
        self._tracks: dict[tuple, _History] = {}
        self._cells: dict[tuple, _History] = {}
        self._hits: dict[tuple, deque] = {}
        self.stats = {"area_dropped": 0, "ghosts": 0, "confirmed": 0, "unconfirmed": 0}

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "EventGate":
        unknown = [k for k in params if k not in PARAM_KEYS]
        if unknown:
            raise ValueError(f"event_gate: khóa không hỗ trợ {unknown} (chỉ {list(PARAM_KEYS)})")
        return cls(**dict(params))

    # ------------------------------------------------------------------ filter
    def filter(self, camera_id: str, detections: Iterable[Detection], width: int, height: int,
               now: float) -> list[Detection]:
        frame_area = float(width * height) or 1.0
        kept = []
        with self._lock:
            self._prune(now)
            for d in detections:
                if not self.min_area <= d.area / frame_area < self.max_area:
                    self.stats["area_dropped"] += 1
                    continue
                if self._is_ghost(camera_id, d, width, height, now):
                    self.stats["ghosts"] += 1
                    continue
                kept.append(d)
        return kept

    def _is_ghost(self, camera_id: str, d: Detection, width: int, height: int, now: float) -> bool:
        if not self.ghost or (self.ghost_classes is not None and d.label not in self.ghost_classes):
            return False
        fx, fy = d.bottom_center[0] / width, d.bottom_center[1] / height
        track_moved = False
        if d.object_id is not None:
            h = self._history(self._tracks, (camera_id, d.object_id), now)
            spread = h.add(fx, fy, now)
            h.moved = h.moved or spread >= self.ghost_move_eps
            track_moved = h.moved
            if self._static(h, spread, now):
                return True
        cell_key = (camera_id, d.label, int(fx / self.ghost_cell), int(fy / self.ghost_cell))
        cell = self._history(self._cells, cell_key, now)
        spread = cell.add(fx, fy, now)
        cell.moved = cell.moved or track_moved      # a mover passed here → thoroughfare, exempt
        return self._static(cell, spread, now)

    def _static(self, h: _History, spread: float, now: float) -> bool:
        return (not h.moved and len(h.xs) >= self.ghost_min_frames
                and now - h.t0 >= self.ghost_min_age_s and spread < self.ghost_var_eps)

    def _history(self, store: dict, key: tuple, now: float) -> _History:
        h = store.get(key)
        if h is None or now - h.last > self.ghost_ttl_s:
            h = store[key] = _History(now, now)
        return h

    def _prune(self, now: float) -> None:
        for store in (self._tracks, self._cells):
            if len(store) > MAX_STATE:
                for k in [k for k, h in store.items() if now - h.last > self.ghost_ttl_s]:
                    del store[k]

    # ------------------------------------------------------------------ confirm
    def confirm(self, camera_id: str, key: Hashable, frame_num: int) -> bool:
        """Record a hit of `key` at `frame_num`; True once ≥ confirm_hits fall in the last confirm_window frames."""
        with self._lock:
            hits = self._hits.setdefault((camera_id, key), deque(maxlen=self.confirm_hits))
            if not hits or hits[-1] != frame_num:
                hits.append(frame_num)
            ok = len(hits) >= self.confirm_hits and frame_num - hits[0] < self.confirm_window
            self.stats["confirmed" if ok else "unconfirmed"] += 1
            return ok

    def drop_camera(self, camera_id: str) -> None:
        with self._lock:
            for store in (self._tracks, self._cells, self._hits):
                for k in [k for k in store if k[0] == camera_id]:
                    del store[k]


class IncidentDeduper:
    """One event per spatial incident (fire, smoke): tracker ids of flickering flames churn every
    frame, so per-id de-duplication floods events. A detection overlapping (box grown by `expand`)
    an incident of the same kind seen within `ttl_s` refreshes it instead of firing again. An
    incident seen within the last `follow_s` is followed further (box grown by `follow_expand`):
    a burning object carried across the frame moves several box sizes between inferences."""

    def __init__(self, ttl_s: float = 30.0, expand: float = 0.5, follow_s: float = 2.0,
                 follow_expand: float = 6.0):
        self.ttl_s, self.expand = ttl_s, expand
        self.follow_s, self.follow_expand = follow_s, follow_expand
        self._incidents: dict[tuple, list[list]] = {}     # (camera, kind) → [[box, last_seen], ...]
        self._lock = threading.Lock()

    def add(self, camera_id: str, kind: str, box: tuple[float, float, float, float], now: float) -> bool:
        """box = (x1, y1, x2, y2). True when this is a new incident (emit), False when known."""
        with self._lock:
            live = [inc for inc in self._incidents.get((camera_id, kind), []) if now - inc[1] <= self.ttl_s]
            self._incidents[(camera_id, kind)] = live
            for inc in live:
                grow = self.follow_expand if now - inc[1] <= self.follow_s else self.expand
                if _overlaps(_grow(inc[0], grow), box):
                    inc[0], inc[1] = box, now
                    return False
            live.append([box, now])
            return True

    def drop_camera(self, camera_id: str) -> None:
        with self._lock:
            for k in [k for k in self._incidents if k[0] == camera_id]:
                del self._incidents[k]


def _grow(box, frac: float):
    dx, dy = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return (box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy)


def _overlaps(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
