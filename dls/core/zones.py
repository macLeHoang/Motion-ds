"""Polygon zones (replacement for nvdsanalytics ROI filtering) + enter/exit tracking per object."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from dls.core.metadata import Detection

Point = tuple[float, float]
MAX_BORN = 4096          # first-seen times kept before pruning ids gone for BORN_TTL_S
BORN_TTL_S = 600.0


@dataclass(frozen=True)
class Zone:
    zone_id: str
    polygon: tuple[Point, ...]
    class_ids: Optional[frozenset[int]] = None    # None → every class
    inverse: bool = False

    def contains(self, x: float, y: float) -> bool:
        inside = _point_in_polygon(x, y, self.polygon)
        return not inside if self.inverse else inside

    def accepts(self, label_id: int) -> bool:
        return self.class_ids is None or label_id in self.class_ids


def _point_in_polygon(x: float, y: float, poly: tuple[Point, ...]) -> bool:
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def zones_from_camera_config(camera_cfg: Mapping[str, Any], frame_w: int, frame_h: int) -> list[Zone]:
    """Build zones from the area_monitoring per-camera config (same schema as the DeepStream app).

    Polygons are given in `config_width` x `config_height` space and scaled to the frame size.
    """
    sx = frame_w / float(camera_cfg.get("config_width", frame_w) or frame_w)
    sy = frame_h / float(camera_cfg.get("config_height", frame_h) or frame_h)
    zones = []
    for roi in camera_cfg.get("roi_filtering", []) or []:
        polygon = roi.get("polygon", [])
        if len(polygon) < 3:
            raise ValueError(f"ROI '{roi.get('roi_id')}' cần ít nhất 3 điểm")
        class_ids = roi.get("class_id")
        zones.append(Zone(
            zone_id=str(roi.get("roi_id", f"roi{len(zones)}")),
            polygon=tuple((float(px) * sx, float(py) * sy) for px, py in polygon),
            class_ids=frozenset(int(c) for c in class_ids) if class_ids is not None else None,
            inverse=bool(roi.get("inverse", False)),
        ))
    return zones


class ZoneTracker:
    """Tracks which tracked objects are inside which zone; emits zone_enter / zone_exit dicts.

    The object's bottom-center (feet) is the reference point. An object counts as out only after
    `exit_timeout_s` outside while still detected ("left"), or `lost_timeout_s` without being
    detected at all ("lost": small far-away people drop out of detection for a few seconds), so
    feet jittering across the zone edge or missed detections are not an exit + re-entry.

    `rebind_dist` (> 0): a NEW id inside the zone whose feet are within rebind_dist × person height
    of an object that is inside but not detected now (lost, still within lost_timeout_s) takes over
    that object silently — no entry, no exit. Motion mode looks at a still worker only now and
    then, and the tracker gives him a new id after the gap (ROI.mp4: 301 entries from ~15 workers).
    A detected object is never taken over, so two people side by side stay two.

    `merge_overlap` (> 0): tracked boxes of one label that overlap by at least this share of the
    smaller box are one object, counted under its oldest id. Motion-mode crops give the same worker
    boxes of different sizes, and the imageless tracker keeps 2–3 tracks on him (ROI.mp4: 10
    "visible" people in a zone holding 5). The younger duplicate is dropped silently.
    """

    def __init__(self, exit_timeout_s: float = 2.0, lost_timeout_s: Optional[float] = None,
                 rebind_dist: float = 0.0, merge_overlap: float = 0.0, enter_after_s: float = 0.0):
        self._exit_timeout = exit_timeout_s
        self._lost_timeout = exit_timeout_s if lost_timeout_s is None else lost_timeout_s
        self._rebind = float(rebind_dist)
        self._merge = float(merge_overlap)
        self._enter_after = float(enter_after_s)
        # (camera_id, zone_id) → {object_id: first time inside} for ids not counted yet
        self._pending: dict[tuple[str, str], dict[int, float]] = {}
        # (camera_id, object_id) → stream time first seen (oldest id wins a merge)
        self._born: dict[tuple[str, int], float] = {}
        # (camera_id, zone_id) → {object_id: last_seen_inside}
        self._inside: dict[tuple[str, str], dict[int, float]] = {}
        # (camera_id, zone_id) → {object_id: (feet x, feet y, height)} where last seen inside
        self._where: dict[tuple[str, str], dict[int, tuple[float, float, float]]] = {}
        self.stats = {"rebound": 0, "merged": 0}
        self._lock = threading.Lock()

    def update(self, camera_id: str, zones: Iterable[Zone], detections: Iterable[Detection],
               now: Optional[float] = None) -> list[dict]:
        now = time.monotonic() if now is None else now
        tracked = [d for d in detections if d.object_id is not None]
        events: list[dict] = []
        with self._lock:
            tracked = self._merge_duplicates(camera_id, tracked, now)
            for zone in zones:
                key = (camera_id, zone.zone_id)
                state = self._inside.setdefault(key, {})
                where = self._where.setdefault(key, {})
                present = {d.object_id: d for d in tracked
                           if zone.accepts(d.label_id) and zone.contains(*d.bottom_center)}
                visible = {d.object_id for d in tracked}
                pending = self._pending.setdefault(key, {})
                for oid in [o for o in pending if o not in present]:
                    del pending[oid]                        # must stay inside without a break
                zone_events: list[dict] = []
                for oid, d in present.items():
                    if oid not in state:
                        old = self._takes_over(d, state, where, visible)
                        if old is not None:
                            del state[old]
                            where.pop(old, None)
                            self.stats["rebound"] += 1
                        elif now - pending.setdefault(oid, now) < self._enter_after:
                            continue                        # not long enough inside yet
                        else:
                            pending.pop(oid, None)
                            zone_events.append({**_zone_event("zone_enter", zone, d.object_id, d.label, "inside"),
                                                "bbox": [round(d.x, 1), round(d.y, 1), round(d.w, 1), round(d.h, 1)]})
                    state[oid] = now
                    where[oid] = (*d.bottom_center, float(d.h))
                for oid in list(state):
                    reason = "left" if oid in visible else "lost"
                    timeout = self._exit_timeout if reason == "left" else self._lost_timeout
                    if oid in present or now - state[oid] <= timeout:
                        continue                # edge jitter / a few missed detections
                    del state[oid]
                    where.pop(oid, None)
                    zone_events.append(_zone_event("zone_exit", zone, oid, "", reason))
                for e in zone_events:
                    e["occupancy"] = len(state)        # people in the zone after this frame
                events.extend(zone_events)
        return events

    def _merge_duplicates(self, camera_id: str, tracked: list[Detection], now: float) -> list[Detection]:
        for d in tracked:
            self._born.setdefault((camera_id, d.object_id), now)
        if len(self._born) > MAX_BORN:
            live = {(camera_id, d.object_id) for d in tracked}
            for k in [k for k, t in self._born.items() if k not in live and now - t > BORN_TTL_S]:
                del self._born[k]
        if self._merge <= 0 or len(tracked) < 2:
            return tracked
        oldest_first = sorted(tracked, key=lambda d: (self._born[(camera_id, d.object_id)], d.object_id))
        kept: list[Detection] = []
        for d in oldest_first:
            if any(k.label_id == d.label_id and _overlap_of_smaller(k, d) >= self._merge for k in kept):
                self.stats["merged"] += 1
                continue
            kept.append(d)
        return kept

    def _takes_over(self, d: Detection, state: dict, where: dict, visible: set) -> Optional[int]:
        """The lost object (inside, not detected now) nearest to d's feet within rebind_dist."""
        if self._rebind <= 0:
            return None
        fx, fy = d.bottom_center
        best, best_dist = None, None
        for oid in state:
            if oid in visible or oid not in where:
                continue
            ox, oy, oh = where[oid]
            dist = ((fx - ox) ** 2 + (fy - oy) ** 2) ** 0.5
            if dist <= self._rebind * max(oh, float(d.h)) and (best_dist is None or dist < best_dist):
                best, best_dist = oid, dist
        return best

    def occupancy(self, camera_id: str) -> dict[str, int]:
        with self._lock:
            return {zid: len(objs) for (cam, zid), objs in self._inside.items() if cam == camera_id}

    def drop_camera(self, camera_id: str) -> None:
        with self._lock:
            for key in [k for k in self._inside if k[0] == camera_id]:
                del self._inside[key]
                self._where.pop(key, None)
            for key in [k for k in self._born if k[0] == camera_id]:
                del self._born[key]
            for key in [k for k in self._pending if k[0] == camera_id]:
                del self._pending[key]


def _overlap_of_smaller(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    small = min(a.area, b.area)
    return ix * iy / small if small > 0 else 0.0


def _zone_event(event_type: str, zone: Zone, object_id: int, label: str, reason: str) -> dict:
    return {"type": event_type, "zone_id": zone.zone_id, "object_id": object_id,
            "label": label, "reason": reason}
