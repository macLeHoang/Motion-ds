"""Event review without labelling: a person marks each alert right / wrong, and a few alert-free
frames "missed something?". Pure Python (cv2 for the pictures); used by dls.tools.event_review.

Item (one picture to judge):
  {"id", "job", "kind": "event" | "sample", "type", "stream_ts", "image", "confidence", "tasks"}
  event   an alert the pipeline raised: is it real?
  sample  a frame with no alert within SAMPLE_GAP_S, shown with what the system believes then
          (`state`: people it counts per zone, alerts it already raised). Alerts fire ONCE per
          entry / incident, so a worker standing in a zone or a fire burning on is not a miss:
          the question is "an incident the system never alerted?" plus, for zones, the real
          head count per zone (optional numbers) to score the system's count.
Verdicts: {"id": {"verdict": "right" | "wrong" | "unsure" | "missed" | "clear", "note", "at",
                  "counts": {zone_id: n}}}, written atomically next to the items.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections import Counter, defaultdict
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

EVENT_VERDICTS = ("right", "wrong", "unsure")
SAMPLE_VERDICTS = ("missed", "clear", "unsure")
SKIP_TYPES = {"camera_offline", "camera_online", "zone_exit"}     # not alerts
SAMPLE_GAP_S = 3.0
EARLIER_S = 30.0          # sample pictures also draw (orange) the alerts of the last EARLIER_S
MAX_COUNT = 999
RED, YELLOW, ORANGE = (0, 0, 255), (0, 220, 255), (0, 140, 255)


def item_id(job: str, event: dict) -> str:
    """Stable across rebuilds of the same run: verdicts survive a re-render of the pictures."""
    key = json.dumps([job, event.get("type"), round(float(event.get("stream_ts") or 0), 1),
                      event.get("bbox"), event.get("roi_id")], sort_keys=True, default=str)
    return f"{job}-{hashlib.sha1(key.encode()).hexdigest()[:12]}"


def reviewable(events: Iterable[dict]) -> list[dict]:
    return [e for e in events if e.get("type") not in SKIP_TYPES and e.get("stream_ts") is not None]


def spread(events: Sequence[dict], per_type: int) -> list[dict]:
    """At most per_type alerts of each type, evenly spaced in time (the precision of a type is
    estimated from them; nobody judges 344 zone entries)."""
    by: dict[str, list[dict]] = defaultdict(list)
    for e in sorted(events, key=lambda e: float(e["stream_ts"])):
        by[e["type"]].append(e)
    out = []
    for evs in by.values():
        if per_type <= 0 or len(evs) <= per_type:
            out += evs
        else:
            out += [evs[round(i * (len(evs) - 1) / (per_type - 1))] for i in range(per_type)] if per_type > 1 else evs[:1]
    return sorted(out, key=lambda e: float(e["stream_ts"]))


def sample_times(duration_s: float, event_times: Sequence[float], n: int,
                 gap_s: float = SAMPLE_GAP_S, start_s: float = 10.0) -> list[float]:
    """Up to n times evenly spread over the alert-free part of the clip, each ≥ gap_s from every
    alert (start_s skips warm-up). Random picks bunched up: 7 of 30 ROI samples fell in 263–288 s."""
    free = [t / 10 for t in range(int(start_s * 10), int(duration_s * 10))
            if all(abs(t / 10 - e) >= gap_s for e in event_times)]
    if not free or n <= 0:
        return []
    if len(free) <= n:
        return [round(t, 1) for t in free]
    if n == 1:
        return [round(free[len(free) // 2], 1)]
    return [round(free[round(i * (len(free) - 1) / (n - 1))], 1) for i in range(n)]


def occupancy_at(events: Sequence[dict], zone_ids: Sequence[str], t: float) -> dict[str, int]:
    """People the system counted in each zone at stream time t (the `occupancy` of the last zone
    event of that zone before t; no event yet → 0)."""
    occ = {z: 0 for z in zone_ids}
    for e in sorted((e for e in events if e.get("zone_id") in occ and "occupancy" in e),
                    key=lambda e: float(e["stream_ts"])):
        if float(e["stream_ts"]) > t:
            break
        occ[e["zone_id"]] = int(e["occupancy"])
    return occ


def alerts_before(events: Sequence[dict], t: float, n: int = 8) -> list[dict]:
    """The last n alerts (not zone entries: the counts show those) raised at or before t, newest first."""
    past = [e for e in reviewable(events) if e["type"] != "zone_enter" and float(e["stream_ts"]) <= t]
    past.sort(key=lambda e: float(e["stream_ts"]), reverse=True)
    return [{"type": e["type"], "stream_ts": round(float(e["stream_ts"]), 1), "bbox": e.get("bbox")}
            for e in past[:n]]


def read_frame(video: str, t: float) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _box_px(bbox: Sequence[float], scale: tuple[float, float]) -> tuple[int, int, int, int]:
    bx, by, bw, bh = (float(v) for v in bbox)              # event bbox: [x, y, w, h] in pipeline pixels
    return int(bx * scale[0]), int(by * scale[1]), int((bx + bw) * scale[0]), int((by + bh) * scale[1])


def draw(frame: np.ndarray, bbox: Optional[Sequence[float]], scale: tuple[float, float], zones: Sequence[dict] = (),
         zone_scale: tuple[float, float] = (1.0, 1.0), caption: str = "", width: int = 1280,
         zone_counts: Optional[Sequence[int]] = None, earlier: Sequence[Sequence[float]] = ()) -> np.ndarray:
    """The full frame (zones yellow, numbered, with the system's count when given; earlier alerts
    orange; the alert box red) and, for a box, a 3× context zoom beside it."""
    img = frame.copy()
    thick = max(2, frame.shape[1] // 400)
    for i, z in enumerate(zones):
        pts = np.array([[x * zone_scale[0], y * zone_scale[1]] for x, y in z.get("polygon", [])], np.int32)
        if len(pts) < 3:
            continue
        cv2.polylines(img, [pts], True, YELLOW, 2)
        tag = f"{i + 1}" + (f": {zone_counts[i]}" if zone_counts is not None else "")
        x, y = int(pts[:, 0].min()) + 4, int(pts[:, 1].min()) + 4
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(img, (x, y), (x + tw + 8, y + th + 10), YELLOW, -1)
        cv2.putText(img, tag, (x + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    for b in earlier:
        x1, y1, x2, y2 = _box_px(b, scale)
        cv2.rectangle(img, (x1, y1), (x2, y2), ORANGE, thick)
    zoom = None
    if bbox:
        x1, y1, x2, y2 = _box_px(bbox, scale)
        cv2.rectangle(img, (x1, y1), (x2, y2), RED, thick)
        pad = max(x2 - x1, y2 - y1)
        X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
        X2, Y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
        if X2 - X1 > 4 and Y2 - Y1 > 4:
            zoom = img[Y1:Y2, X1:X2]
    h = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img, (width, h))
    if zoom is not None:
        zoom = cv2.resize(zoom, (int(zoom.shape[1] * h / zoom.shape[0]), h))
        img = cv2.hconcat([img, zoom])
    if caption:
        cv2.putText(img, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(img, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def atomic_write_json(path: str, data: Any) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


class ReviewStore:
    """items.json (from build) + verdicts.json (from the reviewer), in one directory."""

    def __init__(self, root: str):
        self.root = root
        self._lock = threading.Lock()
        with open(os.path.join(root, "items.json"), encoding="utf-8") as fh:
            self.items: list[dict] = json.load(fh)
        self._by_id = {it["id"]: it for it in self.items}
        path = os.path.join(root, "verdicts.json")
        self.verdicts: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                self.verdicts = json.load(fh)

    def item(self, item_id_: str) -> Optional[dict]:
        return self._by_id.get(item_id_)

    def image_path(self, item_id_: str) -> Optional[str]:
        it = self._by_id.get(item_id_)
        if it is None:
            return None
        path = os.path.realpath(os.path.join(self.root, it["image"]))
        return path if path.startswith(os.path.realpath(self.root) + os.sep) and os.path.exists(path) else None

    def judge(self, item_id_: str, verdict: str, note: str = "", counts: Optional[dict] = None) -> dict:
        it = self._by_id.get(item_id_)
        if it is None:
            raise KeyError(item_id_)
        allowed = EVENT_VERDICTS if it["kind"] == "event" else SAMPLE_VERDICTS
        if verdict not in allowed:
            raise ValueError(f"'{verdict}' không hợp lệ cho {it['kind']}; chỉ {list(allowed)}")
        entry = {"verdict": verdict, "note": str(note)[:500], "at": time.time()}
        if counts:
            entry["counts"] = _valid_counts(it, counts)
        with self._lock:
            self.verdicts[item_id_] = entry
            atomic_write_json(os.path.join(self.root, "verdicts.json"), self.verdicts)
            return self.verdicts[item_id_]

    def report(self) -> dict:
        return report(self.items, self.verdicts)


CARRY_DT_S = 1.5        # a rerun's alert within this of a judged one (same type, zone, place) inherits its verdict
CARRY_DIST = 0.5        # … centres within this × the taller box, heights within CARRY_SIZE ×
CARRY_SIZE = 1.5


def _same_alert(a: dict, b: dict) -> bool:
    if a["job"] != b["job"] or a["type"] != b["type"] or a.get("zone_id") != b.get("zone_id"):
        return False
    if abs(float(a["stream_ts"]) - float(b["stream_ts"])) > CARRY_DT_S:
        return False
    if not a.get("bbox") or not b.get("bbox"):
        return not a.get("bbox") and not b.get("bbox")
    (ax, ay, aw, ah), (bx, by, bw, bh) = a["bbox"], b["bbox"]
    h = max(ah, bh)
    dist = ((ax + aw / 2 - bx - bw / 2) ** 2 + (ay + ah / 2 - by - bh / 2) ** 2) ** 0.5
    return dist <= CARRY_DIST * h and min(ah, bh) * CARRY_SIZE >= h


def carry_verdicts(new_items: Sequence[dict], old_items: Sequence[dict], verdicts: dict) -> dict[str, dict]:
    """Verdicts for new alert items that are the same alert as a judged old one.

    The pipeline is not bit-exact between runs (thread timing moves alerts by a frame or two and
    boxes by a few pixels), so stable ids alone kept 31 of 189 verdicts after a rebuild. Each
    old verdict is used once, by the nearest new alert in time. Samples are not carried: their
    question depends on the state at that moment."""
    pool = [o for o in old_items if o["kind"] == "event" and o["id"] in verdicts]
    used: set[str] = set()
    out: dict[str, dict] = {}
    for it in new_items:
        if it["kind"] != "event" or it["id"] in verdicts:
            continue
        best = min((o for o in pool if o["id"] not in used and _same_alert(it, o)),
                   key=lambda o: abs(float(o["stream_ts"]) - float(it["stream_ts"])), default=None)
        if best is not None:
            used.add(best["id"])
            out[it["id"]] = {**verdicts[best["id"]], "carried_from": best["id"]}
    return out


def _valid_counts(item: dict, counts: dict) -> dict[str, int]:
    zones = ((item.get("state") or {}).get("occupancy") or {})
    if item["kind"] != "sample" or not zones:
        raise ValueError("chỉ khung kiểm tra có vùng mới nhập được số người")
    out = {}
    for z, n in counts.items():
        if z not in zones:
            raise ValueError(f"vùng '{z}' không có trong khung này")
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= MAX_COUNT:
            raise ValueError(f"số người của vùng '{z}' phải là số nguyên 0–{MAX_COUNT}")
        out[z] = n
    return out


def _count_error(items: Sequence[dict], verdicts: dict) -> dict:
    """Per job: the system's zone count vs the reviewer's, over every zone given a number."""
    acc: dict[str, list[float]] = defaultdict(list)
    for it in items:
        given = verdicts.get(it["id"], {}).get("counts") or {}
        system = ((it.get("state") or {}).get("occupancy") or {})
        acc[it["job"]] += [system[z] - n for z, n in given.items() if z in system]
    return {j: {"zones": len(d), "mae": round(sum(abs(x) for x in d) / len(d), 2),
                "bias": round(sum(d) / len(d), 2)} for j, d in sorted(acc.items()) if d}


def report(items: Sequence[dict], verdicts: dict) -> dict:
    """Per alert type: right / wrong / unsure / pending and precision = right / (right + wrong);
    per job: alert-free samples marked missed / clear, and the zone count error (system − real)."""
    by_type: dict[str, Counter] = defaultdict(Counter)
    by_job: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        v = verdicts.get(it["id"], {}).get("verdict", "pending")
        (by_type[it["type"]] if it["kind"] == "event" else by_job[it["job"]])[v] += 1
    types = {}
    for t, c in sorted(by_type.items()):
        judged = c["right"] + c["wrong"]
        types[t] = {"total": sum(c.values()), **{k: c[k] for k in (*EVENT_VERDICTS, "pending")},
                    "precision": round(c["right"] / judged, 3) if judged else None}
    samples = {}
    for j, c in sorted(by_job.items()):
        judged = c["missed"] + c["clear"]
        samples[j] = {"total": sum(c.values()), **{k: c[k] for k in (*SAMPLE_VERDICTS, "pending")},
                      "missed_share": round(c["missed"] / judged, 3) if judged else None}
    done = sum(1 for it in items if it["id"] in verdicts)
    return {"items": len(items), "judged": done, "alerts": types, "samples": samples,
            "counts": _count_error(items, verdicts)}
