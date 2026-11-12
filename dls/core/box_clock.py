"""Where and when was this object last measured? Per (camera, object id): the last detector box.

In motion mode a camera gets an inference every ~0.5–2 s on a busy CPU; in between the imageless
tracker (gvatrack short-term-imageless) hands on a predicted box with confidence 1.0. An alert
raised on such a frame should show the box the detector measured and the time it did (`box_ts`),
or a walking worker stands beside his box in the alert picture (event review, ROI.mp4: "box lệch").
A detector box keeps its real score (< HELD_CONF), which is how the two are told apart; the
alert's confidence is that score too (held boxes all read 1.0).
"""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Iterable

from dls.core.metadata import Detection

HELD_CONF = 0.999       # gvatrack gives the boxes it predicts between detections confidence 1.0
MAX_OBJECTS = 4096      # pruned to objects measured within TTL_S when exceeded
TTL_S = 120.0

XYWH = tuple[float, float, float, float]


class BoxClock:
    def __init__(self) -> None:
        self._measured: dict[tuple[str, int], tuple[XYWH, float, float]] = {}   # box, time, score
        self._lock = threading.Lock()

    def update(self, camera_id: str, detections: Iterable[Detection], now: float) -> None:
        with self._lock:
            for d in detections:
                if d.object_id is not None and d.confidence < HELD_CONF:
                    self._measured[(camera_id, d.object_id)] = ((d.x, d.y, d.w, d.h), now, d.confidence)
            if len(self._measured) > MAX_OBJECTS:
                for k in [k for k, (_, t, _) in self._measured.items() if now - t > TTL_S]:
                    del self._measured[k]

    def measured(self, camera_id: str, det: Detection, now: float) -> dict:
        """{"bbox": [x, y, w, h], "box_ts": t} of det's last detector box (det's own box at now
        when it was never measured, e.g. untracked)."""
        box, ts, _ = self._last(camera_id, det, now)
        return {"bbox": [round(v, 1) for v in box], "box_ts": round(ts, 3)}

    def as_measured(self, camera_id: str, det: Detection) -> Detection:
        """A held (predicted) box moved back to where the detector last saw the object; its
        confidence stays 1.0, so it still reads as held downstream. At 5 fps with an inference
        every ~0.6 s the tracker's motion prediction lands slightly farther from the next
        measurement than the last measured box does (ROI.mp4: IoU 0.68 vs 0.73)."""
        if det.object_id is None or det.confidence < HELD_CONF:
            return det
        with self._lock:
            hit = self._measured.get((camera_id, det.object_id))
        if hit is None:
            return det
        x, y, w, h = hit[0]
        return replace(det, x=x, y=y, w=w, h=h)

    def score(self, camera_id: str, det: Detection) -> float:
        """The detector's score for det (its own confidence when never measured)."""
        return self._last(camera_id, det, 0.0)[2]

    def _last(self, camera_id: str, det: Detection, now: float) -> tuple[XYWH, float, float]:
        with self._lock:
            hit = self._measured.get((camera_id, det.object_id)) if det.object_id is not None else None
        return hit if hit is not None else ((det.x, det.y, det.w, det.h), now, det.confidence)

    def drop_camera(self, camera_id: str) -> None:
        with self._lock:
            for k in [k for k in self._measured if k[0] == camera_id]:
                del self._measured[k]
