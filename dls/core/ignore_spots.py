"""Spots a person marked as false alerts again and again (event review "wrong"), per camera.

A detection of the spot's kind is dropped before any task sees it when it is the same false
thing: its centre is within max(SPOT_TOL × the spot's size, SPOT_TOL_PX) of the spot's centre on
each axis, AND its height is in the spot's height band. On ROI.mp4 a stack of insulators reads as a
22–36 px person over and over (the box wanders ~15 px), while real workers on the same spot were
alerted at 47–90 px: the band learned from the wrong boxes is cut below the nearest right one, so
they still alert. A steel post reads as a ~105 px person with its top at y ≈ 245, while workers of
the same height stand just below it (feet ≥ 20 px lower): the tolerance must stay tight.

Tolerances were chosen on the three review rounds of 2026-10-07, learning on two and testing on
the third: 15 of 47 wrong person alerts dropped, 0 right ones (0.5 × size, the first rule, dropped
2 right alerts of round 2 when learned from rounds 1 and 3).

Camera params (same space as `roi_filtering`: config_width × config_height, else the frame):
    "ignore_boxes": [{"box": [x, y, w, h], "kind": "person", "heights": [lo, hi]}, ...]
kind: person | fire | smoke; heights optional (default: box h / SIZE_RATIO … box h × SIZE_RATIO).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

from dls.core.metadata import Detection

SPOT_TOL = 0.25         # centre offset allowed per axis, as a share of the spot's size …
SPOT_TOL_PX = 15.0      # … or this many pixels at 720 lines, whichever is larger
GROUP_RATIO = 1.3       # wrong boxes grouped into one spot differ in height by at most this
SIZE_RATIO = 1.6        # default height band: [h / ratio, h × ratio] of the spot box
HEIGHT_MARGIN = 1.15    # learned band: wrong heights ± 15 %, kept this far from any right height
MIN_REPEATS = 3         # wrong verdicts at one spot before it is learned (a late box beside a
                        # walking worker reads "wrong" too; ROI.mp4 had two such at one spot)
KINDS = ("person", "fire", "smoke")
KIND_OF_EVENT = {"zone_enter": "person", "helmet_violation": "person", "fire_detected": "fire",
                 "smoke_detected": "smoke"}

Box = tuple[float, float, float, float]        # x1, y1, x2, y2


@dataclass(frozen=True)
class Spot:
    box: Box
    kind: str
    h_min: float
    h_max: float
    tol_px: float = SPOT_TOL_PX


def near(box: Box, spot: Box, tol_px: float = SPOT_TOL_PX) -> bool:
    """box's centre within max(SPOT_TOL × spot size, tol_px) of the spot's centre on each axis."""
    w, h = spot[2] - spot[0], spot[3] - spot[1]
    dx = abs((box[0] + box[2]) / 2 - (spot[0] + spot[2]) / 2)
    dy = abs((box[1] + box[3]) / 2 - (spot[1] + spot[3]) / 2)
    return dx <= max(SPOT_TOL * w, tol_px) and dy <= max(SPOT_TOL * h, tol_px)


def same_spot(box: Box, spot: Spot) -> bool:
    """box looks like the false thing at spot: centre near it and height in its band."""
    return near(box, spot.box, spot.tol_px) and spot.h_min <= box[3] - box[1] <= spot.h_max


def _xyxy(xywh: Sequence[float]) -> Box:
    x, y, w, h = (float(v) for v in xywh)
    return (x, y, x + w, y + h)


def spots_from_params(params: Mapping[str, Any], frame_w: int, frame_h: int) -> list[Spot]:
    """Parse and scale `ignore_boxes` to the frame. Raises ValueError on a malformed entry."""
    raw = params.get("ignore_boxes") or []
    if not isinstance(raw, list):
        raise ValueError("params.ignore_boxes phải là list")
    sx = frame_w / float(params.get("config_width", frame_w) or frame_w)
    sy = frame_h / float(params.get("config_height", frame_h) or frame_h)
    spots = []
    for entry in raw:
        entry = entry if isinstance(entry, Mapping) else {}
        box, kind = entry.get("box"), entry.get("kind", "person")
        if not (isinstance(box, (list, tuple)) and len(box) == 4) or kind not in KINDS:
            raise ValueError(f"ignore_boxes: mục {entry!r} cần box [x, y, w, h] và kind trong {list(KINDS)}")
        x, y, w, h = (float(v) for v in box)
        if w <= 0 or h <= 0:
            raise ValueError(f"ignore_boxes: box {box} phải có w, h > 0")
        heights = entry.get("heights") or [h / SIZE_RATIO, h * SIZE_RATIO]
        if not (isinstance(heights, (list, tuple)) and len(heights) == 2 and 0 < float(heights[0]) < float(heights[1])):
            raise ValueError(f"ignore_boxes: heights {heights!r} phải là [thấp, cao] với 0 < thấp < cao")
        spots.append(Spot((x * sx, y * sy, (x + w) * sx, (y + h) * sy), kind,
                          float(heights[0]) * sy, float(heights[1]) * sy, SPOT_TOL_PX * frame_h / 720.0))
    return spots


def drop_spots(detections: Iterable[Detection], spots: Sequence[Spot],
               kind_of: Mapping[str, str]) -> tuple[list[Detection], int]:
    """Detections not matching a spot of their kind, and how many were dropped.
    kind_of maps a detection label to a spot kind (labels not in it are never dropped)."""
    kept, dropped = [], 0
    for d in detections:
        kind = kind_of.get(d.label)
        box = (d.x, d.y, d.x + d.w, d.y + d.h)
        if kind and any(s.kind == kind and same_spot(box, s) for s in spots):
            dropped += 1
        else:
            kept.append(d)
    return kept, dropped


def learn_spots(judged: Iterable[tuple[str, Sequence[float], str]], min_repeats: int = MIN_REPEATS) -> list[dict]:
    """judged: (event type, bbox [x, y, w, h], verdict) per reviewed alert.

    Wrong boxes are grouped greedily (centre near the group's first box, heights within
    GROUP_RATIO). A group's spot is its
    median box with a height band: the wrong heights ± HEIGHT_MARGIN, cut back so that no box
    judged right near it falls inside (a real person was alerted on there at that height). The
    spot is kept when ≥ min_repeats wrong boxes are still inside the band."""
    wrong: dict[str, list[Box]] = {}
    right: dict[str, list[Box]] = {}
    for etype, bbox, verdict in judged:
        kind = KIND_OF_EVENT.get(etype)
        if kind is None or not bbox:
            continue
        if verdict == "wrong":
            wrong.setdefault(kind, []).append(_xyxy(bbox))
        elif verdict == "right":
            right.setdefault(kind, []).append(_xyxy(bbox))
    out = []
    for kind, boxes in wrong.items():
        groups: list[list[Box]] = []
        for b in boxes:
            home: Optional[list[Box]] = next((g for g in groups if near(b, g[0]) and _similar(b, g[0])), None)
            if home is None:
                groups.append([b])
            else:
                home.append(b)
        for g in groups:
            spot = _spot_of(g, right.get(kind, []), min_repeats)
            if spot is not None:
                out.append({**spot, "kind": kind})
    return out


def _similar(a: Box, b: Box) -> bool:
    ha, hb = a[3] - a[1], b[3] - b[1]
    return max(ha, hb) <= GROUP_RATIO * min(ha, hb)


def _spot_of(group: list[Box], rights: list[Box], min_repeats: int) -> Optional[dict]:
    if len(group) < min_repeats:
        return None
    box = tuple(sorted(c)[len(c) // 2] for c in zip(*group))
    rights = [r for r in rights if near(r, box)]
    heights = sorted(b[3] - b[1] for b in group)
    mid = heights[len(heights) // 2]
    lo, hi = heights[0] / HEIGHT_MARGIN, heights[-1] * HEIGHT_MARGIN
    for r in rights:
        rh = r[3] - r[1]
        if rh >= mid:
            hi = min(hi, rh / HEIGHT_MARGIN)
        else:
            lo = max(lo, rh * HEIGHT_MARGIN)
    inside = [h for h in heights if lo <= h <= hi]
    if lo >= hi or len(inside) < min_repeats:
        return None
    x1, y1, x2, y2 = box
    return {"box": [round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1)],
            "heights": [round(lo, 1), round(hi, 1)], "repeats": len(inside)}
