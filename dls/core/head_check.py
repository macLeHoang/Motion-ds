"""Helmet check: tracked persons' upper bodies, ZOOMED, packed into one model input.

Why: a worker 78 px tall at 720p has a ~20 px head. The model finds a head (helmet or bare) on
1–2 % of persons in full frames (416 or 768) and on 9–12 % in the native-scale motion mosaic,
but on 87 % once the person is ~140 px tall in the model input (ROI.mp4, 139 persons; fire_11_6
41 %: a steep top-down view, heads hidden by shoulders). An upper-body tile (a square of 0.8 ×
the person's height, from 0.15 h above the head) holds the head with enough context; at 140 px
per person a tile is ~112 px, so one 416 input checks up to 9 persons.

Pure (numpy / cv2): MotionMosaic (dls/core/mosaic.py) decides when to run it and runs the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

Box = tuple[float, float, float, float]       # x1, y1, x2, y2

TILE_SIDE = 0.8          # tile side, × the person's height
TILE_ABOVE = 0.15        # tile top above the person's top, × height
SEEN_IOU = 0.3           # a person box matching a checked one within this IoU is the same person
GAP_FILL = 114           # the letterbox grey of the detector


@dataclass(frozen=True)
class Tile:
    person: Box          # frame pixels
    region: tuple[int, int, int, int]   # frame pixels copied
    scale: float         # tile pixels per frame pixel
    x: int               # position in the canvas
    y: int
    w: int
    h: int

    def to_frame(self, box: Sequence[float]) -> Box:
        k = self.scale
        return ((box[0] - self.x) / k + self.region[0], (box[1] - self.y) / k + self.region[1],
                (box[2] - self.x) / k + self.region[0], (box[3] - self.y) / k + self.region[1])

    def holds(self, box: Sequence[float]) -> bool:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return self.x <= cx < self.x + self.w and self.y <= cy < self.y + self.h


def upper_body(person: Box, frame_shape: tuple[int, int]) -> Optional[tuple[int, int, int, int]]:
    """The tile region of a person in frame pixels (clipped), None when nothing is left."""
    fh, fw = frame_shape
    h = person[3] - person[1]
    side = TILE_SIDE * h
    cx = (person[0] + person[2]) / 2
    top = person[1] - TILE_ABOVE * h
    x1, y1 = int(round(max(0.0, cx - side / 2))), int(round(max(0.0, top)))
    x2, y2 = int(round(min(fw, cx + side / 2))), int(round(min(fh, top + side)))
    return (x1, y1, x2, y2) if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def tile_plan(person: Box, frame_shape: tuple[int, int], canvas: int,
              person_px: float) -> Optional[tuple[tuple[int, int, int, int], float, int, int]]:
    """(frame region, scale, tile w, tile h) of a person's zoomed upper body; None when empty."""
    region = upper_body(person, frame_shape)
    h = person[3] - person[1]
    if region is None or h <= 0:
        return None
    k = min(person_px / h, canvas / max(region[2] - region[0], region[3] - region[1]))
    return (region, k, max(1, int(round((region[2] - region[0]) * k))),
            max(1, int(round((region[3] - region[1]) * k))))


def draw_tile(out: np.ndarray, image: np.ndarray, person: Box, plan, x: int, y: int) -> Tile:
    """Copy the planned upper body into `out` at (x, y)."""
    region, k, tw, th = plan
    crop = image[region[1]:region[3], region[0]:region[2]]
    out[y:y + th, x:x + tw] = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    return Tile(tuple(float(v) for v in person), region, k, x, y, tw, th)


def pack_heads(image: np.ndarray, persons: Sequence[Box], canvas: int, person_px: float,
               gap: int = 2) -> tuple[Optional[np.ndarray], list[Tile]]:
    """Persons in order → one canvas × canvas image of zoomed upper-body tiles (shelf packing, left
    to right, top to bottom) and their placements. Persons that do not fit are left out."""
    fh, fw = image.shape[:2]
    out = np.full((canvas, canvas) + image.shape[2:], GAP_FILL, image.dtype)
    tiles: list[Tile] = []
    x = y = row_h = 0
    for p in persons:
        plan = tile_plan(p, (fh, fw), canvas, person_px)
        if plan is None:
            continue
        tw, th = plan[2], plan[3]
        if x + tw > canvas:
            x, y, row_h = 0, y + row_h + gap, 0
        if y + th > canvas:
            break
        tiles.append(draw_tile(out, image, p, plan, x, y))
        x += tw + gap
        row_h = max(row_h, th)
    return (out, tiles) if tiles else (None, [])


def map_heads(detections: Iterable[dict[str, Any]], tiles: Sequence[Tile], head_labels: set[str]) -> list[dict[str, Any]]:
    """Head detections on the canvas → frame detections; anything else, and heads whose centre is
    in no tile, are dropped."""
    out = []
    for d in detections:
        if d["label"] not in head_labels:
            continue
        tile = next((t for t in tiles if t.holds(d["box"])), None)
        if tile is not None:
            out.append({**d, "box": tile.to_frame(d["box"])})
    return out


class HeadSchedule:
    """Which persons to check next: the ones checked longest ago first (never → first), then the
    largest. A person is recognised across checks by box IoU (ids live downstream, in gvatrack)."""

    def __init__(self, every_s: float, forget_s: float = 30.0):
        self.every_s = float(every_s)
        self.forget_s = float(forget_s)
        self._last_run: Optional[float] = None
        self._checked: list[tuple[Box, float]] = []

    def due(self, now: float) -> bool:
        return self.every_s > 0 and (self._last_run is None or now - self._last_run >= self.every_s - 1e-6)

    def order(self, persons: Sequence[Box], now: float) -> list[Box]:
        self._checked = [(b, t) for b, t in self._checked if now - t <= self.forget_s]

        def last(p: Box) -> float:
            return max((t for b, t in self._checked if _iou(b, p) >= SEEN_IOU), default=float("-inf"))
        return sorted(persons, key=lambda p: (last(p), -(p[3] - p[1])))

    def done(self, checked: Sequence[Box], now: float) -> None:
        self._last_run = now
        self._checked = [(b, t) for b, t in self._checked if all(_iou(b, c) < SEEN_IOU for c in checked)]
        self._checked += [(tuple(c), now) for c in checked]


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
