"""Crop geometry for motion-mosaic inference (dls/core/mosaic.py).

Motion boxes (+ boxes of objects detected a moment ago) are merged, padded a little for context and
cropped at NATIVE resolution; everything else in the frame (meaningless background) is dropped. The
crops are shelf-packed into ONE near-square mosaic image — the detector input is square, so a square
mosaic needs the least letterbox resize. The detector runs once on the mosaic and every box is
mapped back through the crop holding its centre (a box whose centre falls in a gap is dropped).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import cv2
import numpy as np

Box = tuple[int, int, int, int]
GAP_FILL = 114                 # YOLO letterbox grey: neutral to the detector


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Inclusive-pixel IoU (Frigate's +1 arithmetic)."""
    if a[0] > b[2] or b[0] > a[2] or a[1] > b[3] or b[1] > a[3]:
        return 0.0
    inter = (max(0, min(a[2], b[2]) - max(a[0], b[0]) + 1)
             * max(0, min(a[3], b[3]) - max(a[1], b[1]) + 1))
    union = (a[2] - a[0] + 1) * (a[3] - a[1] + 1) + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter
    return inter / float(union) if inter else 0.0


def expand_box(box: Sequence[float], frac: float, frame_shape: tuple[int, int]) -> Box:
    """Grow each side by frac × width/height, clamped to the frame."""
    dx, dy = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return (int(max(0, box[0] - dx)), int(max(0, box[1] - dy)),
            int(min(frame_shape[1], box[2] + dx)), int(min(frame_shape[0], box[3] + dy)))


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def merge_overlapping(boxes: Sequence[Sequence[float]], max_growth: Optional[float] = None) -> list[Box]:
    """Union boxes that overlap until none do (a motion blob of the legs + the kept whole-person box).
    `max_growth`: merge two only when their bounding rectangle is at most this many times the area
    they really cover — chained unions otherwise grow into crops that are mostly background (two
    workers far apart along a pole became one 591×720 crop). Unmerged crops may then overlap; the
    caller removes duplicate detections after mapping back (dedupe_detections)."""
    out = [tuple(int(v) for v in b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if not (a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]):
                    continue
                union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                if max_growth is not None:
                    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
                    if _area(union) > max_growth * (_area(a) + _area(b) - inter):
                        continue
                out[i] = union
                del out[j]
                merged = True
                break
            if merged:
                break
    return out


def dedupe_detections(dets: Sequence[dict[str, Any]], iou_min: float = 0.5) -> list[dict[str, Any]]:
    """Same label, overlapping (IoU ≥ iou_min or one inside the other): keep the best score —
    an object seen in two overlapping crops is reported once."""
    out: list[dict[str, Any]] = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        dup = False
        for k in out:
            if k["label"] != d["label"]:
                continue
            inter = _area((max(d["box"][0], k["box"][0]), max(d["box"][1], k["box"][1]),
                           min(d["box"][2], k["box"][2]), min(d["box"][3], k["box"][3])))
            if iou(d["box"], k["box"]) >= iou_min or inter >= 0.8 * min(_area(d["box"]), _area(k["box"])):
                dup = True
                break
        if not dup:
            out.append(d)
    return out


@dataclass(frozen=True)
class Placement:
    region: Box          # native frame pixels that were copied
    x: int               # position in the mosaic
    y: int
    w: int
    h: int
    scale: float = 1.0   # mosaic pixels per frame pixel (pack_scaled)

    def contains(self, box: Sequence[float]) -> bool:
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        return self.x <= cx < self.x + self.w and self.y <= cy < self.y + self.h

    def to_frame(self, box: Sequence[float]) -> Box:
        rx1, ry1, rx2, ry2 = self.region
        k = self.scale
        fx = lambda v: (v - self.x) / k + rx1      # noqa: E731
        fy = lambda v: (v - self.y) / k + ry1      # noqa: E731
        clamp = lambda v, lo, hi: int(round(max(lo, min(hi, v))))      # noqa: E731
        return (clamp(fx(box[0]), rx1, rx2), clamp(fy(box[1]), ry1, ry2),
                clamp(fx(box[2]), rx1, rx2), clamp(fy(box[3]), ry1, ry2))

    def point_to_frame(self, x: float, y: float) -> tuple[float, float]:
        """A keypoint (face landmark, plate corner): same mapping as boxes, not clamped."""
        return ((x - self.x) / self.scale + self.region[0], (y - self.y) / self.scale + self.region[1])


def _shelf(sizes: Sequence[tuple[int, int]], width: int, gap: int) -> tuple[list[tuple[int, int]], int, int]:
    """Shelf layout (tallest first) inside `width` → (positions, used width, height)."""
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos: list[tuple[int, int]] = [(0, 0)] * len(sizes)
    x = y = shelf_h = used_w = 0
    for i in order:
        w, h = sizes[i]
        if x > 0 and x + w > width:
            y, x, shelf_h = y + shelf_h + gap, 0, 0
        pos[i] = (x, y)
        used_w = max(used_w, x + w)
        x += w + gap
        shelf_h = max(shelf_h, h)
    return pos, used_w, y + shelf_h


def layout(sizes: Sequence[tuple[int, int]], gap: int = 4) -> tuple[list[tuple[int, int]], int, int]:
    """The most square shelf layout of (w, h) sizes → (positions, width, height)."""
    side = math.sqrt(sum((w + gap) * (h + gap) for w, h in sizes))
    widest = max(w for w, _ in sizes)
    best = None
    for f in (0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0):
        pos, used_w, height = _shelf(sizes, max(widest, int(side * f)), gap)
        key = (max(used_w, height), used_w * height)
        if best is None or key < best[0]:
            best = (key, pos, used_w, height)
    return best[1], best[2], best[3]


def cap_zoom(image: np.ndarray, canvas_size: int, max_zoom: float) -> np.ndarray:
    """Pad a small image so a letterbox to `canvas_size` zooms it in at most `max_zoom`."""
    h, w = image.shape[:2]
    target = int(np.ceil(canvas_size / max_zoom)) if max_zoom > 0 else 0
    if max(h, w) >= target:
        return image
    out = np.full((target if h >= w else h, target if w > h else w) + image.shape[2:], GAP_FILL, image.dtype)
    out[:h, :w] = image
    return out


def pack_mosaic(frame: np.ndarray, boxes: Sequence[Box], gap: int = 4) -> tuple[Optional[np.ndarray], list[Placement]]:
    """Crops of `frame` at `boxes` (native scale) → one near-square mosaic + placements."""
    crops = [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in boxes if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
    if not crops:
        return None, []
    pos, used_w, height = layout([(b[2] - b[0], b[3] - b[1]) for b in crops], gap)
    mosaic = np.full((height, used_w) + frame.shape[2:], GAP_FILL, frame.dtype)
    places = []
    for (x, y), (x1, y1, x2, y2) in zip(pos, crops):
        mosaic[y:y + y2 - y1, x:x + x2 - x1] = frame[y1:y2, x1:x2]
        places.append(Placement((x1, y1, x2, y2), x, y, x2 - x1, y2 - y1))
    return mosaic, places


class MaxRects:
    """MaxRects bin packing (best short side fit): rectangles placed one by one into a fixed bin;
    every maximal free rectangle is tracked, so a small item still finds the gap a shelf layout
    leaves beside a tall one. Each item is reserved with `gap` px to its right and bottom."""

    def __init__(self, width: int, height: int, gap: int = 0):
        self.gap = int(gap)
        self.free: list[tuple[int, int, int, int]] = [(0, 0, int(width) + self.gap, int(height) + self.gap)]

    def insert(self, w: int, h: int) -> Optional[tuple[int, int]]:
        w, h = int(w) + self.gap, int(h) + self.gap
        best = None
        for fx, fy, fw, fh in self.free:
            if w <= fw and h <= fh:
                key = (min(fw - w, fh - h), max(fw - w, fh - h))
                if best is None or key < best[0]:
                    best = (key, fx, fy)
        if best is None:
            return None
        _, x, y = best
        self._split((x, y, w, h))
        return x, y

    def _split(self, used: tuple[int, int, int, int]) -> None:
        ux, uy, uw, uh = used
        out = []
        for fx, fy, fw, fh in self.free:
            if ux >= fx + fw or ux + uw <= fx or uy >= fy + fh or uy + uh <= fy:
                out.append((fx, fy, fw, fh))
                continue
            if ux > fx:
                out.append((fx, fy, ux - fx, fh))
            if ux + uw < fx + fw:
                out.append((ux + uw, fy, fx + fw - ux - uw, fh))
            if uy > fy:
                out.append((fx, fy, fw, uy - fy))
            if uy + uh < fy + fh:
                out.append((fx, uy + uh, fw, fy + fh - uy - uh))
        self.free = [r for i, r in enumerate(out)
                     if not any(j != i and _inside(r, o) and (r != o or j < i) for j, o in enumerate(out))]


def _inside(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2] and a[1] + a[3] <= b[1] + b[3]


def _try_pack(sizes: Sequence[tuple[int, int]], canvas: int, gap: int) -> Optional[tuple[list[tuple[int, int]], MaxRects]]:
    bin_ = MaxRects(canvas, canvas, gap)
    pos: list[tuple[int, int]] = [(0, 0)] * len(sizes)
    for i in sorted(range(len(sizes)), key=lambda i: -max(sizes[i])):
        p = bin_.insert(*sizes[i])
        if p is None:
            return None
        pos[i] = p
    return pos, bin_


def pack_scaled(frame: np.ndarray, boxes: Sequence[Box], scales: Sequence[float], canvas: int,
                gap: int = 4, search: int = 10) -> tuple[Optional[np.ndarray], list[Placement], Optional[MaxRects], float]:
    """Crops of `frame` at `boxes`, each resized by its own scale × one common factor g ≤ 1 (the
    largest that fits, binary search), MaxRects-packed into a canvas × canvas image. Returns the
    image, placements (with their scale), the bin (its free space can take more items) and g."""
    items = [((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(s)) for b, s in zip(boxes, scales)
             if b[2] - b[0] >= 2 and b[3] - b[1] >= 2 and s > 0]
    if not items:
        return None, [], None, 0.0

    def sizes(g: float) -> list[tuple[int, int]]:
        return [(max(1, int(round((b[2] - b[0]) * s * g))), max(1, int(round((b[3] - b[1]) * s * g))))
                for b, s in items]

    g, packed = 1.0, _try_pack(sizes(1.0), canvas, gap)
    if packed is None:
        lo, hi = 0.0, 1.0
        for _ in range(search):
            mid = (lo + hi) / 2
            got = _try_pack(sizes(mid), canvas, gap)
            if got is not None:
                lo, packed = mid, got
            else:
                hi = mid
        g = lo
        if packed is None:
            return None, [], None, 0.0
    pos, bin_ = packed
    out = np.full((canvas, canvas) + frame.shape[2:], GAP_FILL, frame.dtype)
    places = []
    for ((x1, y1, x2, y2), s), (x, y), (w, h) in zip(items, pos, sizes(g)):
        out[y:y + h, x:x + w] = cv2.resize(frame[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)
        places.append(Placement((x1, y1, x2, y2), x, y, w, h, w / float(x2 - x1)))
    return out, places, bin_, g


def map_back(detections: Sequence[dict[str, Any]], placements: Sequence[Placement]) -> list[dict[str, Any]]:
    """Mosaic detections → frame detections; a box whose centre is in no crop is dropped."""
    out = []
    for d in detections:
        for p in placements:
            if p.contains(d["box"]):
                mapped = {**d, "box": p.to_frame(d["box"])}
                if d.get("keypoints") is not None:
                    mapped["keypoints"] = [p.point_to_frame(float(x), float(y)) for x, y in d["keypoints"]]
                out.append(mapped)
                break
    return out
