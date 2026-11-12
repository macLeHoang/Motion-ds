"""Detector-side false-positive filter, run once per inference (dls/core/mosaic.py).

A. TrackConfirmer: detections are linked across inferences (same label, centres close relative
   to the box size: a light IoU-free tracker that also follows walkers between inferences 0.5 s
   apart) and a track is released only after ≥ `hits` of the last `window` inferences. A false
   box that flickers on smoke or a steel frame for one or two inferences never reaches the
   tracker / events; a confirmed track stays released through short misses.
B. PerspectiveSize: per camera and per size group (person, helmet…) the expected box HEIGHT as
   a function of the feet y (far = high in the frame = small), learned only from confirmed tracks:
   frame split into horizontal bands → median height per band → weighted line fit. Once learned,
   a box far larger or smaller than expected (a pole taken for a person) is dropped — judged only
   inside the learned y range ± one band; elsewhere boxes are accepted so the range can grow.
   At most SAMPLES_PER_TRACK_BAND samples per track and band (a standing person is one sample
   source, not a hundred). Boxes touching the top/bottom edge are cut, so they are neither judged
   nor learned. The expected height is clamped to the learned heights. The mosaic also
   grows each motion crop to ≥ `min_crop_context` × the expected person height (context).
   Forgotten on a scene change (PTZ move): the caller calls reset_scene().
Pure Python; one instance per camera.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

Det = dict[str, Any]          # {"label", "score", "box": (x1, y1, x2, y2), ...}

PARAM_KEYS = ("confirm_hits", "confirm_window", "confirm_classes", "size_groups", "size_max_ratio",
              "size_min_samples")
MATCH_DIST = 1.0              # centre distance ≤ this × the track box's larger side
MATCH_SCALE = 2.0             # and heights within this factor
MIN_EXPECTED_PX = 4.0
EDGE_PX = 2                   # a box this close to the top/bottom edge is cut: its height is not real
SAMPLES_PER_TRACK_BAND = 3    # a person standing still must not fill a band with copies of itself


# ---------------------------------------------------------------- A: confirmation
@dataclass
class _Track:
    label: str
    box: tuple
    seen: deque = field(default_factory=deque)
    confirmed: bool = False
    sampled: dict = field(default_factory=dict)        # size band → samples given to PerspectiveSize


def _centre(b: Sequence[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _match_cost(track_box: Sequence[float], box: Sequence[float]) -> Optional[float]:
    tw, th = track_box[2] - track_box[0], track_box[3] - track_box[1]
    h = box[3] - box[1]
    if th <= 0 or h <= 0 or not 1 / MATCH_SCALE <= h / th <= MATCH_SCALE:
        return None
    (ax, ay), (bx, by) = _centre(track_box), _centre(box)
    dist = math.hypot(ax - bx, ay - by) / max(tw, th, 1.0)
    return dist if dist <= MATCH_DIST else None


class TrackConfirmer:
    def __init__(self, hits: int = 3, window: int = 5, classes: Optional[Iterable[str]] = None):
        if not 1 <= hits <= window:
            raise ValueError("confirm_hits phải trong [1, confirm_window]")
        self.hits, self.window = int(hits), int(window)
        self.classes = set(classes) if classes else None
        self._tracks: list[_Track] = []
        self._n = -1

    def update(self, dets: Sequence[Det]) -> list[Det]:
        """One inference's detections → those of confirmed tracks (+ classes not confirmed)."""
        return [d for d, _ in self.update_tracks(dets)]

    def update_tracks(self, dets: Sequence[Det]) -> list[tuple[Det, Optional[_Track]]]:
        """As update(), with each released detection's track (None for unconfirmed classes)."""
        self._n += 1
        n = self._n
        self._tracks = [t for t in self._tracks if n - t.seen[-1] < self.window]
        out, todo = [], []
        for d in dets:
            if self.classes is None or d["label"] in self.classes:
                todo.append(d)
            else:
                out.append((d, None))
        pairs = sorted((c, ti, di) for di, d in enumerate(todo) for ti, t in enumerate(self._tracks)
                       if t.label == d["label"] and (c := _match_cost(t.box, d["box"])) is not None)
        used_t, owner = set(), {}
        for _, ti, di in pairs:
            if ti not in used_t and di not in owner:
                used_t.add(ti)
                owner[di] = self._tracks[ti]
        for di, d in enumerate(todo):
            t = owner.get(di)
            if t is None:
                t = _Track(d["label"], d["box"])
                self._tracks.append(t)
            t.box = d["box"]
            t.seen.append(n)
            while t.seen and t.seen[0] <= n - self.window:
                t.seen.popleft()
            t.confirmed = t.confirmed or len(t.seen) >= self.hits
            if t.confirmed:
                out.append((d, t))
        return out


# ---------------------------------------------------------------- B: perspective size
class PerspectiveSize:
    def __init__(self, frame_height: int, bands: int = 12, min_samples: int = 60, min_bands: int = 3,
                 band_min: int = 3, per_band: int = 100, refit_every: int = 20):
        self.frame_height = int(frame_height)
        self.n_bands = int(bands)
        self.min_samples, self.min_bands, self.band_min = int(min_samples), int(min_bands), int(band_min)
        self.per_band, self.refit_every = int(per_band), int(refit_every)
        self.reset()

    def reset(self) -> None:
        self._bands: list[deque] = [deque(maxlen=self.per_band) for _ in range(self.n_bands)]
        self._total = 0
        self._fit: Optional[tuple[float, float, float, float, float, float]] = None
        # slope, intercept, y_lo, y_hi (learned range), h_lo, h_hi (clamp for the fitted line)
        self._dirty = 0

    def band_of(self, y: float) -> int:
        return min(self.n_bands - 1, max(0, int(y * self.n_bands / max(self.frame_height, 1))))

    def add(self, box: Sequence[float]) -> None:
        h = box[3] - box[1]
        if h <= 0:
            return
        self._bands[self.band_of(box[3])].append(float(h))
        self._total += 1
        self._dirty += 1
        if self._fit is None or self._dirty >= self.refit_every:
            self._refit()

    def _refit(self) -> None:
        self._dirty = 0
        used = [(i, b) for i, b in enumerate(self._bands) if len(b) >= self.band_min]
        if self._total < self.min_samples or len(used) < self.min_bands:
            self._fit = None
            return
        step = self.frame_height / self.n_bands
        ys = np.array([(i + 0.5) * step for i, _ in used])
        hs = np.array([float(np.median(b)) for _, b in used])
        w = np.sqrt([len(b) for _, b in used])
        slope, icpt = np.polyfit(ys, hs, 1, w=w)
        self._fit = (float(slope), float(icpt), used[0][0] * step, (used[-1][0] + 1) * step,
                     0.5 * float(hs.min()), 2.0 * float(hs.max()))

    @property
    def ready(self) -> bool:
        return self._fit is not None

    def expected(self, y: float) -> Optional[float]:
        if self._fit is None:
            return None
        slope, icpt, lo, hi, h_lo, h_hi = self._fit
        return max(MIN_EXPECTED_PX, h_lo, min(h_hi, slope * min(max(y, lo), hi) + icpt))

    def plausible(self, box: Sequence[float], max_ratio: float) -> bool:
        """Judged only inside the learned feet-y range (± one band): elsewhere nobody has been seen
        yet, the box is accepted, and once confirmed it extends the range."""
        exp = self.expected(box[3])
        if exp is None:
            return True
        _, _, lo, hi, _, _ = self._fit
        margin = self.frame_height / self.n_bands
        if not lo - margin <= box[3] < hi + margin:
            return True
        ratio = (box[3] - box[1]) / exp
        return 1.0 / max_ratio <= ratio <= max_ratio


# ---------------------------------------------------------------- A + B
class DetectionFilter:
    def __init__(self, frame_height: int, confirm_hits: int = 3, confirm_window: int = 5,
                 confirm_classes: Optional[Iterable[str]] = None,
                 size_groups: Optional[Mapping[str, Sequence[str]]] = None,
                 size_max_ratio: float = 2.5, size_min_samples: int = 60):
        self.confirmer = TrackConfirmer(confirm_hits, confirm_window, confirm_classes)
        groups = dict(size_groups) if size_groups is not None else {"person": ["person"]}
        self._group_of = {label: g for g, labels in groups.items() for label in labels}
        self._sizes = {g: PerspectiveSize(frame_height, min_samples=size_min_samples) for g in groups}
        self.size_max_ratio = float(size_max_ratio)
        self.frame_height = int(frame_height)
        self.stats = {"size_rejected": 0, "unconfirmed": 0, "released": 0}

    @classmethod
    def from_params(cls, frame_height: int, params: Mapping[str, Any]) -> "DetectionFilter":
        unknown = [k for k in params if k not in PARAM_KEYS]
        if unknown:
            raise ValueError(f"bộ lọc phát hiện: khóa không hỗ trợ {unknown} (chỉ {list(PARAM_KEYS)})")
        return cls(frame_height, **dict(params))

    def update(self, dets: Sequence[Det]) -> tuple[list[Det], list[Det]]:
        """One inference → (released detections, candidates worth re-inspecting = size-plausible)."""
        keep = []
        for d in dets:
            size = self._sized(d)
            if size is not None and not size.plausible(d["box"], self.size_max_ratio):
                self.stats["size_rejected"] += 1
            else:
                keep.append(d)
        released = self.confirmer.update_tracks(keep)
        out = [d for d, _ in released]
        for d, track in released:
            size = self._sized(d)
            if size is None or track is None:
                continue
            band = size.band_of(d["box"][3])
            if track.sampled.get(band, 0) < SAMPLES_PER_TRACK_BAND:
                track.sampled[band] = track.sampled.get(band, 0) + 1
                size.add(d["box"])
        self.stats["released"] += len(out)
        self.stats["unconfirmed"] += len(keep) - len(out)
        return out, keep

    def _sized(self, d: Det) -> Optional[PerspectiveSize]:
        """Size model judging this detection; None for other labels and for boxes cut by the
        top/bottom frame edge (someone walking in shows only part of their height)."""
        size = self._sizes.get(self._group_of.get(d["label"]))
        if size is None or d["box"][1] <= EDGE_PX or d["box"][3] >= self.frame_height - EDGE_PX:
            return None
        return size

    def size_ready(self, group: str) -> bool:
        return group in self._sizes and self._sizes[group].ready

    def expected_height(self, group: str, y: float) -> Optional[float]:
        size = self._sizes.get(group)
        return size.expected(y) if size else None

    def reset_scene(self) -> None:
        for size in self._sizes.values():
            size.reset()
