"""Per-camera map of NUISANCE motion, learned online: places that move often but never hold an object.

The OSD clock changes every second, leaves and flags sway, a screen or a water surface flickers:
Frigate reports motion there again and again, and every report costs a mosaic crop. A cell (a
`cell` × `cell` patch of the motion frame) is NUISANCE when
  - its motion frequency (EMA over ~`window_s` of observed frames) is ≥ `min_freq`, and
  - no detection (person, helmet…) touched it in the last `explained_s` seconds.
Motion boxes lying mostly (≥ `cover`) on nuisance cells are dropped before cropping. A work spot
where people really move is not nuisance: their detections keep explaining it. The periodic full
frame still looks everywhere, so a wrongly masked person is found and the cell re-explained.
Pure numpy; boxes in native frame pixels.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

Box = tuple[int, int, int, int]


class NoiseMap:
    def __init__(self, frame_shape: tuple[int, int], grid_h: int = 50, fps: float = 5.0,
                 window_s: float = 60.0, min_freq: float = 0.15, explained_s: float = 60.0,
                 cover: float = 0.6, warmup_s: float = 30.0):
        self.frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        self.gh = int(grid_h)
        self.gw = max(1, int(round(self.gh * self.frame_shape[1] / self.frame_shape[0])))
        self.sy, self.sx = self.gh / self.frame_shape[0], self.gw / self.frame_shape[1]
        self.alpha = 1.0 / max(1.0, fps * window_s)
        self.min_freq, self.explained_s, self.cover = float(min_freq), float(explained_s), float(cover)
        self.warmup = int(fps * warmup_s)
        self.freq = np.zeros((self.gh, self.gw), np.float32)
        self.explained_until = np.full((self.gh, self.gw), -np.inf, np.float32)
        self._n = 0
        self.stats = {"dropped": 0, "kept": 0}

    def _cells(self, box: Sequence[float]) -> tuple[slice, slice]:
        y1, y2 = int(box[1] * self.sy), max(int(box[1] * self.sy) + 1, int(np.ceil(box[3] * self.sy)))
        x1, x2 = int(box[0] * self.sx), max(int(box[0] * self.sx) + 1, int(np.ceil(box[2] * self.sx)))
        return slice(max(0, y1), min(self.gh, y2)), slice(max(0, x1), min(self.gw, x2))

    def observe(self, boxes: Sequence[Box]) -> None:
        """Every observed frame, with ALL its motion boxes (before filtering)."""
        hit = np.zeros_like(self.freq)
        for b in boxes:
            hit[self._cells(b)] = 1.0
        self.freq += self.alpha * (hit - self.freq)
        self._n += 1

    def explain(self, boxes: Sequence[Sequence[float]], now: float) -> None:
        """Detections (any class) make their cells real activity for explained_s."""
        for b in boxes:
            self.explained_until[self._cells(b)] = now + self.explained_s

    def nuisance(self, now: float) -> np.ndarray:
        if self._n < self.warmup:
            return np.zeros_like(self.freq, bool)
        return (self.freq >= self.min_freq) & (self.explained_until < now)

    def filter(self, boxes: Sequence[Box], now: float) -> list[Box]:
        noise = self.nuisance(now)
        if not noise.any():
            self.stats["kept"] += len(boxes)
            return list(boxes)
        out = []
        for b in boxes:
            cells = noise[self._cells(b)]
            if cells.size and cells.mean() >= self.cover:
                self.stats["dropped"] += 1
            else:
                out.append(b)
        self.stats["kept"] += len(out)
        return out

    def reset(self) -> None:
        """New view (PTZ move): what was nuisance there is not here."""
        self.freq[:] = 0
        self.explained_until[:] = -np.inf
        self._n = 0

    def nuisance_fraction(self, now: Optional[float] = None) -> float:
        return float(self.nuisance(now if now is not None else 0.0).mean())
