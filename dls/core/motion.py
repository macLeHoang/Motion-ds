"""Motion detector for the motion → mosaic → inference tier (dls/core/regions.py, pydetect `yolo_motion`).

Ported from Frigate NVR's ImprovedMotionDetector (MIT License, Copyright (c) 2020 Blake Blackshear,
https://github.com/blakeblackshear/frigate, frigate/motion/improved_motion.py @ f3a352ef3) via
dlstreamer-edge-reference/motion/frigate_motion.py, where it was benchmarked against 2-frame diff and
MOG2 on real day/night footage: stock settings (100 px motion frame) kept 92 % day / 100 % night
coverage at ~1.5 ms/frame. The percentile contrast stretch amplifies faint far-field motion before a
fixed threshold; motion that persists ≥ 10 frames is folded into the background (foliage, OSD clock).

`stale_frames` (not upstream): a pixel that differs from the background but has NOT changed between
consecutive frames for that many frames is a stale background (lights switched on, a door left open,
IR→colour switch, a PTZ preset, a parked car), not motion: the background takes the frame there at
once. Upstream folds it in at frame_alpha 0.01, so for ~20 s every such area was a motion box (a
mosaic crop, or a crowded full-frame fallback). Real motion changes between frames and is kept.
`protect(boxes)` (recent detections) keeps the upstream slow absorption there, so a worker standing at
a bench is not relearned; no whole-frame rebase during the startup contrast ease-in.

Caller contract (same as Frigate): ignore the returned boxes while `calibrating` is True (startup,
lightning, IR switch) — the mosaic detector falls back to full-frame inference then.
Differences from upstream: no PTZ, no debug dumps, mask from normalised polygons, cv2 blur fallback
when scipy is missing (1 px border difference).
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # noqa: BLE001
    gaussian_filter = None

Box = tuple[int, int, int, int]


class MotionDetector:
    """Running-average motion detector with conditional background absorption. One per camera."""

    SCENE_CHANGE = 0.15        # share of the frame relearned at once → rebase background + contrast

    def __init__(self, frame_shape: tuple[int, int], frame_height: int = 100, threshold: int = 30,
                 contour_area: int = 10, frame_alpha: float = 0.01, lightning_threshold: float = 0.8,
                 improve_contrast: bool = True, mask_polygons: Optional[Sequence] = None,
                 contrast_frame_history: int = 50, stale_frames: int = 0):
        self.frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        self.h = int(frame_height)
        self.w = self.h * self.frame_shape[1] // self.frame_shape[0]
        self.resize_factor = self.frame_shape[0] / self.h
        self.threshold = int(threshold)
        # kept at 10 px also on a 200 px motion frame: scaling it by (h/100)² (40 px) drops the motion
        # of small far-away workers (ROI.mp4: person recall 0.69 → 0.54)
        self.contour_area = int(contour_area)
        self.stale_frames = int(stale_frames)
        self.frame_alpha = float(frame_alpha)
        self.lightning_threshold = float(lightning_threshold)
        self.improve_contrast = bool(improve_contrast)
        # Frigate seeds the contrast history at (0, 255) so the stretch eases in.
        self.contrast_values = np.zeros((int(contrast_frame_history), 2), np.uint8)
        self.contrast_values[:, 1:2] = 255
        self.contrast_values_index = 0
        self.motion_frame_count = 0
        self.calibrating = True
        self.avg_frame = np.zeros((self.h, self.w), np.float32)
        self._mask_idx: Optional[tuple] = None
        self.update_mask(mask_polygons)

    def update_mask(self, mask_polygons: Optional[Sequence]) -> None:
        """Normalised polygons ([[x, y], ...] in 0..1); inside = ignored. Resets the background."""
        raster = np.ones((self.h, self.w), np.uint8)
        for poly in mask_polygons or []:
            pts = np.array([[round(float(x) * self.w), round(float(y) * self.h)] for x, y in poly], np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(raster, [pts], 0)
        self._mask_idx = np.where(raster == 0)
        self.avg_frame = np.zeros((self.h, self.w), np.float32)
        self.calibrating = True
        self.motion_frame_count = 0
        self._prev: Optional[np.ndarray] = None
        self._stale = np.zeros((self.h, self.w), np.uint16)
        self._protected = np.zeros((self.h, self.w), bool)
        self._frames = 0
        self._seen = 0  # frames since the last reset, for the contrast warm-up
        self.relearned_px = 0
        self.rebases = 0

    def detect(self, gray: np.ndarray) -> list[Box]:
        """Grayscale native frame → motion boxes (x1, y1, x2, y2) in native pixels."""
        small = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        resized = self._prepare(small)
        delta = cv2.absdiff(resized, cv2.convertScaleAbs(self.avg_frame))
        if self.stale_frames:
            self._relearn_stale(small, resized, delta)
        thresh = cv2.dilate(cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1], None, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[Box] = []
        total_area = 0.0
        f = self.resize_factor
        for c in contours:
            area = cv2.contourArea(c)
            total_area += area
            if area > self.contour_area:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((int(x * f), int(y * f), int((x + w) * f), int((y + h) * f)))

        pct_motion = total_area / (self.h * self.w)
        # The (0, 255) contrast seed keeps changing the stretched image until the history is full, which reads as
        # motion on a static scene: calibration ends only once every history entry comes from real frames.
        self._seen += 1
        settled = not self.improve_contrast or self._seen >= len(self.contrast_values)
        if pct_motion < 0.05 and len(boxes) <= 4 and settled:
            self.calibrating = False
        if self.calibrating or pct_motion > self.lightning_threshold:
            self.calibrating = True

        # conditional absorption: persistent motion (≥ 10 frames) is folded into the background
        alpha = 0.2 if self.calibrating else self.frame_alpha
        if boxes:
            self.motion_frame_count += 1
            if self.motion_frame_count >= 10:
                cv2.accumulateWeighted(resized.astype(np.float32), self.avg_frame, alpha)
        else:
            cv2.accumulateWeighted(resized.astype(np.float32), self.avg_frame, alpha)
            self.motion_frame_count = 0
        return boxes

    def protect(self, boxes: Sequence[Sequence[float]]) -> None:
        """Native-pixel boxes of recent detections: no stale relearning there (replaces the last set)."""
        raster = np.zeros((self.h, self.w), bool)
        f = self.resize_factor
        for b in boxes:
            x1, y1 = max(0, int(b[0] / f)), max(0, int(b[1] / f))
            x2, y2 = min(self.w, int(np.ceil(b[2] / f)) + 1), min(self.h, int(np.ceil(b[3] / f)) + 1)
            raster[y1:y2, x1:x2] = True
        self._protected = raster

    def _prepare(self, small: np.ndarray, update_contrast: bool = True) -> np.ndarray:
        if self.improve_contrast:
            small = self._stretch(small, update_contrast)
        if self._mask_idx is not None and len(self._mask_idx[0]):
            small[self._mask_idx] = 0              # after contrast, as upstream
        if gaussian_filter is not None:
            return gaussian_filter(small, sigma=1, radius=1)
        return cv2.GaussianBlur(small, (3, 3), 1)

    def _relearn_stale(self, small: np.ndarray, resized: np.ndarray, delta: np.ndarray) -> None:
        """Count frames each pixel is off-background yet still; snap the background there (next frame).
        A large still change (lights, IR switch, PTZ) rebases the whole frame, contrast included:
        the stretch would otherwise drift for 50 frames and drag the rest over the threshold."""
        # half the threshold: the contrast stretch drifts for ~50 frames after a lighting change and
        # would push freshly relearned pixels back over the threshold one by one
        self._frames += 1
        half = max(1, self.threshold // 2)
        changed = (delta > half) & ~self._protected
        if self._prev is None:
            still = np.zeros_like(changed)
        else:
            still = cv2.absdiff(resized, self._prev) < half
        self._prev = resized
        self._stale = np.where(changed & still, np.minimum(self._stale + 1, 60000), 0).astype(np.uint16)
        snap = self._stale >= self.stale_frames
        # no whole-frame rebase while the startup contrast eases in: it would cut the ease-in short
        if snap.mean() >= self.SCENE_CHANGE and self._frames > len(self.contrast_values):
            self._rebase(small)
        elif snap.any():
            self.avg_frame[snap] = resized[snap]
            self._stale[snap] = 0
            self.relearned_px += int(snap.sum())

    def _rebase(self, small: np.ndarray) -> None:
        last = self.contrast_values[(self.contrast_values_index - 1) % len(self.contrast_values)].copy()
        self.contrast_values[:] = last
        rebased = self._prepare(small.copy(), update_contrast=False)
        self.avg_frame[:] = rebased
        self._prev = rebased
        self._stale[:] = 0
        self.relearned_px += self.h * self.w
        self.rebases += 1

    def _stretch(self, resized: np.ndarray, update: bool = True) -> np.ndarray:
        mn = np.percentile(resized, 4).astype(np.uint8)
        mx = np.percentile(resized, 96).astype(np.uint8)
        if mn >= mx:                                # single-colour frame: skip, as upstream
            return resized
        if update:
            self.contrast_values[self.contrast_values_index] = [mn, mx]
            self.contrast_values_index = (self.contrast_values_index + 1) % len(self.contrast_values)
        avg_min, avg_max = np.mean(self.contrast_values, axis=0)
        resized = np.clip(resized, avg_min, avg_max)
        return (((resized - avg_min) / (avg_max - avg_min)) * 255).astype(np.uint8)

    def stats(self) -> dict:
        return {"calibrating": bool(self.calibrating), "motion_frame_count": int(self.motion_frame_count),
                "motion_frame": f"{self.w}x{self.h}"}
