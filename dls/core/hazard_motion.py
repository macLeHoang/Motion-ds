"""Fire / smoke CANDIDATES from cheap image analysis, on every frame, without the detector.

The detector needs the whole scene to see fire and smoke (isolated crops lose them), and smoke
barely registers as motion (Frigate's background absorbs it, ~2 % of its pixels pass the motion
threshold). Running full frames often enough for 20+ cameras is too expensive, so this module
points at WHERE to look; the detector confirms on a context crop (dls/core/mosaic.py).

Works on a small copy of the frame (`width` px wide, 5+ fps):
  FIRE   fire-coloured pixels (red/orange/yellow, bright; R ≥ G > B) that FLICKER IN PLACE:
         a component of pixels fire-coloured in ≥ 2 of the last `history` frames, holding a
         yellow-WHITE HOT CORE (R, G high and B ≥ fire_core_blue: an orange uniform has a much
         lower G, a yellow helmet or vest almost no blue; ≥ fire_core_px
         and ≥ fire_core_frac of the component), whose brightness varies frame to frame (std ≥
         fire_flicker; ≥ fire_flicker_hot when the core is ≥ fire_hot_frac: a furnace or a steady
         blaze glows more than it flickers), and whose box stays put
         (a walking orange uniform moves, a flame does not), for ≥ fire_persist frames.
  SMOKE  per block, against a SLOW background (`bg_alpha`, frozen where smoke is suspected so
         a smouldering fire is not absorbed), the SEMI-TRANSPARENCY test: behind smoke the
         background pattern is still there — the high-pass pattern of the frame correlates with
         the background's (NCC ≥ smoke_ncc) and no texture was added (contrast ratio ≤
         smoke_contrast; a thin plume barely lowers it) — whereas a person or a car hides the
         pattern (NCC low, contrast up). Also: the block got greyer (saturation down by
         ≥ smoke_desat, or the background is grey already), it got BRIGHTER (≥ smoke_min_delta
         after removing the global light change: grey / white smoke; black smoke is not handled)
         and it CHURNS (std of its brightness over the last smoke_fluct_frames ≥ smoke_fluct): a
         lighting step, a shadow or an object left behind changes once and then stays.
         Blocks crossed by a solid moving object (frame-to-frame change ≥ object_activity: a
         person or a car; smoke drifts slowly) are not judged for object_hold frames, then their
         background snaps to the current view (no ghost). Most of the frame changing at once
         (camera_moved: PTZ, shake) restarts the background.
         Optional smoke_activity: frame-to-frame change (smoke drifts). Blocks without
         background texture cannot be judged. Evidence accumulates per block (+1 / −decay); a block counts at
         ≥ smoke_score; ≥ smoke_min_blocks adjacent blocks make a candidate.
Pure numpy / OpenCV; one instance per camera; candidates in native frame pixels.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    kind: str                              # "fire" | "smoke"
    box: tuple[int, int, int, int]         # native frame pixels
    score: float


class HazardCandidates:
    def __init__(self, frame_shape: tuple[int, int], width: int = 480, history: int = 8,
                 fire_min_px: int = 6, fire_flicker: float = 10.0, fire_persist: int = 5,
                 fire_max_shift: float = 0.5, fire_core_px: int = 3, fire_core_frac: float = 0.05,
                 fire_hot_frac: float = 0.15, fire_flicker_hot: float = 5.0, fire_core_blue: int = 90,
                 block: int = 12, bg_alpha: float = 0.004, smoke_max_sat: int = 80,
                 smoke_min_delta: float = 3.0, smoke_ncc: float = 0.6, smoke_contrast: float = 1.3,
                 smoke_min_texture: float = 2.5, smoke_activity: float = 0.0, smoke_desat: float = 2.0,
                 smoke_grey_bg: float = 25.0, smoke_fluct: float = 1.5, smoke_fluct_frames: int = 15,
                 smoke_score: float = 12.0, smoke_decay: float = 0.5, smoke_min_blocks: int = 3,
                 object_activity: float = 6.0, object_hold: int = 15, camera_moved: float = 0.3,
                 warmup: int = 25):
        self.frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        self.w = int(width)
        self.h = max(8, int(round(self.w * self.frame_shape[0] / self.frame_shape[1])))
        self.sx, self.sy = self.frame_shape[1] / self.w, self.frame_shape[0] / self.h
        self.history = int(history)
        self.fire_min_px, self.fire_flicker = int(fire_min_px), float(fire_flicker)
        self.fire_persist, self.fire_max_shift = int(fire_persist), float(fire_max_shift)
        self.fire_core_px, self.fire_core_frac = int(fire_core_px), float(fire_core_frac)
        self.fire_hot_frac, self.fire_flicker_hot = float(fire_hot_frac), float(fire_flicker_hot)
        self.fire_core_blue = int(fire_core_blue)
        self.block = int(block)
        self.bg_alpha = float(bg_alpha)
        self.smoke_max_sat, self.smoke_min_delta = int(smoke_max_sat), float(smoke_min_delta)
        self.smoke_ncc, self.smoke_contrast = float(smoke_ncc), float(smoke_contrast)
        self.smoke_min_texture, self.smoke_activity = float(smoke_min_texture), float(smoke_activity)
        self.smoke_desat, self.smoke_grey_bg = float(smoke_desat), float(smoke_grey_bg)
        self.smoke_fluct = float(smoke_fluct)
        self._means: deque = deque(maxlen=int(smoke_fluct_frames))   # block means, last ~3 s
        self.smoke_score = float(smoke_score)
        self.smoke_decay, self.smoke_min_blocks = float(smoke_decay), int(smoke_min_blocks)
        self.object_activity, self.object_hold = float(object_activity), int(object_hold)
        self.camera_moved = float(camera_moved)
        self._occ: Optional[np.ndarray] = None              # frames left: a solid object moved here
        self.warmup = int(warmup)
        self._v: deque = deque(maxlen=self.history)         # brightness (V) history
        self._fire: deque = deque(maxlen=self.history)      # fire-colour masks
        self._tracks: list[list] = []                       # [box(small px), hits]
        self._bg: Optional[np.ndarray] = None               # slow grey background (float32)
        self._bg_sat: Optional[np.ndarray] = None           # slow saturation background
        self._prev: Optional[np.ndarray] = None             # previous grey frame
        self._score: Optional[np.ndarray] = None            # smoke evidence per block
        self._n = 0
        self._moved = False
        self.stats = {"frames": 0, "fire_candidates": 0, "smoke_candidates": 0, "camera_moved": 0}

    # ------------------------------------------------------------------ public
    def update(self, image: np.ndarray, camera_moving: bool = False) -> list[Candidate]:
        """One frame (BGR / BGRx native) → current candidates. `camera_moving` (e.g. the motion
        detector is calibrating: PTZ, shake, lightning) restarts both analyses; most of the frame
        changing at once does too."""
        self._n += 1
        self.stats["frames"] += 1
        small = cv2.resize(np.ascontiguousarray(image[:, :, :3]), (self.w, self.h), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self._moved = bool(camera_moving)
        smoke = self._smoke_candidates(hsv, grey)
        if self._moved:                                    # nothing stays put while the view moves
            self._v.clear()
            self._fire.clear()
            self._tracks = []
            return []
        out = self._fire_candidates(small, hsv) + smoke
        for c in out:
            self.stats[f"{c.kind}_candidates"] += 1
        return out

    # ------------------------------------------------------------------ fire
    def _fire_mask(self, small: np.ndarray, hsv: np.ndarray) -> np.ndarray:
        b, g, r = (small[:, :, i].astype(np.int16) for i in range(3))
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        coloured = (r >= 150) & (r >= g) & (g > b) & (r - b >= 60) & ((h <= 35) | (h >= 170)) & (s >= 60)
        # the white-hot core of a flame next to coloured flame pixels
        core = (v >= 235) & (s < 60) & (cv2.dilate(coloured.astype(np.uint8), None, iterations=2) > 0)
        return coloured | core

    def _fire_candidates(self, small: np.ndarray, hsv: np.ndarray) -> list[Candidate]:
        self._v.append(hsv[:, :, 2].astype(np.float32))
        self._fire.append(self._fire_mask(small, hsv))
        if len(self._fire) < 3:
            return []
        seen = np.sum(np.stack(self._fire), axis=0) >= 2
        n, _, st, _ = cv2.connectedComponentsWithStats(seen.astype(np.uint8))
        flicker = np.std(np.stack(self._v), axis=0)
        b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
        # yellow-WHITE: burning; a yellow helmet or vest is saturated yellow (blue near 0)
        hot = (r >= 220) & (g >= 180) & (b >= self.fire_core_blue)
        found = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area < self.fire_min_px:
                continue
            core = int(hot[max(0, y - 1):y + h + 1, max(0, x - 1):x + w + 1].sum())
            if core < self.fire_core_px or core < self.fire_core_frac * area:
                continue
            region = flicker[y:y + h, x:x + w][seen[y:y + h, x:x + w]]
            # a big glowing core (furnace, steady blaze) flickers less than a small flame
            need = self.fire_flicker_hot if core >= self.fire_hot_frac * area else self.fire_flicker
            if region.size and float(np.mean(region)) >= need:
                found.append(((x, y, x + w, y + h), float(np.mean(region))))
        return self._persist(found)

    def _persist(self, found: list) -> list[Candidate]:
        """A candidate must stay put for fire_persist frames (a flame flickers in place)."""
        tracks = []
        for box, score in found:
            prev = next((t for t in self._tracks if _overlap(t[0], box) >= self.fire_max_shift), None)
            tracks.append([box, (prev[1] + 1) if prev else 1, score])
        self._tracks = tracks
        return [Candidate("fire", self._native(b), s) for b, hits, s in tracks if hits >= self.fire_persist]

    # ------------------------------------------------------------------ smoke
    def _smoke_candidates(self, hsv: np.ndarray, grey: np.ndarray) -> list[Candidate]:
        k = self.block
        bh, bw = self.h // k, self.w // k
        g = grey[:bh * k, :bw * k]
        sat = hsv[:bh * k, :bw * k, 1].astype(np.float32)
        if self._bg is None or self._moved:
            self._reset(g, sat)
            return []
        hp_f = g - cv2.blur(g, (5, 5))
        hp_b = self._bg - cv2.blur(self._bg, (5, 5))
        sf = np.sqrt(_block_mean(hp_f * hp_f, k))
        sb = np.sqrt(_block_mean(hp_b * hp_b, k))
        ncc = _block_mean(hp_f * hp_b, k) / (sf * sb + 1e-3)
        contrast = sf / (sb + 1e-3)
        activity = _block_mean(np.abs(g - self._prev), k)
        # a person, a car: changes fast AND hides the background pattern; smoke keeps the pattern
        # (and a featureless background cannot tell, so fast change alone counts there)
        moving = (activity >= self.object_activity) & ((ncc < self.smoke_ncc) | (sb < self.smoke_min_texture))
        if (activity >= self.object_activity).mean() > self.camera_moved:   # PTZ move / shake: start over
            self.stats["camera_moved"] += 1
            self._moved = True
            self._reset(g, sat)
            return []
        # where a solid object was moving: no smoke judgement for object_hold frames, then the
        # background snaps to what is there now (no ghost of the object in the slow background) —
        # only where the pattern is still broken: a translucent plume is never snapped into it
        was = self._occ > 0
        self._occ = np.where(moving, self.object_hold, np.maximum(self._occ - 1, 0))
        snap_blocks = was & (self._occ == 0) & (ncc < self.smoke_ncc)
        snap = np.kron(snap_blocks.astype(np.float32), np.ones((k, k), np.float32)) > 0
        self._bg[snap], self._bg_sat[snap] = g[snap], sat[snap]
        mean_now = _block_mean(g, k)
        delta = mean_now - _block_mean(self._bg, k)
        delta -= np.median(delta)                              # global light change (clouds, AGC)
        self._means.append(mean_now - np.median(mean_now))
        # smoke churns: its block brightness keeps changing; a lighting step or a left object does not
        fluct = np.std(np.stack(self._means), axis=0) if len(self._means) >= 3 else np.zeros_like(mean_now)
        s_now, s_bg = _block_mean(sat, k), _block_mean(self._bg_sat, k)
        self._prev = g
        greyer = (s_now <= s_bg - self.smoke_desat) | (s_bg <= self.smoke_grey_bg)
        smoky = ((sb >= self.smoke_min_texture) & (ncc >= self.smoke_ncc) & (contrast <= self.smoke_contrast)
                 & (delta >= self.smoke_min_delta) & (s_now <= self.smoke_max_sat) & greyer
                 & (fluct >= self.smoke_fluct) & (activity >= self.smoke_activity) & (self._occ == 0))
        self._score = np.clip(np.where(smoky, self._score + 1.0, self._score - self.smoke_decay), 0, 3 * self.smoke_score)
        # slow background, frozen where smoke is building up (a smoulder must not be absorbed)
        alpha = 0.2 if self._n <= self.warmup else self.bg_alpha
        # not where smoke builds up (absorbed otherwise) nor under a moving object (its ghost)
        keep = np.kron(((self._score <= 0) & (self._occ == 0)).astype(np.float32), np.ones((k, k), np.float32))
        self._bg += alpha * keep * (g - self._bg)
        self._bg_sat += alpha * keep * (sat - self._bg_sat)
        if self._n <= self.warmup:
            self._score[:] = 0
            return []
        hot = (self._score >= self.smoke_score).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(hot)
        out = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if area >= self.smoke_min_blocks:
                score = float(self._score[y:y + h, x:x + w].mean())
                out.append(Candidate("smoke", self._native((x * k, y * k, (x + w) * k, (y + h) * k)), score))
        return out

    def _reset(self, g: np.ndarray, sat: np.ndarray) -> None:
        k = self.block
        self._bg, self._bg_sat, self._prev = g.copy(), sat.copy(), g.copy()
        self._score = np.zeros((g.shape[0] // k, g.shape[1] // k), np.float32)
        self._occ = np.zeros_like(self._score)
        self._means.clear()
        self._n = 0                                         # warm up again

    # ------------------------------------------------------------------ helpers
    def _native(self, b: tuple) -> tuple[int, int, int, int]:
        return (int(b[0] * self.sx), int(b[1] * self.sy), int(b[2] * self.sx), int(b[3] * self.sy))


def _block_mean(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k, a.shape[1] // k
    return a[:h * k, :w * k].reshape(h, k, w, k).mean(axis=(1, 3))


def _overlap(a: tuple, b: tuple) -> float:
    """Intersection over the smaller box (a flame's box breathes; IoU would drop)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1
    return ix * iy / small
