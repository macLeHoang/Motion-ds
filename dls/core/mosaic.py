"""Motion → mosaic → inference scheduler (one per camera). Pure: the detector is injected.

Per frame:
  1. MotionDetector (Frigate) on the grayscale frame.
  2. Crops = motion boxes + boxes detected in the last `keep_frames` frames (an object that stops
     moving is still inspected; those boxes are grown a little because the object may have moved),
     merged when they overlap and padded by `crop_pad` for context.
  3. The crops — and nothing else of the frame — are packed into ONE near-square mosaic at native
     resolution; the detector runs ONCE on it and boxes are mapped back to the frame. NOTE: the
     detector's letterbox fits the mosaic to the model input both ways — a mosaic smaller than
     the input is ZOOMED IN (ROI.mp4: median ×1.28), not just shrunk when larger.
  4. Full frame instead when the motion background is calibrating (startup, lightning, IR switch),
     periodically (`full_frame_every_s` of stream time, else every `full_frame_interval` frames:
     stationary objects, a smouldering fire absorbed into the background) or when the scene is
     crowded: the mosaic would be shrunk at least as much as the whole frame, so cropping gains
     nothing. A mosaic smaller than the model input is zoomed in at most `max_zoom` (padded).
  5. No motion and nothing kept → no inference at all.
  6. Optional `det_filter` (dls/core/det_filter.py), fed once per inference: only detections of
     tracks confirmed over N of M inferences are returned, boxes far off the learned perspective
     size are dropped (and not re-inspected), and each crop grows to ≥ `min_crop_context` × the
     expected person height there (a tight crop on a smoke wisp loses the context that tells the
     model it is not a person). A scene change (calibration ends on a different background, e.g.
     PTZ move) forgets the learned sizes.
  7. Motion runs on EVERY frame (Frigate's constants assume ~5 fps: at 1 fps clouds and swaying
     cables become "motion" over half the frame), but inference is capped at `infer_max_fps` per
     stream second (`now` = buffer PTS); frames over the cap only update the motion background.
  8. `full_frame_labels` (fire, smoke): the model needs the whole scene for them (isolated crops,
     even 5× the box, mostly fail), so they are trusted only from full-frame inferences. Seen in a
     mosaic they are dropped but make the NEXT inference a full frame; a full-frame hit is
     followed by `confirm_full` more full frames (the event gate gets real repeated hits).
  9. Optional `gate` (dls/core/scheduler.py, shared by the cameras of one device): each inference
     asks for a turn with a priority — CONFIRM (fire suspected) > FULL_DUE (periodic scan,
     calibration) > MOSAIC (persons). No turn → this frame is skipped, the request ages.
 10. Optional `hazard` (dls/core/hazard_motion.py): fire / smoke CANDIDATES from image analysis
     on every `hazard_every`-th observed frame. The strongest one is checked by the detector on a
     context crop (hazard_crop_px square, default 768 px native; full frame when the candidate
     does not fit), trusted like a full frame; at most one check per hazard_check_s;
     a check that finds nothing cools that spot down for hazard_cooldown_s. So the periodic
     full-frame scan can be rare (full_frame_every_s 30–60 s) without waiting for it.
 11. Optional helmet check (`head_check_s` > 0, dls/core/head_check.py): every head_check_s the
     upper bodies of the persons of the latest sighting, zoomed to ~head_person_px tall, are
     packed into one model input (up to ~9) — longest-unchecked first. Only the head classes
     (`head_labels`) come back from it; persons and the filters are left as they were.
 12. Optional scaled mosaic (`crop_target_px` > 0): instead of native crops letterboxed as one
     image, each crop is zoomed so a person standing at its bottom (learned perspective) is
     ~crop_target_px tall (≤ max_zoom; all shrunk alike when they do not fit) and MaxRects-packed
     into the model input itself. With `heads_in_mosaic` the space left takes the zoomed upper
     bodies of step 11, so the helmet check needs no inference of its own (and no separate one
     runs). ROI.mp4, 1 camera: person recall vs 768 0.59 → 0.71, persons with a head box 0.46 →
     0.63; 5 cameras: persons equal, heads 0.05 → 0.08. Scaling alone did not help; tiles of 110 /
     120 px and re-inspecting still workers less often changed nothing measurable.

`infer(image) -> [{"label", "score", "box": (x1, y1, x2, y2)}]` in the coordinates of `image`.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from dls.core.det_filter import DetectionFilter
from dls.core.hazard_motion import HazardCandidates
from dls.core.head_check import HeadSchedule, draw_tile, map_heads, pack_heads, tile_plan
from dls.core.motion import MotionDetector
from dls.core.noise_map import NoiseMap
from dls.core.regions import (cap_zoom, dedupe_detections, expand_box, iou, map_back, merge_overlapping,
                              pack_mosaic, pack_scaled)
from dls.core.scheduler import CONFIRM, FULL_DUE, MOSAIC

KEEP_REPLACE_IOU = 0.3
SCENE_CHANGE_DIFF = 20.0     # mean |Δ| of the motion background (grey levels) that means a new view
SCENE_SETTLE_FRAMES = 25    # ~5 s at 5 fps without calibration = the background has converged
PENDING_MAX = 200           # motion boxes kept between two inferences (observe → process)
KEEP_EXPAND = 0.25          # kept boxes grow by this fraction per side (object moved since)
PERSON_LIKE = {"person"}
PERSON_RECENT_S = 2.0       # helmet check: persons seen within this of the latest sighting …
PERSON_STALE_S = 5.0        # … when that sighting is at most this old
CUT_EDGE_PX = 3             # a box this close to an inner crop edge was cut by the crop
CUT_EXPAND = 0.5            # its surroundings, re-inspected next time, grow by this per side
InferFn = Callable[[np.ndarray], list[dict[str, Any]]]


class MotionMosaic:
    def __init__(self, frame_shape: tuple[int, int], canvas_size: int, infer: InferFn,
                 full_frame_interval: int = 50, keep_frames: int = 10, crop_pad: float = 0.15, gap: int = 4,
                 classes: Optional[Iterable[str]] = None, mask_polygons: Optional[Sequence] = None,
                 motion_frame_height: int = 100, infer_max_fps: float = 0.0,
                 det_filter: Optional[DetectionFilter] = None, min_crop_context: float = 0.0,
                 full_frame_every_s: float = 0.0, full_frame_labels: Optional[Iterable[str]] = None,
                 confirm_full: int = 2, max_zoom: float = 2.0, gate: Optional[Any] = None,
                 hazard: Any = None, hazard_every: int = 1, hazard_check_s: float = 3.0,
                 hazard_cooldown_s: float = 10.0, hazard_context: float = 4.0, hazard_min_side: float = 0.5,
                 hazard_crop_px: int = 768, merge_growth: Optional[float] = 1.3, kept_max_aspect: float = 4.5,
                 noise_filter: Any = None, crop_context_aspect: float = 1.0,
                 mosaic_min_score: Optional[Mapping[str, float]] = None, drop_cut: bool = False,
                 static_fp: Any = None, motion_stale_frames: int = 0, head_check_s: float = 0.0,
                 head_person_px: float = 140.0, head_labels: Optional[Iterable[str]] = None,
                 crop_target_px: float = 0.0, heads_in_mosaic: bool = False):
        self.frame_shape = (int(frame_shape[0]), int(frame_shape[1]))
        self.canvas_size = int(canvas_size)            # model input side (square)
        self.full_frame_interval = int(full_frame_interval)
        self.full_frame_every_s = float(full_frame_every_s)
        self.full_frame_labels = set(full_frame_labels or ())
        self.confirm_full = int(confirm_full)
        self.max_zoom = float(max_zoom)
        self.keep_frames = int(keep_frames)
        self.crop_pad = float(crop_pad)
        self.gap = int(gap)
        self.classes = set(classes) if classes else None
        self.motion = MotionDetector(self.frame_shape, frame_height=motion_frame_height,
                                     mask_polygons=mask_polygons, stale_frames=motion_stale_frames)
        self._infer = infer
        self.det_filter = det_filter
        self.min_crop_context = float(min_crop_context)
        self.gate = gate                               # .acquire(priority) -> bool, .release()
        self._motion_lock = threading.Lock()           # observe() runs on the camera thread
        self._hazard_lock = threading.Lock()           # HazardCandidates is not thread-safe
        # fire / smoke candidates from image analysis (dls/core/hazard_motion.py): checked by the
        # detector on a context crop instead of waiting for the periodic full frame
        self.hazard = None
        if hazard:
            self.hazard = HazardCandidates(self.frame_shape, **(dict(hazard) if isinstance(hazard, Mapping) else {}))
        self.hazard_every, self.hazard_check_s = max(1, int(hazard_every)), float(hazard_check_s)
        self.hazard_cooldown_s, self.hazard_context = float(hazard_cooldown_s), float(hazard_context)
        self.hazard_min_side = float(hazard_min_side)
        self.hazard_crop_px = int(hazard_crop_px)
        self.merge_growth = merge_growth                # None = union every overlap (old behaviour)
        # nuisance motion learned online (OSD clock, swaying trees): dls/core/noise_map.py
        self.noise = None
        if noise_filter:
            self.noise = NoiseMap(self.frame_shape, **(dict(noise_filter) if isinstance(noise_filter, Mapping) else {}))
        self.kept_max_aspect = float(kept_max_aspect)
        self.crop_context_aspect = float(crop_context_aspect)
        # stricter scores for mosaic detections (tight crops lack context: low-score boxes on
        # insulators, rods); full frames keep the detector threshold
        self.mosaic_min_score = dict(mosaic_min_score or {})
        self.drop_cut = bool(drop_cut)
        # weak person boxes on things that never move (dls docstring of _drop_static_weak)
        self.static_fp = None
        if static_fp:
            self.static_fp = {"score": 0.4, "release": 0.5, "hits": 3, "ttl_s": 60.0,
                              **(dict(static_fp) if isinstance(static_fp, Mapping) else {})}
        self._static: list[dict] = []
        self.head_labels = set(head_labels or ())
        self.head_person_px = float(head_person_px)
        self.heads = HeadSchedule(head_check_s) if head_check_s > 0 and self.head_labels else None
        self._persons: list[tuple[tuple, float]] = []  # (box, time) of persons recently detected
        self.crop_target_px = float(crop_target_px)
        self.heads_in_mosaic = bool(heads_in_mosaic) and self.heads is not None
        self._hz_frame = 0
        self._hz_cands: list = []
        self._hz_cool: list[tuple[tuple, float]] = []  # (region, until): checked, nothing there
        self._hz_last: Optional[float] = None
        self._hz_focus: Optional[tuple] = None         # crop of the last hit, re-checked to confirm
        self._observed = 0
        self._pending: list = []                       # motion boxes observed since the last process()
        self._settled = 0
        self._recalibrated = False
        self._scene_ref: Optional[np.ndarray] = None
        self._min_interval = 1.0 / infer_max_fps if infer_max_fps > 0 else 0.0
        self._last_infer: Optional[float] = None
        self._last_full: Optional[float] = None
        self._now: Optional[float] = None
        self._suspect = False                          # full_frame_labels seen in a mosaic
        self._confirm_left = 0                         # full frames still owed after a hit
        self._counted = 0
        self._kept: dict[tuple[int, int, int, int], int] = {}     # box → frames left
        self._frame = 0
        self.stats = {"frames": 0, "full_frame": 0, "mosaic": 0, "skipped": 0, "crowded": 0, "rate_limited": 0,
                      "crops": 0, "inferences": 0, "mosaic_px": 0, "frame_px": 0, "scene_changes": 0,
                      "denied": 0, "confirm_full": 0, "suspects": 0, "full_gap_max_s": 0.0, "full_gap_sum_s": 0.0,
                      "observed": 0, "hazard_candidates": 0, "hazard_checks": 0, "hazard_hits": 0,
                      "hazard_cooled": 0, "noise_dropped": 0, "cut_set_aside": 0,
                      "static_dropped": 0, "weak_cut_dropped": 0, "head_checks": 0, "head_tiles": 0,
                      "heads_found": 0, "mosaic_head_tiles": 0, "mosaic_heads_found": 0}

    def observe(self, image: np.ndarray, now: Optional[float] = None) -> None:
        """Motion only, for EVERY frame, called upstream of the inference (dls/gst/chain.py probe
        before the leaky queue). While an inference runs (~1 s on a small CPU) the inference
        thread sees no frames; without this the motion detector would run at ~1 fps, where half
        the frame "moves" and every inference turns into a full frame. Boxes accumulate until the
        next process() call, which uses them instead of running motion itself."""
        gray = _gray(image)
        with self._motion_lock:
            boxes = self._denoise(self.motion.detect(gray), now)
            self._check_scene()
            self._observed += 1
            # boxes seen while calibrating are meaningless (the caller runs full frame then)
            self._pending = [] if self.motion.calibrating else (self._pending + list(boxes))[-PENDING_MAX:]
        self._hazard_update(image)

    def process(self, image: np.ndarray, now: Optional[float] = None) -> list[dict[str, Any]]:
        """`now` = stream time (s) for the inference cap and `full_frame_every_s`; None → no cap."""
        self._frame += 1
        self._now = now
        self.stats["frames"] += 1
        with self._motion_lock:
            observed, motion = self._observed, self._pending
            self._observed, self._pending = 0, []
        if not observed:                                        # no observer: motion here, on this frame
            gray = _gray(image)
            with self._motion_lock:
                motion = self._denoise(self.motion.detect(gray), now)
                self._check_scene()
            self._hazard_update(image)
        else:
            self.stats["observed"] += observed
        if now is not None and self._min_interval and self._last_infer is not None \
                and now - self._last_infer < self._min_interval - 1e-6:
            self.stats["rate_limited"] += 1
            return []
        result = self._decide(image, motion)
        if self.stats["inferences"] != self._counted:
            self._counted = self.stats["inferences"]
            self._last_infer = now
        return result

    def _full_due(self) -> bool:
        if self.full_frame_every_s > 0 and self._now is not None:
            return self._last_full is None or self._now - self._last_full >= self.full_frame_every_s - 1e-6
        return self.full_frame_interval > 0 and self._frame % self.full_frame_interval == 0

    def _decide(self, image: np.ndarray, motion: list) -> list[dict[str, Any]]:
        if self._confirm_left > 0 and self._hz_focus is not None and not self._suspect:
            return self._run_hazard(image, None)          # confirm a crop hit on the same crop
        if self._suspect or self._confirm_left > 0:
            return self._run_full(image, CONFIRM)
        if self.motion.calibrating or self._full_due():
            return self._run_full(image, FULL_DUE)
        candidate = self._hazard_due()
        if candidate is not None:
            return self._run_hazard(image, candidate)
        persons = self._recent_persons()
        # heads_in_mosaic: heads only ride along (a separate check on a busy device took a third
        # of the inferences from persons: 5 cameras, person recall 0.125 → 0.072)
        if self.heads is not None and not self.heads_in_mosaic and persons and self.heads.due(self._clock()):
            return self._run_heads(image, persons)

        kept = list(self._kept)
        if not motion and not kept:
            self.stats["skipped"] += 1
            return []
        # each region grows ONCE (the largest of: its padding, the kept-box margin, the minimum
        # context), then overlapping regions merge only when that adds little background
        regions = ([expand_box(b, self.crop_pad, self.frame_shape) for b in motion]
                   + [expand_box(b, max(KEEP_EXPAND, self.crop_pad), self.frame_shape) for b in kept])
        crops = merge_overlapping([self._with_context(b) for b in regions], max_growth=self.merge_growth)
        if self.crop_target_px > 0:
            return self._run_scaled(image, crops, motion)
        mosaic, places = pack_mosaic(image, crops, self.gap)
        if mosaic is None:
            self.stats["skipped"] += 1
            return []
        if self._shrink(mosaic.shape[:2]) <= self._shrink(self.frame_shape):
            return self._run_full(image, MOSAIC, crowded=True)
        if not self._turn(MOSAIC):
            return []
        try:
            raw = self._infer(self._cap_zoom(mosaic))
        finally:
            self._release()
        self.stats["mosaic"] += 1
        self.stats["inferences"] += 1
        self.stats["crops"] += len(places)
        self.stats["mosaic_px"] += mosaic.shape[0] * mosaic.shape[1]
        self.stats["frame_px"] += self.frame_shape[0] * self.frame_shape[1]
        dets = dedupe_detections(map_back(self._filter(raw), places))
        dets = [d for d in dets if d["score"] >= self.mosaic_min_score.get(d["label"], 0.0)]
        if self.drop_cut:
            dets = self._set_aside_cut(dets, places)
        if self.static_fp is not None:
            dets = self._drop_static_weak(dets, places, motion)
        return self._finish(dets, full=False)

    def _crop_scale(self, crop: tuple) -> float:
        """Zoom that brings a person standing at the crop's bottom (learned perspective) to
        crop_target_px, capped at max_zoom; native (1) while sizes are not learned yet."""
        expected = self.det_filter.expected_height("person", crop[3]) if self.det_filter is not None else None
        if not expected:
            return 1.0
        return min(self.max_zoom if self.max_zoom > 0 else 1.0, self.crop_target_px / expected)

    def _run_scaled(self, image: np.ndarray, crops: list, motion: list) -> list[dict[str, Any]]:
        """Each crop zoomed so persons in it are ~crop_target_px tall (all shrunk alike when they
        do not fit), MaxRects-packed into the model input; the space left takes zoomed upper
        bodies for the helmet check (heads_in_mosaic). Crowded — every crop would be seen no
        larger than in the letterboxed full frame — → full frame."""
        scales = [self._crop_scale(c) for c in crops]
        canvas, places, bin_, _ = pack_scaled(image, crops, scales, self.canvas_size, self.gap)
        if canvas is None:
            self.stats["skipped"] += 1
            return []
        if max(p.scale for p in places) <= self._shrink(self.frame_shape):
            return self._run_full(image, MOSAIC, crowded=True)
        tiles = self._add_head_tiles(canvas, image, bin_) if self.heads_in_mosaic else []
        if not self._turn(MOSAIC):
            return []
        try:
            raw = self._filter(self._infer(canvas))
        finally:
            self._release()
        self.stats["mosaic"] += 1
        self.stats["inferences"] += 1
        self.stats["crops"] += len(places)
        self.stats["mosaic_px"] += sum(p.w * p.h for p in places)
        self.stats["frame_px"] += self.frame_shape[0] * self.frame_shape[1]
        in_tile = [any(t.holds(d["box"]) for t in tiles) for d in raw]
        dets = dedupe_detections(map_back([d for d, t in zip(raw, in_tile) if not t], places))
        dets = [d for d in dets if d["score"] >= self.mosaic_min_score.get(d["label"], 0.0)]
        if self.drop_cut:
            dets = self._set_aside_cut(dets, places)
        if self.static_fp is not None:
            dets = self._drop_static_weak(dets, places, motion)
        out = self._finish(dets, full=False)
        if not tiles:
            return out
        self.heads.done([t.person for t in tiles], self._clock())
        heads = map_heads([d for d, t in zip(raw, in_tile) if t], tiles, self.head_labels)
        self.stats["mosaic_head_tiles"] += len(tiles)
        self.stats["mosaic_heads_found"] += len(heads)
        # a head seen both in a person crop and in a tile: the better score stays
        best = {id(d) for d in dedupe_detections([d for d in out if d["label"] in self.head_labels] + heads)}
        return [d for d in out + heads if d["label"] not in self.head_labels or id(d) in best]

    def _add_head_tiles(self, canvas: np.ndarray, image: np.ndarray, bin_) -> list:
        """Zoomed upper bodies of recent persons into the canvas' free space, longest-unchecked
        first; a person whose tile does not fit is left for a later mosaic (or the separate check)."""
        persons = self._recent_persons()
        if not persons or bin_ is None:
            return []
        tiles = []
        for p in self.heads.order(persons, self._clock()):
            plan = tile_plan(p, self.frame_shape, self.canvas_size, self.head_person_px)
            if plan is None:
                continue
            at = bin_.insert(plan[2], plan[3])
            if at is not None:
                tiles.append(draw_tile(canvas, image, p, plan, at[0], at[1]))
        return tiles

    def _recent_persons(self) -> list[tuple]:
        """Persons of the latest sighting (within PERSON_RECENT_S of it), if it is ≤ PERSON_STALE_S
        old: on a busy device a camera infers every ~3 s, so "the last 2 s" would hold nobody."""
        if not self._persons:
            return []
        newest = max(seen for _, seen in self._persons)
        if self._clock() - newest > PERSON_STALE_S:
            return []
        return [b for b, seen in self._persons if newest - seen <= PERSON_RECENT_S]

    def _run_heads(self, image: np.ndarray, persons: list[tuple]) -> list[dict[str, Any]]:
        """Zoomed upper bodies in one inference; only the heads found come back (frame pixels)."""
        t = self._clock()
        canvas, tiles = pack_heads(image, self.heads.order(persons, t), self.canvas_size, self.head_person_px)
        if canvas is None or not self._turn(MOSAIC):
            return []
        try:
            raw = self._infer(canvas)
        finally:
            self._release()
        self.heads.done([tile.person for tile in tiles], t)
        heads = map_heads(self._filter(raw), tiles, self.head_labels)
        self.stats["head_checks"] += 1
        self.stats["inferences"] += 1
        self.stats["head_tiles"] += len(tiles)
        self.stats["heads_found"] += len(heads)
        return heads

    def _run_full(self, image: np.ndarray, priority: int, crowded: bool = False) -> list[dict[str, Any]]:
        if not self._turn(priority):
            return []
        try:
            if crowded:
                self.stats["crowded"] += 1
            if priority == CONFIRM:
                self.stats["confirm_full"] += 1
            dets = self._full(image)
        finally:
            self._release()
        return self._finish(dets, full=True)

    def _drop_static_weak(self, dets: list[dict[str, Any]], places, motion: list) -> list[dict[str, Any]]:
        """Weak "person" boxes that never move are things (insulator stacks, rods): the model
        sees a person there at 0.25–0.35, the box is kept for re-inspection, the crop comes back
        and the same false box returns — a loop. A weak box with no motion touching it, seen at
        the same spot in `hits` inferences in a row, marks that spot: boxes there are neither
        reported nor kept. Any motion on the spot or a confident box there (≥ release) clears it;
        an unseen spot is forgotten after ttl_s. A weak box cut by its crop is dropped at once."""
        cfg, now = self.static_fp, self._clock()
        fh, fw = self.frame_shape
        self._static = [s for s in self._static if now - s["seen"] <= cfg["ttl_s"]]
        out = []
        for d in dets:
            b = d["box"]
            if d["label"] not in PERSON_LIKE:
                out.append(d)
                continue
            spot = next((s for s in self._static if iou(s["box"], b) >= 0.5), None)
            moving = any(_overlap(b, m) > 0.1 for m in motion)
            if d["score"] >= cfg["release"] or moving:
                if spot is not None:
                    self._static.remove(spot)
                out.append(d)
                continue
            if d["score"] >= cfg["score"]:
                out.append(d)
                continue
            region = next((p.region for p in places if p.region[0] <= (b[0] + b[2]) / 2 < p.region[2]
                           and p.region[1] <= (b[1] + b[3]) / 2 < p.region[3]), None)
            if region is not None and _cut_by(b, region, fw, fh):
                self.stats["weak_cut_dropped"] += 1          # half an object, weakly: not a person
                continue
            if spot is None:
                spot = {"box": tuple(b), "hits": 0, "seen": now}
                self._static.append(spot)
            spot["hits"] += 1
            spot["seen"] = now
            spot["box"] = tuple(b)
            if spot["hits"] >= cfg["hits"]:
                self.stats["static_dropped"] += 1
                continue
            out.append(d)
        return out

    def _set_aside_cut(self, dets: list[dict[str, Any]], places) -> list[dict[str, Any]]:
        """A box touching an INNER edge of its crop is an object cut in half (half a person, a
        fragment of a rod): not reported; its surroundings are re-inspected on the next mosaic
        with a crop large enough to hold it whole."""
        fh, fw = self.frame_shape
        whole = []
        for d in dets:
            b = d["box"]
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            region = next((p.region for p in places
                           if p.region[0] <= cx < p.region[2] and p.region[1] <= cy < p.region[3]), None)
            if region is not None and _cut_by(b, region, fw, fh):
                self.stats["cut_set_aside"] += 1
                grown = expand_box(b, CUT_EXPAND, self.frame_shape)
                self._kept[tuple(int(v) for v in grown)] = self.keep_frames
            else:
                whole.append(d)
        return whole

    def _denoise(self, boxes: list, now: Optional[float]) -> list:
        """Learn and drop nuisance motion (caller holds _motion_lock)."""
        if self.noise is None or self.motion.calibrating:
            return boxes
        self.noise.observe(boxes)
        kept = self.noise.filter(boxes, now if now is not None else self._frame / 5.0)
        self.stats["noise_dropped"] += len(boxes) - len(kept)
        return kept

    # ------------------------------------------------------------------ fire / smoke candidates
    def _hazard_update(self, image: np.ndarray) -> None:
        if self.hazard is None:
            return
        # the observer (camera thread) and process() without an observed frame can both get
        # here: HazardCandidates keeps frame histories that one call clears while another stacks
        with self._hazard_lock:
            self._hz_frame += 1
            if self._hz_frame % self.hazard_every:
                return
            cands = self.hazard.update(image, camera_moving=self.motion.calibrating)
        with self._motion_lock:
            self._hz_cands = cands
        self.stats["hazard_candidates"] += len(cands)

    def _clock(self) -> float:
        return self._now if self._now is not None else self._frame / 5.0

    def _hazard_due(self):
        """The strongest candidate not cooled down (checked and empty), at most one check per
        hazard_check_s."""
        if self.hazard is None or not self.full_frame_labels:
            return None
        t = self._clock()
        self._hz_cool = [(b, until) for b, until in self._hz_cool if until > t]
        with self._motion_lock:
            cands = [c for c in self._hz_cands if not any(_overlap(c.box, b) >= 0.3 for b, _ in self._hz_cool)]
        # a "smoke" block that is mostly a tracked object (a person walking off) is not smoke
        cands = [c for c in cands if c.kind != "smoke" or _covered(c.box, self._kept) < 0.6]
        if not cands or (self._hz_last is not None and t - self._hz_last < self.hazard_check_s):
            return None
        return max(cands, key=lambda c: c.score)

    def _context_region(self, box: tuple) -> Optional[tuple[int, int, int, int]]:
        """Square crop around the candidate; None = use the full frame.
        hazard_crop_px > 0: a FIXED native size (768 px measured best: in a 416 model the smoke
        then has about the scale it has when a 768 model sees the whole 1080p frame — 6/8 smoke
        frames vs 0/8 for 416 full frame); capped at the frame's short side; full frame when the
        candidate does not fit. 0: ≥ hazard_context × the candidate, ≥ hazard_min_side × height."""
        fh, fw = self.frame_shape
        bw, bh = box[2] - box[0], box[3] - box[1]
        if self.hazard_crop_px > 0:
            side = min(self.hazard_crop_px, fh, fw)
            if max(bw, bh) * 1.2 > side:
                return None
        else:
            side = int(max(self.hazard_context * max(bw, bh), self.hazard_min_side * fh))
            if side * side >= 0.5 * fh * fw or side >= min(fh, fw):
                return None
        x1, x2 = _span((box[0] + box[2]) // 2, (box[0] + box[2]) // 2, side, fw)
        y1, y2 = _span((box[1] + box[3]) // 2, (box[1] + box[3]) // 2, side, fh)
        return x1, y1, x2, y2

    def _run_hazard(self, image: np.ndarray, cand) -> list[dict[str, Any]]:
        """Check a candidate on its context crop; `cand` None = confirm the last hit on the same
        crop (a 416 full frame would not see the smoke the crop showed)."""
        region = self._context_region(cand.box) if cand is not None else self._hz_focus[0]
        if not self._turn(CONFIRM):
            return []
        try:
            self.stats["hazard_checks"] += 1
            self.stats["inferences"] += 1
            if region is None:
                self.stats["full_frame"] += 1
                dets = self._filter(self._infer(image))
            else:
                x1, y1, x2, y2 = region
                crop = self._filter(self._infer(self._cap_zoom(image[y1:y2, x1:x2])))
                dets = [{**d, "box": (d["box"][0] + x1, d["box"][1] + y1, d["box"][2] + x1, d["box"][3] + y1)}
                        for d in crop if (d["box"][0] + d["box"][2]) / 2 < x2 - x1 and (d["box"][1] + d["box"][3]) / 2 < y2 - y1]
        finally:
            self._release()
        t = self._clock()
        self._hz_last = t
        hit = any(d["label"] in self.full_frame_labels for d in dets)
        if cand is None:
            if self._confirm_left <= 1:
                self._hz_focus = None                # burst over (decremented in _finish)
        elif hit:
            self.stats["hazard_hits"] += 1
            self._hz_focus = (region, cand.box) if self._confirm_left == 0 else self._hz_focus
        else:                                        # nothing there: leave this spot alone a while
            self._hz_cool.append((cand.box, t + self.hazard_cooldown_s))
            self.stats["hazard_cooled"] += 1
        return self._finish(dets, full=True)       # a context crop is trusted like a full frame

    def _turn(self, priority: int) -> bool:
        if self.gate is None or self.gate.acquire(priority):
            return True
        self.stats["denied"] += 1
        return False

    def _release(self) -> None:
        if self.gate is not None:
            self.gate.release()

    def _cap_zoom(self, mosaic: np.ndarray) -> np.ndarray:
        """Pad a small mosaic so the detector's letterbox zooms it in at most `max_zoom`."""
        return cap_zoom(mosaic, self.canvas_size, self.max_zoom)

    def _check_scene(self) -> None:
        """Forget learned sizes when the background SETTLES on a different view (PTZ moved).
        Lightning, an IR switch or a shake recalibrate too, but the background comes back the same.
        Only a settled background (SCENE_SETTLE_FRAMES without calibration) is compared: right
        after a recalibration — and at startup — it is still converging."""
        if self.det_filter is None:
            return
        if self.motion.calibrating:
            self._settled = 0
            self._recalibrated = True
            return
        self._settled += 1
        if self._settled < SCENE_SETTLE_FRAMES:
            return
        bg = self.motion.avg_frame
        if (self._recalibrated and self._scene_ref is not None
                and float(np.mean(np.abs(bg - self._scene_ref))) > SCENE_CHANGE_DIFF):
            self.det_filter.reset_scene()
            if self.noise is not None:
                self.noise.reset()
            self.stats["scene_changes"] += 1
        self._recalibrated = False
        self._scene_ref = bg.copy()

    def _with_context(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Grow a crop that is smaller than a person standing there (learned perspective) to a
        PERSON-SHAPED box: min_crop_context × the expected height, crop_context_aspect × that
        wide. A near worker whose legs alone move is then seen whole, while a far worker's crop
        stays tight (its motion box is already person-sized)."""
        if self.det_filter is None or self.min_crop_context <= 0:
            return box
        expected = self.det_filter.expected_height("person", box[3])
        if expected is None:
            return box
        fh, fw = self.frame_shape
        h = int(round(self.min_crop_context * expected))
        return _grow_to(box, int(round(self.crop_context_aspect * h)), fw, fh, side_h=h)

    def _shrink(self, shape: tuple[int, int]) -> float:
        """Letterbox scale the detector applies to an image of this shape (capped at 1)."""
        return min(1.0, self.canvas_size / max(shape))

    def _full(self, image: np.ndarray) -> list[dict[str, Any]]:
        if self._now is not None and self._last_full is not None:
            gap = self._now - self._last_full                  # how stale the fire / smoke scan got
            self.stats["full_gap_max_s"] = round(max(self.stats["full_gap_max_s"], gap), 2)
            self.stats["full_gap_sum_s"] = round(self.stats["full_gap_sum_s"] + gap, 2)
        self._last_full = self._now
        self.stats["full_frame"] += 1
        self.stats["inferences"] += 1
        return self._filter(self._infer(image))

    def _filter(self, dets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return dets if self.classes is None else [d for d in dets if d["label"] in self.classes]

    def _finish(self, dets: list[dict[str, Any]], full: bool = True) -> list[dict[str, Any]]:
        """Fire / smoke bookkeeping, then the filter; this inference's candidates replace the
        kept boxes they overlap."""
        dets = self._full_frame_labels(dets, full)
        out, keep = self.det_filter.update(dets) if self.det_filter is not None else (dets, dets)
        # a "person" taller than kept_max_aspect × its width (along a pole) is not re-inspected:
        # it would stretch a crop over the whole frame height
        fresh = [tuple(int(v) for v in d["box"]) for d in keep if d["label"] not in PERSON_LIKE
                 or self.kept_max_aspect <= 0
                 or (d["box"][3] - d["box"][1]) <= self.kept_max_aspect * max(1.0, d["box"][2] - d["box"][0])]
        self._kept = {b: n - 1 for b, n in self._kept.items()
                      if n > 1 and all(iou(b, f) <= KEEP_REPLACE_IOU for f in fresh)}
        self._kept.update({f: self.keep_frames for f in fresh})
        with self._motion_lock:                             # a worker standing still is not relearned
            self.motion.protect(list(self._kept))
        if self.heads is not None:
            self._remember_persons([tuple(d["box"]) for d in out if d["label"] in PERSON_LIKE])
        if self.noise is not None and out:                  # CONFIRMED objects: their cells are not nuisance
            with self._motion_lock:                         # (a false box on a swaying tree must not count)
                self.noise.explain([d["box"] for d in out], self._clock())
        return list(out)

    def _remember_persons(self, boxes: list[tuple]) -> None:
        t = self._clock()
        old = [(b, seen) for b, seen in self._persons
               if t - seen <= PERSON_STALE_S and all(iou(b, n) < KEEP_REPLACE_IOU for n in boxes)]
        self._persons = old + [(b, t) for b in boxes]

    def _full_frame_labels(self, dets: list[dict[str, Any]], full: bool) -> list[dict[str, Any]]:
        if not self.full_frame_labels:
            return dets
        hit = any(d["label"] in self.full_frame_labels for d in dets)
        if full:
            self._suspect = False
            if self._confirm_left > 0:
                self._confirm_left -= 1
            elif hit:
                self._confirm_left = self.confirm_full
            return dets
        if hit:
            self._suspect = True
            self.stats["suspects"] += 1
        return [d for d in dets if d["label"] not in self.full_frame_labels]


def _gray(image: np.ndarray) -> np.ndarray:
    """BGR, BGRx (camera caps, straight from the mapped buffer) or already grey."""
    if image.ndim == 2:
        return image
    if not image.flags["C_CONTIGUOUS"]:                   # a mapped buffer may have padded rows
        image = np.ascontiguousarray(image)
    return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY)


def _cut_by(box: Sequence[float], region: Sequence[int], fw: int, fh: int) -> bool:
    """Box touches a side of its crop that is not the frame border."""
    e = CUT_EDGE_PX
    return ((box[0] - region[0] <= e and region[0] > 0) or (region[2] - box[2] <= e and region[2] < fw)
            or (box[1] - region[1] <= e and region[1] > 0) or (region[3] - box[3] <= e and region[3] < fh))


def _covered(box: Sequence[float], others) -> float:
    """Fraction of `box` covered by the union-free sum of `others` (capped at 1)."""
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    inter = sum(max(0.0, min(box[2], o[2]) - max(box[0], o[0])) * max(0.0, min(box[3], o[3]) - max(box[1], o[1]))
                for o in others)
    return min(1.0, inter / area)


def _overlap(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over the smaller box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) or 1.0
    return ix * iy / small


def _grow_to(box: tuple[int, int, int, int], side: int, fw: int, fh: int,
             side_h: Optional[int] = None) -> tuple[int, int, int, int]:
    """Grow box symmetrically so it is ≥ `side` px wide and ≥ `side_h` (default `side`) px high,
    shifted to stay inside the frame."""
    x1, x2 = _span(box[0], box[2], side, fw)
    y1, y2 = _span(box[1], box[3], side if side_h is None else side_h, fh)
    return x1, y1, x2, y2


def _span(lo: int, hi: int, side: int, limit: int) -> tuple[int, int]:
    if hi - lo >= side:
        return lo, hi
    side = min(side, limit)
    lo = max(0, (lo + hi - side) // 2)
    hi = min(limit, lo + side)
    return hi - side, hi
