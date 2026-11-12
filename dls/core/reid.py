"""Re-ID track-id remap: fold churned tracker ids into stable ids using appearance + position.

Measured on this project: DL Streamer's preview `deep-sort` gave MORE id switches than the IoU
tracker at low fps (12–16 vs 9 ids on the same clip). Same conclusion as
dlstreamer-edge-reference (ID-1 / ID-4): keep `zero-term-imageless` (accurate boxes, no lead) and
fold ids afterwards — "never publish a raw id as if it were stable".

Per camera:
- a tracker id seen before keeps its stable id (and adds its embedding to the gallery);
- a NEW tracker id is compared with tracks lost within `lost_ttl_s` (never with tracks visible in
  the same frame): position gate (distance ≤ max_speed × gap, as a fraction of the frame
  diagonal) + appearance gate (max cosine over the lost track's gallery ≥ cos_threshold);
- a new id without an embedding yet gets a provisional stable id and may merge ONCE, within
  `merge_window_s`, when its first embedding arrives.
Embeddings come from `gvainference` as a tensor on the ROI.

Model choice (measured, dls/tools/reid_eval.py on people-detection.mp4, gvainference ≡ reference
preprocessing): mars-small128 is useless here — two DIFFERENT people seen in the same frame had a
HIGHER cosine (median 0.945) than the same person 2 s apart (0.845), fp32 and int8 alike.
OMZ person-reidentification-retail-0277 separates them (same 0.48 / different 0.25, p95 0.41);
cos_threshold 0.55 folded only correct pairs, 0.4 already merged two different people.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Optional

import numpy as np

from dls.core.metadata import Detection

CONFIG_KEYS = ("cos_threshold", "lost_ttl_s", "max_speed", "base_radius", "gallery_size",
               "merge_window_s", "classes", "tensor_name")


def reid_config_errors(cfg: Any) -> list[str]:
    if not isinstance(cfg, Mapping):
        return ["reid: phải là object"]
    errors = [f"reid: khóa không hỗ trợ '{k}' (chỉ {list(CONFIG_KEYS)})" for k in cfg if k not in CONFIG_KEYS]
    cos = cfg.get("cos_threshold", 0.55)
    if not isinstance(cos, (int, float)) or not 0 < cos <= 1:
        errors.append("reid: cos_threshold phải trong (0, 1]")
    for key in ("lost_ttl_s", "max_speed", "base_radius", "merge_window_s"):
        v = cfg.get(key, 1)
        if not isinstance(v, (int, float)) or v < 0:
            errors.append(f"reid: {key} phải là số >= 0")
    if not isinstance(cfg.get("gallery_size", 1), int) or cfg.get("gallery_size", 1) < 1:
        errors.append("reid: gallery_size phải là số nguyên >= 1")
    return errors


@dataclass
class _Track:
    stable_id: int
    gallery: deque
    last_seen: float
    last_center: tuple[float, float]
    first_seen: float
    raw_ids: set = field(default_factory=set)


@dataclass
class _Raw:
    stable_id: int
    first_seen: float
    has_embedding: bool = False


class ReidRemap:
    def __init__(self, cos_threshold: float = 0.55, lost_ttl_s: float = 15.0, max_speed: float = 0.6,
                 base_radius: float = 0.1, gallery_size: int = 10, merge_window_s: float = 3.0,
                 classes: Optional[Iterable[str]] = None, tensor_name: Optional[str] = None):
        self.cos_threshold = cos_threshold
        self.lost_ttl_s = lost_ttl_s
        self.max_speed = max_speed            # fraction of the frame diagonal per second
        self.base_radius = base_radius        # always-allowed displacement (fraction of diagonal)
        self.gallery_size = gallery_size
        self.merge_window_s = merge_window_s
        self.classes = set(classes) if classes else None
        self.tensor_name = tensor_name
        self._lock = threading.Lock()
        self._tracks: dict[str, dict[int, _Track]] = {}     # camera → stable id → track
        self._raw: dict[str, dict[int, _Raw]] = {}          # camera → tracker id → mapping
        self._next: dict[str, int] = {}
        self._folded = 0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "ReidRemap":
        errors = reid_config_errors(cfg)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(**dict(cfg))

    # ------------------------------------------------------------------ public
    def update(self, camera_id: str, detections: Iterable[Detection], width: int, height: int,
               now: float) -> dict[int, int]:
        """Returns {tracker id: stable id} for this frame's tracked detections of the configured classes."""
        diag = math.hypot(width, height) or 1.0
        with self._lock:
            tracks = self._tracks.setdefault(camera_id, {})
            raws = self._raw.setdefault(camera_id, {})
            self._expire(camera_id, now)
            dets = [d for d in detections if d.object_id is not None
                    and (self.classes is None or d.label in self.classes)]
            visible: set[int] = {raws[d.object_id].stable_id for d in dets if d.object_id in raws}
            mapping: dict[int, int] = {}
            for d in dets:
                emb = self._embedding(d)
                center = (d.center[0] / diag, d.center[1] / diag)
                raw = raws.get(d.object_id)
                if raw is None:
                    sid = self._match(tracks, visible, emb, center, now) if emb is not None else None
                    if sid is None:
                        sid = self._new_track(camera_id, tracks, center, now)
                    else:
                        self._folded += 1
                    raw = raws[d.object_id] = _Raw(sid, now, emb is not None)
                    visible.add(sid)
                elif emb is not None and not raw.has_embedding:
                    raw.has_embedding = True
                    if now - raw.first_seen <= self.merge_window_s:
                        sid = self._match(tracks, visible - {raw.stable_id}, emb, center, now)
                        if sid is not None and sid != raw.stable_id:
                            tracks.pop(raw.stable_id, None)
                            raw.stable_id = sid
                            visible.add(sid)
                            self._folded += 1
                track = tracks[raw.stable_id]
                track.last_seen, track.last_center = now, center
                track.raw_ids.add(d.object_id)
                if emb is not None:
                    track.gallery.append(emb)
                mapping[d.object_id] = raw.stable_id
            return mapping

    def drop_camera(self, camera_id: str) -> None:
        with self._lock:
            for store in (self._tracks, self._raw, self._next):
                store.pop(camera_id, None)

    def stats(self) -> dict:
        with self._lock:
            return {"cameras": len(self._tracks), "folded": self._folded,
                    "stable_tracks": sum(len(t) for t in self._tracks.values())}

    def gallery_len(self, camera_id: str, stable_id: int) -> int:
        return len(self._tracks[camera_id][stable_id].gallery)

    # ------------------------------------------------------------------ internals
    def _embedding(self, d: Detection) -> Optional[np.ndarray]:
        t = d.tensor(self.tensor_name) if self.tensor_name else d.first_tensor_with_data()
        if t is None or t.data is None or t.data.size == 0:
            return None
        v = np.asarray(t.data, dtype=np.float32).ravel()
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else None

    def _new_track(self, camera_id: str, tracks: dict, center, now: float) -> int:
        sid = self._next.get(camera_id, 0) + 1
        self._next[camera_id] = sid
        tracks[sid] = _Track(sid, deque(maxlen=self.gallery_size), now, center, now)
        return sid

    def _match(self, tracks: dict, visible: set, emb: np.ndarray, center, now: float) -> Optional[int]:
        best, best_cos = None, self.cos_threshold
        for sid, tr in tracks.items():
            if sid in visible or not tr.gallery or now - tr.last_seen > self.lost_ttl_s:
                continue
            gap = max(0.0, now - tr.last_seen)
            dist = math.dist(center, tr.last_center)
            if dist > self.base_radius + self.max_speed * gap:
                continue
            cos = max(float(emb @ g) for g in tr.gallery)
            if cos >= best_cos:
                best, best_cos = sid, cos
        return best

    def _expire(self, camera_id: str, now: float) -> None:
        tracks, raws = self._tracks[camera_id], self._raw[camera_id]
        dead = [sid for sid, tr in tracks.items() if now - tr.last_seen > self.lost_ttl_s]
        for sid in dead:
            for rid in tracks[sid].raw_ids:
                raws.pop(rid, None)
            del tracks[sid]


class RemappedFrame:
    """FrameView whose detections carry stable (re-ID folded) ids instead of raw tracker ids."""

    def __init__(self, frame, mapping: Mapping[int, int]):
        self._frame = frame
        self._mapping = mapping
        self.width = getattr(frame, "width", 0)
        self.height = getattr(frame, "height", 0)

    def detections(self) -> list[Detection]:
        return [replace(d, object_id=self._mapping[d.object_id]) if d.object_id in self._mapping else d
                for d in self._frame.detections()]

    def image_bgr(self):
        return self._frame.image_bgr()

    def set_label(self, detection: Detection, text: str) -> None:
        self._frame.set_label(detection, text)
