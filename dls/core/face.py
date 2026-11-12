"""Face recognition logic: embedding database, identity confirmation by streak, per-camera tracks.

Ported from apps/face/processor.py (DeepStream) without the pyds/OSD parts.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

logger = logging.getLogger(__name__)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class FaceDatabase:
    """Read-only gallery: one or more L2-normalised embeddings ("faces") per person; a probe
    matches the person owning its nearest face (L2). Several faces per person (front, profile,
    glasses, mask…) let one person be recognised from more angles.

    File (features_json, the DeepStream format extended):
        {"<name>": {"feature": [f, ...] | [[f, ...], [f, ...]], "face_ids": ["1", "2"], "avatar": "...",
                    "display_name": "..."}}
    <name> is the person's key (the VMS user id); `display_name` (optional) is what people read.
    A flat `feature` is one face. The DeepStream backend reads only the first face of a list.
    """

    def __init__(self, names: tuple[str, ...], matrix: Optional[np.ndarray],
                 avatars: Mapping[str, str], owner: Optional[np.ndarray] = None,
                 face_ids: Optional[tuple[str, ...]] = None):
        self._names = names
        self._matrix = matrix
        self._avatars = dict(avatars)
        self._display: dict[str, str] = {}
        n = 0 if matrix is None else matrix.shape[0]
        self._owner = owner if owner is not None else np.arange(n)
        self._face_ids = face_ids if face_ids is not None else tuple("1" for _ in range(n))

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "FaceDatabase":
        if not os.path.exists(path):
            logger.warning("[FaceDB] features file not found: %s", path)
            return cls((), None, {})
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        db = cls.from_entries(data, source=path)
        logger.info("[FaceDB] loaded %d people (%d faces) from %s", len(db), db.face_count, path)
        return db

    @classmethod
    def from_entries(cls, data: Mapping[str, Any], source: str = "gallery") -> "FaceDatabase":
        names, vectors, owner, face_ids, avatars, display = [], [], [], [], {}, {}
        for name, info in data.items():
            faces = faces_of(info)
            if not faces:
                continue
            ids = face_ids_of(info, len(faces))
            names.append(name)
            for vec, fid in zip(faces, ids):
                vectors.append(l2_normalize(np.asarray(vec, dtype=np.float32)))
                owner.append(len(names) - 1)
                face_ids.append(fid)
            if "avatar" in info:
                avatars[name] = info["avatar"]
            if info.get("display_name"):
                display[name] = str(info["display_name"])
        if len({v.shape for v in vectors}) > 1:
            raise ValueError(f"[FaceDB] mixed embedding dimensions in {source}")
        matrix = np.vstack(vectors) if vectors else None
        db = cls(tuple(names), matrix, avatars, np.asarray(owner, dtype=np.int64), tuple(face_ids))
        db._display = display
        return db

    @classmethod
    def from_embeddings(cls, embeddings: Mapping[str, np.ndarray],
                        avatars: Optional[Mapping[str, str]] = None) -> "FaceDatabase":
        """name → one embedding (1-d) or several faces (2-d, one per row)."""
        data = {n: {"feature": np.atleast_2d(np.asarray(e, dtype=np.float32)).tolist()} for n, e in embeddings.items()}
        db = cls.from_entries(data)
        db._avatars = dict(avatars or {})
        return db

    def entries(self) -> dict[str, dict[str, Any]]:
        """The file form of this gallery (see the class docstring)."""
        out: dict[str, dict[str, Any]] = {}
        for p, name in enumerate(self._names):
            rows = [i for i in range(len(self._owner)) if self._owner[i] == p]
            feats = [self._matrix[i].tolist() for i in rows]
            entry: dict[str, Any] = {"feature": feats[0] if len(feats) == 1 else feats,
                                     "face_ids": [self._face_ids[i] for i in rows]}
            if name in self._avatars:
                entry["avatar"] = self._avatars[name]
            if name in self._display:
                entry["display_name"] = self._display[name]
            out[name] = entry
        return out

    def save(self, path: str) -> None:
        write_json_atomic(path, self.entries())

    # -- queries --------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def matrix(self) -> Optional[np.ndarray]:
        """Every face (one row each); `owner[i]` is the person of row i."""
        return self._matrix

    @property
    def owner(self) -> np.ndarray:
        return self._owner

    @property
    def face_count(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])

    @property
    def dim(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[1])

    def avatar(self, name: str) -> Optional[str]:
        return self._avatars.get(name)

    def display_name(self, name: str) -> str:
        return self._display.get(name, name)

    def __len__(self) -> int:
        return len(self._names)

    def match(self, embedding: np.ndarray) -> tuple[int, float]:
        """(person index, L2 distance to that person's nearest face); (-1, inf) if empty or
        dimension mismatch."""
        if self._matrix is None or embedding.shape[-1] != self.dim:
            return -1, float("inf")
        distances = np.linalg.norm(self._matrix - embedding.reshape(1, -1), axis=1)
        row = int(np.argmin(distances))
        return int(self._owner[row]), float(distances[row])


def faces_of(info: Mapping[str, Any]) -> list[list[float]]:
    """The faces of one gallery entry: a flat `feature` is one face, a list of lists several."""
    feat = info.get("feature") if isinstance(info, Mapping) else None
    if not feat:
        return []
    return [list(f) for f in feat] if isinstance(feat[0], (list, tuple)) else [list(feat)]


def face_ids_of(info: Mapping[str, Any], n: int) -> list[str]:
    ids = [str(i) for i in (info.get("face_ids") or [])]
    if len(ids) != n or len(set(ids)) != n:
        ids = [str(k + 1) for k in range(n)]
    return ids


def write_json_atomic(path: str, payload: Any) -> None:
    """Write then rename: a reader (a recognition branch reloading) never sees half a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


@dataclass
class TrackedFace:
    """Identity of one tracked face; confirmed after `min_streak` consecutive matches."""
    object_id: int
    l2_threshold: float = 1.0
    min_streak: int = 3

    label: Optional[str] = None
    score: float = 0.0
    last_seen: float = 0.0
    _person: Any = None
    _distances: list[float] = field(default_factory=list)

    def add_match(self, person: Any, distance: float) -> bool:
        """Record one match of `person` (a name; None / a negative index = no match). True when
        the identity is (re)confirmed. Names, not gallery indices: the gallery may be reloaded
        between two matches of one track."""
        if person is None or (isinstance(person, int) and person < 0) or distance > self.l2_threshold:
            return False
        if person != self._person:
            self._person, self._distances = person, [distance]
        else:
            self._distances.append(distance)
        return len(self._distances) >= self.min_streak

    def confirm(self, name: str) -> bool:
        """Set the label. True only on the first confirmation of this track."""
        is_new = self.label is None
        self.label = name
        self.score = sum(self._distances) / len(self._distances) if self._distances else 0.0
        self._person, self._distances = None, []
        return is_new


class FaceTrackRegistry:
    """TrackedFace per (camera_id, object_id); tracks idle for `max_idle_s` are pruned."""

    def __init__(self, params: Mapping[str, Any], max_idle_s: float = 10.0):
        self._threshold = float(params.get("l2_threshold", 1.0))
        self._min_streak = int(params.get("min_streak", 3))
        self._max_idle = max_idle_s
        self._tracks: dict[tuple[str, int], TrackedFace] = {}
        self._lock = threading.Lock()

    def get_or_create(self, camera_id: str, object_id: int, now: Optional[float] = None) -> TrackedFace:
        now = time.monotonic() if now is None else now
        key = (camera_id, object_id)
        with self._lock:
            track = self._tracks.get(key)
            if track is None:
                track = TrackedFace(object_id, self._threshold, self._min_streak)
                self._tracks[key] = track
            track.last_seen = now
            return track

    def prune(self, now: Optional[float] = None) -> list[tuple[str, int]]:
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = [k for k, t in self._tracks.items() if now - t.last_seen > self._max_idle]
            for k in stale:
                del self._tracks[k]
        return stale

    def drop_camera(self, camera_id: str) -> None:
        with self._lock:
            for key in [k for k in self._tracks if k[0] == camera_id]:
                del self._tracks[key]

    def stats(self) -> dict[str, int]:
        with self._lock:
            total = len(self._tracks)
            confirmed = sum(1 for t in self._tracks.values() if t.label)
        return {"total": total, "confirmed": confirmed, "pending": total - confirmed}
