"""Edit the face gallery file (features_json) for the /api/faces endpoints.

One person may hold several faces (front, profile, glasses, mask…); each has an id unique within
that person. Every change rewrites the file atomically; recognition branches notice the new file
and reload it (FaceRecognitionProcessor), so no restart is needed. File format: dls/core/face.py
FaceDatabase. Pure: embeddings come in, no models here.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Optional, Sequence

import numpy as np

from dls.core.face import face_ids_of, faces_of, l2_normalize, write_json_atomic

MAX_FACES_PER_PERSON = 50
MAX_NAME_LEN = 64


class FaceGallery:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ read
    def _read(self) -> dict[str, dict[str, Any]]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"{self.path}: không phải object JSON {{tên: {{feature: ...}}}}")
        return data

    def list(self) -> dict[str, Any]:
        with self._lock:
            data = self._read()
        people = {}
        dim = 0
        for name, info in data.items():
            faces = faces_of(info)
            if not faces:
                continue
            dim = dim or len(faces[0])
            people[name] = {"faces": face_ids_of(info, len(faces)), "avatar": info.get("avatar")}
        return {"people": people, "count": len(people), "faces": sum(len(p["faces"]) for p in people.values()),
                "dim": dim, "file": self.path}

    # ------------------------------------------------------------------ write
    def add_faces(self, name: str, vectors: Sequence[Sequence[float]], avatar: Optional[str] = None) -> list[str]:
        """Add faces to `name` (created if new). Returns the new face ids."""
        name = validate_name(name)
        new = [l2_normalize(np.asarray(v, dtype=np.float32)) for v in vectors]
        if not new:
            raise ValueError("Cần ít nhất 1 khuôn mặt")
        if any(not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0 for v in new):
            raise ValueError("Vector khuôn mặt không hợp lệ (NaN / toàn 0)")
        if len({v.shape for v in new}) > 1:
            raise ValueError("Các vector khuôn mặt khác số chiều")
        with self._lock:
            data = self._read()
            dim = next((len(faces_of(i)[0]) for i in data.values() if faces_of(i)), None)
            if dim is not None and new[0].shape[0] != dim:
                raise ValueError(f"Vector {new[0].shape[0]} chiều, gallery đang dùng {dim} chiều (model khác?)")
            info = data.get(name) or {}
            faces = faces_of(info)
            ids = face_ids_of(info, len(faces))
            if len(faces) + len(new) > MAX_FACES_PER_PERSON:
                raise ValueError(f"'{name}' đã có {len(faces)} khuôn mặt; tối đa {MAX_FACES_PER_PERSON}")
            next_id = max((int(i) for i in ids if i.isdigit()), default=0) + 1
            added = [str(next_id + k) for k in range(len(new))]
            faces += [v.tolist() for v in new]
            ids += added
            entry: dict[str, Any] = {"feature": faces[0] if len(faces) == 1 else faces, "face_ids": ids}
            avatar = avatar if avatar is not None else info.get("avatar")
            if avatar is not None:
                entry["avatar"] = avatar
            data[name] = entry
            write_json_atomic(self.path, data)
        return added

    def remove_person(self, name: str) -> bool:
        with self._lock:
            data = self._read()
            if name not in data:
                return False
            del data[name]
            write_json_atomic(self.path, data)
            return True

    def remove_face(self, name: str, face_id: str) -> bool:
        """Remove one face; the person goes too when it was the last one."""
        with self._lock:
            data = self._read()
            info = data.get(name)
            if info is None:
                return False
            faces = faces_of(info)
            ids = face_ids_of(info, len(faces))
            if face_id not in ids:
                return False
            k = ids.index(face_id)
            del faces[k], ids[k]
            if not faces:
                del data[name]
            else:
                data[name] = {**{x: y for x, y in info.items() if x not in ("feature", "face_ids")},
                              "feature": faces[0] if len(faces) == 1 else faces, "face_ids": ids}
            write_json_atomic(self.path, data)
            return True


def validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name or len(name) > MAX_NAME_LEN or any(c in name for c in "/\\\n\r\t"):
        raise ValueError(f"Tên không hợp lệ (1–{MAX_NAME_LEN} ký tự, không có / \\ hay xuống dòng)")
    return name
