"""Event pictures for a VMS: JPEGs written beside the event, path carried in event["images"].

Off unless configured (dls.main: DLS_SNAPSHOT_DIR). A processor passes `snapshot=Snapshot(...)`
to `emit`; only then is the frame copied and drawn (on the streaming thread, a few ms); the JPEG
is encoded and written on a background thread, so a slow disk never stalls a camera. A full
queue drops the picture (the event still goes out, without images).

Layout (same as the DeepStream VMS integration): <dir>/<kind>/<camera>_<tag>_<YYYYmmdd_HHMMSS_ffffff>.jpg
The path in the event is `public_root` + the part under <dir> when public_root is set (the VMS
reads the file on a shared disk where the container's path differs), else the absolute path.
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

QUEUE_MAX = 256
JPEG_QUALITY = 90
CLEAN_EVERY_S = 3600.0
GREEN, RED, WHITE, BLACK = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)

Box = Sequence[float]          # x, y, w, h (frame pixels)


@dataclass
class Snapshot:
    """What to picture: `frame` (a FrameView or a BGR array), boxes to draw, or `crop` (x, y, w, h)
    to cut out with `crop_pad` around it instead of the full frame."""
    frame: Any
    kind: str                                   # sub-directory: face, plate, fire_smoke, ppe, <task>…
    tag: str = ""                               # file-name part: user id, plate text, label_track…
    boxes: list[tuple[Box, str]] = field(default_factory=list)
    crop: Optional[Box] = None
    crop_pad: float = 0.1
    color: tuple[int, int, int] = RED


def _image_of(frame: Any) -> Optional[np.ndarray]:
    if isinstance(frame, np.ndarray):
        img = frame
    else:
        img = frame.image_bgr() if frame is not None else None
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return np.ascontiguousarray(img).copy()


def draw(img: np.ndarray, boxes: Sequence[tuple[Box, str]], color=RED) -> np.ndarray:
    for (x, y, w, h), label in boxes:
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(img, (x1 - 2, ty - th - 4), (x1 + tw + 2, ty + 4), BLACK, -1)
            cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    return img


def crop(img: np.ndarray, box: Box, pad: float) -> Optional[np.ndarray]:
    x, y, w, h = box
    fh, fw = img.shape[:2]
    x1, y1 = max(0, int(x - w * pad)), max(0, int(y - h * pad))
    x2, y2 = min(fw, int(x + w + w * pad)), min(fh, int(y + h + h * pad))
    return img[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else None


def _safe(text: str) -> str:
    return re.sub(r"[^\w.-]", "_", str(text))[:64] or "x"


class SnapshotWriter:
    def __init__(self, root: str, public_root: Optional[str] = None, keep_days: float = 7.0):
        self.root = os.path.abspath(root)
        self.public_root = public_root.rstrip("/") if public_root else None
        self.keep_days = float(keep_days)
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="Snapshots", daemon=True)
        self.stats = {"written": 0, "dropped": 0, "failed": 0, "cleaned": 0, "error": None}
        self._last_clean = 0.0
        self._last_cause: Optional[tuple] = None
        self._check_root()
        self._thread.start()

    def _check_root(self) -> None:
        """A directory Docker created as root (missing host path) fails every picture: say it once, at start."""
        try:
            os.makedirs(self.root, exist_ok=True)
            writable = os.access(self.root, os.W_OK | os.X_OK)
        except OSError:
            writable = False
        if not writable:
            self.stats["error"] = (f"{self.root} not writable by uid {os.getuid()}: "
                                   "event pictures are not saved (chown the host directory)")
            logger.error("[SNAP] %s", self.stats["error"])

    def submit_for(self, camera_id: str, snap: Snapshot) -> Optional[str]:
        """Draw now (the frame buffer is reused after the probe returns), write later.
        Returns the path the file will have, or None when there is nothing to picture."""
        img = _image_of(snap.frame)
        if img is None:
            return None
        if snap.crop is not None:
            img = crop(img, snap.crop, snap.crop_pad)
            if img is None:
                return None
        elif snap.boxes:
            img = draw(img, snap.boxes, snap.color)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = "_".join(p for p in (_safe(camera_id), _safe(snap.tag) if snap.tag else "", stamp) if p)
        return self._enqueue(img, os.path.join(_safe(snap.kind), name + ".jpg"))

    def _enqueue(self, img: np.ndarray, rel: str) -> Optional[str]:
        try:
            self._queue.put_nowait((img, rel))
        except queue.Full:
            self.stats["dropped"] += 1
            return None
        if self.public_root:
            return f"{self.public_root}/{rel}"
        return os.path.join(self.root, rel)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                img, rel = self._queue.get(timeout=0.5)
            except queue.Empty:
                self._maybe_clean()
                continue
            path = os.path.join(self.root, rel)
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    raise RuntimeError("imencode failed")
                tmp = path + ".part"
                with open(tmp, "wb") as f:
                    f.write(buf.tobytes())
                os.replace(tmp, path)            # the VMS never opens half a file
                self.stats["written"] += 1
            except Exception as e:
                self.stats["failed"] += 1
                cause = (type(e).__name__, getattr(e, "errno", None))
                self.stats["error"] = f"{type(e).__name__}: {e}"
                if cause != self._last_cause:             # each new cause once, not once per event
                    self._last_cause = cause
                    logger.warning("[SNAP] writing %s failed (%s); further identical failures are only counted",
                                   path, self.stats["error"])
                continue
            if self.stats["error"]:
                logger.info("[SNAP] pictures written again")
                self.stats["error"] = None
                self._last_cause = None
            self._maybe_clean()

    def _maybe_clean(self) -> None:
        """Delete pictures older than keep_days (hourly)."""
        now = time.time()
        if self.keep_days <= 0 or now - self._last_clean < CLEAN_EVERY_S:
            return
        self._last_clean = now
        limit = now - self.keep_days * 86400
        for dirpath, _, files in os.walk(self.root):
            for f in files:
                p = os.path.join(dirpath, f)
                try:
                    if f.endswith(".jpg") and os.path.getmtime(p) < limit:
                        os.remove(p)
                        self.stats["cleaned"] += 1
                except OSError:
                    pass

    def flush(self, timeout: float = 5.0) -> None:
        end = time.time() + timeout
        while not self._queue.empty() and time.time() < end:
            time.sleep(0.02)
        time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


_writer: Optional[SnapshotWriter] = None


def configure(writer: Optional[SnapshotWriter]) -> None:
    global _writer
    _writer = writer


def writer() -> Optional[SnapshotWriter]:
    return _writer
