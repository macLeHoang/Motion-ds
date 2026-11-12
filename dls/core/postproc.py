"""Python decoders for models whose DeepStream parsers were custom C++ libraries.

Faithful ports of:
- data/face/models/scrfd640/custom_parse_7_1/nvdsparsebbox.cpp (SCRFD: 2 anchors/cell, strides
  8/16/32, distance→box, 5 landmarks, NMS 0.4, letterbox padded right/bottom)
- DeepStream-Yolo-Pose nvdsparsepose_Yolo.cpp (rows [x1,y1,x2,y2,score,kx,ky,...], NMS 0.45,
  symmetric letterbox)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import cv2
import numpy as np

SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS_PER_CELL = 2


@dataclass(frozen=True)
class Letterbox:
    """frame = (net - pad) / scale"""
    scale: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0

    def to_frame(self, points: np.ndarray) -> np.ndarray:
        out = np.asarray(points, dtype=np.float32).copy()
        out[..., 0] = (out[..., 0] - self.pad_x) / self.scale
        out[..., 1] = (out[..., 1] - self.pad_y) / self.scale
        return out


@dataclass(frozen=True, eq=False)
class RawDetection:
    box: np.ndarray            # [x1, y1, x2, y2] in frame pixels
    score: float
    keypoints: np.ndarray      # [K, 2] in frame pixels
    label_id: int = 0


def letterbox(image: np.ndarray, net_w: int, net_h: int, symmetric: bool,
              pad_value: int = 0) -> tuple[np.ndarray, Letterbox]:
    h, w = image.shape[:2]
    scale = min(net_w / w, net_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pad_x = (net_w - new_w) / 2 if symmetric else 0.0
    pad_y = (net_h - new_h) / 2 if symmetric else 0.0
    out = np.full((net_h, net_w, image.shape[2]), pad_value, dtype=np.uint8)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0, y0 = int(pad_x), int(pad_y)
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out, Letterbox(scale=scale, pad_x=float(x0), pad_y=float(y0))


def to_nchw(image_bgr: np.ndarray, mean: float, scale: float, rgb: bool = True) -> np.ndarray:
    img = image_bgr[:, :, ::-1] if rgb else image_bgr
    return ((img.astype(np.float32) - mean) / scale).transpose(2, 0, 1)[None]


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS on [x1,y1,x2,y2]; returns kept indices ordered by score."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = (boxes[:, i] for i in range(4))
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        overlap = np.where(union > 0, inter / np.maximum(union, 1e-9), 0)
        order = order[1:][overlap <= iou]
    return np.asarray(keep, dtype=np.int64)


@lru_cache(maxsize=16)
def scrfd_anchor_centers(net_w: int, net_h: int, stride: int) -> np.ndarray:
    gx, gy = np.meshgrid(np.arange(net_w // stride), np.arange(net_h // stride))
    centers = np.stack([gx, gy], axis=-1).reshape(-1, 2).astype(np.float32) * stride
    return np.repeat(centers, SCRFD_ANCHORS_PER_CELL, axis=0)


def decode_scrfd(outputs: Sequence[np.ndarray], net_w: int, net_h: int, lb: Letterbox,
                 threshold: float = 0.7, iou: float = 0.4, max_det: int = 50) -> list[RawDetection]:
    """outputs = [score_8, score_16, score_32, bbox_8.., kps_8..] (model output order)."""
    boxes, scores, kpss = [], [], []
    for i, stride in enumerate(SCRFD_STRIDES):
        score = outputs[i].reshape(-1)
        keep = score > threshold
        if not keep.any():
            continue
        centers = scrfd_anchor_centers(net_w, net_h, stride)[keep]
        dist = outputs[i + 3].reshape(-1, 4)[keep] * stride
        kps = outputs[i + 6].reshape(-1, 10)[keep].reshape(-1, 5, 2) * stride
        boxes.append(np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], axis=1))
        kpss.append(kps + centers[:, None, :])
        scores.append(score[keep])
    if not boxes:
        return []
    boxes_np, scores_np, kps_np = np.concatenate(boxes), np.concatenate(scores), np.concatenate(kpss)
    kept = nms(boxes_np, scores_np, iou)[:max_det]
    return [RawDetection(lb.to_frame(boxes_np[k].reshape(2, 2)).reshape(4), float(scores_np[k]),
                         lb.to_frame(kps_np[k])) for k in kept]


def decode_yolo_pose(output: np.ndarray, lb: Letterbox, threshold: float = 0.25, iou: float = 0.45,
                     net_w: int = 640, net_h: int = 640, max_det: int = 100) -> list[RawDetection]:
    rows = output.reshape(-1, output.shape[-1])
    rows = rows[rows[:, 4] >= threshold]
    if rows.size == 0:
        return []
    boxes = rows[:, :4].copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, net_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, net_h)
    valid = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
    rows, boxes = rows[valid], boxes[valid]
    kept = nms(boxes, rows[:, 4], iou)[:max_det]
    out = []
    for k in kept:
        kps = rows[k, 5:].reshape(-1, 2).copy()
        kps[:, 0] = np.clip(kps[:, 0], 0, net_w)
        kps[:, 1] = np.clip(kps[:, 1], 0, net_h)
        out.append(RawDetection(lb.to_frame(boxes[k].reshape(2, 2)).reshape(4), float(rows[k, 4]),
                                lb.to_frame(kps)))
    return out


def decode_yolo(output: np.ndarray, lb: Letterbox, threshold: float = 0.25, iou: float = 0.45,
                max_det: int = 300, nms_groups: Optional[Sequence[Sequence[int]]] = None) -> list[RawDetection]:
    """Raw YOLOv8/11 head [1, 4+C, N] (cx, cy, w, h in net pixels, sigmoid class scores) → per-class NMS.
    `nms_groups`: classes suppressed TOGETHER (one head is either "no helmet" or "white helmet",
    never both); other classes keep per-class NMS (a person and a helmet overlap legitimately)."""
    out = output.reshape(output.shape[-2], output.shape[-1])
    scores = out[4:]
    label = scores.argmax(axis=0)
    conf = scores.max(axis=0)
    keep = conf >= threshold
    if not keep.any():
        return []
    cx, cy, w, h = out[:4, keep]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    conf, label = conf[keep], label[keep]
    group = label.copy()
    for g, members in enumerate(nms_groups or ()):
        group[np.isin(label, list(members))] = -1 - g          # shared NMS key for the group
    result: list[RawDetection] = []
    for key in np.unique(group):
        idx = np.nonzero(group == key)[0]
        for k in nms(boxes[idx], conf[idx], iou):
            i = idx[k]
            result.append(RawDetection(lb.to_frame(boxes[i].reshape(2, 2)).reshape(4), float(conf[i]),
                                       np.zeros((0, 2), np.float32), int(label[i])))
    result.sort(key=lambda d: -d.score)
    return result[:max_det]


def clip_detections(dets: Sequence[RawDetection], width: int, height: int,
                    min_size: float = 1.0) -> list[RawDetection]:
    """Clip boxes to the image; drop those left smaller than min_size (e.g. boxes decoded in the
    letterbox padding, which map outside the frame)."""
    out = []
    for d in dets:
        box = np.clip(d.box, [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
        if box[2] - box[0] >= min_size and box[3] - box[1] >= min_size:
            out.append(RawDetection(box, d.score, d.keypoints, d.label_id))
    return out
