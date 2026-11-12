"""License-plate geometry + PaddleOCR CTC decoding (ported from apps/plate, no CUDA/pyds)."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

DEFAULT_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"
TWO_LINE_MAX_ASPECT = 2.0       # width/height below this → square (2-line) plate
MIN_WARP_SIZE = 10


class CTCLabelDecode:
    """Greedy CTC decoder: argmax per step, merge repeats, drop blank (index 0)."""

    def __init__(self, character_dict_path: Optional[str], use_space_char: bool = False):
        if character_dict_path is None:
            chars = list(DEFAULT_ALPHABET)
        else:
            with open(character_dict_path, encoding="utf-8") as f:
                chars = [line.rstrip("\r\n") for line in f]
            if use_space_char:
                chars.append(" ")
        self.characters: tuple[str, ...] = ("blank", *chars)

    def __call__(self, preds: np.ndarray) -> list[tuple[str, float]]:
        idx = preds.argmax(axis=2)
        prob = preds.max(axis=2)
        results = []
        for b in range(idx.shape[0]):
            keep = np.ones(idx.shape[1], dtype=bool)
            keep[1:] = idx[b, 1:] != idx[b, :-1]
            keep &= idx[b] != 0
            text = "".join(self.characters[i] for i in idx[b][keep] if i < len(self.characters))
            confs = prob[b][keep]
            results.append((text, float(np.mean(confs)) if confs.size else 0.0))
        return results


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def warp_plate(frame: np.ndarray, points: np.ndarray) -> Optional[np.ndarray]:
    """Perspective-rectify the quadrilateral `points` into an axis-aligned image."""
    p = order_points(points)
    width = int(max(np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[2] - p[3])))
    height = int(max(np.linalg.norm(p[0] - p[3]), np.linalg.norm(p[1] - p[2])))
    if width < MIN_WARP_SIZE or height < MIN_WARP_SIZE:
        return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(p, dst)
    return cv2.warpPerspective(frame, matrix, (width, height))


def split_two_line(plate: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Square-ish plates are 2-line: return (top, bottom) halves of equal size, else None."""
    h, w = plate.shape[:2]
    if h == 0 or w == 0 or w / h >= TWO_LINE_MAX_ASPECT:
        return None
    mid = h // 2
    top, bottom = plate[:mid], plate[mid:]
    bottom = cv2.resize(bottom, (top.shape[1], top.shape[0]))
    return top, bottom


def crop_box(frame: np.ndarray, box: Sequence[float], pad_ratio: float = 0.05) -> Optional[np.ndarray]:
    """Crop (x, y, w, h) with relative padding, clipped to the frame. None if empty."""
    x, y, w, h = box
    px, py = w * pad_ratio, h * pad_ratio
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, int(round(x - px))), max(0, int(round(y - py)))
    x2, y2 = min(fw, int(round(x + w + px))), min(fh, int(round(y + h + py)))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def preprocess_ocr(image: np.ndarray, height: int = 48, width: int = 320) -> np.ndarray:
    """PaddleOCR rec preprocessing: keep aspect, scale to [-1, 1], right-pad to `width` (CHW)."""
    h, w = image.shape[:2]
    resize_w = min(width, max(1, int(np.ceil(height * w / max(h, 1)))))
    resized = cv2.resize(image, (resize_w, height)).astype(np.float32)
    resized = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
    out = np.zeros((3, height, width), dtype=np.float32)
    out[:, :, :resize_w] = resized
    return out
