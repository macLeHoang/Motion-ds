"""Backend-neutral inference metadata handed to processors.

Processors only see these types, never pyds / gstgva objects, so their logic can be unit-tested
with fakes and reused if the backend changes again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

import numpy as np


@dataclass(frozen=True, eq=False)
class Tensor:
    """One inference result attached to a detection (classification label or raw output)."""
    name: str
    label: str = ""
    label_id: int = -1
    confidence: float = 0.0
    data: Optional[np.ndarray] = None
    layer_name: str = ""


@dataclass(frozen=True, eq=False)
class Detection:
    """Bounding box in frame pixels (x, y = top-left)."""
    x: float
    y: float
    w: float
    h: float
    label: str = ""
    label_id: int = -1
    confidence: float = 0.0
    object_id: Optional[int] = None
    tensors: tuple[Tensor, ...] = ()
    index: int = -1                    # position in the frame's region list (for annotations)
    keypoints: Optional[np.ndarray] = None   # [K, 2] frame pixels (SCRFD landmarks, plate corners)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h)

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def box(self) -> tuple[float, float, float, float]:
        """(x1, y1, x2, y2)"""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def contains_center_of(self, other: "Detection") -> bool:
        cx, cy = other.center
        x1, y1, x2, y2 = self.box
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def tensor(self, name: str) -> Optional[Tensor]:
        return next((t for t in self.tensors if t.name == name), None)

    def first_tensor_with_data(self) -> Optional[Tensor]:
        return next((t for t in self.tensors if t.data is not None and t.data.size > 0), None)


@dataclass(frozen=True)
class StreamContext:
    """Which camera/branch a frame belongs to (DL Streamer runs probes per stream, not per batch)."""
    camera_id: str
    branch: str
    source_id: int
    frame_num: int
    pts: int
    camera_params: Mapping[str, Any] = field(default_factory=dict)


class FrameView(Protocol):
    width: int
    height: int

    def detections(self) -> list[Detection]: ...

    def image_bgr(self) -> Optional[np.ndarray]: ...

    def set_label(self, detection: Detection, text: str) -> None: ...
