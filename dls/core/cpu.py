"""CPU budget: how many cores the pipeline may use, and how big inference thread pools may be.

Two complementary limits (see dlstreamer-edge-reference INF-3 / CPU governance):
- `cpuset` (affinity): a hard limit applied to the whole process with sched_setaffinity — every
  GStreamer, OpenVINO/TBB and Python thread inherits it.
- `max_cores`: a budget used to size OpenVINO thread pools. Left unset, TBB grows its pool to all
  host cores regardless of quota → threads pile up and get throttled (load1 ≫ CPU%). For a hard
  fractional quota use the container limit as well (docker compose `cpus`, env DLS_CPUS).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def parse_cpuset(spec: str) -> Optional[set[int]]:
    """'0-2,5' → {0, 1, 2, 5}; '' → None (no affinity)."""
    spec = (spec or "").strip()
    if not spec:
        return None
    cores: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"cpuset không hợp lệ: {spec!r}")
        if "-" in part:
            lo, _, hi = part.partition("-")
            if not lo.isdigit() or not hi.isdigit() or int(lo) > int(hi):
                raise ValueError(f"cpuset không hợp lệ: {spec!r}")
            cores.update(range(int(lo), int(hi) + 1))
        elif part.isdigit():
            cores.add(int(part))
        else:
            raise ValueError(f"cpuset không hợp lệ: {spec!r}")
    return cores


@dataclass(frozen=True)
class CpuBudget:
    cores: float                       # usable cores (affinity size, further capped by max_cores)
    affinity: Optional[set[int]]       # cores to pin the process to (None = all)
    threads: int                       # total inference threads that fit the budget

    @classmethod
    def resolve(cls, max_cores: float, cpuset: str, available: Optional[int] = None) -> "CpuBudget":
        available = available or os.cpu_count() or 1
        affinity = parse_cpuset(cpuset)
        if affinity is not None and max(affinity) >= available:
            raise ValueError(f"cpuset {cpuset!r} chứa core không tồn tại (máy có {available} core)")
        cores = float(len(affinity)) if affinity else float(available)
        if max_cores and max_cores > 0:
            cores = min(cores, float(max_cores))
        return cls(cores=cores, affinity=affinity, threads=max(1, round(cores)))

    def apply_affinity(self) -> None:
        if self.affinity:
            os.sched_setaffinity(0, self.affinity)


def inference_threads(budget: CpuBudget, engines: int) -> int:
    """Threads per inference engine so that all engines together fit the budget."""
    if engines <= 0:
        return budget.threads
    return max(1, budget.threads // engines)


# Process-wide inference thread count (set once at startup by dls.main; 0 = OpenVINO default).
_inference_threads = 0


def set_inference_threads(n: int) -> None:
    global _inference_threads
    _inference_threads = max(0, int(n))


def get_inference_threads() -> int:
    return _inference_threads


# Concurrent inferences per device for the scheduled detectors (pydetect yolo_motion). On a small
# CPU one inference at a time is fastest (two concurrent ones: 0.45 vs 0.71 inf/s on the i3); a big
# CPU needs several OpenVINO streams to use its cores.
THREADS_PER_SLOT = 7
_infer_slots = 1


def auto_infer_slots(threads: int) -> int:
    return 1 if threads <= 8 else max(1, threads // THREADS_PER_SLOT)


def set_infer_slots(n: int) -> None:
    global _infer_slots
    _infer_slots = max(1, int(n))


def get_infer_slots() -> int:
    return _infer_slots


def compile_config(extra: Optional[dict] = None) -> dict:
    """pydetect compile_model() config: LATENCY with one slot, else THROUGHPUT with one stream per
    slot (OpenVINO splits INFERENCE_NUM_THREADS between the streams)."""
    cfg = dict(extra or {})
    if _infer_slots > 1:
        cfg.update({"PERFORMANCE_HINT": "THROUGHPUT", "NUM_STREAMS": str(_infer_slots)})
    else:
        cfg.setdefault("PERFORMANCE_HINT", "LATENCY")
    return openvino_config(cfg)


def openvino_config(extra: Optional[dict] = None) -> dict:
    """compile_model() config honouring the CPU budget."""
    cfg = dict(extra or {})
    if _inference_threads > 0:
        cfg.setdefault("INFERENCE_NUM_THREADS", _inference_threads)
    return cfg


def ie_config_property(existing: Optional[str] = None) -> Optional[str]:
    """gva* `ie-config` value with INFERENCE_NUM_THREADS added (keeps an explicit setting)."""
    if _inference_threads <= 0 or (existing and "INFERENCE_NUM_THREADS" in existing):
        return existing
    item = f"INFERENCE_NUM_THREADS={_inference_threads}"
    return f"{existing},{item}" if existing else item
