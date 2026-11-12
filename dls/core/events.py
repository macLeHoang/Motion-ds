"""Event bus (ring buffer + optional JSONL + subscribers) and event de-duplication."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from collections import Counter, deque
from typing import Any, Callable, Hashable, Optional

import numpy as np

logger = logging.getLogger(__name__)

Subscriber = Callable[[dict], None]


def _iso_local(ts: float) -> str:
    """Local time with its UTC offset (2026-10-05T14:03:12+07:00): unambiguous for the receiver."""
    t = time.localtime(ts)
    off = time.strftime("%z", t)
    return time.strftime("%Y-%m-%dT%H:%M:%S", t) + (f"{off[:3]}:{off[3:]}" if off else "")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class EventBus:
    """Thread-safe store of recent events. Processors publish; the API reads; subscribers (the
    webhook) get every event. `seq` starts from the epoch in ms, so it keeps increasing across
    restarts and a client polling `since_seq` never re-reads or misses after one. Each event also
    gets a unique `event_id` (receivers de-duplicate webhook retries with it). The optional JSONL
    archive rotates at `jsonl_max_bytes` keeping `jsonl_backups` old files."""

    def __init__(self, maxlen: int = 1000, jsonl_path: Optional[str] = None,
                 jsonl_max_bytes: int = 100 * 1024 * 1024, jsonl_backups: int = 5):
        self._events: deque[dict] = deque(maxlen=maxlen)
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._seq = time.time_ns() // 1_000_000
        self._subscribers: list[Subscriber] = []
        self._jsonl = None
        self._jsonl_path = jsonl_path
        self._jsonl_max, self._jsonl_backups = int(jsonl_max_bytes), int(jsonl_backups)
        if jsonl_path:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            self._jsonl = open(jsonl_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def subscribe(self, callback: Subscriber) -> None:
        self._subscribers.append(callback)

    def publish(self, event: dict) -> dict:
        now = time.time()
        with self._lock:
            self._seq += 1
            stored = {**event, "seq": self._seq, "event_id": uuid.uuid4().hex, "ts": now,
                      "time": _iso_local(now)}
            self._events.append(stored)
            self._counts[stored.get("type", "unknown")] += 1
            if self._jsonl is not None:
                try:
                    self._jsonl.write(json.dumps(stored, default=_json_default) + "\n")
                    if self._jsonl.tell() >= self._jsonl_max:
                        self._rotate()
                except (OSError, ValueError):
                    logger.exception("[EVENT] JSONL write failed")
        logger.info("[EVENT] %s cam=%s %s", stored.get("type"), stored.get("camera_id"),
                    {k: v for k, v in stored.items() if k not in ("type", "camera_id", "avatar")})
        for callback in list(self._subscribers):
            try:
                callback(stored)
            except Exception:
                logger.exception("[EVENT] subscriber failed")
        return stored

    def _rotate(self) -> None:
        """events.jsonl → .1 → .2 … (caller holds the lock)."""
        self._jsonl.close()
        path = self._jsonl_path
        for i in range(self._jsonl_backups - 1, 0, -1):
            if os.path.exists(f"{path}.{i}"):
                os.replace(f"{path}.{i}", f"{path}.{i + 1}")
        if self._jsonl_backups > 0:
            os.replace(path, f"{path}.1")
        else:
            os.remove(path)
        self._jsonl = open(path, "a", encoding="utf-8", buffering=1)

    def query(self, event_type: Optional[str] = None, camera_id: Optional[str] = None,
              branch: Optional[str] = None, since_seq: int = 0, limit: int = 100) -> list[dict]:
        with self._lock:
            items = [
                e for e in self._events
                if e["seq"] > since_seq
                and (event_type is None or e.get("type") == event_type)
                and (camera_id is None or e.get("camera_id") == camera_id)
                and (branch is None or e.get("branch") == branch)
            ]
        return items[-limit:] if limit > 0 else items

    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def close(self) -> None:
        with self._lock:
            if self._jsonl is not None:
                self._jsonl.close()
                self._jsonl = None


class EventDeduper:
    """Suppress repeated events for the same key while it keeps being seen.

    A key fires again only after it has not been seen for `ttl_s` seconds.
    """

    def __init__(self, ttl_s: float = 30.0):
        self._ttl = ttl_s
        self._last_seen: dict[Hashable, float] = {}
        self._lock = threading.Lock()

    def add(self, key: Hashable, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            last = self._last_seen.get(key)
            self._last_seen[key] = now
            return last is None or now - last > self._ttl

    def prune(self, now: Optional[float] = None) -> list[Hashable]:
        now = time.monotonic() if now is None else now
        with self._lock:
            stale = [k for k, t in self._last_seen.items() if now - t > self._ttl]
            for k in stale:
                del self._last_seen[k]
        return stale

    def drop_where(self, predicate: Callable[[Hashable], bool]) -> int:
        """Forget matching keys (e.g. all keys of a removed camera: tracker ids restart at 1)."""
        with self._lock:
            keys = [k for k in self._last_seen if predicate(k)]
            for k in keys:
                del self._last_seen[k]
        return len(keys)

    def __len__(self) -> int:
        return len(self._last_seen)
