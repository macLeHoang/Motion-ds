"""Desired state on disk: what the REST API created (dynamic branches, cameras and their per-branch
params) survives a restart, a crash or a power cut. Pure file handling, no GStreamer.

Format (version 1):
    {"version": 1, "saved_at": <epoch>,
     "branches": [{"name", "config", "processor", "max_cameras"}],          # dynamic branches only
     "cameras":  [{"camera_id", "uri", "source": {...}, "branches": {branch: params}}]}
Branch order inside a camera matters: the first one is used to add the camera, the others link.
The file holds camera URIs WITH credentials, so it is written 0600, atomically (tmp + rename).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

VERSION = 1


def empty_state() -> dict:
    return {"version": VERSION, "branches": [], "cameras": []}


def save_state(path: str, state: dict) -> None:
    """Atomic write: a crash mid-write leaves the previous file intact."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    data = json.dumps({**state, "version": VERSION, "saved_at": time.time()}, ensure_ascii=False, indent=1)
    fd, tmp = tempfile.mkstemp(prefix=".state-", dir=directory)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_state(path: str) -> Optional[dict]:
    """None when there is no file. A corrupt file is moved aside (never silently overwritten)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        if not isinstance(state, dict) or state.get("version") != VERSION:
            raise ValueError(f"version {state.get('version') if isinstance(state, dict) else '?'}")
        state.setdefault("branches", [])
        state.setdefault("cameras", [])
        return state
    except (OSError, ValueError) as e:
        bad = f"{path}.bad-{int(time.time())}"
        os.replace(path, bad)
        logger.error("[STATE] %s unreadable (%s) → moved to %s, starting empty", path, e, bad)
        return None


def merge_pending(state: dict, pending: dict) -> dict:
    """Live state + entries that could not be restored yet (kept, retried, never dropped)."""
    names = {b["name"] for b in state["branches"]}
    cams = {c["camera_id"] for c in state["cameras"]}
    return {**state,
            "branches": state["branches"] + [b for n, b in pending.get("branches", {}).items() if n not in names],
            "cameras": state["cameras"] + [c for i, c in pending.get("cameras", {}).items() if i not in cams]}


def camera_entry(camera_id: str, uri: str, source: Optional[dict], branches: dict[str, Any]) -> dict:
    return {"camera_id": camera_id, "uri": uri, "source": dict(source or {}),
            "branches": {b: dict(p or {}) for b, p in branches.items()}}
