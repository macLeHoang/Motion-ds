"""Camera URI helpers: scheme allow-list and credential masking for logs/API responses."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit

DEFAULT_SCHEMES = ("rtsp", "rtsps", "file")


def allowed_schemes() -> tuple[str, ...]:
    env = os.environ.get("DLS_URI_SCHEMES")
    return tuple(s.strip().lower() for s in env.split(",") if s.strip()) if env else DEFAULT_SCHEMES


def scheme_allowed(uri: str) -> bool:
    return urlsplit(uri).scheme.lower() in allowed_schemes()


def mask_uri(uri: str) -> str:
    """rtsp://user:pass@host/x → rtsp://user:***@host/x (password never leaves the process)."""
    parts = urlsplit(uri)
    if parts.password is None:
        return uri
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    netloc = f"{parts.username}:***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
