"""Source health policy: reconnect backoff with parking, and a cheap TCP probe before a rebuild.

From dlstreamer-edge-reference: a dead camera must not cost a full pipeline rebuild every few
seconds forever — after `park_after_failures` consecutive failures it is parked (retried every
`park_s`), and a rebuild is only attempted when the RTSP host:port accepts a TCP connection.
"""

from __future__ import annotations

import socket
from urllib.parse import urlsplit

from dls.core.config import SourceSettings

DEFAULT_PORTS = {"rtsp": 554, "rtsps": 322}


def tcp_probe(uri: str, timeout_s: float = 2.0) -> bool:
    """True when the source's host:port accepts TCP (non-network sources always pass)."""
    parts = urlsplit(uri)
    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS or not parts.hostname:
        return True
    try:
        with socket.create_connection((parts.hostname, parts.port or DEFAULT_PORTS[scheme]), timeout=timeout_s):
            return True
    except OSError:
        return False


def next_backoff(failures: int, settings: SourceSettings) -> float:
    """Delay before reconnect attempt number `failures` (1-based): doubling, capped, then parked."""
    if failures >= settings.park_after_failures:
        return float(settings.park_s)
    return float(min(settings.reconnect_max_s, settings.reconnect_initial_s * 2 ** max(0, failures - 1)))
