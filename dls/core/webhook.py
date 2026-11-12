"""Webhook delivery of events with a durable retry queue (SQLite, standard library only).

Every published event is inserted into the queue (on the publishing thread, ~1 ms) and only
removed after the receiver answered 2xx, so a network outage, a receiver restart, a crash or a
power cut does not lose alerts. A worker thread POSTs the oldest due events:
  - body: the event object (batch_size 1) or {"events": [...]} (batch_size > 1), JSON UTF-8;
  - headers: Content-Type, X-VMS-Event-Id (same value as event["event_id"]: the receiver
    de-duplicates retries), optional Authorization (`token`) and X-VMS-Signature =
    sha256=<hex HMAC of the body with `secret`>;
  - failure (connection error, timeout, non-2xx) → retry with exponential backoff
    (backoff_s … max_backoff_s), in order: newer events wait behind the oldest one;
    an event older than max_age_s is dropped (logged);
  - the queue is bounded (max_queue rows): when full the OLDEST events are dropped (logged).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

PostFn = Callable[[str, bytes, dict], int]        # (url, body, headers) → HTTP status


def http_post(url: str, body: bytes, headers: dict, timeout: float = 10.0) -> int:
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 — operator-set URL
            return int(resp.status)
    except urllib.error.HTTPError as e:
        return int(e.code)


class WebhookSender:
    def __init__(self, url: str, db_path: str, token: Optional[str] = None, secret: Optional[str] = None,
                 batch_size: int = 1, timeout_s: float = 10.0, backoff_s: float = 2.0, max_backoff_s: float = 300.0,
                 max_age_s: float = 86400.0, max_queue: int = 100_000, post: Optional[PostFn] = None,
                 clock: Callable[[], float] = time.time):
        if not url.startswith(("http://", "https://")):
            raise ValueError("DLS_WEBHOOK_URL phải bắt đầu bằng http:// hoặc https://")
        self.url, self.token, self.secret = url, token, secret
        self.batch_size, self.timeout_s = max(1, int(batch_size)), float(timeout_s)
        self.backoff_s, self.max_backoff_s, self.max_age_s = float(backoff_s), float(max_backoff_s), float(max_age_s)
        self.max_queue = int(max_queue)
        self._post = post or (lambda u, b, h: http_post(u, b, h, self.timeout_s))
        self._clock = clock
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS outbox (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                         "created REAL, payload TEXT, attempts INTEGER DEFAULT 0, next_at REAL)")
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = {"delivered": 0, "failed_attempts": 0, "dropped_full": 0, "dropped_old": 0,
                      "last_error": None, "last_delivered_at": None}

    # ------------------------------------------------------------------ producer side
    def enqueue(self, event: dict) -> None:
        """EventBus subscriber: durable insert, then wake the worker."""
        payload = json.dumps(event, ensure_ascii=False, default=str)
        now = self._clock()
        with self._lock:
            self._db.execute("INSERT INTO outbox (created, payload, next_at) VALUES (?, ?, ?)", (now, payload, now))
            excess = self.queued() - self.max_queue
            if excess > 0:
                self._db.execute("DELETE FROM outbox WHERE id IN (SELECT id FROM outbox ORDER BY id LIMIT ?)", (excess,))
                self.stats["dropped_full"] += excess
                logger.error("[WEBHOOK] queue full (%d): dropped %d oldest events", self.max_queue, excess)
        self._wake.set()

    def queued(self) -> int:
        return int(self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0])

    # ------------------------------------------------------------------ worker
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="Webhook", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)
        with self._lock:
            self._db.close()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                wait = self.deliver_due()
            except Exception:
                logger.exception("[WEBHOOK] delivery loop error")
                wait = self.backoff_s
            self._wake.wait(wait)
            self._wake.clear()

    def deliver_due(self) -> float:
        """Send due events; returns seconds until the next attempt is due (for the worker wait)."""
        now = self._clock()
        with self._lock:
            old = self._db.execute("DELETE FROM outbox WHERE created < ?", (now - self.max_age_s,)).rowcount
            # strict order: while the oldest event backs off, newer ones wait behind it
            rows = self._db.execute("SELECT id, payload, attempts, next_at FROM outbox ORDER BY id LIMIT ?",
                                    (self.batch_size,)).fetchall()
        if old:
            self.stats["dropped_old"] += old
            logger.error("[WEBHOOK] dropped %d events older than %.0f s (receiver unreachable)", old, self.max_age_s)
        if not rows:
            return 3600.0
        if rows[0][3] > now:
            return rows[0][3] - now
        events = [json.loads(r[1]) for r in rows]
        body = json.dumps(events[0] if self.batch_size == 1 else {"events": events}, ensure_ascii=False).encode()
        headers = {"Content-Type": "application/json; charset=utf-8",
                   "X-VMS-Event-Id": ",".join(str(e.get("event_id", "")) for e in events)}
        if self.token:
            headers["Authorization"] = self.token if " " in self.token else f"Bearer {self.token}"
        if self.secret:
            headers["X-VMS-Signature"] = "sha256=" + hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
        try:
            status = self._post(self.url, body, headers)
            error = None if 200 <= status < 300 else f"HTTP {status}"
        except Exception as e:                       # connection refused, timeout, DNS…
            error = f"{type(e).__name__}: {e}"
        ids = [r[0] for r in rows]
        with self._lock:
            if error is None:
                self._db.execute(f"DELETE FROM outbox WHERE id IN ({','.join('?' * len(ids))})", ids)
                self.stats["delivered"] += len(ids)
                self.stats["last_delivered_at"] = now
                return 0.0                           # maybe more due: loop again at once
            attempts = max(r[2] for r in rows) + 1
            delay = min(self.max_backoff_s, self.backoff_s * 2 ** (attempts - 1))
            self._db.execute(f"UPDATE outbox SET attempts = attempts + 1, next_at = ? WHERE id IN "
                             f"({','.join('?' * len(ids))})", [now + delay, *ids])
            queued = self.queued()
        self.stats["failed_attempts"] += 1
        self.stats["last_error"] = error
        logger.warning("[WEBHOOK] %s → retry in %.0f s (attempt %d, %d queued)", error, delay, attempts, queued)
        return delay

    def status(self) -> dict[str, Any]:
        with self._lock:
            queued = self.queued()
        return {"url": self.url, "queued": queued, "alive": self.is_alive(), **self.stats}
