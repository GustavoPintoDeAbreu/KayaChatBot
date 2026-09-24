"""Raspberry Pi gateway relay endpoint logic.

The gateway journals WhatsApp events and forwards them one at a time over
POST /whatsapp/relay, retrying until acknowledged.  This module provides
parsing, deduplication and backlog tracking without loading the model.
"""

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class RelayState:
    """Last relayed event applied, per gateway journal. Persisted atomically."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._journal_id: str = ""
        self._last_applied_seq: int = 0
        self._load()

    def is_duplicate(self, journal_id: str, seq: int) -> bool:
        """Return True when *journal_id/seq* has already been applied."""
        with self._lock:
            return (
                journal_id == self._journal_id
                and journal_id != ""
                and seq <= self._last_applied_seq
            )

    def mark_applied(self, journal_id: str, seq: int) -> None:
        """Record that *journal_id/seq* was applied successfully."""
        with self._lock:
            if self._journal_id != "" and journal_id != self._journal_id:
                logger.warning(
                    "gateway journal changed from %s to %s; "
                    "accepting its sequence from the start",
                    self._journal_id,
                    journal_id,
                )
                self._journal_id = journal_id
                self._last_applied_seq = 0
            self._journal_id = journal_id
            self._last_applied_seq = seq
            self._save()

    @property
    def journal_id(self) -> str:
        with self._lock:
            return self._journal_id

    @property
    def last_applied_seq(self) -> int:
        with self._lock:
            return self._last_applied_seq

    def _load(self) -> None:
        try:
            with open(self._path, "r") as fh:
                data = json.load(fh)
            self._journal_id = data.get("journal_id", "") or ""
            self._last_applied_seq = int(data.get("last_applied_seq", 0) or 0)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self._journal_id = ""
            self._last_applied_seq = 0

    def _save(self) -> None:
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = self._path + ".tmp"
        data = {
            "journal_id": self._journal_id,
            "last_applied_seq": self._last_applied_seq,
        }
        with open(tmp, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, self._path)


@dataclass
class RelayEnvelope:
    """Parsed relay payload with metadata from the gateway."""

    journal_id: str
    seq: int
    received_at: float
    replayed: bool
    deferred_reply: bool
    backlog_remaining: int
    event: Dict[str, Any]


def parse_envelope(body: Dict[str, Any]) -> RelayEnvelope:
    """Parse and validate a relay POST body, raising *ValueError* on bad input."""
    if not isinstance(body, dict):
        raise ValueError("body must be a dict")

    kaya_relay = body.get("kaya_relay")
    if not isinstance(kaya_relay, dict):
        raise ValueError("missing or invalid 'kaya_relay' section")

    event = body.get("event")
    if not isinstance(event, dict):
        raise ValueError("missing or invalid 'event' section")

    journal_id = kaya_relay.get("journal_id", "")
    if not journal_id:
        raise ValueError("journal_id must not be empty")

    seq = kaya_relay.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 1:
        raise ValueError(f"seq must be an int >= 1, got {seq!r}")

    received_at = float(kaya_relay.get("received_at", 0) or 0)
    replayed = bool(kaya_relay.get("replayed", False))
    deferred_reply = bool(kaya_relay.get("deferred_reply", False))
    backlog_remaining = int(kaya_relay.get("backlog_remaining", 0) or 0)

    return RelayEnvelope(
        journal_id=journal_id,
        seq=seq,
        received_at=received_at,
        replayed=replayed,
        deferred_reply=deferred_reply,
        backlog_remaining=backlog_remaining,
        event=event,
    )


class BacklogTracker:
    """Whether the gateway is still draining a backlog, so ingestion can wait for it."""

    def __init__(
        self,
        stale_after_seconds: float = 600.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._stale_after = stale_after_seconds
        self._now = now
        self._lock = threading.Lock()
        self._backlog_remaining: int = 0
        self._last_update: float = 0.0

    def update(self, backlog_remaining: int) -> None:
        with self._lock:
            self._backlog_remaining = backlog_remaining
            self._last_update = self._now()

    def draining(self) -> bool:
        with self._lock:
            if self._backlog_remaining <= 0:
                return False
            elapsed = self._now() - self._last_update
            return elapsed < self._stale_after
