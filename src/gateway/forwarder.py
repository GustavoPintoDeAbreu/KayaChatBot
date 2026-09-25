"""Forward journal events to the PC's relay endpoint.

Events are delivered in sequence-number order.  The forwarder retries on
failure with exponential back-off and skips delivery while the PC is
unreachable (as reported by ``PcMonitor``).
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

import httpx

from src.gateway.journal import Journal, JournalEvent
from src.gateway.monitor import PcMonitor, PcState

logger = logging.getLogger(__name__)


@dataclass
class ForwardResult:
    """Whether one delivery attempt was acked, and why not."""
    delivered: bool
    error: str = ""


class Forwarder:
    """Pull pending events from the journal and POST them to the PC."""

    def __init__(
        self,
        journal: Journal,
        monitor: PcMonitor,
        *,
        pc_url: str,
        relay_token: str,
        media_base_url: str,
        client: Optional[httpx.AsyncClient] = None,
        timeout: float = 900.0,
        deferred_after_seconds: float = 60.0,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        retention_seconds: float = 7 * 24 * 3600,
    ) -> None:
        self._journal = journal
        self._monitor = monitor
        self._pc_url = pc_url
        self._relay_token = relay_token
        self._media_base_url = media_base_url
        self._client = client or httpx.AsyncClient()
        self._timeout = timeout
        self._deferred_after = deferred_after_seconds
        self._now = now
        self._sleep = sleep
        self._retention_seconds = retention_seconds
        self._wake = asyncio.Event()
        self._last_purge: float = now()

    def wake(self) -> None:
        """Signal ``run()`` to check for pending work now."""
        self._wake.set()

    def envelope(self, entry: JournalEvent) -> Dict[str, Any]:
        """Build the relay envelope for *entry*."""
        event = copy.deepcopy(entry.event)

        if entry.media_status == "stored" and entry.media_path:
            media = event.setdefault("payload", {}).setdefault("media", {})
            media["url"] = (f"{self._media_base_url}/media/{entry.seq}/"
                            f"{Path(entry.media_path).name}")

        now = self._now()
        replayed = now - entry.received_at > self._deferred_after

        relay = {
            "journal_id": self._journal.journal_id,
            "seq": entry.seq,
            "received_at": entry.received_at,
            "replayed": replayed,
            "deferred_reply": (
                replayed
                and entry.addressed
                and not entry.backlog
                and not self._journal.newer_addressed_from(
                    entry.seq, entry.chat_id, entry.sender_id
                )
            ),
            "backlog_remaining": max(0, self._journal.pending_count() - 1),
        }

        return {"kaya_relay": relay, "event": event}

    async def deliver_one(self, entry: JournalEvent) -> ForwardResult:
        """POST the envelope for *entry* to the relay endpoint.

        Returns ``ForwardResult(delivered=True)`` on success, or
        ``ForwardResult(delivered=False, error=...)`` on failure.
        """
        url = f"{self._pc_url}/whatsapp/relay"
        envelope = self.envelope(entry)

        try:
            response = await self._client.post(
                url,
                json=envelope,
                headers={"X-Relay-Token": self._relay_token},
                timeout=self._timeout,
            )

            if response.status_code != 200:
                error = f"HTTP {response.status_code}"
                self._journal.record_attempt(entry.seq, error)
                return ForwardResult(delivered=False, error=error)

            body = response.json()
            ack = body.get("ack")
            if ack != entry.seq:
                error = f"ack mismatch: expected {entry.seq}, got {ack}"
                self._journal.record_attempt(entry.seq, error)
                return ForwardResult(delivered=False, error=error)

            self._journal.mark_delivered(entry.seq, self._now())
            logger.info("Delivered event seq=%d to PC", entry.seq)
            return ForwardResult(delivered=True)

        except (httpx.HTTPError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._journal.record_attempt(entry.seq, error)
            return ForwardResult(delivered=False, error=error)

    def _purge_if_due(self) -> None:
        """Delete delivered events past retention, at most every ten minutes."""
        now = self._now()
        if now - self._last_purge < 600:
            return
        self._last_purge = now
        counts = self._journal.purge_delivered(now - self._retention_seconds)
        if counts["events"]:
            logger.info("purged delivered events: %s", counts)

    async def _idle(self) -> None:
        """Wait for ``wake()`` or five seconds, whichever comes first."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        self._wake.clear()

    async def run(self, stop: asyncio.Event) -> None:
        """Deliver pending events, oldest first, until *stop* is set."""
        while not stop.is_set():
            try:
                self._purge_if_due()
                entry = self._journal.next_pending()
                if entry is None or self._monitor.state is not PcState.ONLINE:
                    await self._idle()
                    continue
                result = await self.deliver_one(entry)
                if not result.delivered:
                    logger.warning("delivery of seq=%d failed: %s", entry.seq, result.error)
                    await self._sleep(min(60, 2 ** min(entry.attempts, 6)))
            except Exception:  # noqa: BLE001 — the forwarder must outlive any one bad event
                logger.exception("forwarder loop error")
                await self._sleep(5)
