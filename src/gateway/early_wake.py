"""Wake the PC earlier than its scheduled wake, when the idea pipeline asks.

The PC's nightly shutdown honours a wake request the idea pipeline writes
(``~/.wake-at``) and posts the chosen time as ``wake_at`` with
``/pc/going-down``. The gateway keeps it in the journal's meta table, so a
restart does not lose it, and when the time is due it writes the same
``pc-wake.request`` file ``/homelabon`` writes, so the existing
``pc-wake-request.path`` unit sends the Wake-on-LAN.

This module imports nothing heavy: the gateway runs on the Pi
(``tests/gateway/test_torch_free.py``).
"""
import asyncio
import datetime
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src.gateway.journal import Journal
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule
from src.gateway.homelab import WAKE_REQUEST_NAME

logger = logging.getLogger(__name__)

META_KEY = "pc_early_wake"
STALE_SECONDS = 3600.0


class EarlyWake:
    def __init__(self, journal: Journal, monitor: PcMonitor, schedule: PowerSchedule,
                 data_dir: str, now: Callable[[], float] = time.time) -> None:
        self._journal = journal
        self._monitor = monitor
        self._schedule = schedule
        self._now = now
        self._request_path = Path(data_dir) / WAKE_REQUEST_NAME

    def pending(self) -> Optional[float]:
        """The epoch the PC asked to wake at, kept across gateway restarts."""
        raw = self._journal.get_meta(META_KEY, "")
        try:
            value = float(raw)
        except ValueError:
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        return value

    def accept(self, wake_at: Any) -> Optional[float]:
        """Keep ``wake_at`` when the schedule honours it from now; clear it otherwise.

        The PC already ran the same check before posting, but the gateway is the
        one that sends the packet, so it must not trust a time it would not
        honour itself.
        """
        value = None
        if not isinstance(wake_at, bool) and isinstance(wake_at, (int, float)):
            candidate = float(wake_at)
            if math.isfinite(candidate) and candidate > 0:
                tz = self._schedule.timezone
                honoured = self._schedule.early_wake(
                    datetime.datetime.fromtimestamp(self._now(), tz),
                    datetime.datetime.fromtimestamp(candidate, tz))
                if honoured is not None:
                    value = float(honoured.timestamp())
        if value is None:
            self._journal.set_meta(META_KEY, "")
            return None
        self._journal.set_meta(META_KEY, str(value))
        logger.info("early wake accepted: wake the PC at epoch %s", value)
        return value

    def tick(self) -> None:
        """Send the Wake-on-LAN once the accepted time is due."""
        at = self.pending()
        if at is None or self._now() < at:
            return
        self._journal.set_meta(META_KEY, "")
        if self._now() - at > STALE_SECONDS:
            logger.info("early wake at epoch %s is stale, skipped", at)
            return
        if self._monitor.state is PcState.ONLINE:
            logger.info("early wake dropped: the PC is already on")
            return
        try:
            self._request_path.write_text(str(self._now()))
        except OSError:
            logger.exception("could not write %s", self._request_path)
            return
        logger.info("early wake due: asked for the PC to wake now")

    async def watch(self, stop: asyncio.Event, interval: float = 30.0) -> None:
        """Check the due time until *stop* is set."""
        while not stop.is_set():
            await asyncio.to_thread(self.tick)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
