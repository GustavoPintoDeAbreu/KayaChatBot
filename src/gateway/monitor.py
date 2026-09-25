"""Periodic health probe for the GPU PC.

The gateway monitors whether the PC is reachable and whether its application
is running.  State transitions are logged and made available as properties so
that the forwarder and the autoreply module can make decisions.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import time
from enum import Enum
from typing import Callable, Optional

import httpx
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)


class PcState(str, Enum):
    """How the PC looks from the Pi. APP_DOWN is a deploy, not an outage."""
    ONLINE = "online"
    APP_DOWN = "app_down"
    OFFLINE = "offline"
    GOING_DOWN = "going_down"


class PcMonitor:
    """Probe the PC's health endpoint and track state transitions.

    ``probe_port`` is the TCP port used as a fallback when the HTTP health
    check fails (default 7860, the Gradio web-server port).  ``check_seconds``
    controls the default ``run()`` cadence.
    """

    def __init__(
        self,
        pc_url: str,
        *,
        probe_port: int = 7860,
        check_seconds: float = 10.0,
        now: Callable[[], float] = time.time,
        probe: Optional[Callable[[], PcState]] = None,
    ) -> None:
        self._pc_url = pc_url
        self._probe_port = probe_port
        self._check_seconds = check_seconds
        self._now = now
        self._probe_fn = probe
        self._hostname = urlsplit(pc_url).hostname or "localhost"
        self._state: PcState = PcState.OFFLINE
        self._offline_since: Optional[float] = None
        self._going_down: bool = False

    @property
    def state(self) -> PcState:
        return self._state

    @property
    def offline_since(self) -> Optional[float]:
        return self._offline_since

    def announce_going_down(self) -> None:
        """Mark the PC as going down; persists until ONLINE is observed again."""
        now = self._now()
        if self._offline_since is None:
            self._offline_since = now
        self._going_down = True
        self._state = PcState.GOING_DOWN
        logger.info("PC announced going down")

    def observe(self, observed: PcState) -> PcState:
        """Apply one probe result and return the new state."""
        now = self._now()
        old_state = self._state

        if observed is PcState.ONLINE:
            self._state = PcState.ONLINE
            self._offline_since = None
            self._going_down = False

        elif observed is PcState.APP_DOWN:
            self._state = PcState.GOING_DOWN if self._going_down else PcState.APP_DOWN

        elif observed is PcState.OFFLINE:
            self._state = PcState.GOING_DOWN if self._going_down else PcState.OFFLINE
            if self._offline_since is None:
                self._offline_since = now

        elif observed is PcState.GOING_DOWN:
            self._state = PcState.GOING_DOWN
            if self._offline_since is None:
                self._offline_since = now

        if self._state is not old_state:
            logger.info("PC state: %s -> %s", old_state.value, self._state.value)
        return self._state

    def check(self) -> PcState:
        """Run the default probe once and observe the result."""
        if self._probe_fn is not None:
            result = self._probe_fn()
        else:
            result = _default_probe(self._pc_url, self._probe_port)
        return self.observe(result)

    async def run(self, stop: asyncio.Event) -> None:
        """Call ``check`` every ``check_seconds`` until *stop* is set."""
        while not stop.is_set():
            await asyncio.to_thread(self.check)
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._check_seconds)
            except asyncio.TimeoutError:
                pass

    def unreachable_for(self) -> float:
        """Seconds since the PC became unreachable, 0 when reachable."""
        if self._offline_since is None:
            return 0.0
        return self._now() - self._offline_since


def _default_probe(pc_url: str, probe_port: int) -> PcState:
    """Probe the PC via HTTP health, falling back to TCP connect.

    200 from the health endpoint means ONLINE.  A connection to the TCP
    port (open or refused) means APP_DOWN.  Any other network error means
    OFFLINE.
    """
    try:
        response = httpx.get(f"{pc_url}/whatsapp/health", timeout=3)
        if response.status_code == 200:
            return PcState.ONLINE
    except httpx.HTTPError:
        pass

    try:
        with socket.create_connection(
            (urlsplit(pc_url).hostname or "localhost", probe_port), timeout=3
        ):
            return PcState.APP_DOWN
    except ConnectionRefusedError:
        return PcState.APP_DOWN
    except OSError:
        return PcState.OFFLINE
