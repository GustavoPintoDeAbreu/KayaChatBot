"""Offline reply for addressed messages when the PC is unreachable.

When the PC is down the gateway must still answer addressed messages once
per offline period so the sender knows the bot is alive and when to expect
a real reply.
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Callable, Optional

from src.gateway.journal import Journal, JournalEvent
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule

logger = logging.getLogger(__name__)


def offline_text(next_wake: datetime.datetime, now: datetime.datetime) -> str:
    """Return the Portuguese offline message for *next_wake* relative to *now*."""
    time_str = next_wake.strftime("%H:%M")
    if next_wake.date() == now.date():
        return f"Estou desligado agora. Volto às {time_str} e respondo-te nessa altura."
    tomorrow = now.date() + datetime.timedelta(days=1)
    if next_wake.date() == tomorrow:
        return f"Estou desligado agora. Volto amanhã às {time_str} e respondo-te nessa altura."
    day_month = next_wake.strftime("%d/%m")
    return f"Estou desligado agora. Volto no dia {day_month} às {time_str} e respondo-te nessa altura."


class OfflineResponder:
    """Send a single offline notice per chat per offline period.

    ``send_text`` is called with ``(chat_id, text, message_id_or_none)``.
    It should quote the original message (WhatsApp reply).
    """

    def __init__(
        self,
        journal: Journal,
        send_text: Callable[[str, str, Optional[str]], Any],
        schedule: PowerSchedule,
        monitor: PcMonitor,
        *,
        grace_seconds: float = 90.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._journal = journal
        self._send_text = send_text
        self._schedule = schedule
        self._monitor = monitor
        self._grace_seconds = grace_seconds
        self._now = now

    def maybe_reply(self, entry: JournalEvent) -> bool:
        """Send an offline reply when conditions are met.

        Returns ``True`` when a reply was sent.
        """
        if not entry.addressed or entry.backlog or entry.event_type != "message":
            return False

        monitor = self._monitor
        state = monitor.state
        is_offline = (
            state is PcState.GOING_DOWN
            or (state is PcState.OFFLINE and monitor.unreachable_for() >= self._grace_seconds)
        )
        if not is_offline:
            return False

        if monitor.offline_since is None:
            return False

        if self._journal.autoreply_sent(entry.chat_id, monitor.offline_since):
            return False

        now = datetime.datetime.fromtimestamp(self._now(), self._schedule.timezone)
        text = offline_text(self._schedule.next_wake(now), now)

        try:
            self._send_text(entry.chat_id, text, entry.message_id or None)
            self._journal.record_autoreply(entry.chat_id, monitor.offline_since, self._now())
            self._journal.mark_auto_replied(entry.seq, self._now())
            logger.info("Sent offline reply to %s", entry.chat_id)
            return True
        except Exception:
            logger.exception("Failed to send offline reply to %s", entry.chat_id)
            return False
