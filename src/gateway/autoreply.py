"""The fixed WhatsApp line sent when the bot cannot answer.

Two situations, two texts, no model:

- the PC is off by schedule (or announced its shutdown): say so, give the hours
  from config.yaml ``power``, and say when it is back;
- the PC should be answering but is not (the app has been down past a deploy's
  length, the PC reported itself degraded at boot, or it is unreachable during
  scheduled hours): say it is down and that the message is kept.

One line per chat per outage. The message itself stays in the journal and is
answered properly when the PC returns (see the forwarder's deferred replies).
No dashes anywhere: the group asked for none, and a canned string is the one
place a prompt rule cannot reach.
"""
from __future__ import annotations

import datetime
import logging
import time
from typing import Any, Callable, Optional

from src.gateway.journal import Journal, JournalEvent
from src.gateway.monitor import PcMonitor, PcState
from src.gateway.schedule import PowerSchedule, schedule_sentence

logger = logging.getLogger(__name__)

SCHEDULED = "scheduled"
FAULT = "fault"

FAULT_TEXT = ("Estou em baixo neste momento e não consigo responder. "
              "Já fiquei com a tua mensagem e respondo assim que voltar.")


def offline_text(next_wake: datetime.datetime, now: datetime.datetime, hours: str = "") -> str:
    """The scheduled-off line: off now, the hours, and when it is back."""
    clock = next_wake.strftime("%H:%M")
    if next_wake.date() == now.date():
        back = f"às {clock}"
    elif next_wake.date() == now.date() + datetime.timedelta(days=1):
        back = f"amanhã às {clock}"
    else:
        back = f"no dia {next_wake.strftime('%d/%m')} às {clock}"
    schedule = f" O meu horário é {hours}." if hours else ""
    return f"Estou desligado agora.{schedule} Volto {back} e respondo-te nessa altura."


class OfflineResponder:
    """Decide whether the bot is unavailable, and send the fixed line once per chat per outage.

    ``send_text`` is called with ``(chat_id, text, message_id_or_none)`` and
    quotes the original message.
    """

    def __init__(
        self,
        journal: Journal,
        send_text: Callable[[str, str, Optional[str]], Any],
        schedule: PowerSchedule,
        monitor: PcMonitor,
        *,
        grace_seconds: float = 90.0,
        app_down_grace_seconds: float = 300.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        self._journal = journal
        self._send_text = send_text
        self._schedule = schedule
        self._monitor = monitor
        self._grace_seconds = grace_seconds
        self._app_down_grace_seconds = app_down_grace_seconds
        self._now = now

    def outage(self) -> Optional[str]:
        """``SCHEDULED``, ``FAULT``, or None while the bot can (or may soon) answer.

        App down for less than ``app_down_grace_seconds`` is a deploy or a boot
        still starting its containers: the message is buffered and answered
        within minutes, so saying "I am down" would be wrong. Past it, the bot is
        down whatever the reason, which is the case that stayed silent before.
        """
        monitor = self._monitor
        state = monitor.state
        if state is PcState.ONLINE or monitor.down_since is None:
            return None
        if state is PcState.GOING_DOWN:
            return SCHEDULED
        if monitor.degraded:
            return FAULT
        if state is PcState.OFFLINE and monitor.unreachable_for() >= self._grace_seconds:
            local_now = datetime.datetime.fromtimestamp(self._now(), self._schedule.timezone)
            return FAULT if self._schedule.is_scheduled_on(local_now) else SCHEDULED
        if state is PcState.APP_DOWN and monitor.down_for() >= self._app_down_grace_seconds:
            return FAULT
        return None

    def text_for(self, outage: str) -> str:
        """The fixed line for an outage kind."""
        if outage == FAULT:
            return FAULT_TEXT
        now = datetime.datetime.fromtimestamp(self._now(), self._schedule.timezone)
        return offline_text(self._schedule.next_wake(now), now, schedule_sentence(self._schedule))

    def maybe_reply(self, entry: JournalEvent) -> bool:
        """Send the line for this message if the bot is unavailable. True when sent."""
        if not entry.addressed or entry.backlog or entry.event_type != "message":
            return False
        outage = self.outage()
        down_since = self._monitor.down_since
        if outage is None or down_since is None:
            return False
        if self._journal.autoreply_sent(entry.chat_id, down_since):
            return False
        try:
            self._send_text(entry.chat_id, self.text_for(outage), entry.message_id or None)
        except Exception:  # noqa: BLE001 — nothing recorded, so a later message may try again
            logger.exception("offline reply to %s failed", entry.chat_id)
            return False
        self._journal.record_autoreply(entry.chat_id, down_since, self._now())
        self._journal.mark_auto_replied(entry.seq, self._now())
        logger.info("sent the %s line to %s", outage, entry.chat_id)
        return True
