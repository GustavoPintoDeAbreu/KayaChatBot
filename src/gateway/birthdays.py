"""Birthday wishes from the Pi: a fixed line at midnight, a nudge at noon.

The Pi is on at midnight and the PC may not be, so the greeting lives here and
needs no model: it is a line picked from ``chat.birthdays.messages``, with the
member tagged. At noon a second line goes out only if nobody in the group has
said parabéns to that member since midnight, which the journal already knows.

The PC owns the dates (``/aniversario``, ``scripts/mine_birthdays.py``). This
pulls the PC's roster (``GET /whatsapp/relay/birthdays``) whenever the PC
answers and keeps the last copy on disk, so a birthday still goes out with the
PC off. Each slot is sent at most once per member per year; a failed send is
retried on the next tick while the slot's window is open.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from zoneinfo import ZoneInfo

from src.chat import birthdays
from src.chat.whatsapp_adapter import _normalize_jid, parse_waha_message
from src.gateway.journal import Journal

logger = logging.getLogger(__name__)

ROSTER_REFRESH_SECONDS = 1800.0

SendText = Callable[[str, str, List[str]], Any]
FetchRoster = Callable[[], Optional[Dict[str, Any]]]


def tag_for(member: str, entry: Dict[str, Any]) -> tuple:
    """``(text, mentions)``: ``@<id>`` plus the id when it is known, else the name."""
    jid = str(entry.get("jid") or "")
    if not jid:
        return member, []
    return "@" + jid.split("@", 1)[0], [jid]


class BirthdayGreeter:
    """Sends the midnight line and the conditional noon nudge to one chat."""

    def __init__(self, *, chat_id: str, data_dir: str, config: Dict[str, Any],
                 journal: Journal, send_text: SendText, fetch_roster: FetchRoster,
                 timezone: ZoneInfo, now: Callable[[], float] = time.time,
                 rng: Optional[random.Random] = None) -> None:
        self.chat_id = chat_id
        self.journal = journal
        self._send_text = send_text
        self._fetch_roster = fetch_roster
        self._timezone = timezone
        self._now = now
        self._rng = rng or random.Random()
        self._settings = birthdays.settings(config)
        self._roster_path = Path(data_dir) / "birthdays.json"
        self.state = birthdays.GreetingState(Path(data_dir) / "birthday_state.json")
        self._roster_fetched_at = 0.0
        self.last_check: Optional[float] = None
        self.last_sent: str = ""

    # ── the roster ────────────────────────────────────────────────────────────
    def roster(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self._roster_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def store_roster(self, roster: Dict[str, Any]) -> None:
        self._roster_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._roster_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(roster, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self._roster_path)

    def refresh_roster(self, force: bool = False) -> bool:
        """Pull the PC's roster at most every half hour. False when the PC did not answer."""
        if not force and self._now() - self._roster_fetched_at < ROSTER_REFRESH_SECONDS:
            return True
        try:
            roster = self._fetch_roster()
        except Exception as exc:  # noqa: BLE001 — the PC being off is normal
            logger.debug("birthday roster not refreshed: %s", exc)
            return False
        if not isinstance(roster, dict):
            return False
        self.store_roster(roster)
        self._roster_fetched_at = self._now()
        return True

    # ── who was congratulated ────────────────────────────────────────────────
    def congratulated(self, member: str, entry: Dict[str, Any], since: float,
                      alone: bool) -> bool:
        """Whether a group message since ``since`` said parabéns to ``member``.

        With one birthday today any parabéns counts. With two, the message has
        to name the member (or an alias), tag them, or quote a line that does.
        """
        names = [member, *(entry.get("aliases") or [])]
        pattern = re.compile(r"\b(" + "|".join(
            re.escape(birthdays._fold(name)) for name in names if len(name) >= 3) + r")\w*\b") \
            if any(len(name) >= 3 for name in names) else None
        jid = _normalize_jid(entry.get("jid"))
        for event in self.journal.messages_since(since):
            msg = parse_waha_message(event.event)
            if msg is None or msg.from_me or not msg.is_group:
                continue
            if not birthdays._CONGRATS.search(msg.text or ""):
                continue
            if alone:
                return True
            folded = birthdays._fold(f"{msg.text}\n{msg.quoted_text}")
            if pattern is not None and pattern.search(folded):
                return True
            if jid and (jid in msg.mentioned_ids
                        or jid.split("@", 1)[0] in folded):
                return True
        return False

    # ── one pass ──────────────────────────────────────────────────────────────
    def tick(self) -> List[str]:
        """Send whatever is due now. Returns ``["member:slot sent|skipped"]``."""
        now = datetime.datetime.fromtimestamp(self._now(), self._timezone)
        self.last_check = self._now()
        roster = self.roster()
        dates = {member: (str(entry.get("date") or ""), "")
                 for member, entry in roster.items() if entry.get("date")}
        today = birthdays.birthdays_on(dates, now.date())
        midnight = datetime.datetime.combine(now.date(), datetime.time(0, 0),
                                             tzinfo=self._timezone).timestamp()
        done = []
        for slot in birthdays.SLOTS:
            start, until = self._settings["windows"][slot]
            for member in birthdays.due(dates, self.state, now, slot, start, until):
                entry = roster.get(member, {})
                if slot == "noon" and self.congratulated(member, entry, midnight,
                                                         alone=len(today) == 1):
                    self.state.mark(now.year, f"{member}:{slot}", "skipped: already congratulated")
                    done.append(f"{member}:{slot} skipped")
                    continue
                tag, mentions = tag_for(member, entry)
                text = self._rng.choice(self._settings["messages"][slot]).format(tag=tag)
                try:
                    self._send_text(self.chat_id, text, mentions)
                except Exception as exc:  # noqa: BLE001 — retried on the next tick
                    logger.warning("birthday %s for %s not sent: %s", slot, member, exc)
                    continue
                self.state.mark(now.year, f"{member}:{slot}", "sent")
                self.last_sent = f"{now.isoformat(timespec='minutes')} {member}:{slot}"
                logger.info("birthday %s sent for %s", slot, member)
                done.append(f"{member}:{slot} sent")
        return done

    def status(self) -> Dict[str, Any]:
        return {"members": len(self.roster()), "last_check": self.last_check,
                "last_sent": self.last_sent}

    async def watch(self, stop: asyncio.Event, interval: float = 60.0) -> None:
        """Refresh the roster and send what is due, every ``interval`` seconds."""
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.refresh_roster)
                await asyncio.to_thread(self.tick)
            except Exception as exc:  # noqa: BLE001 — must never take the gateway down
                logger.warning("birthday check failed: %s", exc)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
