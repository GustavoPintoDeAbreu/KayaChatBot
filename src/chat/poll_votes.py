"""Votes on the polls Kaya posts, and the reminders that go with them.

Kaya creates native WhatsApp polls (the sibling conversation-poll feature); this
module is the other half. The Pi journals WAHA's ``poll.vote`` events, the PC
records who voted on each poll in ``data/poll_votes.json``, and a prod-only
scheduler @-mentions the people who have not voted every few hours until everyone
has, the poll is old enough to stop, or nobody can be identified.

This ships ``enabled: false``, and turning it on needs a WAHA change first.
Checked against the Pi's WAHA (noweb-arm-2026.8.2) on 2026-10-15: NOWEB decrypts
a vote by looking up the poll message **by id alone**, and only the persistent
store can do that (``NowebPersistentStore.loadMessage`` with no jid); the
in-memory store keys messages by chat, so every vote is dropped with "poll
creation message not found" and no event reaches the webhook. The session needs
``config.noweb.store.enabled: true`` before any vote arrives.

Verified against WAHA's DTOs: ``poll.vote`` (and ``poll.vote.failed``) carry
``payload.poll`` (id, to, from, fromMe) and ``payload.vote`` (the same plus
``selectedOptions`` and ``timestamp``). Still unverified until a real event is
captured on the Pi:
* A poll's id arrives serialised (``true_<chat>_<key>``) in some deliveries and
  bare in others; ``poll_key`` makes the two compare equal.
* ``selectedOptions: []`` means the vote was retracted, not a vote for nothing.

This module must stay importable by the Pi gateway, so it pulls in nothing but
the standard library and the adapter's pure JID helpers.
"""
from __future__ import annotations

import datetime
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from src.chat.whatsapp_adapter import _normalize_jid, _phone_from_alt
from src.chat.waha_client import extract_sent_id
from src.chat.birthdays import _atomic_write

POLL_VOTE_EVENTS = ("poll.vote",)
POLL_VOTE_FAILED_EVENTS = ("poll.vote.failed",)

NO_POLL_TEXT = "Não tenho nenhuma sondagem minha a decorrer nesta conversa."


def poll_key(message_id: str) -> str:
    """A poll id that compares equal whether it arrived serialised or bare.

    WhatsApp serialises message ids as ``<bool>_<chat>_<key>[_<sender>]``; the
    key is the segment that is neither a chat JID nor a bare true/false.
    """
    for segment in (message_id or "").split("_"):
        if "@" not in segment and segment not in ("true", "false"):
            return segment
    return (message_id or "").strip()


@dataclass
class PollVote:
    """One vote (or retraction) on one poll, parsed from a WAHA event."""

    poll_id: str
    chat_id: str
    voter_id: str
    options: List[str]
    timestamp: Optional[int]
    poll_from_me: bool
    vote_from_me: bool


def parse_waha_poll_vote(event: Dict[str, Any]) -> Optional[PollVote]:
    """The vote in a ``poll.vote`` event, or ``None`` when it is not one.

    ``poll.vote.failed`` parses to ``None`` on purpose: the caller logs it and
    moves on, because there is no voter to record.
    """
    if event.get("event") not in POLL_VOTE_EVENTS:
        return None
    payload = event.get("payload") or {}
    vote = payload.get("vote")
    poll = payload.get("poll")
    if not isinstance(vote, dict) or not isinstance(poll, dict):
        return None
    poll_id = poll_key(extract_sent_id({"id": poll.get("id")}))
    if not poll_id:
        return None

    voter = _normalize_jid(vote.get("participant") or vote.get("author")
                           or vote.get("from"))
    if not voter or voter.endswith("@g.us"):
        return None

    chat_id = ""
    for candidate in (vote.get("to"), vote.get("from"),
                      poll.get("to"), poll.get("from")):
        normalized = _normalize_jid(candidate)
        if normalized.endswith("@g.us"):
            chat_id = normalized
            break
    if not chat_id:
        chat_id = voter  # a DM: the poll lives in the voter's chat with the bot

    options = [str(option) for option in (vote.get("selectedOptions") or [])]
    timestamp = vote.get("timestamp")
    if isinstance(timestamp, int):
        timestamp = timestamp // 1000 if timestamp > 10**12 else timestamp
    else:
        timestamp = None

    return PollVote(
        poll_id=poll_id,
        chat_id=chat_id,
        voter_id=voter,
        options=options,
        timestamp=timestamp,
        poll_from_me=bool(poll.get("fromMe")),
        vote_from_me=bool(vote.get("fromMe")),
    )


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    """``whatsapp.poll_votes`` with its defaults (off until proven on the Pi)."""
    block = (config.get("whatsapp", {}) or {}).get("poll_votes", {}) or {}
    remind = block.get("remind", {}) or {}
    result = {
        "enabled": bool(block.get("enabled", False)),
        "store": str(block.get("store", "data/poll_votes.json")),
        "electorate_days": float(block.get("electorate_days", 60)),
        "status_max_age_hours": float(block.get("status_max_age_hours", 168)),
        "remind": {
            "enabled": bool(remind.get("enabled", True)),
            "dry_run": bool(remind.get("dry_run", True)),
            "interval_hours": float(remind.get("interval_hours", 6)),
            "stop_after_hours": float(remind.get("stop_after_hours", 24)),
            "min_gap_hours": float(remind.get("min_gap_hours", 3)),
            "send_between": list(remind.get("send_between") or ["10:00", "22:00"]),
        },
        "timezone": str((config.get("power", {}) or {}).get(
            "timezone", "Europe/Lisbon")),
    }
    return result


@dataclass
class Voter:
    """One person who could have voted, and how to reach them."""

    name: str
    ids: Set[str] = field(default_factory=set)
    mention_jid: str = ""


class PollRegistry:
    """Who voted on each poll Kaya created, kept in one JSON file.

    Every method loads, mutates and atomically saves under one lock, because the
    webhook thread records votes and the reminder thread marks slots.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        if not isinstance(data, dict) or not isinstance(data.get("polls"), dict):
            data = {"polls": {}}
        return data

    def _save(self, data: Dict[str, Any]) -> None:
        _atomic_write(self.path, data)

    def register(self, chat_id: str, poll_id: str, options: List[str],
                 created_at: int, source: str = "explicit") -> str:
        """Record a poll Kaya created. Idempotent: re-registering keeps the votes."""
        key = poll_key(poll_id)
        with self._lock:
            data = self._load()
            record = data["polls"].get(key)
            if record is None:
                record = {
                    "chat_id": chat_id,
                    "options": list(options),
                    "created_at": int(created_at),
                    "source": source,
                    "votes": {},
                    "reminded_slots": [],
                    "last_reminder_at": None,
                    "closed": None,
                }
                data["polls"][key] = record
            else:
                if not record.get("options"):
                    record["options"] = list(options)
                if not record.get("created_at"):
                    record["created_at"] = int(created_at)
                if source == "explicit":
                    # An explicit registration knows more than a vote-guessed one.
                    record["source"] = "explicit"
                    record["created_at"] = int(created_at)
            self._save(data)
        return key

    def get(self, poll_id: str) -> Optional[Dict[str, Any]]:
        """A copy of the record for a poll (by bare or serialised id), or None."""
        key = poll_key(poll_id)
        with self._lock:
            record = self._load()["polls"].get(key)
        if record is None:
            return None
        result = dict(record)
        result["id"] = key
        return result

    def latest_for_chat(self, chat_id: str, now_ts: int,
                        max_age_hours: float) -> Optional[Dict[str, Any]]:
        """The newest poll in this chat that is still young enough to ask about."""
        cutoff = int(now_ts) - int(max_age_hours * 3600)
        newest: Optional[Dict[str, Any]] = None
        with self._lock:
            data = self._load()["polls"]
        for key, record in data.items():
            if record.get("chat_id") != chat_id:
                continue
            created = int(record.get("created_at") or 0)
            if created < cutoff:
                continue
            if newest is None or created > int(newest.get("created_at") or 0):
                candidate = dict(record)
                candidate["id"] = key
                newest = candidate
        return newest

    def record_vote(self, vote: PollVote, voter_name: str, now_ts: int) -> bool:
        """Store one vote. False when it changed nothing.

        A vote on a poll Kaya created but nobody told us about yet auto-registers
        it (``source="vote"``); a vote on somebody else's poll is ignored.
        """
        if vote.vote_from_me:
            return False
        key = poll_key(vote.poll_id)
        with self._lock:
            data = self._load()
            record = data["polls"].get(key)
            if record is None:
                if not vote.poll_from_me:
                    return False
                record = {
                    "chat_id": vote.chat_id,
                    "options": list(vote.options),
                    "created_at": int(vote.timestamp or now_ts),
                    "source": "vote",
                    "votes": {},
                    "reminded_slots": [],
                    "last_reminder_at": None,
                    "closed": None,
                }
                data["polls"][key] = record
            stored_at = int(vote.timestamp or now_ts)
            existing = record["votes"].get(_phone_from_alt(vote.voter_id))
            if existing and int(existing.get("at") or 0) > stored_at:
                return False
            record["votes"][_phone_from_alt(vote.voter_id)] = {
                "name": voter_name,
                "options": list(vote.options),
                "at": stored_at,
            }
            for option in vote.options:
                if option not in record["options"]:
                    record["options"].append(option)
            self._save(data)
        return True

    def open_polls(self, now_ts: int, stop_after_hours: float) -> List[Dict[str, Any]]:
        """The polls still worth reminding about, closing whatever is past due."""
        cutoff = int(now_ts) - int(stop_after_hours * 3600)
        open_polls: List[Dict[str, Any]] = []
        changed = False
        with self._lock:
            data = self._load()
            for key, record in data["polls"].items():
                if record.get("closed"):
                    continue
                created = int(record.get("created_at") or 0)
                if created < cutoff:
                    record["closed"] = "expired"
                    changed = True
                    continue
                candidate = dict(record)
                candidate["id"] = key
                open_polls.append(candidate)
            if changed:
                self._save(data)
        return open_polls

    def mark_reminded(self, poll_id: str, slot: int, sent_at: int) -> None:
        key = poll_key(poll_id)
        with self._lock:
            data = self._load()
            record = data["polls"].get(key)
            if record is None:
                return
            if slot not in record["reminded_slots"]:
                record["reminded_slots"].append(slot)
            record["last_reminder_at"] = int(sent_at)
            self._save(data)

    def mark_slot_skipped(self, poll_id: str, slot: int) -> None:
        """Record a slot as handled without a reminder having been sent.

        A slot skipped for lack of an identifiable electorate must not count as a
        reminder: it sets nothing in ``last_reminder_at``, so the min-gap check
        cannot hold back the next real one.
        """
        key = poll_key(poll_id)
        with self._lock:
            data = self._load()
            record = data["polls"].get(key)
            if record is None:
                return
            if slot not in record["reminded_slots"]:
                record["reminded_slots"].append(slot)
                self._save(data)

    def close(self, poll_id: str, reason: str) -> None:
        key = poll_key(poll_id)
        with self._lock:
            data = self._load()
            record = data["polls"].get(key)
            if record is None:
                return
            record["closed"] = reason
            self._save(data)


def electorate(records: Sequence[Dict[str, Any]], chat_id: str, since_ts: int,
               bot_ids: Set[str], is_member) -> List[Voter]:
    """Who could have voted: everyone who wrote in that chat since ``since_ts``.

    ``records`` are message-log rows (``src/data/message_log.py``). The bot's own
    lines are out, and so are any other bot identities. A member is grouped by
    name; an unknown sender is grouped by phone, so two ids for one member merge
    into one voter.
    """
    voters: List[Voter] = []
    by_key: Dict[str, Voter] = {}
    for record in records:
        if record.get("chat_id") != chat_id:
            continue
        if int(record.get("timestamp") or 0) < int(since_ts):
            continue
        sender = str(record.get("sender") or "")
        sender_id = str(record.get("sender_id") or "")
        # A phone is only a phone when it is one: ``sender_phone`` is, but the
        # local part of an ``@lid`` id is not (most of this group's contacts are
        # keyed by lid), so the lid must not be dressed up as a phone.
        phone = _phone_from_alt(record.get("sender_phone") or "")
        if sender == "Kaya Bot" or _phone_from_alt(sender_id) in bot_ids:
            continue
        if is_member(sender):
            key = sender
        else:
            key = phone or sender_id.split("@", 1)[0]
        if not key:
            continue
        voter = by_key.get(key)
        if voter is None:
            voter = Voter(name=sender if is_member(sender) else "",
                          mention_jid=(f"{phone}@c.us" if phone else _normalize_jid(sender_id)))
            by_key[key] = voter
            voters.append(voter)
        if phone:
            voter.ids.add(phone)
            # A phone is a better mention target than a lid: if the first record
            # only had an @lid id, upgrade the mention once a phone turns up.
            if not voter.mention_jid.endswith("@c.us"):
                voter.mention_jid = f"{phone}@c.us"
        # The id's local part (a phone or a lid) is how a vote is keyed, so it
        # must stay in ``ids`` for ``has_voted`` to match — but it is an id, not
        # a phone, and must not feed ``mention_jid``.
        local = sender_id.split("@", 1)[0]
        if local:
            voter.ids.add(local)
        if is_member(sender) and not voter.name:
            voter.name = sender
    return voters


def has_voted(poll: Dict[str, Any], voter: Voter) -> bool:
    """Whether this person already voted, by any of their ids or by name.

    A retracted vote (``options`` empty) does not count: the person is back in
    "Falta votar" and gets reminded again.
    """
    votes = poll.get("votes") or {}
    for key, vote in votes.items():
        if not vote.get("options"):
            continue
        if key in voter.ids:
            return True
        name = str(vote.get("name") or "")
        if name and name == voter.name:
            return True
    return False


def missing_voters(poll: Dict[str, Any], voters: Sequence[Voter]) -> List[Voter]:
    return [voter for voter in voters if not has_voted(poll, voter)]


def tally(poll: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """``(option, [who picked it])`` per option, most votes first.

    Options with zero votes stay in, in the poll's order; ties keep the poll's
    option order.
    """
    votes = poll.get("votes") or {}
    counts: Dict[str, List[str]] = {option: [] for option in poll.get("options") or []}
    for key, vote in votes.items():
        name = str(vote.get("name") or "") or key
        for option in vote.get("options") or []:
            counts.setdefault(str(option), []).append(name)
    # one sort: most votes first, and a tie keeps the poll's option order
    return sorted(counts.items(), key=lambda item: (-len(item[1]),
                                                    list(counts).index(item[0])))


def in_window(now: datetime.datetime, send_between: Sequence[str]) -> bool:
    """Whether ``now`` falls inside the send window (which may wrap midnight)."""
    start = datetime.time.fromisoformat(str(send_between[0]))
    end = datetime.time.fromisoformat(str(send_between[1]))
    current = now.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end


def due_slot(poll: Dict[str, Any], now: datetime.datetime,
             remind_cfg: Dict[str, Any]) -> Optional[int]:
    """The one reminder slot that is due right now, if any.

    Only the LATEST slot is ever returned: a PC that was off for a day does not
    fire the missed slots one after another, and quiet hours defer rather than
    queue.
    """
    created_at = int(poll.get("created_at") or 0)
    elapsed = int(now.timestamp()) - created_at
    interval = int(float(remind_cfg.get("interval_hours", 6)) * 3600)
    stop_after = int(float(remind_cfg.get("stop_after_hours", 24)) * 3600)
    if elapsed >= stop_after:
        return None
    if interval <= 0:
        return None
    k = elapsed // interval
    if k < 1:
        return None
    slot = created_at + k * interval
    if slot in (poll.get("reminded_slots") or []):
        return None
    last = poll.get("last_reminder_at")
    if last and int(now.timestamp()) - int(last) < int(
            float(remind_cfg.get("min_gap_hours", 3)) * 3600):
        return None
    if not in_window(now, remind_cfg.get("send_between") or ["10:00", "22:00"]):
        return None
    return slot


def status_text(poll: Dict[str, Any], voters: Sequence[Voter],
                timezone: str = "Europe/Lisbon") -> str:
    """The answer to "quem está a ganhar?", built from the recorded votes.

    Never generated: a count the model made up is worse than no count. No dashes
    anywhere, the group's rule. The date is rendered in ``timezone`` (the
    configured ``power.timezone``), not a zone baked into the record.
    """
    votes = poll.get("votes") or {}
    voted = sum(1 for vote in votes.values() if vote.get("options"))
    created = int(poll.get("created_at") or 0)
    when = datetime.datetime.fromtimestamp(
        created, tz=ZoneInfo(timezone)).strftime("%d/%m")
    lines = [f"Sondagem de {when}, {voted} pessoas votaram:"]
    for option, names in tally(poll):
        if names:
            lines.append(f"{option}: {len(names)} ({', '.join(names)})")
        else:
            lines.append(f"{option}: 0")
    missing = missing_voters(poll, voters)
    if voters and missing:
        lines.append("Falta votar: " + ", ".join(
            voter.name or voter.mention_jid.split("@", 1)[0] for voter in missing) + ".")
    elif voters and not missing:
        lines.append("Já votaram todos.")
    return "\n".join(lines)


def reminder_text(poll: Dict[str, Any],
                  missing: Sequence[Voter]) -> Tuple[str, str, List[str]]:
    """The reminder line, three ways: tokens, names, and the JIDs to mention.

    ``text_with_tokens`` carries ``@<local>`` where the mention goes; WhatsApp
    resolves it through the ``mentions`` list, which is ``mention_jids``.
    """
    tokens = " ".join(f"@{voter.mention_jid.split('@', 1)[0]}" for voter in missing)
    names = ", ".join(voter.name or voter.mention_jid.split("@", 1)[0]
                      for voter in missing)
    jids = [voter.mention_jid for voter in missing if voter.mention_jid]
    first = f"Ainda falta votar na sondagem: {tokens}."
    second = f"Ainda falta votar na sondagem: {names}."
    ranked = tally(poll)
    leading = [option for option, names_ in ranked if names_]
    if leading:
        top = len(ranked[0][1])
        tied = [option for option, names_ in ranked if len(names_) == top]
        if len(tied) == 1:
            lead = f"Vai à frente: {tied[0]} ({top})."
        else:
            lead = f"Empate entre {' e '.join(tied)} ({top})."
        return first + "\n" + lead, second + "\n" + lead, jids
    return first, second, jids
