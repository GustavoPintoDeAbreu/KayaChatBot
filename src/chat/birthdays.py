"""Whose birthday it is, and saying parabéns once.

The group congratulates each other in bursts: on 2025-06-29 five people wrote
"Happy birthday Fred" within hours. The bot had no idea why. The profiles held
birthdays only as loose prose ("carnall had a birthday on february 25"), which no
code can act on, so on the day it could neither join in nor answer "quando faz
anos o Pedro?".

Dates come from four places, most trusted first:

1. ``self``: the member said it (``/aniversario 8/9``).
2. ``profile``: a ``birthday`` field in ``group_members.json``, set by hand or
   accepted through ``scripts/review_bios.py``.
3. ``about``: the member's WhatsApp About text.
4. ``mined``: a burst of parabéns in the history, used only once a person has
   confirmed it. Bursts are noisy: a birth, a new job and a wedding all get one,
   and Manuel has bursts on two different days.

The store is its own file, ``data/birthdays.json``, so the live bot never writes
into the profiles.

The greeting is sent by the Pi gateway, which is on at midnight when the PC
may not be (``src/gateway/birthdays.py``). It is fixed text, no model: one line
at 00:00, and a second at 12:00 only if nobody in the group has said parabéns
by then. Each slot goes out once per member per year (``birthday_state.json``).
The PC owns the dates and hands the Pi a roster (``roster``) of who, when, and
which WhatsApp id to tag.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

BASE_DIR = Path(__file__).resolve().parents[2]

SOURCES = ("self", "profile", "about", "mined")

MONTHS_PT = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho",
             "agosto", "setembro", "outubro", "novembro", "dezembro")
_MONTH_NAMES = {
    **{name: index for index, name in enumerate(MONTHS_PT, start=1)},
    "marco": 3,
    **{name: index for index, name in enumerate(
        ("january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"), start=1)},
}

_NUMERIC = re.compile(r"\b(\d{1,2})\s*[/.\-]\s*(\d{1,2})(?:\s*[/.\-]\s*\d{2,4})?\b")
_WORDED = re.compile(r"\b(\d{1,2})\s*(?:de\s+|of\s+)?([a-zçã]+)\b", re.IGNORECASE)
_WORDED_EN = re.compile(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.IGNORECASE)

_CONGRATS = re.compile(
    r"parab[ée]ns|feliz(es)? anivers|happy\s*b(irth)?-?day|\bhbd\b|bom anivers",
    re.IGNORECASE)
_THANKS = re.compile(r"\bobrigad|\bthank|\bthx\b|\bvaleu\b", re.IGNORECASE)


def _valid(month: int, day: int) -> bool:
    try:
        date(2024, month, day)
    except ValueError:
        return False
    return True


def parse_date(text: str) -> Optional[str]:
    """``"MM-DD"`` from what somebody typed, day first. None if there is no date.

    Accepts "8/9", "08-09", "8.9.1994", "8 de setembro", "8 setembro" and
    "September 8". Day first because this is Portugal: "8/9" is the 8th of
    September.
    """
    lowered = (text or "").lower()
    match = _NUMERIC.search(lowered)
    if match:
        day, month = int(match.group(1)), int(match.group(2))
        return f"{month:02d}-{day:02d}" if _valid(month, day) else None
    for match in _WORDED.finditer(lowered):
        month = _MONTH_NAMES.get(match.group(2))
        day = int(match.group(1))
        if month and _valid(month, day):
            return f"{month:02d}-{day:02d}"
    for match in _WORDED_EN.finditer(lowered):
        month = _MONTH_NAMES.get(match.group(1))
        day = int(match.group(2))
        if month and _valid(month, day):
            return f"{month:02d}-{day:02d}"
    return None


def spoken_date(month_day: str) -> str:
    """``"09-08"`` → ``"8 de setembro"``."""
    month, day = (int(part) for part in month_day.split("-"))
    return f"{day} de {MONTHS_PT[month - 1]}"


def falls_on(month_day: str, today: date) -> bool:
    """Whether a birthday is today. 29 February is kept on the 28th in other years."""
    if month_day == "02-29" and not _is_leap(today.year):
        return (today.month, today.day) == (2, 28)
    return month_day == f"{today.month:02d}-{today.day:02d}"


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _resolve(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else BASE_DIR / path


SLOTS = ("midnight", "noon")

DEFAULT_MESSAGES = {
    "midnight": ["Parabéns {tag}! 🎂 Muitos anos de vida, grande abraço do Kaya."],
    "noon": ["Ainda ninguém deu os parabéns ao {tag}? 👀🎂"],
}
DEFAULT_WINDOWS = {
    "midnight": ("00:00", "11:00"),
    "noon": ("12:00", "18:00"),
}


def settings(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = ((config.get("chat", {}) or {}).get("birthdays", {}) or {})
    messages = raw.get("messages", {}) or {}
    windows = raw.get("windows", {}) or {}
    return {
        "enabled": bool(raw.get("enabled", False)),
        "store": raw.get("store", "data/birthdays.json"),
        "state": raw.get("state", "data/birthday_state.json"),
        "messages": {slot: [str(line) for line in (messages.get(slot) or DEFAULT_MESSAGES[slot])]
                     for slot in SLOTS},
        "windows": {slot: tuple(windows.get(slot) or DEFAULT_WINDOWS[slot]) for slot in SLOTS},
    }


def _atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class BirthdayStore:
    """``data/birthdays.json``: ``{member: {source: {date, at}}}``.

    Every source is kept, not just the winner, so a later "/aniversario" can
    overrule a mined date without losing where the mined one came from.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def set(self, member: str, month_day: str, source: str) -> None:
        if source not in SOURCES:
            raise ValueError(f"unknown birthday source {source!r}")
        data = self.load()
        data.setdefault(member, {})[source] = {
            "date": month_day, "at": datetime.now().isoformat(timespec="seconds")}
        _atomic_write(self.path, data)


def resolved_dates(store: Dict[str, Dict[str, Dict[str, str]]],
                   members_data: Optional[Dict[str, Any]] = None) -> Dict[str, Tuple[str, str]]:
    """``{member: (MM-DD, source)}`` using the most trusted source each member has."""
    candidates: Dict[str, Dict[str, str]] = defaultdict(dict)
    for member, by_source in (store or {}).items():
        for source, entry in (by_source or {}).items():
            if isinstance(entry, dict) and entry.get("date"):
                candidates[member][source] = entry["date"]
    for member in (members_data or {}).get("members", []) or []:
        birthday = parse_date(str(member.get("birthday") or ""))
        if member.get("name") and birthday:
            candidates[member["name"]]["profile"] = birthday
    resolved = {}
    for member, by_source in candidates.items():
        for source in SOURCES:
            if source in by_source:
                resolved[member] = (by_source[source], source)
                break
    return resolved


def load_dates(config: Dict[str, Any], config_dir: Optional[Path] = None) -> Dict[str, Tuple[str, str]]:
    """Every known birthday, from the store and the profiles. ``{}`` on any failure."""
    cfg = settings(config)
    store = BirthdayStore(_resolve(cfg["store"])).load()
    return resolved_dates(store, _members(config, config_dir))


def _members(config: Dict[str, Any], config_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    members_file = (config.get("data", {}) or {}).get("group_members_file")
    if not members_file:
        return None
    path = Path(members_file)
    if not path.is_absolute():
        path = (config_dir or BASE_DIR) / path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def birthdays_on(dates: Dict[str, Tuple[str, str]], today: date) -> List[str]:
    return sorted(member for member, (month_day, _) in dates.items()
                  if falls_on(month_day, today))


def today_line(dates: Dict[str, Tuple[str, str]], today: date) -> str:
    """The line for the system prompt on somebody's birthday, or ``""``."""
    names = birthdays_on(dates, today)
    if not names:
        return ""
    return f"Hoje faz anos: {', '.join(names)}."


def profile_line(month_day: str) -> str:
    return f"Faz anos a {spoken_date(month_day)}."


# ── the greeting ────────────────────────────────────────────────────────────
class GreetingState:
    """``data/birthday_state.json``: which ``YYYY:member`` greetings have gone out."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> Dict[str, str]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def sent(self, year: int, member: str) -> bool:
        return f"{year}:{member}" in self._load()

    def mark(self, year: int, member: str, where: str) -> None:
        data = self._load()
        data[f"{year}:{member}"] = f"{datetime.now().isoformat(timespec='seconds')} {where}"
        _atomic_write(self.path, data)


def _minutes(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


def due(dates: Dict[str, Tuple[str, str]], state: "GreetingState", now: datetime,
        slot: str, start: str, until: str) -> List[str]:
    """Members whose ``slot`` message is due right now.

    Birthday today, inside the slot's window, not yet sent this year. The window's
    end is the catch-up limit: the Pi may be rebooting at midnight, but a
    "parabéns" at 11:30 reads as an afterthought next to the noon nudge.
    """
    minute = now.hour * 60 + now.minute
    if not _minutes(start) <= minute < _minutes(until):
        return []
    return [member for member in birthdays_on(dates, now.date())
            if not state.sent(now.year, f"{member}:{slot}")]


def roster(config: Dict[str, Any], config_dir: Optional[Path] = None,
           contacts: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    """What the Pi needs to greet: ``{member: {date, aliases, jid}}``.

    ``contacts`` is ``whatsapp_contacts.json`` (id → member). The id to tag is
    the member's ``@lid`` when there is one, since that is what WhatsApp puts in
    a group mention now, else their phone as ``@c.us``. No id means the greeting
    names them instead of tagging them.
    """
    dates = load_dates(config, config_dir)
    members = _members(config, config_dir) or {}
    aliases = {member.get("name"): list(member.get("aliases") or [])
               for member in members.get("members", []) or []}
    ids: Dict[str, List[str]] = defaultdict(list)
    for raw_id, name in (contacts or {}).items():
        ids[name].append(str(raw_id))
    result = {}
    for member, (month_day, _) in dates.items():
        known = ids.get(member, [])
        lid = next((value for value in known if value.endswith("@lid")), "")
        phone = next((value for value in known if value.isdigit()), "")
        result[member] = {
            "date": month_day,
            "aliases": aliases.get(member, []),
            "jid": lid or (f"{phone}@c.us" if phone else ""),
        }
    return result


# ── mining the history ──────────────────────────────────────────────────────
def _fold(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFKD", text.lower())
                   if not unicodedata.combining(char))


@dataclass
class Candidate:
    """A proposed birthday: who, which day, and the evidence behind it."""
    member: str
    month_day: str
    years: List[int] = field(default_factory=list)
    senders: int = 0
    evidence: List[str] = field(default_factory=list)

    @property
    def confidence(self) -> str:
        if len(set(self.years)) >= 2:
            return "high"
        return "medium" if self.senders >= 4 else "low"


def _name_patterns(aliases: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    patterns = {}
    for member, names in aliases.items():
        words = sorted({_fold(name) for name in [member, *names] if len(name) >= 3},
                       key=len, reverse=True)
        if words:
            patterns[member] = re.compile(
                r"\b(" + "|".join(re.escape(word) for word in words) + r")\w*\b")
    return patterns


def mine(rows: Iterable[Tuple[datetime, str, str]], aliases: Dict[str, List[str]],
         min_senders: int = 3) -> List[Candidate]:
    """Birthday candidates from ``(when, sender, text)`` rows, senders already canonical.

    A day counts when at least ``min_senders`` different people congratulate
    somebody. Who: the member most named in those messages (never counting a
    sender naming themselves), with a member thanking people that day as a
    tiebreak. A day with nobody identifiable is dropped rather than guessed.
    Repeated in another year, the same day becomes a high-confidence candidate.
    """
    patterns = _name_patterns(aliases)
    by_day: Dict[date, List[Tuple[str, str]]] = defaultdict(list)
    for when, sender, text in rows:
        by_day[when.date()].append((sender or "", text or ""))

    found: Dict[Tuple[str, str], Candidate] = {}
    for day, messages in sorted(by_day.items()):
        congrats = [(sender, text) for sender, text in messages if _CONGRATS.search(text)]
        senders = {sender for sender, _ in congrats}
        if len(senders) < min_senders:
            continue
        votes: Dict[str, float] = defaultdict(float)
        for sender, text in congrats:
            folded = _fold(text)
            for member, pattern in patterns.items():
                if member != sender and pattern.search(folded):
                    votes[member] += 1
        for sender, text in messages:
            if sender not in senders and _THANKS.search(text) and sender in patterns:
                votes[sender] += 0.5
        if not votes:
            continue
        member = max(votes, key=votes.get)
        month_day = f"{day.month:02d}-{day.day:02d}"
        candidate = found.setdefault((member, month_day), Candidate(member, month_day))
        candidate.years.append(day.year)
        candidate.senders = max(candidate.senders, len(senders))
        candidate.evidence.extend(
            f"{day.isoformat()} {sender}: {text[:100]}" for sender, text in congrats[:4])
    return merge_adjacent(list(found.values()))


def merge_adjacent(candidates: List[Candidate]) -> List[Candidate]:
    """Fold a member's day-after burst into the day before it.

    Bernardo's bursts landed on 09-08 in 2024 and 09-09 in 2025, which is one
    birthday straddling midnight and late wishers. The earlier day wins because
    late wishes happen and early ones do not.
    """
    by_member: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_member[candidate.member].append(candidate)
    merged: List[Candidate] = []
    for member, items in by_member.items():
        items.sort(key=lambda item: item.month_day)
        kept: List[Candidate] = []
        for item in items:
            if kept and _next_day(kept[-1].month_day) == item.month_day:
                kept[-1].years.extend(item.years)
                kept[-1].senders = max(kept[-1].senders, item.senders)
                kept[-1].evidence.extend(item.evidence)
            else:
                kept.append(item)
        merged.extend(kept)
    return sorted(merged, key=lambda item: (item.member, item.month_day))


def _next_day(month_day: str) -> str:
    from datetime import timedelta

    month, day = (int(part) for part in month_day.split("-"))
    following = date(2024, month, day) + timedelta(days=1)
    return f"{following.month:02d}-{following.day:02d}"
