"""When a profile fact was learned, and whether it may no longer be true (torch-free).

Profile facts were plain strings, so a job from years ago read as today's: on
2026-11-25 a roast built a whole paragraph on a member's "hardware startup" when he
had long moved on to an engineering job. Each member now carries ``fact_meta``,
``{<fact text>: {"learned_at": <epoch>, "source": "gate"|"his yes"|"manual"}}``,
written wherever a fact is added or reworded (``bio_gate``, ``review_bios.py``).
A fact with no entry is a legacy one: nobody knows when it was learned.

Only facts that go stale count: a job, a home, a relationship (``kind``).

* **Asked** (``is_stale``): one that is undated or older than ``STALE_DAYS`` goes in
  Gustavo's weekly 👤 list ("Still true?"), at most ``RECHECK_PER_WEEK`` a week,
  undated first, then the oldest. His keep dates it again; his drop or rewording
  changes the profile.
* **Marked** (``is_old``): only one with a known date older than ``STALE_DAYS`` is
  shown to the model with ``STALE_MARK``. An undated fact is shown as before: most
  legacy facts are still true, and marking all of them would put every member's job
  in the past tense until the re-check had worked through them.
"""
from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

FACT_META = "fact_meta"
STALE_DAYS = 365
RECHECK_PER_WEEK = 3
STALE_MARK = "(info antiga, pode já não ser verdade)"

_KINDS = (
    ("job", re.compile(
        r"(?i)\b(works?|worked|working|job|employ\w*|startup|start-up|company|empresa|"
        r"trabalh\w*|emprego|engineer\w*|engenheir\w*|developer|programador\w*|editor\w*|"
        r"freelanc\w*|founder|fundador\w*|ceo|cto|manager|consultant|consultor\w*|"
        r"studies|studying|student|estud\w*|intern\w*|est[aá]gio)\b")),
    ("home", re.compile(
        r"(?i)\b(lives?|lived|living|moved|mudou-se|vive|morava|mora|casa em|house in|"
        r"apartment|apartamento|flat in|splits (?:his|her) time|based in)\b")),
    ("relationship", re.compile(
        r"(?i)\b(married|casad\w*|girlfriend|boyfriend|namorad\w*|wife|husband|mulher|marido|"
        r"partner|engaged|noiv\w*|divorc\w*|single|solteir\w*)\b")),
)


def kind(fact: str) -> str:
    """"job", "home", "relationship", or "" for a fact that does not go stale."""
    for name, pattern in _KINDS:
        if pattern.search(fact or ""):
            return name
    return ""


def _now() -> float:
    return time.time()


def learned_at(member: Dict[str, Any], fact: str) -> Optional[float]:
    entry = (member.get(FACT_META) or {}).get(fact) or {}
    try:
        return float(entry["learned_at"])
    except (KeyError, TypeError, ValueError):
        return None


def stamp(member: Dict[str, Any], fact: str, source: str, when: Optional[float] = None) -> None:
    """Record that ``fact`` was learned (or confirmed) now, by ``source``."""
    meta = dict(member.get(FACT_META) or {})
    meta[fact] = {"learned_at": float(when if when is not None else _now()), "source": source}
    member[FACT_META] = meta


def forget(member: Dict[str, Any], fact: str) -> None:
    meta = dict(member.get(FACT_META) or {})
    if meta.pop(fact, None) is not None:
        member[FACT_META] = meta


def is_stale(member: Dict[str, Any], fact: str, now: Optional[float] = None,
             days: int = STALE_DAYS) -> bool:
    """A job/home/relationship fact that is undated or older than ``days``."""
    if not kind(fact):
        return False
    when = learned_at(member, fact)
    return when is None or ((now if now is not None else _now()) - when) > days * 86400


def is_old(member: Dict[str, Any], fact: str, now: Optional[float] = None,
           days: int = STALE_DAYS) -> bool:
    """A job/home/relationship fact with a known date older than ``days``."""
    when = learned_at(member, fact)
    return bool(kind(fact)) and when is not None and \
        ((now if now is not None else _now()) - when) > days * 86400


def mark(member: Dict[str, Any], fact: str, now: Optional[float] = None) -> str:
    """The fact as the model sees it: with ``STALE_MARK`` when it is known to be old."""
    text = str(fact).strip().rstrip(".")
    return f"{text} {STALE_MARK}." if is_old(member, fact, now) else f"{text}."


def stale_facts(members: List[Dict[str, Any]], now: Optional[float] = None,
                limit: int = RECHECK_PER_WEEK) -> List[Dict[str, Any]]:
    """Up to ``limit`` facts to re-check, undated first, then the oldest.

    An unanswered one comes back the next week: nothing about it has changed.
    """
    from src.data.bio_gate import _item_id

    found = []
    for member in members:
        name = member.get("name") or ""
        for fact in member.get("key_facts") or []:
            if not str(fact).strip() or not is_stale(member, fact, now):
                continue
            item_id = _item_id(name, fact)
            when = learned_at(member, fact)
            found.append({"id": item_id, "member": name, "fact": fact,
                          "learned_at": when, "kind": kind(fact)})
    found.sort(key=lambda item: (item["learned_at"] is not None, item["learned_at"] or 0,
                                 item["member"], item["fact"]))
    return found[:limit]


def backfill(members: List[Dict[str, Any]], applied: List[Dict[str, Any]]) -> int:
    """Date the facts the gate applied (``bio_queue.json`` "applied", ``moved_at``).

    The rest stay undated: nobody knows when they were learned. Returns how many
    facts got a date; a fact already dated is left alone."""
    from src.data.bio_gate import same_fact

    dated = 0
    by_name = {m.get("name"): m for m in members}
    for item in applied:
        member = by_name.get(item.get("member"))
        when = item.get("moved_at") or item.get("first_seen")
        if member is None or not when:
            continue
        for fact in member.get("key_facts") or []:
            if learned_at(member, fact) is None and same_fact(fact, item.get("fact") or ""):
                stamp(member, fact, "gate" if item.get("applied") == "auto" else
                      str(item.get("applied") or "gate"), float(when))
                dated += 1
    return dated
