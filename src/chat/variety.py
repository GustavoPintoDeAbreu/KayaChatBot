"""Stop the bot telling the same joke about the same person forever.

Asked "roast me", Peter got the same four beats every time across three days and
four separate turns: he lives between Rotterdam and Queijas, he edits other
people's videos, he eats Five Guys, he posts concert videos as though he were a
music critic. Romano got "só manda stickers" and "acha-se analista político por
ler tweets" five times. Rafa got "ginásio próprio" and "o Iñaki no sparring"
five times.

Two causes, both fixed elsewhere and both worth naming here:

* the WhatsApp system prompt was built **once at import**, so the member profiles
  — including the shuffle that was supposed to vary them — were byte-identical
  for the whole uptime, and
* every one of a member's key_facts was in that prompt every turn, so the model
  reliably reached for the same most-roastable two.

This module is the third piece: what the bot has ALREADY said about someone. The
interaction log already records ``reply_members`` per turn, so the material is on
disk; it just was never read back. A turn that is about to talk about Peter gets
told, in one line, what it said about Peter last time and not to do it again.

Only open-ended answers get any of this. A count, a date, "o que faz o Gil" must
come out the same every time it is asked — variety there is just being wrong in
a new way.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.chat.response_utils import truncate_history_line

logger = logging.getLogger(__name__)

# Modes whose answers are opinions rather than facts. `factual` is deliberately
# absent, and so is the CMD_COUNT path that borrows it.
OPEN_ENDED = ("banter", "mixed", "roast")


def is_open_ended(mode: str, command: Optional[str] = None) -> bool:
    """Whether this turn is an opinion the bot may vary, or a fact it may not."""
    if command:
        return False
    return mode in OPEN_ENDED


# Below this many words a reply carries no angle to avoid — "Yo Peter, what's
# up man?" is not material, and putting it in the hint costs tokens and teaches
# the model nothing.
MIN_WORDS = 8

# A reply that named half the group — a ranking, a roster, "top 3 intelectuais" —
# is not material ABOUT any one of them, and quoting it back teaches the model
# nothing except to avoid a list it was not going to write anyway.
MAX_MEMBERS_NAMED = 5


def _flatten(text: str, max_words: int) -> str:
    """One line, no newlines. The log holds multi-line replies and the hint is a
    single parenthetical — a raw count table pasted in wrecks both."""
    return truncate_history_line(" ".join(text.split()), max_words)


def recent_lines_about(members: Sequence[str], rows: Iterable[Dict[str, Any]],
                       limit: int = 3, max_words: int = 25) -> Dict[str, List[str]]:
    """The last few things the bot said about each of ``members``.

    ``rows`` is the interaction log, oldest first — whatever
    ``metrics.load_interactions`` returns. Matching is on ``reply_members``,
    which the engine already writes, rather than on searching the text: the
    engine's own name detection is the same one that decided the reply was about
    that person, so the two agree by construction.

    Only OPEN-ENDED turns count as material. A tally of who said a word names
    every member in the group and is not a joke anyone can repeat; without this
    filter the hint for any member was three count-tables and a greeting.
    """
    wanted = {name.lower(): name for name in members if name}
    found: Dict[str, List[str]] = {name: [] for name in wanted.values()}
    if not wanted:
        return {}

    for row in reversed(list(rows)):
        if row.get("route_mode") not in OPEN_ENDED or row.get("route_command"):
            continue
        reply = (row.get("assistant_response") or "").strip()
        if len(reply.split()) < MIN_WORDS:
            continue
        named = row.get("reply_members") or []
        if len(named) > MAX_MEMBERS_NAMED:
            continue
        for name in named:
            canonical = wanted.get(str(name).lower())
            if not canonical or len(found[canonical]) >= limit:
                continue
            line = _flatten(reply, max_words)
            if line not in found[canonical]:
                found[canonical].append(line)
    return {name: lines for name, lines in found.items() if lines}


def build_hint(said: Dict[str, List[str]]) -> str:
    """The nudge appended to the user turn, or "" when there is nothing to avoid.

    Deliberately shows the actual sentences rather than an abstracted list of
    "angles". The failure was verbatim repetition, and the cheapest thing that
    prevents it is putting the previous words in front of the model — the same
    trick ``previous_bot_replies`` already uses for the last few turns of a chat,
    which is per-chat and per-session and so never caught a roast repeated three
    days later.
    """
    if not said:
        return ""
    blocks = []
    for name, lines in said.items():
        joined = " | ".join(lines)
        blocks.append(f"sobre o {name} já disseste: {joined}")
    return ("\n\n(Não repitas o material que já usaste — " + "; ".join(blocks) +
            ". Diz outra coisa, com outro ângulo e outras palavras. Se já não "
            "houver ângulo novo, sê breve em vez de repetir.)")


def hint_for(members: Sequence[str], rows: Iterable[Dict[str, Any]],
             limit: int = 3) -> str:
    """``recent_lines_about`` and ``build_hint`` in one call. "" on any failure."""
    try:
        return build_hint(recent_lines_about(members, rows, limit=limit))
    except Exception as exc:  # noqa: BLE001 — a hint is never worth a dropped reply
        logger.warning("could not build the variety hint: %s", exc)
        return ""
