"""Who a roast is about, decided by code (torch-free).

On 2026-11-25 a member asked to be roasted, giving his full name and saying which
member with the same first name he was not. The roast hint took every member named
in the message as a target, and that first name is another member, so the bot was
told to roast both and did, in the third person, the second half from the other
member's stale profile. The names in a self-roast introduce the asker or rule someone out; they are
never extra targets.

- ``is_self_aimed``: the request is to roast the person writing ("roast me",
  "faz-me um roast", "goza comigo"). The target is then the speaker and nobody else.
- ``strip_negated``: for an aimed roast, the words right after a negation ("not the
  X", "não o X", "nem o X") are removed before names are looked up, so "roast the
  Gil, not Rafa" is about Gil.
"""
from __future__ import annotations

import re

# "me"/"mim"/"comigo" as the object of a roast verb, in either order, EN and PT.
# Kept to the object pronoun next to the verb: "roast the guy who sent me this" is
# not self-aimed, so a bare "me" anywhere is not enough.
_VERBS = (r"roast(?:a|ar|ing)?|insult[ae]?|insultar|goza|gozar|destr[oó]i|destruir|"
          r"arrasa|arrasar|humilha|humilhar|burn|cook|flame|faz(?:e|er)?\s+(?:um\s+)?roast")
_SELF = re.compile(
    rf"\b(?:{_VERBS})\s*(?:-|\s)?\s*(?:me|mim|myself)\b"        # roast me, goza-me, insulta-me
    rf"|\b(?:me|mim)\s+(?:um\s+)?(?:{_VERBS})\b"                  # me roast, me faz um roast
    rf"|\bfaz(?:e|er)?\s*-?\s*me\s+(?:um\s+)?roast\b"             # faz-me um roast
    rf"|\b(?:{_VERBS})\b[^.?!]{{0,20}}\b(?:comigo|a\s+mim)\b"      # goza comigo, roast a mim
    rf"|\broast\s+of\s+me\b|\bmy\s+roast\b|\bo\s+meu\s+roast\b",
    re.IGNORECASE,
)

# A negation and up to four words after it: the span a name there is excluded by.
_NEGATED = re.compile(
    r"\b(?:not|nor|n[aã]o|nem)\b(?:\s+(?:sou|é|e|is|am|are|esse|este))?"
    r"(?:\s+[\w'’À-ÿ-]+){1,4}",
    re.IGNORECASE,
)


def is_self_aimed(text: str) -> bool:
    """True when the message asks for a roast of the person who wrote it."""
    return bool(_SELF.search(text or ""))


def strip_negated(text: str) -> str:
    """``text`` with every negated span removed, for looking up who a roast is about."""
    return _NEGATED.sub(" ", text or "")


def self_roast_hint(speaker: str) -> str:
    """The hint for a roast the speaker asked for about themselves."""
    return (f"\n\n(O roast é sobre o {speaker}, que to pediu: é só sobre ele. Os outros "
            "nomes na mensagem dele servem para dizer quem ele é ou quem ele não é, "
            "nunca são alvos. Fala diretamente com ele, na segunda pessoa (tu, ou you "
            "se ele escreveu em inglês), e não fales de mais ninguém do grupo.)")
