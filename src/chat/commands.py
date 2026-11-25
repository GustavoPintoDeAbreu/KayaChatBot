"""The WhatsApp commands people can use, read from the lab's registry (torch-free).

Until 2026-11-25 the command list lived in three places that drifted: the parser,
a hand-written ``/help`` text on the Pi, and a fixed "Tenho /bug, /feedback,
/aniversario e /clear" line for a wrong command. ``/help`` only worked for the
owner, the wrong-command line never mentioned it, and nothing made a new feature
add itself to either.

Now every repo declares its commands in its ``lab.yaml`` (``commands:``), labwatch's
catalog collects them (``/api/commands``), and this module turns them into the
``/help`` reply and the "did you mean" line for the chat asking:

* ``where``: ``group`` and/or ``dm``; ``who``: ``everyone`` or ``owner``.
* a group sees the group commands only, whoever asks; a member's DM sees the
  everyone DM commands; the owner's DM sees everything.
* an entry has ``cmd`` (a ``/word``, with ``aliases`` and ``args``) or ``say``
  (something said in plain words, e.g. "responde só em áudio").

Each repo has a test that its parser and its ``lab.yaml`` agree, so the list
cannot go stale again.
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

WHERE = ("group", "dm")
WHO = ("everyone", "owner")


@dataclass(frozen=True)
class Command:
    cmd: str = ""
    say: str = ""
    aliases: Tuple[str, ...] = ()
    args: str = ""
    where: Tuple[str, ...] = ("group", "dm")
    who: str = "everyone"
    summary: str = ""
    product: str = ""

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(t.lower() for t in (self.cmd, *self.aliases) if t)


def from_entries(entries: Iterable[Dict[str, Any]], product: str = "") -> List[Command]:
    """Commands from ``lab.yaml`` / ``/api/commands`` rows; malformed rows are skipped."""
    found = []
    for row in entries or []:
        if not isinstance(row, dict):
            continue
        cmd, say = str(row.get("cmd") or "").strip(), str(row.get("say") or "").strip()
        where = tuple(w for w in (row.get("where") or WHERE) if w in WHERE)
        who = row.get("who") if row.get("who") in WHO else "everyone"
        if (cmd and not cmd.startswith("/")) or not (cmd or say) or not where:
            continue
        found.append(Command(cmd=cmd, say=say,
                             aliases=tuple(str(a) for a in row.get("aliases") or ()),
                             args=str(row.get("args") or ""), where=where, who=who,
                             summary=str(row.get("summary") or "").strip(),
                             product=str(row.get("product") or product)))
    return found


def load_lab_yaml(path: Path) -> List[Command]:
    """One repo's commands from its ``lab.yaml``; [] when it cannot be read."""
    import yaml

    try:
        doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return []
    return from_entries(doc.get("commands") or [], str(doc.get("product") or ""))


def merge(*lists: Iterable[Command]) -> List[Command]:
    """The first list wins for a command (or a ``say``) present in several."""
    seen, out = set(), []
    for commands in lists:
        for command in commands:
            key = command.cmd.lower() or command.say.lower()
            if key not in seen:
                seen.add(key)
                out.append(command)
    return out


def visible(commands: Iterable[Command], where: str, is_owner: bool) -> List[Command]:
    """What this chat can use: a group never sees owner commands, whoever asks."""
    owner_ok = is_owner and where == "dm"
    return [c for c in commands if where in c.where and (c.who == "everyone" or owner_ok)]


def _line(command: Command) -> str:
    if command.say:
        return f'- "{command.say}" — {command.summary}'
    head = command.cmd + (f" {command.args}" if command.args else "")
    if command.aliases:
        head += f" ({', '.join(command.aliases)})"
    return f"- {head} — {command.summary}"


def help_text(commands: Iterable[Command], where: str, is_owner: bool) -> str:
    """The ``/help`` reply for this chat: slash commands first, then plain-word ones."""
    usable = visible(commands, where, is_owner)

    def section(rows: List[Command]) -> List[str]:
        return [_line(c) for c in rows if c.cmd] + [_line(c) for c in rows if c.say]

    lines = ["Kaya — o que podes pedir"]
    if where == "group":
        lines += ["", "Aqui no grupo:", *section(usable)]
    elif is_owner:
        everyone = [c for c in usable if c.who == "everyone"]
        mine = [c for c in usable if c.who == "owner"]
        group = [c for c in commands if "group" in c.where and c.who == "everyone"]
        lines += ["", "No grupo:", *section(group)]
        dm_only = [c for c in everyone if c not in group]
        if dm_only:
            lines += ["", "Por DM:", *section(dm_only)]
        if mine:
            lines += ["", "Por DM (só tu):", *section(mine)]
    else:
        lines += ["", "Por DM:", *section(usable)]
    return "\n".join(lines)


def suggest(word: str, commands: Iterable[Command], where: str, is_owner: bool) -> Optional[str]:
    """The closest command this chat can use, or None."""
    word = (word or "").strip().lower()
    by_token = {t: c.cmd for c in visible(commands, where, is_owner) for t in c.tokens}
    close = difflib.get_close_matches(word, list(by_token), n=1, cutoff=0.6)
    return by_token[close[0]] if close else None


def unknown_reply(word: str, commands: Iterable[Command], where: str, is_owner: bool) -> str:
    """The fixed answer to a ``/word`` nobody handles: a guess, and ``/help``."""
    guess = suggest(word, commands, where, is_owner)
    shown = (word or "").split()[0] if word else "esse comando"
    if guess:
        return f"Não conheço {shown}. Querias dizer {guess}? Manda /help para veres tudo o que há."
    return f"Não conheço {shown}. Manda /help para veres os comandos que há aqui."


def tokens(commands: Iterable[Command]) -> set:
    return {t for c in commands for t in c.tokens}
